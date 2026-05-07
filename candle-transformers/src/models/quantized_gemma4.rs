//! Gemma 4 quantized decoder — the QMatMul/GGUF counterpart to
//! [`gemma4::text::TextModel`].
//!
//! **Scope of this port:**
//! - RMSNorm input + post-attention + pre-feedforward + post-feedforward
//! - GQA self-attention with q_norm / k_norm / v_norm
//! - p-RoPE (partial 25%) on full layers, standard RoPE on sliding
//! - SwiGLU MLP (gate + up + down)
//! - KV-cache (Normal for full layers, Rotating for sliding)
//! - **KV-share donor / receiver attention** (canonical Gemma 4 layout —
//!   layers 15..34 share K/V from donors 0..14 of the matching
//!   sliding/full type). Donors stash post-cache `(k, v)` in a shared
//!   store; receivers skip their own k_proj/v_proj/k_norm/RoPE and
//!   read from the donor entry instead.
//! - **LaurelBlock** — low-rank residual augmentation merged into the
//!   attention output via `(attn + laurel) * (1/√2)`.
//! - **layer_scalar** — per-layer learned gain on the residual stream.
//!   Required on Gemma 4 E2B; without it `abs_max` runs away.
//! - **activation sparsity** (Gaussian-topk) — pre-activation gate
//!   thresholding using `mean+std*z` with `z = sqrt(2)*erfinv(2p-1)`.
//! - **PLE side-channel** — per-layer-embed table (or projection alone
//!   on memory-constrained hosts) → `gate(h) → act → * per_layer_input
//!   → projection → norm → +residual`. Wired through every
//!   `DecoderLayer`; falls back to a no-op when the GGUF doesn't
//!   carry `per_layer_model_proj.weight`.
//! - **AltUp (Alternating Updates)** — multi-stream forward with
//!   `altup_num_inputs` parallel hidden streams. Top-level
//!   `altup_projections` build the stack from the input embeddings;
//!   each `DecoderLayer` runs predict → activate (attn+laurel+MLP on
//!   the active stream) → correct over the full stack; top-level
//!   `altup_unembed_projections` collapse the stack back to a single
//!   stream before the final RmsNorm + lm_head. Falls back to the
//!   classic single-stream path when the GGUF doesn't carry the
//!   AltUp tensors.
//!
//! Every Gemma 4 / Gemma 3n auxiliary tower is now ported. The remaining
//! validation step is reference correctness against an actual
//! Ollama-published gemma4:e2b GGUF — the tensor name conventions
//! (especially for AltUp / PLE / Laurel) are llama.cpp-style guesses
//! and may need adjustment when the first real GGUF is loaded.
//!
//! Output won't match Gemma 4 reference exactly with the auxiliary
//! towers off — this scaffold exists to exercise the QMatMul + GGUF
//! integration end-to-end on the WGPU q4_k.pwgsl kernel. Adding the
//! auxiliary towers is straightforward Linear → QMatMul translation;
//! the holdout is verifying each tower against a reference output,
//! which needs an Ollama-published gemma4:e2b GGUF + matching
//! reference run.

use std::borrow::Cow;
use std::sync::Arc;

use candle::quantized::{gguf_file, QMatMul, QTensor};
use candle::{CpuStorage, DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Activation;

/// Diag trace inside `Model::forward` — only compiled in for the
/// chat-pwa wasm build via `wasm-trace-quantized-gemma4`. Every call
/// emits one line to `console.log`. Use to bisect wasm traps that
/// fire mid-forward without a Rust panic stack (e.g. rayon thread-init
/// `unreachable`, OOM, simd128 fallthrough panics).
#[cfg(all(target_arch = "wasm32", feature = "wasm-trace-quantized-gemma4"))]
macro_rules! wasm_trace {
    ($($t:tt)*) => {{
        web_sys::console::log_1(&format!("[qgemma4/trace] {}", format!($($t)*)).into());
    }};
}
#[cfg(not(all(target_arch = "wasm32", feature = "wasm-trace-quantized-gemma4")))]
macro_rules! wasm_trace {
    ($($t:tt)*) => {{
        // no-op on native and on wasm without the trace feature
        let _ = format_args!($($t)*);
    }};
}

/// Row-wise embedding lookup against a quantized table — equivalent to
/// `ggml_get_rows(qtensor, ids)` in llama.cpp. Holds the QTensor as-is
/// and dequantizes only the rows referenced by each forward call.
///
/// Why this exists: `Embedding::new(qtensor.dequantize(...))` materializes
/// the full table in f32. For Gemma 4 E2B's `per_layer_token_embd` the
/// shape is `[262144, num_layers * per_layer]` ≈ 2.0 G elements ≈ 8 GB
/// f32, which overflows wasm32's 4 GB address space (`raw_vec`'s
/// `capacity_overflow`). Keeping the table quantized (~1.1 GB at Q4_K_M)
/// and dequantizing per-row at lookup is what makes Gemma 4 E2B fit
/// in-browser.
///
/// Requires `cols % block_size == 0` so each row aligns to a whole
/// number of quantization blocks (true for Gemma 4 E2B's PLE: 7680 cols
/// / 256 block_size = 30 blocks/row).
#[derive(Debug, Clone)]
struct QEmbedding {
    qtensor: Arc<QTensor>,
    rows: usize,
    cols: usize,
    bytes_per_row: usize,
}

impl QEmbedding {
    fn new(qtensor: QTensor) -> Result<Self> {
        let dims = qtensor.shape().dims();
        if dims.len() != 2 {
            candle::bail!(
                "QEmbedding expects a 2D quantized tensor, got shape {:?}",
                dims
            );
        }
        let rows = dims[0];
        let cols = dims[1];
        let dtype = qtensor.dtype();
        let block_size = dtype.block_size();
        let type_size = dtype.type_size();
        if !cols.is_multiple_of(block_size) {
            candle::bail!(
                "QEmbedding: cols={} not a multiple of block_size={} for dtype {:?}",
                cols,
                block_size,
                dtype
            );
        }
        let blocks_per_row = cols / block_size;
        let bytes_per_row = blocks_per_row * type_size;
        Ok(Self {
            qtensor: Arc::new(qtensor),
            rows,
            cols,
            bytes_per_row,
        })
    }

    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        let target_dev = indices.device().clone();
        let indices_cpu = if target_dev.is_cpu() {
            indices.clone()
        } else {
            indices.to_device(&Device::Cpu)?
        };
        let flat = indices_cpu.flatten_all()?;
        let ids: Vec<u32> = flat.to_vec1::<u32>()?;
        let raw = self.qtensor.data()?;
        let dtype = self.qtensor.dtype();
        let mut out = Vec::<f32>::with_capacity(ids.len() * self.cols);
        for &id in &ids {
            let id_us = id as usize;
            if id_us >= self.rows {
                candle::bail!(
                    "QEmbedding lookup: index {} out of range (rows={})",
                    id,
                    self.rows
                );
            }
            let start = id_us * self.bytes_per_row;
            let end = start + self.bytes_per_row;
            let row_bytes = &raw[start..end];
            let qtype = dtype.from_data(Cow::Borrowed(row_bytes));
            let storage = qtype.dequantize(self.cols)?;
            match storage {
                CpuStorage::F32(v) => out.extend_from_slice(&v),
                _ => candle::bail!("QEmbedding: dequantize returned non-F32 storage"),
            }
        }
        let mut out_shape: Vec<usize> = indices.dims().to_vec();
        out_shape.push(self.cols);
        let result = Tensor::from_vec(out, out_shape, &Device::Cpu)?;
        if target_dev.is_cpu() {
            Ok(result)
        } else {
            result.to_device(&target_dev)
        }
    }
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

/// Per-step shared K/V store. Donor layers write `(k, v)` here after
/// their own cache append; receiver layers read by donor index. Cleared
/// at the start of every `ModelWeights::forward` so a single forward
/// pass owns the store. Tensors are reference-counted under the hood,
/// so the clones are cheap.
type SharedKvStore = Vec<Option<(Tensor, Tensor)>>;

use crate::models::gemma4::config::Gemma4TextConfig;
use crate::quantized_nn::RmsNorm;

// ── Pure-RMS V norm (no learned weight) ─────────────────────────────────────

fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.contiguous()?.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32
        .broadcast_div(&rms)?
        .contiguous()?
        .to_dtype(original_dtype)
}

// ── RoPE tables ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, head_dim: usize, rope_theta: f64, max_seq_len: usize, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (rope_theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self { sin: freqs.sin()?, cos: freqs.cos()? })
    }

    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct ProportionalRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl ProportionalRotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let rope_angles = (partial_rotary_factor * head_dim as f64 / 2.0) as usize;
        let half_dim = head_dim / 2;
        let mut inv_freq_vec = Vec::with_capacity(half_dim);
        for i in 0..rope_angles {
            inv_freq_vec.push(1f32 / (rope_theta as f32).powf((2 * i) as f32 / head_dim as f32));
        }
        inv_freq_vec.extend(std::iter::repeat_n(0f32, half_dim - rope_angles));
        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half_dim), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    /// `icdf_normal(sparsity)`. `0.0` disables sparsity for this layer.
    /// When non-zero, gate goes through a Gaussian top-K threshold:
    /// `mean+std*z`-based ReLU before the activation. Mirrors HF's
    /// `Gemma3nTextMLP.gaussian_topk`.
    sparsity_threshold_z: f64,
}

impl MLP {
    fn forward_inner(&self, xs: &Tensor) -> Result<Tensor> {
        let mut gate = self.gate_proj.forward(xs)?;
        if self.sparsity_threshold_z != 0.0 {
            let original_dtype = gate.dtype();
            let gate_f32 = gate.to_dtype(DType::F32)?;
            let mean = gate_f32.mean_keepdim(D::Minus1)?;
            let var = gate_f32.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
            let std = var.sqrt()?;
            let threshold = (mean + (std * self.sparsity_threshold_z)?)?;
            let sparse = gate_f32.broadcast_sub(&threshold)?;
            let zero = Tensor::zeros_like(&sparse)?;
            let sparse = sparse.maximum(&zero)?;
            gate = sparse.to_dtype(original_dtype)?;
        }
        let lhs = gate.apply(&self.act_fn)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * up)?)
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> { self.forward_inner(xs) }
}

/// Cheap erfinv approximation accurate to ~5e-4 across [-1, 1]. Sufficient
/// for the sparsity threshold which is itself a multiplicative factor
/// applied to a noisy mean+std estimate. Source: Winitzki, "A handy
/// approximation for the error function and its inverse" (2008).
fn erfinv_approx(x: f64) -> f64 {
    if x.abs() >= 1.0 {
        return x.signum() * f64::INFINITY;
    }
    let a = 0.147;
    let ln = (1.0 - x * x).ln();
    let term1 = 2.0 / (std::f64::consts::PI * a) + ln / 2.0;
    let term2 = ln / a;
    let inner = (term1 * term1 - term2).sqrt() - term1;
    x.signum() * inner.sqrt()
}

// ── LaurelBlock (Learned Augmented Residual Layer) ─────────────────────────
//
// Low-rank residual augmentation applied alongside the attention output.
// Mirrors HF `Gemma3nTextLaurelBlock`: project hidden_size → laurel_rank →
// hidden_size, RmsNorm, add to the attention output before the residual.

#[derive(Debug, Clone)]
struct LaurelBlock {
    linear_left: QMatMul,
    linear_right: QMatMul,
    post_laurel_norm: RmsNorm,
}

impl LaurelBlock {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let l = self.linear_left.forward(hidden)?;
        let l = self.linear_right.forward(&l)?;
        let n = self.post_laurel_norm.forward(&l)?;
        hidden + n
    }
}

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: QMatMul,
    /// `None` for receiver layers — they read K from the donor.
    k_proj: Option<QMatMul>,
    /// `None` for receiver layers — they read V from the donor.
    v_proj: Option<QMatMul>,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    /// `None` for receiver layers — RoPE was already applied at the donor.
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    /// Sliding-window size for `flash_attn_decode`'s implicit mask.
    /// `Some(w)` for sliding layers, `None` for full-attention layers.
    /// Equals `cfg.effective_sliding_window()` when `is_sliding`.
    sliding_window: Option<u32>,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    /// `Some(donor_idx)` makes this a receiver — reads `(k, v)` from
    /// `shared_kv_store[donor_idx]`. `None` makes it a donor — runs
    /// its own k_proj/v_proj/RoPE/cache append, then writes to
    /// `shared_kv_store[layer_idx]`.
    donor_layer_idx: Option<usize>,
    layer_idx: usize,
}

impl Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv_store: &mut SharedKvStore,
    ) -> Result<Tensor> {
        let li = self.layer_idx;
        let (b_sz, q_len, _) = xs.dims3()?;
        wasm_trace!("L{li}.attn enter is_sliding={} donor={:?}", self.is_sliding, self.donor_layer_idx);

        // Q is always projected from the layer's own input — receivers
        // share K/V with their donor but keep their own Q. RmsNorm
        // requires contiguous input; the reshape+transpose may yield a
        // non-contiguous tensor (q_len > 1 prefill case).
        wasm_trace!("L{li}.attn q_proj");
        let q = self.q_proj.forward(xs)?;
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        wasm_trace!("L{li}.attn q_norm");
        let q = self.q_norm.forward(&q)?;

        // Q's RoPE comes from THIS layer's RoPE table either way. For
        // donors we'll co-rotate q+k; for receivers we rotate only q
        // and read k_full pre-rotated from the donor.
        let (q, k_full, v_full) = if let Some(donor_idx) = self.donor_layer_idx {
            // Receiver branch: rotate q only. apply_rotary_emb_qkv
            // requires both inputs to share shape, so feed q in twice
            // and discard the second result.
            let q_clone = q.clone();
            let (q_rot, _) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&q, &q_clone, seqlen_offset)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&q, &q_clone, seqlen_offset)?
            };
            let donor = shared_kv_store
                .get(donor_idx)
                .and_then(|x| x.as_ref())
                .ok_or_else(|| candle::Error::Msg(format!(
                    "quantized_gemma4: KV-shared layer {} has no donor entry at index {} \
                     (donor must execute earlier in the same forward pass)",
                    self.layer_idx, donor_idx,
                )))?;
            (q_rot, donor.0.clone(), donor.1.clone())
        } else {
            // Donor branch: project + RoPE-rotate q and k jointly,
            // append k/v to the cache, stash post-append for any
            // receiver later in this forward pass.
            let k_proj = self.k_proj.as_ref().ok_or_else(|| candle::Error::Msg(
                "quantized_gemma4: donor layer is missing its k_proj".into(),
            ))?;
            let v_proj = self.v_proj.as_ref().ok_or_else(|| candle::Error::Msg(
                "quantized_gemma4: donor layer is missing its v_proj".into(),
            ))?;
            let k_norm = self.k_norm.as_ref().ok_or_else(|| candle::Error::Msg(
                "quantized_gemma4: donor layer is missing its k_norm".into(),
            ))?;
            wasm_trace!("L{li}.attn donor: k_proj+v_proj");
            let k = k_proj.forward(xs)?;
            let v = v_proj.forward(xs)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            wasm_trace!("L{li}.attn donor: k_norm");
            let k = k_norm.forward(&k)?;
            wasm_trace!("L{li}.attn donor: v_norm");
            let v = v_norm(&v, self.rms_norm_eps)?;
            // q and k must share the rotated dimension; co-rotate.
            wasm_trace!("L{li}.attn donor: rotary_emb_qkv");
            let (q_rot, k_rot) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
            };
            let v = v.contiguous()?;
            let k = k_rot.contiguous()?;
            wasm_trace!("L{li}.attn donor: kv_cache.append");
            let (k_full, v_full) = match &mut self.kv_cache {
                KvCache::Normal(c) => c.append(&k, &v)?,
                KvCache::Rotating(c) => c.append(&k, &v)?,
            };
            shared_kv_store[self.layer_idx] = Some((k_full.clone(), v_full.clone()));
            (q_rot, k_full, v_full)
        };

        wasm_trace!("L{li}.attn repeat_kv");
        let k = crate::utils::repeat_kv(k_full, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v_full, self.num_kv_groups)?.contiguous()?;

        // Gemma 4 sets pre-softmax scale to 1.0 — q_norm/k_norm produce
        // unit-magnitude queries/keys so the dot-products are already
        // O(1). No divide here.
        //
        // Decode (q_len == 1): use the fused flash-attention kernel so
        // the `[B, H, 1, kv_len]` attn-weights tensor never materializes.
        // The mask is implicit (causal + optional sliding window). For
        // prefill (q_len > 1) we keep the standard 3-step path — the
        // fused kernel is decode-only.
        let attn_output = if q_len == 1 {
            wasm_trace!("L{li}.attn flash_attn_decode");
            candle_nn::flash_attn::flash_attn_decode(&q, &k, &v, self.sliding_window)?
        } else {
            wasm_trace!("L{li}.attn qk_matmul");
            let attn_weights = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
            let mask = if self.is_sliding { sliding_attention_mask } else { attention_mask };
            let attn_weights = match mask {
                Some(m) => attn_weights.broadcast_add(m)?,
                None => attn_weights,
            };
            wasm_trace!("L{li}.attn softmax");
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            wasm_trace!("L{li}.attn av_matmul");
            attn_weights.matmul(&v)?
        };

        wasm_trace!("L{li}.attn o_proj");
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        let result = self.o_proj.forward(&attn_output);
        wasm_trace!("L{li}.attn exit");
        result
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

// ── AltUp (Alternating Updates) ────────────────────────────────────────────
//
// Maintains `altup_num_inputs` parallel hidden streams. Each layer
// runs predict → activate (the layer's main attn+MLP path on the
// active stream) → correct over the full stack, returning the
// corrected stack for the next layer. Mirrors HF
// `Gemma3nTextAltUp`.

#[derive(Debug, Clone)]
struct AltUp {
    correct_output_scale: Tensor,
    correction_coefs: QMatMul,
    prediction_coefs: QMatMul,
    modality_router: QMatMul,
    router_norm: RmsNorm,
    /// `1 / hidden_size` — applied to the modality vector before the
    /// router projection.
    router_input_scale: f64,
    altup_num_inputs: usize,
    altup_active_idx: usize,
}

impl AltUp {
    /// `tanh(modality_router(router_norm(x) * router_input_scale))`.
    fn compute_router_modalities(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.router_norm.forward(x)?;
        let scaled = (normed * self.router_input_scale)?;
        let routed = self.modality_router.forward(&scaled)?;
        routed.tanh()
    }

    /// `predict(stack) -> stack + (predicted_residual_per_input)`.
    /// `stack` shape: `[num_inputs, B, T, hidden]`.
    fn predict(&self, stack: &Tensor) -> Result<Tensor> {
        let active = stack.i(self.altup_active_idx)?;
        let modalities = self.compute_router_modalities(&active)?; // [B, T, num_inputs]

        let coefs = self.prediction_coefs.forward(&modalities)?; // [B, T, num_inputs²]
        let bt = coefs.dims();
        let (b, t) = (bt[0], bt[1]);
        let coefs = coefs
            .reshape((b, t, self.altup_num_inputs, self.altup_num_inputs))?
            .transpose(2, 3)?;

        let permuted = stack.permute((1, 2, 3, 0))?.contiguous()?;
        let predicted = permuted.matmul(&coefs.contiguous()?)?;
        let predicted = predicted.permute((3, 0, 1, 2))?.contiguous()?;
        let result = (predicted + stack)?;
        result.contiguous()
    }

    /// `correct(predictions, activated)`. Predictions: `[num_inputs, B, T, H]`,
    /// activated: `[B, T, H]`. Returns `[num_inputs, B, T, H]`.
    fn correct(&self, predictions: &Tensor, activated: &Tensor) -> Result<Tensor> {
        let modalities = self.compute_router_modalities(activated)?; // [B, T, num_inputs]
        let active_pred = predictions.i(self.altup_active_idx)?;
        let innovation = (activated - &active_pred)?;
        let innovation = innovation
            .unsqueeze(0)?
            .broadcast_as(predictions.shape())?
            .contiguous()?;

        let coefs = self.correction_coefs.forward(&modalities)?; // [B, T, num_inputs]
        let coefs = (coefs + 1.0)?;
        let coefs = coefs.permute((2, 0, 1))?.unsqueeze(3)?.contiguous()?;

        let scaled = innovation.broadcast_mul(&coefs)?;
        let corrected = (scaled + predictions)?;
        corrected.contiguous()
    }

    /// Multiplies the active stream by the learnable per-feature scale.
    fn scale_corrected_output(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.correct_output_scale)
    }
}

// ── PerLayerEmbedding (PLE side-channel) ────────────────────────────────────
//
// Computed once per step from input_ids + inputs_embeds, returns a
// `[B, T, num_layers, hidden_per_layer]` table. DecoderLayer slices
// `[B, T, hidden_per_layer]` out of it via `per_layer_input.i((.., .., layer_idx, ..))`
// and uses it as the side-channel `gate(h) * per_layer_input` →
// projection → norm → +residual block at the end of each layer.

#[derive(Debug, Clone)]
struct PerLayerEmbedding {
    /// `[vocab_per_layer, num_layers * hidden_per_layer]`, lookup
    /// is row-wise on the quantized table (mirrors llama.cpp's
    /// `ggml_get_rows(tok_embd_per_layer, ids)`). Optional — large
    /// publications may skip this layer to save memory; without it
    /// the side-channel is the projection alone (loses per-token
    /// PLE signal).
    embed_tokens_per_layer: Option<QEmbedding>,
    /// `[hidden_size, num_layers * hidden_per_layer]` projection of
    /// the main `inputs_embeds` into the same flat space.
    per_layer_model_projection: QMatMul,
    /// RmsNorm over the inner `hidden_per_layer` axis (weight shape
    /// `[hidden_per_layer]`, NOT `[num_layers * hidden_per_layer]`).
    per_layer_projection_norm: RmsNorm,
    /// `1/√hidden_size`, applied to the projection so the merged
    /// signal stays at unit-ish variance.
    per_layer_projection_scale: f64,
    num_hidden_layers: usize,
    hidden_per_layer: usize,
    /// `1/√2`, applied to the merged projection+table sum so the two
    /// sources don't double-up.
    per_layer_input_scale: f64,
}

impl PerLayerEmbedding {
    fn forward(&self, input_ids: &Tensor, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let proj = self.per_layer_model_projection.forward(inputs_embeds)?;
        let proj = (proj * self.per_layer_projection_scale)?;
        let proj = proj.reshape((b, t, self.num_hidden_layers, self.hidden_per_layer))?;
        let proj = self.per_layer_projection_norm.forward(&proj)?;
        let merged = match &self.embed_tokens_per_layer {
            Some(embed) => {
                // QEmbedding does row-wise dequant on CPU and moves the
                // (small) result back to `input_ids.device()`. We then
                // bring it onto `proj.device()` so the broadcast_add is
                // a same-device op.
                let table = embed.forward(input_ids)?;
                let table = if table.device().same_device(proj.device()) {
                    table
                } else {
                    table.to_device(proj.device())?
                };
                let table = table.reshape((b, t, self.num_hidden_layers, self.hidden_per_layer))?;
                // HF wraps embed_tokens_per_layer in
                // Gemma4TextScaledWordEmbedding(embed_scale = sqrt(hidden_per_layer)).
                let table = (table * (self.hidden_per_layer as f64).sqrt())?;
                (proj.broadcast_add(&table)? * self.per_layer_input_scale)?
            }
            None => (proj * self.per_layer_input_scale)?,
        };
        merged.contiguous()
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    /// Laurel low-rank residual augmentation. `None` when
    /// `cfg.laurel_rank == 0` or `cfg.disable_laurel`.
    laurel: Option<LaurelBlock>,
    /// `1/√2`, applied to `(attn_gated + laurel_out)` so the merged
    /// residual stays at unit-ish variance.
    inv_sqrt_2: f64,
    /// Per-layer learned scalar. `None` when the GGUF doesn't carry
    /// this tensor — without it the residual stream loses its
    /// per-layer trained gain (`abs_max` will run away on Gemma 4
    /// E2B). Read from `blk.{i}.layer_scalar`.
    layer_scalar: Option<Tensor>,
    /// PLE side-channel: gate(h) → act → * per_layer_input → projection
    /// → norm → +residual. Only constructed when the model has PLE
    /// enabled. All three tensors must be present for the block to
    /// run; `None` on any one falls back to no side-channel.
    per_layer_input_gate: Option<QMatMul>,
    per_layer_projection: Option<QMatMul>,
    post_per_layer_input_norm: Option<RmsNorm>,
    /// Activation for the PLE gate (matches `cfg.hidden_activation`).
    per_layer_act: Activation,
    /// AltUp wiring. `None` for non-3n configs; the layer falls back
    /// to the classic single-stream forward.
    altup: Option<AltUp>,
    /// When `true` (the trained default for Gemma 3n), AltUp's
    /// corrected active prediction is multiplied by
    /// `correct_output_scale` before feeding into the PLE gate.
    altup_correct_scale: bool,
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv_store: &mut SharedKvStore,
        per_layer_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        let li = self.self_attn.layer_idx;
        if self.altup.is_some() {
            wasm_trace!("L{li} → forward_altup");
            return self.forward_altup(
                xs,
                attention_mask,
                sliding_attention_mask,
                seqlen_offset,
                shared_kv_store,
                per_layer_input,
            );
        }
        // ── Classic / Gemma 4 path (no AltUp wiring) ────────────────────
        wasm_trace!("L{li} input_layernorm");
        let residual = xs;
        let normed_input = self.input_layernorm.forward(xs)?;
        wasm_trace!("L{li} self_attn.forward");
        let attn = self.self_attn.forward(
            &normed_input,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
            shared_kv_store,
        )?;
        wasm_trace!("L{li} post_attention_layernorm");
        let attn = self.post_attention_layernorm.forward(&attn)?;
        // LaurelBlock merges with the attention output before the
        // first residual add: `attn = (attn + laurel(normed_input)) * inv_sqrt_2`.
        let attn = if let Some(laurel) = &self.laurel {
            wasm_trace!("L{li} laurel");
            let l = laurel.forward(&normed_input)?;
            ((attn + l)? * self.inv_sqrt_2)?
        } else {
            attn
        };
        let xs = (attn + residual)?;
        wasm_trace!("L{li} attn+residual");

        let residual = &xs;
        wasm_trace!("L{li} pre_ff_norm");
        let normed = self.pre_feedforward_layernorm.forward(&xs)?;
        wasm_trace!("L{li} mlp.forward");
        let mlp_out = self.mlp.forward(&normed)?;
        wasm_trace!("L{li} post_ff_norm");
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        let xs = (residual + mlp_out)?;
        wasm_trace!("L{li} mlp+residual");

        // PLE side-channel — gate(h) → act → * per_layer_input → proj
        // → norm → +residual. All four components must be present.
        let xs = if let (Some(gate), Some(proj), Some(norm), Some(per_layer_input)) = (
            self.per_layer_input_gate.as_ref(),
            self.per_layer_projection.as_ref(),
            self.post_per_layer_input_norm.as_ref(),
            per_layer_input,
        ) {
            wasm_trace!("L{li} ple side-channel");
            let r = xs.clone();
            let g = gate.forward(&xs)?;
            let g = g.apply(&self.per_layer_act)?;
            let g = g.broadcast_mul(per_layer_input)?;
            let g = proj.forward(&g)?;
            let g = norm.forward(&g)?;
            (g + r)?
        } else {
            xs
        };

        // Per-layer learned gain (Gemma 4 specifically — initialised to
        // 1.0 and trained per layer). Without this multiply the
        // residual stream `abs_max` runs away on E2B.
        let result = if let Some(scalar) = &self.layer_scalar {
            wasm_trace!("L{li} layer_scalar");
            xs.broadcast_mul(scalar)
        } else {
            Ok(xs)
        };
        wasm_trace!("L{li} forward exit");
        result
    }

    /// AltUp forward — `xs` is `[num_inputs, B, T, hidden]`, returns
    /// the corrected stack of the same shape. Predict → activate
    /// (attn+laurel+MLP on the active stream) → correct → PLE delta
    /// applied to all-but-active streams.
    fn forward_altup(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv_store: &mut SharedKvStore,
        per_layer_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        let altup = self.altup.as_ref().expect("AltUp branch entered without altup");

        let predictions = altup.predict(xs)?;
        let active = predictions.i(altup.altup_active_idx)?;
        let active_norm = self.input_layernorm.forward(&active)?;

        let laurel_out = if let Some(laurel) = &self.laurel {
            Some(laurel.forward(&active_norm)?)
        } else {
            None
        };

        let attn = self.self_attn.forward(
            &active_norm,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
            shared_kv_store,
        )?;
        let attn = self.post_attention_layernorm.forward(&attn)?;
        let attn_gated = (active + attn)?;
        let attn_laurel = match laurel_out {
            Some(l) => ((attn_gated + l)? * self.inv_sqrt_2)?,
            None => attn_gated,
        };

        let attn_norm = self.pre_feedforward_layernorm.forward(&attn_laurel)?;
        let ffw = self.mlp.forward(&attn_norm)?;
        let ffw = self.post_feedforward_layernorm.forward(&ffw)?;
        let attn_ffw_laurel_gated = (attn_laurel + ffw)?;

        let mut corrected = altup.correct(&predictions, &attn_ffw_laurel_gated)?;

        // PLE consumption: apply the side-channel to corrected[active_idx],
        // then add the result back to corrected[i!=active_idx].
        if let (Some(gate), Some(proj), Some(norm), Some(per_layer_input)) = (
            self.per_layer_input_gate.as_ref(),
            self.per_layer_projection.as_ref(),
            self.post_per_layer_input_norm.as_ref(),
            per_layer_input,
        ) {
            let mut first = corrected.i(altup.altup_active_idx)?.contiguous()?;
            if self.altup_correct_scale {
                first = altup.scale_corrected_output(&first)?;
            }
            let first = gate.forward(&first)?;
            let first = first.apply(&self.per_layer_act)?;
            let first = first.broadcast_mul(per_layer_input)?;
            let first = proj.forward(&first)?;
            let first = norm.forward(&first)?;
            let n = altup.altup_num_inputs;
            let mut slices: Vec<Tensor> = Vec::with_capacity(n);
            for i in 0..n {
                if i == altup.altup_active_idx {
                    slices.push(first.zeros_like()?);
                } else {
                    slices.push(first.clone());
                }
            }
            let delta = Tensor::stack(&slices, 0)?;
            corrected = (corrected + delta)?;
        }

        Ok(corrected)
    }

    fn clear_kv_cache(&mut self) { self.self_attn.clear_kv_cache(); }
}

// ── ModelWeights (top-level) ────────────────────────────────────────────────

pub struct ModelWeights {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    /// PLE side-channel computer. `None` when the GGUF doesn't carry
    /// the PLE projection tensors (most current Ollama publications);
    /// without it DecoderLayer's PLE block is a no-op even if
    /// per_layer_input_gate / per_layer_projection / post_per_layer_input_norm
    /// are present at the layer.
    per_layer_embedding: Option<PerLayerEmbedding>,
    /// `√hidden_size` — Gemma input embeddings are scaled by this
    /// before entering the decoder stack.
    embed_scale: f64,
    /// AltUp project / unproject linears. `Some` when AltUp is wired
    /// (cfg.altup_num_inputs > 1 && !cfg.disable_altup) and all
    /// tensors loaded successfully. `altup_projections.len() ==
    /// altup_unembed_projections.len() == altup_num_inputs - 1` —
    /// the active stream uses the original `inputs_embeds` directly.
    altup_projections: Option<Vec<QMatMul>>,
    altup_unembed_projections: Option<Vec<QMatMul>>,
    altup_num_inputs: usize,
    altup_active_idx: usize,
    device: Device,
    dtype: DType,
    cfg: Gemma4TextConfig,
}

impl ModelWeights {
    /// Construct from an open GGUF reader (`gguf_file::Content::read` already
    /// run, file/cursor still open at the tensor data offset).
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        cfg: &Gemma4TextConfig,
    ) -> Result<Self> {
        let dtype = DType::F32; // RoPE tables computed in f32, broadcast to weight dtype later

        // Full-attention layers use `global_head_dim` (Gemma 4 E2B: 512)
        // with p-RoPE 25%; sliding layers use `head_dim` (256) with
        // standard full-rotation RoPE on the SWA base frequency.
        let rotary_global = Arc::new(ProportionalRotaryEmbedding::new(
            dtype,
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            device,
        )?);
        let rotary_local = Arc::new(RotaryEmbedding::new(
            dtype,
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            device,
        )?);

        // Top-level tensors.
        // `token_embd.weight` for Gemma 4 E2B is `[262144, hidden_size]`
        // — at hidden=2048 it dequantizes to ~2.1 GB f32. On WGPU that
        // overflows the 1 GB `max_storage_buffer_binding_size` cap; on
        // wasm32 + CPU it eats half the 4 GB address space and pushes
        // the rest of the model (PLE Q4_K_M ~1.1 GB + 35 layers Q4_K_M
        // ~1.4 GB + activations) over the ceiling, which traps as
        // "unreachable" before any user-visible log.
        //
        // Use `QEmbedding` (the same trick we use for PLE) — keeps
        // token_embd in its quantized form (~302 MB at Q4_K_M) and
        // dequantizes only the rows looked up per forward call, just
        // like llama.cpp's `ggml_get_rows(model.tok_embd, ids)`. The
        // QTensor lives on CPU regardless of `device`; lookup results
        // get moved to whatever device the indices are on.
        let tok_q = ct.tensor(reader, "token_embd.weight", &Device::Cpu)?;
        let embed_tokens = QEmbedding::new(tok_q)?;
        let norm_q = ct.tensor(reader, "output_norm.weight", device)?;
        let norm = RmsNorm::from_qtensor(norm_q, cfg.rms_norm_eps)?;
        let lm_head_q = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => t,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?, // tied
        };
        let lm_head = QMatMul::from_qtensor(lm_head_q)?;

        // Per-layer.
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{layer_idx}");
            let is_sliding = cfg.is_sliding(layer_idx);
            let (head_dim, num_kv_heads) = if is_sliding {
                (cfg.head_dim, cfg.num_key_value_heads)
            } else {
                let global_kv = cfg.num_global_key_value_heads.unwrap_or(cfg.num_key_value_heads);
                (cfg.global_head_dim, global_kv)
            };
            let num_kv_groups = cfg.num_attention_heads / num_kv_heads;

            let donor_layer_idx = cfg.donor_layer_idx_for(layer_idx);
            let is_receiver = donor_layer_idx.is_some();

            let q_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?)?;
            let o_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?)?;

            // Receivers don't carry their own K/V projections in the
            // canonical Gemma 4 GGUF — they reuse the donor's. If the
            // tensor is present it's harmless dead weight; we just
            // skip the load.
            let k_proj = if is_receiver {
                None
            } else {
                Some(QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?,
                )?)
            };
            let v_proj = if is_receiver {
                None
            } else {
                Some(QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?,
                )?)
            };

            let q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                cfg.rms_norm_eps,
            )?;
            let k_norm = if is_receiver {
                None
            } else {
                Some(RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                    cfg.rms_norm_eps,
                )?)
            };

            let kv_cache = if is_sliding {
                KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(2, cfg.effective_sliding_window()))
            } else {
                // Full-attention layers used to pass `cfg.max_position_embeddings`
                // (32768 for Gemma 4 E2B) directly. `KvCache::Cache::append`
                // lazily allocates `[B, num_kv_heads, max_seq_len, head_dim]`
                // on first call — at head_dim=512, num_kv_heads=4, that's
                // ~536 MB per full layer. Gemma 4 E2B has 7 full layers
                // (one every 5 layers across 35 total), so the cumulative
                // allocation is ~3.75 GB and overflows wasm32's 4 GB
                // address space mid-prefill (typically traps inside L14's
                // first kv_cache.append, the 3rd full layer).
                //
                // Cap the *initial* capacity at `KV_CACHE_INITIAL_CAP`
                // (still grows on demand via `Cache::append`'s
                // grow-and-concat path). 4 KB tokens covers the vast
                // majority of single-turn chats; longer contexts pay one
                // realloc per `KV_CACHE_INITIAL_CAP` tokens. Native
                // builds were also wasting memory at the old size; the
                // cap helps everyone.
                const KV_CACHE_INITIAL_CAP: usize = 4096;
                let initial = cfg.max_position_embeddings.min(KV_CACHE_INITIAL_CAP);
                KvCache::Normal(candle_nn::kv_cache::KvCache::new(2, initial))
            };

            let self_attn = Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                num_heads: cfg.num_attention_heads,
                num_kv_heads,
                num_kv_groups,
                head_dim,
                rms_norm_eps: cfg.rms_norm_eps,
                is_sliding,
                sliding_window: if is_sliding {
                    Some(cfg.effective_sliding_window() as u32)
                } else {
                    None
                },
                rotary_emb_global: rotary_global.clone(),
                rotary_emb_local: rotary_local.clone(),
                kv_cache,
                donor_layer_idx,
                layer_idx,
            };

            let gate_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?)?;
            let up_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?)?;
            let down_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?)?;
            let sparsity = cfg.activation_sparsity_at(layer_idx);
            let sparsity_threshold_z = if sparsity > 0.0 && sparsity < 1.0 {
                std::f64::consts::SQRT_2 * erfinv_approx(2.0 * sparsity - 1.0)
            } else {
                0.0
            };
            let mlp = MLP {
                gate_proj,
                up_proj,
                down_proj,
                act_fn: cfg.hidden_activation,
                sparsity_threshold_z,
            };

            let input_layernorm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                cfg.rms_norm_eps,
            )?;
            let post_attention_layernorm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_attention_norm.weight"), device)?,
                cfg.rms_norm_eps,
            )?;
            let pre_feedforward_layernorm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                cfg.rms_norm_eps,
            )?;
            let post_feedforward_layernorm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                cfg.rms_norm_eps,
            )?;

            // LaurelBlock — gated on cfg.laurel_rank > 0 and not
            // disabled. Tensors are optional; if any one is missing
            // (some publications drop laurel), fall back to None.
            let laurel = if cfg.laurel_rank > 0 && !cfg.disable_laurel {
                let mut try_laurel = || -> Result<LaurelBlock> {
                    let linear_left = QMatMul::from_qtensor(
                        ct.tensor(reader, &format!("{prefix}.laurel_l.weight"), device)?,
                    )?;
                    let linear_right = QMatMul::from_qtensor(
                        ct.tensor(reader, &format!("{prefix}.laurel_r.weight"), device)?,
                    )?;
                    let post_laurel_norm = RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{prefix}.post_laurel_norm.weight"), device)?,
                        cfg.rms_norm_eps,
                    )?;
                    Ok(LaurelBlock { linear_left, linear_right, post_laurel_norm })
                };
                try_laurel().ok()
            } else {
                None
            };

            // Per-layer scalar — single f32 [1] tensor. Ollama's
            // gemma4:e2b publishes it as `layer_output_scale.weight`;
            // older llama.cpp Gemma 3n style was `layer_scalar`. Fall
            // back through both before giving up. Optional: publications
            // without this tensor silently skip the multiply (Gemma 4
            // E2B's residual stream needs it, older Gemmas don't).
            let layer_scalar = ct
                .tensor(reader, &format!("{prefix}.layer_output_scale.weight"), device)
                .or_else(|_| ct.tensor(reader, &format!("{prefix}.layer_scalar"), device))
                .ok()
                .and_then(|t| t.dequantize(device).ok());

            // PLE side-channel — three optional QMatMul projections
            // and a norm. Only constructed when the model has PLE
            // enabled in config and all three tensors are present.
            // Falls back to no side-channel on any missing tensor.
            let ple_gate_enabled =
                cfg.hidden_size_per_layer_input.is_some() && !cfg.disable_per_layer_input_gate;
            let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) =
                if ple_gate_enabled {
                    // Ollama's gemma4:e2b uses bare names
                    // (`inp_gate.weight`, `proj.weight`,
                    // `post_norm.weight`); older llama.cpp gemma3n
                    // publications used `per_layer_inp_gate.weight` /
                    // `per_layer_proj.weight` /
                    // `post_per_layer_input_norm.weight`. Try both.
                    let mut try_load = |new_name: &str, legacy_name: &str| -> Result<QTensor> {
                        ct.tensor(reader, &format!("{prefix}.{new_name}"), device).or_else(
                            |_| ct.tensor(reader, &format!("{prefix}.{legacy_name}"), device),
                        )
                    };
                    let mut try_ple = || -> Result<(QMatMul, QMatMul, RmsNorm)> {
                        let gate = QMatMul::from_qtensor(try_load(
                            "inp_gate.weight",
                            "per_layer_inp_gate.weight",
                        )?)?;
                        let proj = QMatMul::from_qtensor(try_load(
                            "proj.weight",
                            "per_layer_proj.weight",
                        )?)?;
                        let norm = RmsNorm::from_qtensor(
                            try_load(
                                "post_norm.weight",
                                "post_per_layer_input_norm.weight",
                            )?,
                            cfg.rms_norm_eps,
                        )?;
                        Ok((gate, proj, norm))
                    };
                    match try_ple() {
                        Ok((g, p, n)) => (Some(g), Some(p), Some(n)),
                        Err(_) => (None, None, None),
                    }
                } else {
                    (None, None, None)
                };

            // AltUp — Gemma 3n auxiliary tower. Construction is
            // fault-tolerant: any missing tensor falls back to no
            // AltUp wiring on this layer.
            let altup = if cfg.altup_num_inputs > 1 && !cfg.disable_altup {
                let mut try_altup = || -> Result<AltUp> {
                    let correct_output_scale = ct
                        .tensor(reader, &format!("{prefix}.altup_correct_output_scale"), device)?
                        .dequantize(device)?;
                    let correction_coefs = QMatMul::from_qtensor(ct.tensor(
                        reader,
                        &format!("{prefix}.altup_correction_coefs.weight"),
                        device,
                    )?)?;
                    let prediction_coefs = QMatMul::from_qtensor(ct.tensor(
                        reader,
                        &format!("{prefix}.altup_prediction_coefs.weight"),
                        device,
                    )?)?;
                    let modality_router = QMatMul::from_qtensor(ct.tensor(
                        reader,
                        &format!("{prefix}.altup_modality_router.weight"),
                        device,
                    )?)?;
                    let router_norm = RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{prefix}.altup_router_norm.weight"), device)?,
                        cfg.rms_norm_eps,
                    )?;
                    Ok(AltUp {
                        correct_output_scale,
                        correction_coefs,
                        prediction_coefs,
                        modality_router,
                        router_norm,
                        router_input_scale: (cfg.hidden_size as f64).powf(-1.0),
                        altup_num_inputs: cfg.altup_num_inputs,
                        altup_active_idx: cfg.altup_active_idx,
                    })
                };
                try_altup().ok()
            } else {
                None
            };

            layers.push(DecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                laurel,
                inv_sqrt_2: (2.0_f64).powf(-0.5),
                layer_scalar,
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
                per_layer_act: cfg.hidden_activation,
                altup,
                altup_correct_scale: cfg.altup_correct_scale,
            });
        }

        // Top-level AltUp project / unproject linears (one per
        // non-active stream). Names follow llama.cpp convention:
        // `altup_proj.{i}.weight` and `altup_unembd_proj.{i}.weight`
        // for i in 0..altup_num_inputs-1.
        let (altup_projections, altup_unembed_projections) =
            if cfg.altup_num_inputs > 1 && !cfg.disable_altup {
                let mut try_top_altup = || -> Result<(Vec<QMatMul>, Vec<QMatMul>)> {
                    let n = cfg.altup_num_inputs - 1;
                    let mut projs = Vec::with_capacity(n);
                    let mut unembeds = Vec::with_capacity(n);
                    for i in 0..n {
                        let proj = QMatMul::from_qtensor(ct.tensor(
                            reader,
                            &format!("altup_proj.{i}.weight"),
                            device,
                        )?)?;
                        let unembed = QMatMul::from_qtensor(ct.tensor(
                            reader,
                            &format!("altup_unembd_proj.{i}.weight"),
                            device,
                        )?)?;
                        projs.push(proj);
                        unembeds.push(unembed);
                    }
                    Ok((projs, unembeds))
                };
                match try_top_altup() {
                    Ok((p, u)) => (Some(p), Some(u)),
                    Err(_) => (None, None),
                }
            } else {
                (None, None)
            };

        // Build the PLE side-channel computer if config enables it.
        // Loads `per_layer_model_proj.weight` (the [hidden, num_layers *
        // hidden_per_layer] projection), `per_layer_projection_norm.weight`,
        // and optionally `embed_tokens_per_layer.weight` (the big VxLxH
        // table; on memory-constrained targets like wasm32 chat-pwa
        // this is skipped and the side-channel falls back to projection-only).
        let per_layer_embedding = if let (Some(hidden_per_layer), false) = (
            cfg.hidden_size_per_layer_input,
            cfg.disable_per_layer_input_gate,
        ) {
            let mut try_ple = || -> Result<PerLayerEmbedding> {
                let total = cfg.num_hidden_layers * hidden_per_layer;
                // Ollama's gemma4:e2b: `per_layer_model_proj.weight`,
                // `per_layer_proj_norm.weight`, `per_layer_token_embd.weight`.
                // Older llama.cpp gemma3n: `_projection`, `_norm`,
                // `embed_tokens_per_layer`. Accept both.
                let per_layer_model_projection = QMatMul::from_qtensor(
                    ct.tensor(reader, "per_layer_model_proj.weight", device)
                        .or_else(|_| {
                            ct.tensor(reader, "per_layer_model_projection.weight", device)
                        })?,
                )?;
                let per_layer_projection_norm = RmsNorm::from_qtensor(
                    ct.tensor(reader, "per_layer_proj_norm.weight", device).or_else(|_| {
                        ct.tensor(reader, "per_layer_projection_norm.weight", device)
                    })?,
                    cfg.rms_norm_eps,
                )?;
                // Per-layer token embed is `[vocab × num_layers × per_layer]`
                // — for Gemma 4 E2B this dequantizes to ~8 GB f32, which
                // overflows wasm32's 4 GB address space. Even on native
                // WGPU it would bust `max_storage_buffer_binding_size`.
                // Keep the QTensor on CPU and use `QEmbedding` for
                // row-wise dequant at lookup time (mirrors llama.cpp's
                // `ggml_get_rows`). Allocation footprint stays at the
                // quantized size (~1.1 GB at Q4_K_M).
                let _ = total; // shape now derived from QTensor inside QEmbedding
                let embed_tokens_per_layer = ct
                    .tensor(reader, "per_layer_token_embd.weight", &Device::Cpu)
                    .or_else(|_| ct.tensor(reader, "embed_tokens_per_layer.weight", &Device::Cpu))
                    .ok()
                    .and_then(|t| QEmbedding::new(t).ok());
                Ok(PerLayerEmbedding {
                    embed_tokens_per_layer,
                    per_layer_model_projection,
                    per_layer_projection_norm,
                    per_layer_projection_scale: (cfg.hidden_size as f64).powf(-0.5),
                    num_hidden_layers: cfg.num_hidden_layers,
                    hidden_per_layer,
                    per_layer_input_scale: (2.0_f64).powf(-0.5),
                })
            };
            try_ple().ok()
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            per_layer_embedding,
            embed_scale: (cfg.hidden_size as f64).sqrt(),
            altup_projections,
            altup_unembed_projections,
            altup_num_inputs: cfg.altup_num_inputs,
            altup_active_idx: cfg.altup_active_idx,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
        })
    }

    /// One forward step. `xs` is `[b, q_len]` token ids; result is
    /// `[b, q_len, vocab_size]` logits.
    pub fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, q_len) = xs.dims2()?;
        wasm_trace!("forward enter b={b_sz} q_len={q_len} seqlen_offset={seqlen_offset}");
        // QEmbedding does the row-wise dequant on CPU and returns the
        // (small) result on `xs.device()`. Move to `self.device` so
        // every downstream op sees a same-device tensor.
        wasm_trace!("embed_tokens.forward (QEmbedding row-wise dequant)");
        let inputs_embeds = self.embed_tokens.forward(xs)?;
        let inputs_embeds = if inputs_embeds.device().same_device(&self.device) {
            inputs_embeds
        } else {
            inputs_embeds.to_device(&self.device)?
        };
        wasm_trace!("inputs_embeds ready dtype={:?}", inputs_embeds.dtype());
        // Gemma scale: input embeddings are multiplied by sqrt(hidden_size).
        let mut hidden = (inputs_embeds.clone() * self.embed_scale)?;
        wasm_trace!("embed_scale done");

        let attention_mask = if q_len <= 1 {
            None
        } else {
            wasm_trace!("prepare_attention_mask");
            Some(prepare_attention_mask(b_sz, q_len, seqlen_offset, &self.device, hidden.dtype())?)
        };
        let sliding_attention_mask = if q_len <= 1 {
            None
        } else {
            wasm_trace!("prepare_sliding_attention_mask");
            Some(prepare_sliding_attention_mask(
                b_sz,
                q_len,
                seqlen_offset,
                self.cfg.effective_sliding_window(),
                &self.device,
                hidden.dtype(),
            )?)
        };

        // PLE side-channel — compute the full `[B, T, num_layers,
        // hidden_per_layer]` table once per step and slice per layer
        // inside the loop.
        let per_layer_table = match &self.per_layer_embedding {
            Some(ple) => {
                wasm_trace!("per_layer_embedding.forward");
                Some(ple.forward(xs, &inputs_embeds)?)
            }
            None => {
                wasm_trace!("per_layer_embedding: None");
                None
            }
        };
        wasm_trace!("per_layer_table done");

        // Per-step shared K/V store for KV-shared receivers. Keyed by
        // donor layer index — donors write here after their cache
        // append, receivers read by donor index later in the same
        // forward pass. Reset every step.
        let mut shared_kv_store: SharedKvStore =
            (0..self.cfg.num_hidden_layers).map(|_| None).collect();

        let altup_active = self.altup_projections.is_some()
            && self.altup_unembed_projections.is_some()
            && self.altup_num_inputs > 1;
        wasm_trace!("altup_active={altup_active} num_layers={}", self.layers.len());

        let hidden = if altup_active {
            // Project the original hidden into a stack of altup_num_inputs
            // streams. Active stream is `hidden`; other streams come
            // from `altup_projections[i]`.
            let projs = self.altup_projections.as_ref().unwrap();
            let mut streams: Vec<Tensor> = Vec::with_capacity(self.altup_num_inputs);
            for i in 0..self.altup_num_inputs {
                if i == self.altup_active_idx {
                    streams.push(hidden.clone());
                } else {
                    let idx = if i < self.altup_active_idx { i } else { i - 1 };
                    streams.push(projs[idx].forward(&hidden)?);
                }
            }
            let mut stack = Tensor::stack(&streams, 0)?.contiguous()?;
            wasm_trace!("altup stack built, entering layer loop");
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                wasm_trace!("altup layer {layer_idx} enter");
                let per_layer_input = match &per_layer_table {
                    Some(t) => Some(t.i((.., .., layer_idx, ..))?.contiguous()?),
                    None => None,
                };
                stack = layer.forward(
                    &stack,
                    attention_mask.as_ref(),
                    sliding_attention_mask.as_ref(),
                    seqlen_offset,
                    &mut shared_kv_store,
                    per_layer_input.as_ref(),
                )?;
                wasm_trace!("altup layer {layer_idx} done");
            }
            // Unproject: `out = active + sum(unembed_projections[i](stack[i]))`
            // for i != active_idx. The trained model treats the active
            // stream as the canonical prediction and the others as
            // residual corrections.
            let unembeds = self.altup_unembed_projections.as_ref().unwrap();
            let active_out = stack.i(self.altup_active_idx)?;
            let mut acc = active_out.contiguous()?;
            for i in 0..self.altup_num_inputs {
                if i == self.altup_active_idx {
                    continue;
                }
                let idx = if i < self.altup_active_idx { i } else { i - 1 };
                let stream = stack.i(i)?;
                let unembedded = unembeds[idx].forward(&stream)?;
                acc = (acc + unembedded)?;
            }
            wasm_trace!("altup unembed done, final norm");
            self.norm.forward(&acc)?
        } else {
            wasm_trace!("classic single-stream layer loop");
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                wasm_trace!("layer {layer_idx} enter");
                let per_layer_input = match &per_layer_table {
                    Some(t) => Some(t.i((.., .., layer_idx, ..))?.contiguous()?),
                    None => None,
                };
                hidden = layer.forward(
                    &hidden,
                    attention_mask.as_ref(),
                    sliding_attention_mask.as_ref(),
                    seqlen_offset,
                    &mut shared_kv_store,
                    per_layer_input.as_ref(),
                )?;
                wasm_trace!("layer {layer_idx} done");
            }
            wasm_trace!("final norm");
            self.norm.forward(&hidden)?
        };
        // Take the last token only for autoregressive sampling — same as
        // gemma4/text.rs.
        wasm_trace!("slice last token");
        let hidden = hidden.i((.., q_len - 1, ..))?.unsqueeze(1)?;
        wasm_trace!("lm_head.forward");
        let logits = self.lm_head.forward(&hidden)?;
        wasm_trace!("lm_head done");
        // Final-logit softcap: tanh(logits / softcap) * softcap.
        let result = match self.cfg.final_logit_softcapping {
            Some(sc) if sc > 0.0 => {
                wasm_trace!("logit softcap (sc={sc})");
                let scaled = (logits / sc)?;
                let capped = scaled.tanh()?;
                capped * sc
            }
            _ => Ok(logits),
        };
        wasm_trace!("forward exit");
        result
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

// ── Attention masks ─────────────────────────────────────────────────────────

fn prepare_attention_mask(
    b_sz: usize,
    seq_len: usize,
    seqlen_offset: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let total_len = seqlen_offset + seq_len;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..total_len).map(move |j| {
            if j > seqlen_offset + i { f32::NEG_INFINITY } else { 0.0 }
        }))
        .collect();
    Tensor::from_slice(&mask, (seq_len, total_len), device)?
        .expand((b_sz, 1, seq_len, total_len))?
        .to_dtype(dtype)
}

fn prepare_sliding_attention_mask(
    b_sz: usize,
    seq_len: usize,
    seqlen_offset: usize,
    sliding_window: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let total_len = seqlen_offset + seq_len;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..total_len).map(move |j| {
            let abs_i = seqlen_offset + i;
            if j > abs_i {
                f32::NEG_INFINITY
            } else if abs_i.saturating_sub(j) >= sliding_window {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        }))
        .collect();
    Tensor::from_slice(&mask, (seq_len, total_len), device)?
        .expand((b_sz, 1, seq_len, total_len))?
        .to_dtype(dtype)
}
