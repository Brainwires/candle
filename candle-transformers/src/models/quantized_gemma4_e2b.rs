//! Faithful Rust port of Ollama's Go gemma4 model architecture
//! (`/Users/nightness/Source/ollama/x/models/gemma4/gemma4.go`),
//! restricted to the gemma4:e2b text-only path. Drops Laurel /
//! AltUp / MoE / activation-sparsity scaffolding present in
//! `quantized_gemma4` (none of which Ollama implements).
//!
//! Translation notes — Ollama Go → Rust mapping:
//! - `Model.Forward`             → `ModelWeights::forward`
//! - `computePLEInputs`          → `PerLayerEmbedding::forward`
//! - `sliceLayerDim`             → inline `t.i((.., .., layer_idx, ..))`
//! - `DecoderLayer.Forward`      → `DecoderLayer::forward`
//! - `Attention.Forward`         → `Attention::forward`
//! - `MLP.Forward`               → `Mlp::forward`
//! - `parseTextConfig` RoPE math → `RotaryEmbedding::new` /
//!   `ProportionalRotaryEmbedding::new`
//!
//! Bug fix vs `quantized_gemma4.rs`: `layer_scalar` is applied **only
//! to full-attention layers**, matching Ollama lines 1203-1204. The
//! older module applied it to every layer, which compounded over the
//! sliding stack.

use std::borrow::Cow;
use std::sync::Arc;

use candle::quantized::{gguf_file, QMatMul, QTensor};
use candle::{CpuStorage, DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Activation;

use super::gemma4::config::Gemma4TextConfig;
use crate::quantized_nn::RmsNorm;

// ─────────────────────────────────────────────────────────────────────
// Bisect dump infrastructure (env-var gated, native only).
// Mirrors the pattern in quantized_gemma4.rs so the same diff harness
// (gemma4_bisect_diff) can compare the two ports against each other.
// ─────────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
mod bisect_dump {
    use super::*;
    use std::io::Write;

    pub fn enabled() -> Option<std::path::PathBuf> {
        std::env::var_os("CANDLE_BISECT_DUMP_DIR").map(std::path::PathBuf::from)
    }

    pub fn dump(
        dir: &std::path::Path,
        step: usize,
        layer: usize,
        label: &str,
        tensor: &Tensor,
    ) -> Result<()> {
        std::fs::create_dir_all(dir)
            .map_err(|e| candle::Error::Msg(format!("bisect mkdir: {e}")))?;
        let f32_t = tensor.contiguous()?.to_dtype(DType::F32)?;
        let dims = f32_t.dims().to_vec();
        let flat: Vec<f32> = f32_t.flatten_all()?.to_vec1::<f32>()?;
        let path = dir.join(format!("step{step:04}_layer{layer:03}_{label}.bin"));
        let mut f = std::fs::File::create(&path)
            .map_err(|e| candle::Error::Msg(format!("bisect open: {e}")))?;
        f.write_all(b"BST1")
            .map_err(|e| candle::Error::Msg(format!("bisect write magic: {e}")))?;
        f.write_all(&[dims.len() as u8])
            .map_err(|e| candle::Error::Msg(format!("bisect write rank: {e}")))?;
        for d in &dims {
            f.write_all(&(*d as u64).to_le_bytes())
                .map_err(|e| candle::Error::Msg(format!("bisect write dim: {e}")))?;
        }
        f.write_all(&[0u8])
            .map_err(|e| candle::Error::Msg(format!("bisect write dtype: {e}")))?;
        let mut buf: Vec<u8> = Vec::with_capacity(flat.len() * 4);
        for v in &flat {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&buf)
            .map_err(|e| candle::Error::Msg(format!("bisect write data: {e}")))?;
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
mod bisect_dump {
    use super::*;
    pub fn enabled() -> Option<std::path::PathBuf> {
        None
    }
    pub fn dump(
        _dir: &std::path::Path,
        _step: usize,
        _layer: usize,
        _label: &str,
        _tensor: &Tensor,
    ) -> Result<()> {
        Ok(())
    }
}

macro_rules! bdump {
    ($dir:expr, $step:expr, $layer:expr, $label:expr, $tensor:expr) => {{
        if let Some(dir) = &$dir {
            let _ = bisect_dump::dump(dir, $step, $layer, $label, $tensor);
        }
    }};
}

// ─────────────────────────────────────────────────────────────────────
// 1. RmsNorm wrapper (Ollama: mlx.RMSNormFn)
// ─────────────────────────────────────────────────────────────────────
//
// Ollama uses `mlx.RMSNormFn(x, weight, eps)` everywhere; with
// `weight = nil` it computes a pure RMSNorm (no learned scale).
//
// candle's `quantized_nn::RmsNorm` is the learned-scale variant; for
// the unweighted case (V-norm in attention, `per_layer_projection_norm`
// when applied to a [..., hidden_per_layer] tensor) we use a free
// function `rms_norm_no_weight`.

fn rms_norm_no_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = x.dtype();
    let xf = x.contiguous()?.to_dtype(DType::F32)?;
    let mean_sq = xf.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    xf.broadcast_div(&rms)?
        .contiguous()?
        .to_dtype(original_dtype)
}

// ─────────────────────────────────────────────────────────────────────
// 2. RotaryEmbedding (Ollama: RoPEWithFreqs)
// ─────────────────────────────────────────────────────────────────────
//
// Two flavors:
// - `RotaryEmbedding`: standard full-rotation RoPE used by sliding
//   layers (Ollama lines 1213-1215, 1234 with rope_freqs == nil).
// - `ProportionalRotaryEmbedding`: partial-rotary RoPE used by full
//   layers when `rope_parameters.full_attention.partial_rotary_factor`
//   is set (Ollama lines 411-430). The non-rotated tail of the head
//   uses 0 frequency so the cos/sin pair is (1, 0) — identity rotation.

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (rope_theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
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
        // Mirrors Ollama lines 417-430: `dims=global_head_dim`,
        // `half_dim = dims/2`, `rope_angles = partial * dims / 2`.
        // Non-rotated tail uses a sentinel large value so MLX's
        // reciprocal collapses to ~0 → cos=1, sin=0 (identity).
        // Here we just store 0 directly and skip the inversion since we
        // build inv_freq ourselves.
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

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ─────────────────────────────────────────────────────────────────────
// QEmbedding — row-wise dequantize for huge embedding tables
// ─────────────────────────────────────────────────────────────────────
//
// Same trick as `quantized_gemma4.rs::QEmbedding`: keeps `token_embd`
// and `per_layer_token_embd` quantized on CPU, dequantizes only the
// rows looked up at forward time. Required to fit Gemma 4 E2B's
// per-layer-token-embed table inside wasm32's 4 GB linear memory.

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
        let storage_size = qtensor.storage_size_in_bytes() as u64;
        let expected_size = (rows as u64) * (bytes_per_row as u64);
        if storage_size != expected_size {
            candle::bail!(
                "QEmbedding storage truncation: dtype={dtype:?} shape=[{rows},{cols}] \
                 expected_bytes={expected_size} actual_bytes={storage_size}"
            );
        }
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
            if end > raw.len() {
                candle::bail!(
                    "QEmbedding row OOB: id={id} start={start} end={end} \
                     bytes_per_row={} rows={} raw.len={}",
                    self.bytes_per_row,
                    self.rows,
                    raw.len(),
                );
            }
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

// ─────────────────────────────────────────────────────────────────────
// KV cache (cat-based, with optional sliding-window trim)
// ─────────────────────────────────────────────────────────────────────
//
// We deliberately avoid `candle_nn::kv_cache::KvCache` (slice_set-based)
// in favor of `ConcatKvCache` (cat-based) for the same wgpu coherence
// reason as the legacy module: under Chrome/Dawn → Tint → MSL the
// `copy2d` writes are not visible to subsequent reads, manifesting as
// duplicate-token decode bugs.

#[derive(Debug, Clone)]
struct KvCache {
    inner: candle_nn::kv_cache::ConcatKvCache,
    sliding_window: Option<usize>,
}

impl KvCache {
    fn new(sliding_window: Option<usize>) -> Self {
        Self {
            inner: candle_nn::kv_cache::ConcatKvCache::new(2),
            sliding_window,
        }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k_full, v_full) = self.inner.append(k, v)?;
        if let Some(window) = self.sliding_window {
            let len = k_full.dim(2)?;
            if len > window {
                let start = len - window;
                let k_trim = k_full.narrow(2, start, window)?.contiguous()?;
                let v_trim = v_full.narrow(2, start, window)?.contiguous()?;
                self.inner.reset();
                let (k_full, v_full) = self.inner.append(&k_trim, &v_trim)?;
                return Ok((k_full, v_full));
            }
        }
        Ok((k_full, v_full))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Per-step shared K/V store. Donor layers store post-cache (k, v) here;
/// receiver layers (KV-share) read by donor index. Cleared every forward.
type SharedKvStore = Vec<Option<(Tensor, Tensor)>>;

// ─────────────────────────────────────────────────────────────────────
// 3. PerLayerEmbedding (Ollama: Model.computePLEInputs, lines 1120-1148)
// ─────────────────────────────────────────────────────────────────────
//
// Computed once per forward, returns a `[B, L, num_layers, hidden_per_layer]`
// table. DecoderLayer slices its `[B, L, hidden_per_layer]` per layer.
//
// Algorithm (Ollama lines 1122-1147):
// 1. ple_emb = embed_tokens_per_layer(input_ids) * sqrt(hidden_per_layer)
// 2. ple_emb reshape → [B, L, num_layers, hidden_per_layer]
// 3. ple_proj = per_layer_model_projection(h) * (1/sqrt(hidden_size))
// 4. ple_proj reshape → [B, L, num_layers, hidden_per_layer]
// 5. ple_proj = RMSNormFn(ple_proj, per_layer_proj_norm_weight, eps)
//      → applied over the inner (hidden_per_layer) axis
// 6. combined = (ple_proj + ple_emb) * 2^(-0.5)

#[derive(Debug, Clone)]
struct PerLayerEmbedding {
    /// Optional — when absent (memory-constrained hosts) the side-channel
    /// is projection-only. Ollama's mainline always has it; the chat-pwa
    /// wasm path skips the table when it's too large for 4 GB linear mem.
    embed_tokens_per_layer: Option<QEmbedding>,
    per_layer_model_projection: QMatMul,
    per_layer_projection_norm: RmsNorm,
    per_layer_projection_scale: f64, // 1/sqrt(hidden_size)
    per_layer_input_scale: f64,      // 2^(-0.5)
    embed_scale_per_layer: f64,      // sqrt(hidden_per_layer)
    num_hidden_layers: usize,
    hidden_per_layer: usize,
}

impl PerLayerEmbedding {
    fn forward(&self, input_ids: &Tensor, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;

        // Hidden-state projection.
        let proj = self.per_layer_model_projection.forward(inputs_embeds)?;
        let proj = (proj * self.per_layer_projection_scale)?;
        let proj = proj.reshape((b, t, self.num_hidden_layers, self.hidden_per_layer))?;
        let proj = self.per_layer_projection_norm.forward(&proj)?;

        // Token embed lookup + sqrt(hidden_per_layer) scale.
        let merged = match &self.embed_tokens_per_layer {
            Some(embed) => {
                let table = embed.forward(input_ids)?;
                let table = if table.device().same_device(proj.device()) {
                    table
                } else {
                    table.to_device(proj.device())?
                };
                let table =
                    table.reshape((b, t, self.num_hidden_layers, self.hidden_per_layer))?;
                let table = (table * self.embed_scale_per_layer)?;
                (proj.broadcast_add(&table)? * self.per_layer_input_scale)?
            }
            None => (proj * self.per_layer_input_scale)?,
        };
        merged.contiguous()
    }
}

// ─────────────────────────────────────────────────────────────────────
// 4. Attention (Ollama: Attention struct + Forward, lines 110-122 + 1210-1332)
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: QMatMul,
    /// `None` for KV-share receiver layers — they read K from the donor.
    k_proj: Option<QMatMul>,
    /// `None` for receivers; also `None` for K=V full-attention layers
    /// (Ollama lines 1249-1256: `value_states = key_states` raw).
    v_proj: Option<QMatMul>,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    /// `None` for receivers (RoPE'd K is read from donor).
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    rotary_global: Arc<ProportionalRotaryEmbedding>,
    rotary_local: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    /// `Some(donor_idx)` makes this a receiver. `None` makes this a
    /// donor (it owns its own K/V projections + cache).
    donor_layer_idx: Option<usize>,
    layer_idx: usize,
    /// `true` when this layer's K is reused as V (no separate v_proj).
    /// Mirrors Ollama's `cfg.AttentionKEqV && !isSliding && a.VProj == nil`.
    k_eq_v: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv_store: &mut SharedKvStore,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let bdir = bisect_dump::enabled();
        let step = seqlen_offset;
        let li = self.layer_idx;

        // Q path — always projected from this layer's input (Ollama 1223-1228).
        let q = self.q_proj.forward(xs)?;
        bdump!(bdir, step, li, "A01_q_proj", &q);
        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = self.q_norm.forward(&q)?;
        bdump!(bdir, step, li, "A02_q_norm", &q);

        // K/V path — donor vs receiver split (Ollama 1236-1276).
        let (q, k_full, v_full) = if let Some(donor_idx) = self.donor_layer_idx {
            // Receiver: rotate Q only, read K/V verbatim from donor.
            let q_clone = q.clone();
            let (q_rot, _) = if self.is_sliding {
                self.rotary_local.apply(&q, &q_clone, seqlen_offset)?
            } else {
                self.rotary_global.apply(&q, &q_clone, seqlen_offset)?
            };
            let donor = shared_kv_store
                .get(donor_idx)
                .and_then(|x| x.as_ref())
                .ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "quantized_gemma4_e2b: KV-shared layer {} has no donor entry at \
                         index {} (donor must run earlier in the same forward pass)",
                        self.layer_idx, donor_idx,
                    ))
                })?;
            (q_rot, donor.0.clone(), donor.1.clone())
        } else {
            // Donor: own K/V projections, RoPE on Q+K, V-norm, cache append.
            let k_proj = self.k_proj.as_ref().ok_or_else(|| {
                candle::Error::Msg("quantized_gemma4_e2b: donor missing k_proj".into())
            })?;
            let k_norm = self.k_norm.as_ref().ok_or_else(|| {
                candle::Error::Msg("quantized_gemma4_e2b: donor missing k_norm".into())
            })?;

            let k = k_proj.forward(xs)?;
            bdump!(bdir, step, li, "A03_k_proj", &k);
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            // V path: either project then reshape, or alias K (raw,
            // pre-norm) when this is a K=V full-attention layer
            // (Ollama lines 1249-1256). `v` shape mirrors `k`.
            let v_raw = if let Some(v_proj) = self.v_proj.as_ref() {
                let v = v_proj.forward(xs)?;
                bdump!(bdir, step, li, "A04_v_proj", &v);
                v.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?
            } else {
                debug_assert!(self.k_eq_v, "v_proj is None but k_eq_v=false");
                k.clone()
            };

            let k = k_norm.forward(&k)?;
            bdump!(bdir, step, li, "A05_k_norm", &k);

            // RoPE on Q+K (Ollama 1234, 1262) — must share head_dim and
            // candle's `rope` op requires same shape, so we co-rotate.
            // Q has num_heads heads, K has num_kv_heads heads, so we
            // rotate them separately by feeding K paired with itself.
            let (q_rot, _) = if self.is_sliding {
                self.rotary_local.apply(&q, &q, seqlen_offset)?
            } else {
                self.rotary_global.apply(&q, &q, seqlen_offset)?
            };
            let k_clone = k.clone();
            let (k_rot, _) = if self.is_sliding {
                self.rotary_local.apply(&k, &k_clone, seqlen_offset)?
            } else {
                self.rotary_global.apply(&k, &k_clone, seqlen_offset)?
            };
            bdump!(bdir, step, li, "A07_q_rope", &q_rot);
            bdump!(bdir, step, li, "A08_k_rope", &k_rot);

            // V-norm (Ollama line 1265) — pure RMS, no learned weight.
            let v = rms_norm_no_weight(&v_raw, self.rms_norm_eps)?;
            bdump!(bdir, step, li, "A06_v_norm", &v);

            // Cache append (Ollama 1267-1276).
            let v = v.contiguous()?;
            let k = k_rot.contiguous()?;
            let (k_full, v_full) = self.kv_cache.append(&k, &v)?;
            bdump!(bdir, step, li, "A09_k_cache", &k_full);
            bdump!(bdir, step, li, "A10_v_cache", &v_full);
            shared_kv_store[self.layer_idx] = Some((k_full.clone(), v_full.clone()));
            (q_rot, k_full, v_full)
        };

        // GQA expansion: repeat KV heads up to num_attention_heads.
        let k = crate::utils::repeat_kv(k_full, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v_full, self.num_kv_groups)?.contiguous()?;
        bdump!(bdir, step, li, "A11_k_after_repeat", &k);
        bdump!(bdir, step, li, "A12_v_after_repeat", &v);

        // SDPA. scale = 1.0 always (Ollama 394-395: `SlidingScale = 1.0`,
        // `FullScale = 1.0`). Q-norm + K-norm produce unit-magnitude.
        let attn_weights = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        bdump!(bdir, step, li, "A13_qk_scores", &attn_weights);
        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };
        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        bdump!(bdir, step, li, "A14_qk_masked", &attn_weights);
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        bdump!(bdir, step, li, "A15_softmax", &attn_weights);
        let attn_output = attn_weights.matmul(&v)?;
        bdump!(bdir, step, li, "A16_attn_v", &attn_output);

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        let out = self.o_proj.forward(&attn_output)?;
        bdump!(bdir, step, li, "A17_o_proj", &out);
        Ok(out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────
// 5. MLP (Ollama: MLP struct + Forward, lines 124-129 + 1334-1338)
// ─────────────────────────────────────────────────────────────────────
//
// SwiGLU / GeGLU: `down(act(gate(x)) * up(x))`. No sparsity, no MoE.
// Ollama uses `mlx.GeGLU(gate, up)` which expands to GELU(gate)*up; we
// honor `cfg.hidden_activation` here so it can swap between GELU/SiLU
// without touching the port.

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let lhs = gate.apply(&self.act_fn)?;
        let up = self.up_proj.forward(xs)?;
        let mul = (lhs * up)?;
        self.down_proj.forward(&mul)
    }
}

// ─────────────────────────────────────────────────────────────────────
// 6. DecoderLayer (Ollama: DecoderLayer struct + Forward, lines 285-324 + 1160-1208)
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    /// PLE side-channel — gated GeGLU+projection+norm+residual block
    /// (Ollama lines 1194-1200). All three Optional fields must be
    /// present for the block to fire.
    per_layer_input_gate: Option<QMatMul>,
    per_layer_projection: Option<QMatMul>,
    post_per_layer_input_norm: Option<RmsNorm>,
    /// Per-layer learned scalar applied **only to full-attention layers**
    /// (Ollama lines 1203-1205). When `None`, no scaling is done.
    layer_scalar: Option<Tensor>,
    /// True for full-attention layers; gates `layer_scalar` application.
    is_full_attention: bool,
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
        let bdir = bisect_dump::enabled();
        let layer_idx = self.self_attn.layer_idx;
        let step = seqlen_offset;

        // Block 1: attention (Ollama 1161-1164).
        let residual = xs;
        bdump!(bdir, step, layer_idx, "00_input", xs);
        let normed_input = self.input_layernorm.forward(xs)?;
        bdump!(bdir, step, layer_idx, "01_input_layernorm", &normed_input);
        let attn = self.self_attn.forward(
            &normed_input,
            attention_mask,
            sliding_attention_mask,
            seqlen_offset,
            shared_kv_store,
        )?;
        bdump!(bdir, step, layer_idx, "02_self_attn", &attn);
        let attn = self.post_attention_layernorm.forward(&attn)?;
        bdump!(bdir, step, layer_idx, "03_post_attn_norm", &attn);
        let h = (residual + attn)?;
        bdump!(bdir, step, layer_idx, "05_after_residual_1", &h);

        // Block 2: MLP (Ollama 1187-1191).
        let residual = &h;
        let normed = self.pre_feedforward_layernorm.forward(&h)?;
        bdump!(bdir, step, layer_idx, "06_pre_ff_norm", &normed);
        let mlp_out = self.mlp.forward(&normed)?;
        bdump!(bdir, step, layer_idx, "07_mlp", &mlp_out);
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        bdump!(bdir, step, layer_idx, "08_post_ff_norm", &mlp_out);
        let h = (residual + mlp_out)?;
        bdump!(bdir, step, layer_idx, "09_after_residual_2", &h);

        // Block 3: PLE injection (Ollama 1194-1200).
        // Ollama uses `mlx.GeGLU(gate(h), pleInput)` which is
        // `gelu(gate(h)) * pleInput`. We honor cfg.hidden_activation in
        // place of hard-coded GELU so SiLU-trained variants work too.
        let h = if let (Some(gate), Some(proj), Some(norm), Some(ple_in)) = (
            self.per_layer_input_gate.as_ref(),
            self.per_layer_projection.as_ref(),
            self.post_per_layer_input_norm.as_ref(),
            per_layer_input,
        ) {
            let r = h.clone();
            let g = gate.forward(&h)?;
            let g = g.apply(&self.mlp.act_fn)?;
            let g = g.broadcast_mul(ple_in)?;
            let g = proj.forward(&g)?;
            let g = norm.forward(&g)?;
            (r + g)?
        } else {
            h
        };
        bdump!(bdir, step, layer_idx, "10_after_ple", &h);

        // Block 4: layer_scalar — Ollama 1203-1205 applies it
        // unconditionally based on tensor presence; we restrict to
        // full-attention layers per the bug fix in the plan.
        let out = if self.is_full_attention {
            if let Some(scalar) = &self.layer_scalar {
                h.broadcast_mul(scalar)?
            } else {
                h
            }
        } else {
            h
        };
        bdump!(bdir, step, layer_idx, "11_layer_out", &out);
        Ok(out)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ─────────────────────────────────────────────────────────────────────
// 7. ModelWeights (Ollama: Model struct + Forward, lines 326-346 + 1019-1080)
// ─────────────────────────────────────────────────────────────────────

pub struct ModelWeights {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    per_layer_embedding: Option<PerLayerEmbedding>,
    embed_scale: f64, // sqrt(hidden_size) — Ollama EmbedScale
    device: Device,
    cfg: Gemma4TextConfig,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        cfg: &Gemma4TextConfig,
    ) -> Result<Self> {
        let dtype = DType::F32;

        // Per Ollama lines 397-431: full layers may use partial-rotary
        // proportional RoPE; sliding layers always use full-rotation
        // RoPE on the SWA base frequency.
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
        let tok_q = ct.tensor(reader, "token_embd.weight", &Device::Cpu)?;
        let embed_tokens = QEmbedding::new(tok_q)?;
        let norm_q = ct.tensor(reader, "output_norm.weight", device)?;
        let norm = RmsNorm::from_qtensor(norm_q, cfg.rms_norm_eps)?;
        let lm_head_q = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => t,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
        };
        let lm_head = QMatMul::from_qtensor(lm_head_q)?;

        // Decoder layers.
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{layer_idx}");
            let is_sliding = cfg.is_sliding(layer_idx);
            let is_full_attention = !is_sliding;

            // Head dim & KV head count are layer-type dependent
            // (Ollama 1212-1221, 1239-1242).
            let (head_dim, mut num_kv_heads) = if is_sliding {
                (cfg.head_dim, cfg.num_key_value_heads)
            } else {
                let global_kv = cfg
                    .num_global_key_value_heads
                    .unwrap_or(cfg.num_key_value_heads);
                (cfg.global_head_dim, global_kv)
            };

            let donor_layer_idx = cfg.donor_layer_idx_for(layer_idx);
            let is_receiver = donor_layer_idx.is_some();

            let q_proj = QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?,
            )?;
            let o_proj = QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?,
            )?;

            // K projection — receivers omit; donors require.
            let k_proj = if is_receiver {
                None
            } else {
                Some(QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?,
                )?)
            };

            // V projection — `None` means K=V (donor full layer with
            // attention_k_eq_v) OR receiver. Ollama (998) treats a
            // missing v_proj on full layers as the K=V configuration.
            let v_proj_loaded = if is_receiver {
                None
            } else {
                ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)
                    .ok()
            };
            let k_eq_v = !is_receiver && v_proj_loaded.is_none() && is_full_attention;
            // If donor is a K=V full layer, override num_kv_heads to
            // num_global_key_value_heads (already done above).
            // If a v_proj IS present but the per-layer K head count
            // differs (rare), prefer num_kv_heads as set.
            let _ = &mut num_kv_heads; // kept for symmetry with Ollama's branch

            let v_proj = match v_proj_loaded {
                Some(t) => Some(QMatMul::from_qtensor(t)?),
                None => None,
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
                KvCache::new(Some(cfg.effective_sliding_window()))
            } else {
                KvCache::new(None)
            };

            let num_kv_groups = cfg.num_attention_heads / num_kv_heads;
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
                rotary_global: rotary_global.clone(),
                rotary_local: rotary_local.clone(),
                kv_cache,
                donor_layer_idx,
                layer_idx,
                k_eq_v,
            };

            // MLP — pure SwiGLU / GeGLU, no sparsity (Ollama 1334-1338).
            let gate_proj = QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?,
            )?;
            let up_proj = QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?,
            )?;
            let down_proj = QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?,
            )?;
            let mlp = Mlp {
                gate_proj,
                up_proj,
                down_proj,
                act_fn: cfg.hidden_activation,
            };

            // Norms (Ollama 699-710).
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

            // PLE side-channel weights (Ollama 964-974). All three
            // tensors are required; if any one is missing, the block
            // is skipped silently.
            let ple_enabled = cfg.hidden_size_per_layer_input.is_some();
            let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) =
                if ple_enabled {
                    // Tensor name compatibility: Ollama gemma4:e2b uses
                    // bare names; older llama.cpp gemma3n style adds
                    // `per_layer_` prefix. Try both.
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
                            try_load("post_norm.weight", "post_per_layer_input_norm.weight")?,
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

            // Layer scalar — single [1] tensor. Ollama publishes as
            // `layer_output_scale.weight`; older Gemma 3n style was
            // `layer_scalar`. Try both. Optional.
            let layer_scalar = ct
                .tensor(reader, &format!("{prefix}.layer_output_scale.weight"), device)
                .or_else(|_| ct.tensor(reader, &format!("{prefix}.layer_scalar"), device))
                .ok()
                .and_then(|t| t.dequantize(device).ok());

            layers.push(DecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                per_layer_input_gate,
                per_layer_projection,
                post_per_layer_input_norm,
                layer_scalar,
                is_full_attention,
            });
        }

        // Top-level PLE inputs computer (Ollama 660-677).
        let per_layer_embedding = if let Some(hidden_per_layer) = cfg.hidden_size_per_layer_input {
            let mut try_ple = || -> Result<PerLayerEmbedding> {
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
                // Per-layer token embed table — ~4.7 GB on Gemma 4 E2B
                // bf16; on wasm32 we skip it (wouldn't fit in 4 GB
                // address space) and fall back to projection-only PLE.
                #[cfg(target_arch = "wasm32")]
                let embed_tokens_per_layer: Option<QEmbedding> = None;
                #[cfg(not(target_arch = "wasm32"))]
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
                    per_layer_input_scale: (2.0_f64).powf(-0.5),
                    embed_scale_per_layer: (hidden_per_layer as f64).sqrt(),
                    num_hidden_layers: cfg.num_hidden_layers,
                    hidden_per_layer,
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
            device: device.clone(),
            cfg: cfg.clone(),
        })
    }

    /// One forward step. `xs` is `[B, q_len]` token ids; result is
    /// `[B, 1, vocab_size]` logits (last token only, for autoregressive
    /// sampling — same convention as the legacy module).
    pub fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, q_len) = xs.dims2()?;

        // Embed lookup → embed scale (sqrt(hidden_size)) — Ollama 1023-1024.
        let inputs_embeds = self.embed_tokens.forward(xs)?;
        let inputs_embeds = if inputs_embeds.device().same_device(&self.device) {
            inputs_embeds
        } else {
            inputs_embeds.to_device(&self.device)?
        };
        let mut hidden = (inputs_embeds.clone() * self.embed_scale)?;

        // Attention masks — only built for prefill (q_len > 1).
        let attention_mask = if q_len <= 1 {
            None
        } else {
            Some(prepare_attention_mask(
                b_sz,
                q_len,
                seqlen_offset,
                &self.device,
                hidden.dtype(),
            )?)
        };
        let sliding_attention_mask = if q_len <= 1 {
            None
        } else {
            Some(prepare_sliding_attention_mask(
                b_sz,
                q_len,
                seqlen_offset,
                self.cfg.effective_sliding_window(),
                &self.device,
                hidden.dtype(),
            )?)
        };

        // PLE inputs precomputed once per forward (Ollama 1027-1030).
        let per_layer_table = match &self.per_layer_embedding {
            Some(ple) => Some(ple.forward(xs, &inputs_embeds)?),
            None => None,
        };

        // KV-share per-step store (Ollama 1034-1037).
        let mut shared_kv_store: SharedKvStore =
            (0..self.cfg.num_hidden_layers).map(|_| None).collect();

        // Layer loop (Ollama 1039-1066).
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
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
        }

        // Final norm (Ollama 1068).
        let hidden = self.norm.forward(&hidden)?;
        {
            let bdir = bisect_dump::enabled();
            bdump!(bdir, seqlen_offset, 999usize, "98_post_norm", &hidden);
        }
        // Last-token slice for autoregressive sampling.
        let hidden = hidden.i((.., q_len - 1, ..))?.unsqueeze(1)?;
        let logits = self.lm_head.forward(&hidden)?;
        {
            let bdir = bisect_dump::enabled();
            bdump!(bdir, seqlen_offset, 999usize, "99_logits_pre_softcap", &logits);
        }
        // Final-logit softcap (Ollama 1074-1077): tanh(logits/cap)*cap.
        // F32 upcast for tanh precision (BF16/F16 saturate poorly).
        match self.cfg.final_logit_softcapping {
            Some(sc) if sc > 0.0 => {
                let logits_f32 = logits.to_dtype(DType::F32)?;
                let scaled = (logits_f32 / sc)?;
                let capped = scaled.tanh()?;
                capped * sc
            }
            _ => Ok(logits),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Attention masks (causal + sliding-window)
// ─────────────────────────────────────────────────────────────────────
//
// Same shape as the legacy module — float `0.0` / `-inf` mask added
// pre-softmax. Sliding mask additionally zeroes attention outside the
// window.

fn prepare_attention_mask(
    b_sz: usize,
    seq_len: usize,
    seqlen_offset: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let total_len = seqlen_offset + seq_len;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..total_len).map(move |j| {
                if j > seqlen_offset + i {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
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
        .flat_map(|i| {
            (0..total_len).map(move |j| {
                let abs_i = seqlen_offset + i;
                if j > abs_i {
                    f32::NEG_INFINITY
                } else if abs_i.saturating_sub(j) >= sliding_window {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, total_len), device)?
        .expand((b_sz, 1, seq_len, total_len))?
        .to_dtype(dtype)
}
