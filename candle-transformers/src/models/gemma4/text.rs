//! Gemma 4 text decoder.
//!
//! and following the candle gemma3.rs patterns.

use std::sync::Arc;

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

// ── RmsNorm (Gemma 3n / Gemma 4 — plain `weight` gain, no +1 offset) ───────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        // `.contiguous()` before `.to_dtype` because the WGPU `to_dtype`
        // kernel rejects strided sources (`wgpu_backend/ops/cast.rs:65`).
        // Callers like `q_norm(q.transpose(1,2))` produce non-contiguous
        // input. On CPU/CUDA `.contiguous()` is essentially a no-op when
        // the layout is already contiguous, so this is safe across
        // backends.
        let x = x.contiguous()?.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        // Gemma 3n / Gemma 4 use plain `weight` as the gain (HF
        // `Gemma3nRMSNorm`: `weight = nn.Parameter(torch.ones(dim))`,
        // forward applies `normed * weight`). This is **different** from
        // Gemma 1/2's `(1 + weight)` convention where weights are trained
        // around 0. Applying `(1 + weight)` to Gemma 3n weights doubles
        // every norm output → Q and K both ~2× → attention scores ~4× →
        // BF16 softmax overflow at the first global layer → all-NaN logits.
        x_normed
            .contiguous()?
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)
    }
}

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    // `.contiguous()` before each `to_dtype` for WGPU compatibility
    // (see RmsNorm::forward comment).
    let v_f32 = v.contiguous()?.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32
        .broadcast_div(&rms)?
        .contiguous()?
        .to_dtype(original_dtype)
}

// ── RotaryEmbedding (standard, for sliding layers) ──────────────────────────

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, head_dim: usize, rope_theta: f64, max_seq_len: usize, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
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

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── ProportionalRotaryEmbedding (for global/full layers) ────────────────────

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
        // Pad with zeros for non-rotated dimensions -> cos=1, sin=0 -> identity
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

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
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
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
    /// Pre-activation Gaussian-topk threshold offset
    /// `icdf_normal(sparsity)`. `0.0` disables sparsity for this layer.
    /// Computed once at construction from the layer's
    /// `cfg.activation_sparsity_at(layer_idx)`.
    sparsity_threshold_z: f64,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        bias: bool,
        sparsity: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("gate_proj"))?;
        let up_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("up_proj"))?;
        let down_proj = linear_bias(intermediate_size, hidden_size, bias, vb.pp("down_proj"))?;
        let sparsity_threshold_z = if sparsity > 0.0 && sparsity < 1.0 {
            // icdf of standard normal at p = sparsity. Using
            // sqrt(2) * erfinv(2*sparsity - 1).
            std::f64::consts::SQRT_2 * erfinv_approx(2.0 * sparsity - 1.0)
        } else {
            0.0
        };
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
            sparsity_threshold_z,
        })
    }

    /// `gate(x)` with optional Gaussian-topk sparsity applied before the
    /// activation. Equivalent to HF `Gemma3nTextMLP.gaussian_topk` + the
    /// surrounding gate/up/down flow.
    fn forward_inner(&self, xs: &Tensor) -> Result<Tensor> {
        let mut gate = xs.apply(&self.gate_proj)?;
        if self.sparsity_threshold_z != 0.0 {
            // threshold = mean + std * z, applied per-token along the
            // intermediate axis. mean / std computed in f32 for stability.
            let original_dtype = gate.dtype();
            let gate_f32 = gate.to_dtype(DType::F32)?;
            let mean = gate_f32.mean_keepdim(D::Minus1)?;
            let var = gate_f32
                .broadcast_sub(&mean)?
                .sqr()?
                .mean_keepdim(D::Minus1)?;
            let std = var.sqrt()?;
            let threshold = (mean + (std * self.sparsity_threshold_z)?)?;
            let sparse = gate_f32.broadcast_sub(&threshold)?;
            // ReLU
            let zero = Tensor::zeros_like(&sparse)?;
            let sparse = sparse.maximum(&zero)?;
            gate = sparse.to_dtype(original_dtype)?;
        }
        let lhs = gate.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_inner(xs)
    }
}

/// Cheap erfinv approximation accurate to ~5e-4 across [-1, 1]. Sufficient
/// for AltUp's sparsity threshold which is itself a multiplicative factor
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

// ── Flash attention ─────────────────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

// ── KvCache ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

/// Per-step shared K/V store keyed by donor layer index. Populated by
/// donor `Attention::forward` calls (post-cache-append, pre-`repeat_kv`)
/// and consumed by receiver layers later in the same forward pass.
/// Cleared at the start of every `TextModel::forward_embeds_*`. Tensors
/// are reference-counted (Arc-backed) so the clones are O(1).
pub(crate) type SharedKvStore = Vec<Option<(Tensor, Tensor)>>;

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    /// `None` for receiver layers (KV-shared) — they read K from the donor.
    k_proj: Option<Linear>,
    /// `None` for receiver layers — they read V from the donor.
    v_proj: Option<Linear>,
    o_proj: Linear,
    q_norm: RmsNorm,
    /// `None` for receiver layers — RoPE was already applied at the donor.
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    /// Attention pre-softmax scaling uses `1/√query_pre_attn_scalar`. For
    /// Gemma 3n E2B this equals `head_dim` (both 256), but configurations
    /// where global layers carry a larger `head_dim` keep the trained
    /// scale by reading from the dedicated config field.
    query_pre_attn_scalar: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    /// `Some(donor_idx)` makes this a receiver layer that reads K/V from
    /// `shared_kv_store[donor_idx]` instead of running its own
    /// projections + RoPE + cache append. `None` makes this a donor —
    /// it computes K/V normally and writes to `shared_kv_store[layer_idx]`
    /// after the cache append.
    donor_layer_idx: Option<usize>,
    /// This layer's index — used by donors to write into the shared
    /// store. Receivers read via `donor_layer_idx`.
    layer_idx: usize,
    use_flash_attn: bool,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let bias = cfg.attention_bias;
        let is_sliding = cfg.is_sliding(layer_idx);

        let (head_dim, num_kv_heads) = if is_sliding {
            (cfg.head_dim, cfg.num_key_value_heads)
        } else {
            let global_kv = cfg
                .num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads);
            (cfg.global_head_dim, global_kv)
        };

        let num_kv_groups = num_heads / num_kv_heads;
        let donor_layer_idx = cfg.donor_layer_idx_for(layer_idx);
        let is_receiver = donor_layer_idx.is_some();

        let q_proj = linear_bias(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        // Receivers don't have their own K/V projections in the canonical
        // Gemma 3n checkpoint — the donor's K/V is reused. If a checkpoint
        // does carry the tensors for receiver layers (some exports do),
        // they're harmless dead weight that we simply never call.
        let k_proj = if is_receiver {
            None
        } else {
            Some(linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?)
        };
        let v_proj = if is_receiver {
            None
        } else {
            Some(linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?)
        };
        let o_proj = linear_bias(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = if is_receiver {
            None
        } else {
            Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?)
        };

        // Receivers don't own a KvCache — they read from the donor's
        // post-append store on every step. We still construct a stub
        // KvCache so the `clear_kv_cache` path stays uniform.
        let kv_cache = if is_sliding {
            KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(
                2,
                cfg.effective_sliding_window(),
            ))
        } else {
            KvCache::Normal(candle_nn::kv_cache::KvCache::new(
                2,
                cfg.max_position_embeddings,
            ))
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            rms_norm_eps: cfg.rms_norm_eps,
            is_sliding,
            rotary_emb_global,
            rotary_emb_local,
            kv_cache,
            donor_layer_idx,
            layer_idx,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_kv_store: &mut SharedKvStore,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Q is always projected from the layer's own input — receivers
        // share K/V with their donor but keep their own Q.
        let mut q = self.q_proj.forward(xs)?;
        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        q = self.q_norm.forward(&q)?;

        // ── Donor / receiver branch ─────────────────────────────────────
        // Receivers read post-cache (k_full, v_full) from the donor's
        // entry in the shared store. Donors compute K/V the usual way
        // and write the post-append (pre-`repeat_kv`) tensors back so a
        // later receiver can pick them up.
        let (k_full, v_full) = if let Some(donor_idx) = self.donor_layer_idx {
            let donor = shared_kv_store
                .get(donor_idx)
                .and_then(|x| x.as_ref())
                .ok_or_else(|| candle::Error::Msg(format!(
                    "KV-shared layer {} has no donor entry at index {} \
                     (donor must execute earlier in the same forward pass)",
                    self.layer_idx, donor_idx,
                )))?;
            (donor.0.clone(), donor.1.clone())
        } else {
            let k_proj = self.k_proj.as_ref().ok_or_else(|| candle::Error::Msg(
                "donor layer is missing its k_proj — checkpoint mismatch?".into(),
            ))?;
            let v_proj = self.v_proj.as_ref().ok_or_else(|| candle::Error::Msg(
                "donor layer is missing its v_proj — checkpoint mismatch?".into(),
            ))?;
            let k_norm = self.k_norm.as_ref().ok_or_else(|| candle::Error::Msg(
                "donor layer is missing its k_norm — checkpoint mismatch?".into(),
            ))?;

            let mut k = k_proj.forward(xs)?;
            let v = v_proj.forward(xs)?;
            k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            k = k_norm.forward(&k)?;
            let v = v_norm(&v, self.rms_norm_eps)?;

            // Apply RoPE on (q, k); receivers had their q rotated above
            // separately below.
            let (q_rot, k_rot) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
            };
            q = q_rot;

            let (k_full, v_full) = match &mut self.kv_cache {
                KvCache::Normal(cache) => cache.append(&k_rot, &v)?,
                KvCache::Rotating(cache) => cache.append(&k_rot, &v)?,
            };
            // Persist for any downstream receiver mapped onto this layer.
            // Tensors are Arc-backed so this clone is cheap.
            if self.layer_idx < shared_kv_store.len() {
                shared_kv_store[self.layer_idx] =
                    Some((k_full.clone(), v_full.clone()));
            }
            (k_full, v_full)
        };

        // For receivers, q still needs RoPE — apply on q only.
        let q = if self.donor_layer_idx.is_some() {
            // We need q_only RoPE; use a cheap helper via the same
            // rotary embedding by calling apply_rotary_emb_qkv with
            // a dummy K of the same shape, then discard the K result.
            // To avoid churning, just rotate q via the embedding's
            // single-tensor path: fabricate a zero K of matching shape.
            let dummy_k = q.zeros_like()?;
            let (q_rot, _) = if self.is_sliding {
                self.rotary_emb_local
                    .apply_rotary_emb_qkv(&q, &dummy_k, seqlen_offset)?
            } else {
                self.rotary_emb_global
                    .apply_rotary_emb_qkv(&q, &dummy_k, seqlen_offset)?
            };
            q_rot
        } else {
            q
        };

        let k = crate::utils::repeat_kv(k_full, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v_full, self.num_kv_groups)?.contiguous()?;

        let mask = if self.is_sliding {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let attn_output = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            // Gemma 3 / 3n scale by `query_pre_attn_scalar`, not `head_dim` —
            // they're equal for E2B (both 256) but diverge in variants where
            // global layers carry a larger head_dim.
            let scale = 1f32 / (self.query_pre_attn_scalar as f32).sqrt();
            flash_attn(&q, &k, &v, scale, mask.is_some())?.transpose(1, 2)?
        } else {
            let scale = 1f64 / f64::sqrt(self.query_pre_attn_scalar as f64);
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

            let attn_weights = match mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            // Promote softmax to F32 even when the value dtype is BF16:
            // exp(score) overflows BF16 (max ≈ 65504) at score ≈ 11, which
            // is well within normal pre-softmax range for long-context or
            // sliding-window edge cases. F32 covers Gemma's full range.
            // Cast back to v's dtype before the V matmul so o_proj sees
            // the model's native dtype.
            //
            // `.contiguous()` calls before each `to_dtype` are needed
            // because the WGPU `to_dtype` kernel rejects strided sources
            // (`wgpu_backend/ops/cast.rs:65`). `broadcast_add(mask)` and
            // `softmax_last_dim` may produce non-contiguous views; the
            // single-token forward path skipped this entirely (no mask
            // when seq_len <= 1) so the issue only surfaces with
            // multi-token prompts.
            let v_dtype = v.dtype();
            let attn_weights = attn_weights.contiguous()?.to_dtype(DType::F32)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_weights = attn_weights.contiguous()?.to_dtype(v_dtype)?;
            attn_weights.matmul(&v)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────────

// ── LAuReL (Learned Augmented Residual Layer) ──────────────────────────────
//
// Low-rank residual augmentation applied alongside the attention output.
// Mirrors HF `Gemma3nTextLaurelBlock`: project hidden→rank→hidden,
// RMSNorm, add to the original hidden_states.

#[derive(Debug, Clone)]
struct LaurelBlock {
    linear_left: Linear,
    linear_right: Linear,
    post_laurel_norm: RmsNorm,
}

impl LaurelBlock {
    fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let linear_left =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.laurel_rank, vb.pp("linear_left"))?;
        let linear_right =
            candle_nn::linear_no_bias(cfg.laurel_rank, cfg.hidden_size, vb.pp("linear_right"))?;
        let post_laurel_norm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_laurel_norm"))?;
        Ok(Self {
            linear_left,
            linear_right,
            post_laurel_norm,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let laurel = hidden.apply(&self.linear_left)?.apply(&self.linear_right)?;
        let normed = self.post_laurel_norm.forward(&laurel)?;
        hidden + normed
    }
}

// ── AltUp (Alternating Updates) ────────────────────────────────────────────
//
// Maintains `altup_num_inputs` parallel hidden streams. Each layer
// runs predict → activate (the layer's main attn+MLP path on the active
// stream) → correct over the full stack, returning the corrected stack
// for the next layer. Mirrors HF `Gemma3nTextAltUp`.

#[derive(Debug, Clone)]
struct AltUp {
    correct_output_scale: Tensor,
    correction_coefs: Linear,
    prediction_coefs: Linear,
    modality_router: Linear,
    router_norm: RmsNorm,
    /// `1 / hidden_size` — applied to the modality vector before the
    /// router projection. Buffer in HF speak.
    router_input_scale: f64,
    altup_num_inputs: usize,
    altup_active_idx: usize,
}

impl AltUp {
    fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let correct_output_scale =
            vb.get(cfg.hidden_size, "correct_output_scale")?;
        let correction_coefs = candle_nn::linear_no_bias(
            cfg.altup_num_inputs,
            cfg.altup_num_inputs,
            vb.pp("correction_coefs"),
        )?;
        let prediction_coefs = candle_nn::linear_no_bias(
            cfg.altup_num_inputs,
            cfg.altup_num_inputs * cfg.altup_num_inputs,
            vb.pp("prediction_coefs"),
        )?;
        let modality_router = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.altup_num_inputs,
            vb.pp("modality_router"),
        )?;
        let router_norm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("router_norm"))?;
        Ok(Self {
            correct_output_scale,
            correction_coefs,
            prediction_coefs,
            modality_router,
            router_norm,
            router_input_scale: (cfg.hidden_size as f64).powf(-1.0),
            altup_num_inputs: cfg.altup_num_inputs,
            altup_active_idx: cfg.altup_active_idx,
        })
    }

    /// `tanh(modality_router(router_norm(x) * router_input_scale))`
    fn compute_router_modalities(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.router_norm.forward(x)?;
        let scaled = (normed * self.router_input_scale)?;
        scaled.apply(&self.modality_router)?.tanh()
    }

    /// `predict(stack) -> stack + (predicted_residual_per_input)`
    /// `stack` shape: `[num_inputs, B, T, hidden]`
    fn predict(&self, stack: &Tensor) -> Result<Tensor> {
        let active = stack.i(self.altup_active_idx)?;
        let modalities = self.compute_router_modalities(&active)?; // [B, T, num_inputs]

        // prediction_coefs: [B, T, num_inputs²]
        let coefs = self.prediction_coefs.forward(&modalities)?;
        let bt = coefs.dims();
        let (b, t) = (bt[0], bt[1]);
        // [B, T, num_inputs, num_inputs] then permute last two so matmul gives the right thing
        let coefs = coefs
            .reshape((b, t, self.altup_num_inputs, self.altup_num_inputs))?
            .transpose(2, 3)?;

        // Permute stack [num_inputs, B, T, H] -> [B, T, H, num_inputs]
        let permuted = stack.permute((1, 2, 3, 0))?.contiguous()?;
        // matmul (B, T, H, num_inputs) @ (B, T, num_inputs, num_inputs) -> (B, T, H, num_inputs)
        let predicted = permuted.matmul(&coefs.contiguous()?)?;
        // Undo permute back to [num_inputs, B, T, H]
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
        // Repeat across num_inputs dim: [num_inputs, B, T, H]
        let innovation =
            innovation
                .unsqueeze(0)?
                .broadcast_as(predictions.shape())?
                .contiguous()?;

        let coefs = self.correction_coefs.forward(&modalities)?; // [B, T, num_inputs]
        // +1.0 (the trained "identity" baseline)
        let coefs = (coefs + 1.0)?;
        // Permute [B, T, num_inputs] -> [num_inputs, B, T] then unsqueeze for H broadcast
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

// ── DecoderLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    #[allow(dead_code)]
    is_sliding: bool,

    // ── Gemma 3n additions ──────────────────────────────────────────────
    /// `None` for non-3n configs (no AltUp); decoder behaves classically.
    altup: Option<AltUp>,
    /// `None` for non-3n configs.
    laurel: Option<LaurelBlock>,
    /// Per-layer-input gate Linear (hidden → hidden_per_layer). Only
    /// present when PLE is wired.
    per_layer_input_gate: Option<Linear>,
    /// hidden_per_layer → hidden projection.
    per_layer_projection: Option<Linear>,
    /// Post per-layer-projection RmsNorm.
    post_per_layer_input_norm: Option<RmsNorm>,
    /// Activation function for the per-layer-input gate (matches
    /// `cfg.hidden_activation`). Stored so the gate path can apply the
    /// same nonlinearity the trained model expects.
    per_layer_act: Activation,
    /// `1/√2` constant for the (attn_gated + laurel_out) merge.
    inv_sqrt_2: f64,
    altup_correct_scale: bool,
    /// Per-layer learned scalar applied to the layer output.
    /// `Gemma4TextDecoderLayer` in HF: `register_buffer("layer_scalar",
    /// torch.ones(1))` initialized to 1.0, trained per layer, and applied
    /// as `hidden_states *= layer_scalar` as the final operation of
    /// `forward()`. Without this multiply the residual stream is missing
    /// a per-layer trained gain — empirically observed to send `abs_max`
    /// over 1000 after just one layer on Gemma 4 E2B, then NaN at the
    /// next softmax. `None` when the checkpoint doesn't carry this
    /// tensor (e.g. non-Gemma-4 fallback paths).
    layer_scalar: Option<Tensor>,
}

impl DecoderLayer {
    fn new(
        rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
        rotary_emb_local: Arc<RotaryEmbedding>,
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let is_sliding = cfg.is_sliding(layer_idx);
        let self_attn = Attention::new(
            rotary_emb_global,
            rotary_emb_local,
            cfg,
            layer_idx,
            vb.pp("self_attn"),
        )?;
        let mlp = MLP::new(
            cfg.hidden_size,
            cfg.intermediate_size_at(layer_idx),
            cfg.hidden_activation,
            false,
            cfg.activation_sparsity_at(layer_idx),
            vb.pp("mlp"),
        )?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        // Gemma 3n additions: only construct AltUp / LAuReL / per-layer-
        // input gate when the config carries the corresponding fields.
        // For pure Gemma 3 / Gemma 4 (without AltUp), the layer falls
        // back to the classic pre-norm + attn + post-norm + ffw flow.
        // Each module is also gated by an explicit `disable_*` flag so
        // numeric regressions in the new modules can be bisected at
        // runtime without rebuilding.
        let has_altup = cfg.altup_num_inputs > 1 && !cfg.disable_altup;
        let altup = if has_altup {
            Some(AltUp::new(cfg, vb.pp("altup"))?)
        } else {
            None
        };
        let laurel = if cfg.laurel_rank > 0 && !cfg.disable_laurel {
            // The LaurelBlock weights are present in Gemma 3n
            // checkpoints; if construction fails (e.g. weights missing
            // for a non-3n config) we fall back to no-LAuReL.
            LaurelBlock::new(cfg, vb.pp("laurel")).ok()
        } else {
            None
        };
        let ple_gate_enabled =
            cfg.hidden_size_per_layer_input.is_some() && !cfg.disable_per_layer_input_gate;
        let per_layer_input_gate = if ple_gate_enabled {
            Some(candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.hidden_size_per_layer_input.unwrap(),
                vb.pp("per_layer_input_gate"),
            )?)
        } else {
            None
        };
        let per_layer_projection = if ple_gate_enabled {
            Some(candle_nn::linear_no_bias(
                cfg.hidden_size_per_layer_input.unwrap(),
                cfg.hidden_size,
                vb.pp("per_layer_projection"),
            )?)
        } else {
            None
        };
        let post_per_layer_input_norm = if ple_gate_enabled {
            Some(RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_per_layer_input_norm"),
            )?)
        } else {
            None
        };

        // `layer_scalar` is a [1]-shaped per-layer learned gain stored in
        // the safetensors as `layers.<i>.layer_scalar`. Optional —
        // checkpoints without this tensor (older Gemma generations
        // wrapped through the gemma4 path for shape compat) silently
        // skip the multiply.
        let layer_scalar = vb.get(1, "layer_scalar").ok();

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            is_sliding,
            altup,
            laurel,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            per_layer_act: cfg.hidden_activation,
            inv_sqrt_2: (2.0_f64).powf(-0.5),
            altup_correct_scale: cfg.altup_correct_scale,
            layer_scalar,
        })
    }

    /// Forward pass for the layer.
    ///
    /// Two modes:
    /// - **Classic** (no AltUp wired): `xs` is `[B, T, hidden]`, returned
    ///   value is `[B, T, hidden]`. PLE / per-layer-input gate are
    ///   skipped, and the residual flow is the Gemma 3 path
    ///   (input_layernorm → self_attn → post_attention_layernorm + res
    ///   → pre_feedforward_layernorm → mlp → post_feedforward_layernorm
    ///   + res).
    /// - **AltUp** (Gemma 3n): `xs` is `[num_inputs, B, T, hidden]`, the
    ///   active stream is corrected via attention + MLP, the others via
    ///   the AltUp predict / correct mechanism, and the per-layer-input
    ///   gate adds the PLE side-channel back into the non-active streams.
    ///   Returns `[num_inputs, B, T, hidden]`.
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        per_layer_input: Option<&Tensor>,
        shared_kv_store: &mut SharedKvStore,
        layer_idx: usize,
        intra_hook: &mut Option<&mut dyn FnMut(usize, &str, &Tensor)>,
    ) -> Result<Tensor> {
        macro_rules! checkpoint {
            ($step:expr, $tensor:expr) => {
                if let Some(hook) = intra_hook.as_deref_mut() {
                    hook(layer_idx, $step, $tensor);
                }
            };
        }

        let altup = match &self.altup {
            Some(a) => a,
            None => {
                // Classic / Gemma 4 path — no AltUp wiring. Mirrors
                // `Gemma4TextDecoderLayer.forward` in HF transformers
                // (modular_gemma4.py): three residual blocks (attn, mlp,
                // PLE) followed by a per-layer scalar gain.
                //
                // 1. Attention residual
                let residual = xs;
                checkpoint!("input", residual);
                let h = self.input_layernorm.forward(xs)?;
                checkpoint!("post_input_layernorm", &h);
                let h = self.self_attn.forward(
                    &h,
                    attention_mask,
                    sliding_attention_mask,
                    seqlen_offset,
                    shared_kv_store,
                )?;
                checkpoint!("post_self_attn", &h);
                let h = h.apply(&self.post_attention_layernorm)?;
                checkpoint!("post_attn_layernorm", &h);
                let h = (h + residual)?;
                checkpoint!("post_attn_residual", &h);

                // 2. MLP residual
                let residual = h.clone();
                let h = h.apply(&self.pre_feedforward_layernorm)?;
                checkpoint!("post_pre_ff_layernorm", &h);
                let h = h.apply(&self.mlp)?;
                checkpoint!("post_mlp", &h);
                let h = h.apply(&self.post_feedforward_layernorm)?;
                checkpoint!("post_ff_layernorm", &h);
                let h = (h + &residual)?;
                checkpoint!("post_mlp_residual", &h);

                // 3. Per-layer-input residual (Gemma 4 PLE side-channel).
                // HF: `gate(h) → act → * per_layer_input → projection →
                //      norm → + residual`. Without this block the PLE
                // signal computed before the layer loop is silently
                // dropped, and the layer's residual stream loses the
                // magnitude control built into the trained checkpoint.
                let h = if let (Some(gate), Some(proj), Some(norm), Some(per_layer_input)) = (
                    self.per_layer_input_gate.as_ref(),
                    self.per_layer_projection.as_ref(),
                    self.post_per_layer_input_norm.as_ref(),
                    per_layer_input,
                ) {
                    let residual = h.clone();
                    let g = h.apply(gate)?;
                    checkpoint!("ple/post_gate", &g);
                    let g = g.apply(&self.per_layer_act)?;
                    checkpoint!("ple/post_act", &g);
                    let g = g.broadcast_mul(per_layer_input)?;
                    checkpoint!("ple/post_mul", &g);
                    let g = g.apply(proj)?;
                    checkpoint!("ple/post_proj", &g);
                    let g = norm.forward(&g)?;
                    checkpoint!("ple/post_norm", &g);
                    let h = (g + residual)?;
                    checkpoint!("post_ple_residual", &h);
                    h
                } else {
                    h
                };

                // 4. Per-layer scalar gain. Initialized to 1.0 in HF, but
                // the trained checkpoint carries a learned value per
                // layer that controls the outgoing magnitude. Skipping
                // this multiply was the cause of the runaway
                // `abs_max=1232` observed on layer-0 in the chat-pwa
                // diagnostic harness.
                let h = if let Some(scalar) = &self.layer_scalar {
                    let h = h.broadcast_mul(scalar)?;
                    checkpoint!("post_layer_scalar", &h);
                    h
                } else {
                    h
                };

                return Ok(h);
            }
        };

        // ── AltUp path ───────────────────────────────────────────────
        // xs: [num_inputs, B, T, hidden]
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
        let attn = attn.apply(&self.post_attention_layernorm)?;
        let attn_gated = (active + attn)?;
        let attn_laurel = match laurel_out {
            Some(l) => ((attn_gated + l)? * self.inv_sqrt_2)?,
            None => attn_gated,
        };

        let attn_norm = self.pre_feedforward_layernorm.forward(&attn_laurel)?;
        let ffw = self.mlp.forward(&attn_norm)?;
        let ffw = self.post_feedforward_layernorm.forward(&ffw)?;
        let attn_ffw_laurel_gated = (attn_laurel + ffw)?;

        let corrected = altup.correct(&predictions, &attn_ffw_laurel_gated)?;

        // Per-layer-input gate (Gemma 3n PLE consumption point).
        let mut corrected = corrected;
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
            // gate: [..., hidden] -> [..., hidden_per_layer]
            let first = first.apply(gate)?;
            let first = first.apply(&self.per_layer_act)?;
            let first = first.broadcast_mul(per_layer_input)?;
            // proj: [..., hidden_per_layer] -> [..., hidden]
            let first = first.apply(proj)?;
            let first = norm.forward(&first)?;
            // corrected[1:] += first  (broadcast across non-active streams)
            // We do this by constructing a delta tensor of shape
            // [num_inputs, B, T, H] where index altup_active_idx is zeros
            // and the others are `first`. Then `corrected += delta`.
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

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

// ── Causal mask ─────────────────────────────────────────────────────────────

fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

// ── Per-Layer Embeddings (Gemma 3n) ────────────────────────────────────────
//
// Each token gets an additional small per-layer signal added on top of
// the main residual stream. The forward computes a `[B, T, num_layers,
// hidden_per_layer]` table once and the layer-i decoder takes its slice
// `[B, T, hidden_per_layer]` as a side-channel input. Mirrors HF's
// `Gemma3nTextModel` lines ~2399–2430.

/// Pluggable per-layer-embedding table lookup.
///
/// The default in-memory impl wraps `candle_nn::Embedding` and is what
/// `PerLayerEmbedding::new` constructs when the safetensors file carries
/// a sized-tractable PLE weight. Memory-constrained targets (wasm32) can
/// implement this trait against a streaming source — e.g. an OPFS-backed
/// random-access read of the original safetensors blob — and inject it
/// via [`TextModel::set_per_layer_embed_table`].
///
/// `lookup` returns `[B, T, num_layers * hidden_per_layer]` (matching
/// `candle_nn::Embedding::forward(input_ids)` for an embedding table of
/// size `vocab × (num_layers × hidden_per_layer)`).
pub trait PerLayerEmbedTable: std::fmt::Debug + Send + Sync {
    fn lookup(&self, input_ids: &Tensor) -> Result<Tensor>;
}

/// In-memory wrapper around `candle_nn::Embedding`. Used by the default
/// `PerLayerEmbedding::new` path when the table fits in scope.
#[derive(Debug)]
struct InMemoryPerLayerTable(candle_nn::Embedding);

impl PerLayerEmbedTable for InMemoryPerLayerTable {
    fn lookup(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.0.forward(input_ids)
    }
}

#[derive(Debug, Clone)]
struct PerLayerEmbedding {
    /// Looks up `[B, T] -> [B, T, num_layers * hidden_per_layer]`.
    /// Optional because the canonical Gemma 3n / Gemma 4 weight is huge
    /// (vocab × num_layers × hidden_per_layer = several GB) and may be
    /// skipped at load time on memory-constrained targets (wasm32). When
    /// `None` the table contribution is implicitly zero — the merge
    /// becomes `per_layer_input = per_layer_proj * rsqrt(2)`. That's an
    /// approximation of the trained behavior (we lose the per-token
    /// PLE signal) but produces coherent forward output without
    /// blowing the runtime memory ceiling.
    ///
    /// `Arc<dyn>` so a host (e.g. chat-pwa) can swap in a streaming impl
    /// after construction via [`TextModel::set_per_layer_embed_table`].
    embed_tokens_per_layer: Option<Arc<dyn PerLayerEmbedTable>>,
    /// Projects the main `inputs_embeds` `[B, T, hidden_size]` into the
    /// same `num_layers * hidden_per_layer` space so the two signals
    /// can be summed.
    per_layer_model_projection: Linear,
    /// RMSNorm applied to each per-layer slice (`hidden_per_layer`-wide)
    /// of the reshaped projection. Per HF the norm acts on the LAST
    /// dim of `[B, T, num_layers, hidden_per_layer]`, so its weight is
    /// `[hidden_per_layer]`, not `[num_layers * hidden_per_layer]`.
    per_layer_projection_norm: RmsNorm,
    /// Buffer scale = `1 / √hidden_size` (HF computes this as
    /// `hidden_size ** -0.5`). Multiplied into the projection so the
    /// merged signal stays at unit-ish variance.
    per_layer_projection_scale: f64,
    num_hidden_layers: usize,
    hidden_per_layer: usize,
    /// `1 / √2` — applied to the merged sum so the per-layer signal
    /// doesn't double-up the magnitude when the two sources are
    /// equally scaled. `rsqrt(2.0)` in HF speak.
    per_layer_input_scale: f64,
}

impl PerLayerEmbedding {
    fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_per_layer = cfg
            .hidden_size_per_layer_input
            .ok_or_else(|| candle::Error::Msg(
                "PerLayerEmbedding requires `hidden_size_per_layer_input`".into(),
            ))?;
        let vocab_per_layer = cfg.vocab_size_per_layer_input.unwrap_or(cfg.vocab_size);
        let total = cfg.num_hidden_layers * hidden_per_layer;
        // The embedding table is multiple GB on E2B; only construct it
        // when the VarBuilder backend actually has the tensor in scope.
        // Callers (e.g. chat-pwa) may opt to skip loading it on
        // memory-constrained targets — in which case we silently fall
        // back to the projection-only merge.
        let embed_tokens_per_layer: Option<Arc<dyn PerLayerEmbedTable>> =
            if vb.contains_tensor("embed_tokens_per_layer.weight") {
                let embed = candle_nn::embedding(
                    vocab_per_layer,
                    total,
                    vb.pp("embed_tokens_per_layer"),
                )?;
                Some(Arc::new(InMemoryPerLayerTable(embed)))
            } else {
                None
            };
        let per_layer_model_projection =
            candle_nn::linear_no_bias(cfg.hidden_size, total, vb.pp("per_layer_model_projection"))?;
        let per_layer_projection_norm = RmsNorm::new(
            hidden_per_layer,
            cfg.rms_norm_eps,
            vb.pp("per_layer_projection_norm"),
        )?;
        Ok(Self {
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            per_layer_projection_scale: (cfg.hidden_size as f64).powf(-0.5),
            num_hidden_layers: cfg.num_hidden_layers,
            hidden_per_layer,
            per_layer_input_scale: (2.0_f64).powf(-0.5),
        })
    }

    /// Compute the full per-layer-input table for the current token slice.
    /// Returns `[B, T, num_layers, hidden_per_layer]` with the *same*
    /// dtype as `inputs_embeds`.
    fn forward(&self, input_ids: &Tensor, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;

        // Project inputs_embeds, reshape to (B, T, num_layers, H_per),
        // scale and norm — applied per-slice along the last axis.
        let proj = inputs_embeds.apply(&self.per_layer_model_projection)?;
        let proj = (proj * self.per_layer_projection_scale)?;
        let proj = proj.reshape((b, t, self.num_hidden_layers, self.hidden_per_layer))?;
        let proj = self.per_layer_projection_norm.forward(&proj)?;

        let merged = match &self.embed_tokens_per_layer {
            Some(embed) => {
                let table = embed.lookup(input_ids)?;
                let table =
                    table.reshape((b, t, self.num_hidden_layers, self.hidden_per_layer))?;
                (proj.broadcast_add(&table)? * self.per_layer_input_scale)?
            }
            None => (proj * self.per_layer_input_scale)?,
        };
        merged.contiguous()
    }
}

// ── AltUp magnitude rescale helpers ────────────────────────────────────────

/// Per-token L2 magnitude `sqrt(mean(x**2, dim=-1, keepdim=True))`.
/// Returns a `[..., 1]` tensor in f32 (the magnitude is always computed
/// in f32 to avoid bf16 underflow before the rescale divide).
fn magnitude_per_token(x: &Tensor) -> Result<Tensor> {
    let original_dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?;
    let sq = x32.sqr()?;
    let mean = sq.mean_keepdim(D::Minus1)?;
    let mag = mean.sqrt()?;
    if original_dtype == DType::F32 {
        Ok(mag)
    } else {
        // Magnitude stays in f32 — the consumer divides another f32
        // through it, so we don't bother converting back.
        Ok(mag)
    }
}

/// Rescale `x` so its per-token L2 magnitude matches `target_magnitude`.
/// Both tensors stay in `x`'s dtype, with the divide done in f32 for
/// numerical stability.
///
/// Hardened against the NaN-via-overflow path that the previous
/// implementation hit on Gemma 3n loads:
///
/// 1. The eps floor is folded *inside* the sqrt argument as
///    `sqrt(mean(x²) + eps²)` so `denom >= eps` even when `mean(x²)`
///    underflows to zero in f32 (versus the previous post-sqrt
///    `max(new_mag, 1e-12)` which left a path where the divisor could
///    legitimately be `1e-12` and the resulting scale would explode by
///    ~1e12).
/// 2. The resulting `scale` is clamped to `[0, 65504]` (just above
///    bf16's max-finite, leaving headroom in f32) so even a degenerate
///    near-zero stream can't produce a multiply that overflows the
///    forward dtype on the cast back.
fn rescale_to_magnitude(x: &Tensor, target_magnitude: &Tensor) -> Result<Tensor> {
    const EPS: f64 = 1e-6;
    let original_dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?;
    let mean_sq = x32.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (mean_sq + (EPS * EPS))?.sqrt()?;
    let scale = target_magnitude.broadcast_div(&denom)?;
    // Cap the scale below f32 overflow + bf16 round-trip. 65504 is the
    // largest finite bf16 — even if the scale lands exactly there, the
    // subsequent multiply produces a finite f32 result.
    let scale = scale.clamp(0f32, 65504.0)?;
    let scaled = x32.broadcast_mul(&scale)?;
    scaled.to_dtype(original_dtype)
}

// ── TextModel ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: candle_nn::Embedding,
    /// Optional per-layer-embedding side-channel. `None` for non-Gemma3n
    /// configs that don't carry `hidden_size_per_layer_input`.
    per_layer_embedding: Option<PerLayerEmbedding>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: usize,
    /// AltUp expand projections — `num_altup_inputs - 1` Linears that
    /// derive the non-active starting hidden states from the main
    /// `inputs_embeds`. `None` when AltUp isn't wired.
    altup_projections: Option<Vec<Linear>>,
    /// AltUp consolidation projections used at the very end of the
    /// decoder stack to merge the parallel streams back into a single
    /// `[B, T, hidden]` tensor.
    altup_unembed_projections: Option<Vec<Linear>>,
    altup_num_inputs: usize,
    altup_active_idx: usize,
    /// Number of decoder layers — used to size the per-step shared KV
    /// store allocated inside `forward_embeds_hidden_with_per_layer`.
    num_hidden_layers: usize,
}

impl TextModel {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let rotary_emb_global = Arc::new(ProportionalRotaryEmbedding::new(
            vb.dtype(),
            cfg.global_head_dim,
            cfg.rope_theta,
            cfg.partial_rotary_factor(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);
        let rotary_emb_local = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.rope_local_base_freq(),
            cfg.max_position_embeddings,
            vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb_global.clone(),
                rotary_emb_local.clone(),
                cfg,
                layer_idx,
                vb_l.pp(layer_idx),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            // The default tied behavior shares `embed_tokens.embeddings()`
            // verbatim, which keeps `lm_head` on whatever device the
            // embedding table loaded onto. For Gemma 3n / Gemma 4 the
            // chat-pwa pins `embed_tokens` on CPU (it's ~800 MB and
            // we want to keep WebGPU pressure low), but per-token
            // BF16 matmul against an 800 MB weight on CPU is the
            // single biggest bottleneck — every step pays a 1.6 GB
            // f32-scratch round-trip plus the gemm itself.
            //
            // Untie by explicitly moving the weight onto the
            // VarBuilder's device. For chat-pwa that's the WebGPU
            // device, where the matmul is a single dispatch + a 1 MB
            // logits readback. Memory cost: +800 MB GPU; perf gain:
            // ~10× tokens-per-second.
            let weight = embed_tokens.embeddings().clone();
            let weight = if !weight.device().same_device(vb.device()) {
                weight.to_device(vb.device())?
            } else {
                weight
            };
            Linear::new(weight, None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let per_layer_embedding = if cfg.hidden_size_per_layer_input.is_some() {
            Some(PerLayerEmbedding::new(cfg, vb_m.clone())?)
        } else {
            None
        };

        // AltUp expand / consolidate projections live at the
        // `Gemma3nTextModel` level (HF). Construct only when AltUp is
        // wired (num_inputs > 1) AND the bisection kill-switch is off.
        // Tensor names: `altup_projections.{i}` and
        // `altup_unembed_projections.{i}` for `i in 0..num_inputs-1`.
        let (altup_projections, altup_unembed_projections) = if cfg.altup_num_inputs > 1
            && !cfg.disable_altup
        {
            let n_extra = cfg.altup_num_inputs - 1;
            let vb_proj = vb_m.pp("altup_projections");
            let vb_unproj = vb_m.pp("altup_unembed_projections");
            let mut projs = Vec::with_capacity(n_extra);
            let mut unprojs = Vec::with_capacity(n_extra);
            for i in 0..n_extra {
                projs.push(candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    cfg.hidden_size,
                    vb_proj.pp(i),
                )?);
                unprojs.push(candle_nn::linear_no_bias(
                    cfg.hidden_size,
                    cfg.hidden_size,
                    vb_unproj.pp(i),
                )?);
            }
            (Some(projs), Some(unprojs))
        } else {
            (None, None)
        };

        Ok(Self {
            embed_tokens,
            per_layer_embedding,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            altup_projections,
            altup_unembed_projections,
            altup_num_inputs: cfg.altup_num_inputs,
            altup_active_idx: cfg.altup_active_idx,
            num_hidden_layers: cfg.num_hidden_layers,
        })
    }

    fn create_attention_masks(
        &self,
        batch_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        if seq_len <= 1 {
            return Ok((None, None));
        }
        let mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            None,
            self.dtype,
            &self.device,
        )?;
        let sliding_mask = prepare_decoder_attention_mask(
            batch_size,
            seq_len,
            seqlen_offset,
            Some(self.sliding_window),
            self.dtype,
            &self.device,
        )?;
        Ok((Some(mask), Some(sliding_mask)))
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        xs * (self.hidden_size as f64).sqrt()
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let xs = self.embed_tokens(input_ids)?;
        self.forward_embeds(&xs, seqlen_offset, b_size, seq_len)
    }

    pub fn forward_embeds(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.forward_embeds_with_per_layer(xs, None, seqlen_offset, batch_size, seq_len)
    }

    /// Variant that accepts the precomputed Gemma 3n per-layer-input
    /// table `[B, T, num_layers, hidden_per_layer]`. When `per_layer_inputs`
    /// is `Some`, the layer-i decoder receives its `[..., i, :]` slice.
    pub fn forward_embeds_with_per_layer(
        &mut self,
        xs: &Tensor,
        per_layer_inputs: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let hidden = self.forward_embeds_hidden_with_per_layer(
            xs,
            per_layer_inputs,
            seqlen_offset,
            batch_size,
            seq_len,
        )?;
        let logits = hidden.apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    /// Forward pass through decoder layers + final norm only, returning
    /// hidden states *before* the lm_head projection. Used for mixed-device
    /// execution where embed_tokens / lm_head live on CPU while the decoder
    /// layers run on GPU.
    pub fn forward_embeds_hidden(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.forward_embeds_hidden_with_per_layer(xs, None, seqlen_offset, batch_size, seq_len)
    }

    /// Same as [`Self::forward_embeds_hidden_with_per_layer`] with a
    /// per-layer post-state hook. The hook is called after each decoder
    /// layer with `(layer_idx, current_state)`. AltUp consolidate is
    /// reported via `layer_idx == self.num_hidden_layers`.
    ///
    /// Diagnostic-only — production calls should go through the no-hook
    /// variants. The hook signature is sync; if it needs to do GPU
    /// readback (which on wasm32 requires polling), it should clone the
    /// tensor and defer the readback until after the function returns.
    pub fn forward_embeds_hidden_with_per_layer_hooked(
        &mut self,
        xs: &Tensor,
        per_layer_inputs: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
        mut hook: impl FnMut(usize, &Tensor),
    ) -> Result<Tensor> {
        self.forward_embeds_hidden_inner(
            xs,
            per_layer_inputs,
            seqlen_offset,
            batch_size,
            seq_len,
            Some(&mut hook),
            None,
        )
    }

    /// Same as `*_hooked` plus an *intra-layer* hook fired after every
    /// internal checkpoint within `DecoderLayer::forward` (classic path
    /// only). The intra hook signature is `(layer_idx, step_name, &state)`.
    /// Step names: `input`, `post_input_layernorm`, `post_self_attn`,
    /// `post_attn_layernorm`, `post_attn_residual`, `post_pre_ff_layernorm`,
    /// `post_mlp`, `post_ff_layernorm`, `post_mlp_residual`,
    /// `ple/post_gate`, `ple/post_act`, `ple/post_mul`, `ple/post_proj`,
    /// `ple/post_norm`, `post_ple_residual`, `post_layer_scalar`.
    ///
    /// Diagnostic-only. The intra hook fires on EVERY layer; callers are
    /// expected to filter on `layer_idx` if they only want certain
    /// layers' intermediate states.
    pub fn forward_embeds_hidden_with_intra_hook(
        &mut self,
        xs: &Tensor,
        per_layer_inputs: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
        mut layer_hook: impl FnMut(usize, &Tensor),
        mut intra_hook: impl FnMut(usize, &str, &Tensor),
    ) -> Result<Tensor> {
        self.forward_embeds_hidden_inner(
            xs,
            per_layer_inputs,
            seqlen_offset,
            batch_size,
            seq_len,
            Some(&mut layer_hook),
            Some(&mut intra_hook),
        )
    }

    /// Same as [`Self::forward_embeds_hidden`] with an explicit
    /// per-layer-input table for Gemma 3n.
    pub fn forward_embeds_hidden_with_per_layer(
        &mut self,
        xs: &Tensor,
        per_layer_inputs: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        self.forward_embeds_hidden_inner(
            xs,
            per_layer_inputs,
            seqlen_offset,
            batch_size,
            seq_len,
            None,
            None,
        )
    }

    fn forward_embeds_hidden_inner(
        &mut self,
        xs: &Tensor,
        per_layer_inputs: Option<&Tensor>,
        seqlen_offset: usize,
        batch_size: usize,
        seq_len: usize,
        mut hook: Option<&mut dyn FnMut(usize, &Tensor)>,
        mut intra_hook: Option<&mut dyn FnMut(usize, &str, &Tensor)>,
    ) -> Result<Tensor> {
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        // ── AltUp expand ────────────────────────────────────────────
        // When AltUp is wired (num_inputs > 1) we lift the [B, T, H]
        // input into a [num_inputs, B, T, H] stack: the active stream is
        // the original embeds, the others are produced by altup_projections
        // and then magnitude-rescaled to match the active's per-token
        // L2 magnitude (mirrors HF Gemma3nTextModel.forward).
        let (mut current, altup_active) = if let Some(projs) = &self.altup_projections {
            let target_mag = magnitude_per_token(xs)?;
            let mut streams: Vec<Tensor> = Vec::with_capacity(self.altup_num_inputs);
            for i in 0..self.altup_num_inputs {
                if i == self.altup_active_idx {
                    streams.push(xs.clone());
                    continue;
                }
                let proj_idx = if i < self.altup_active_idx { i } else { i - 1 };
                let projected = xs.apply(&projs[proj_idx])?;
                let rescaled = rescale_to_magnitude(&projected, &target_mag)?;
                streams.push(rescaled);
            }
            let stacked = Tensor::stack(&streams, 0)?;
            (stacked, true)
        } else {
            (xs.clone(), false)
        };

        // Per-step KV-share scratch: donor layers populate, receivers read.
        // Cleared (allocated fresh) per call so the (k_full, v_full) entries
        // always reflect the current step's history.
        let mut shared_kv_store: SharedKvStore = vec![None; self.num_hidden_layers];

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let per_layer_slice = match per_layer_inputs {
                Some(table) => Some(table.narrow(2, layer_idx, 1)?.squeeze(2)?),
                None => None,
            };
            current = layer.forward(
                &current,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
                per_layer_slice.as_ref(),
                &mut shared_kv_store,
                layer_idx,
                &mut intra_hook,
            )?;
            if let Some(h) = hook.as_mut() {
                h(layer_idx, &current);
            }
        }

        // ── AltUp consolidate ──────────────────────────────────────
        let final_hidden = if altup_active {
            let unprojs = self
                .altup_unembed_projections
                .as_ref()
                .expect("altup_projections set without altup_unembed_projections");
            let active = current.i(self.altup_active_idx)?;
            let target_mag = magnitude_per_token(&active)?;
            let mut streams: Vec<Tensor> = Vec::with_capacity(self.altup_num_inputs);
            for i in 0..self.altup_num_inputs {
                if i == self.altup_active_idx {
                    streams.push(active.clone());
                    continue;
                }
                let proj_idx = if i < self.altup_active_idx { i } else { i - 1 };
                let projected = current.i(i)?.apply(&unprojs[proj_idx])?;
                let rescaled = rescale_to_magnitude(&projected, &target_mag)?;
                streams.push(rescaled);
            }
            let stacked = Tensor::stack(&streams, 0)?;
            stacked.mean(0)?
        } else {
            current
        };

        if let Some(h) = hook.as_mut() {
            h(self.num_hidden_layers, &final_hidden);
        }

        final_hidden.narrow(1, seq_len - 1, 1)?.apply(&self.norm)
    }

    /// Per-layer `layer_scalar` tensors in load order, for diagnostic
    /// readback. `None` for layers where the checkpoint did not carry
    /// the tensor (older Gemma generations). Length always equals
    /// `num_hidden_layers`.
    pub fn layer_scalars(&self) -> Vec<Option<Tensor>> {
        self.layers
            .iter()
            .map(|l| l.layer_scalar.clone())
            .collect()
    }

    /// Compute the Gemma 3n per-layer-input table once for a given
    /// `(input_ids, inputs_embeds)` pair. Returns `None` when the model
    /// wasn't constructed with PLE.
    pub fn compute_per_layer_inputs(
        &self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Option<Tensor>> {
        match &self.per_layer_embedding {
            Some(ple) => Ok(Some(ple.forward(input_ids, inputs_embeds)?)),
            None => Ok(None),
        }
    }

    /// Project hidden states to logits via the lm_head linear layer.
    /// Applies final logit softcapping if configured.
    pub fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let logits = hidden.apply(&self.lm_head)?;
        match self.final_logit_softcapping {
            None => Ok(logits),
            Some(sc) => Ok(((logits / sc)?.tanh()? * sc)?),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    /// Inject a custom per-layer-embedding table after construction.
    /// Used by hosts (chat-pwa) that skip the in-memory PLE weight at
    /// load time and back it with a streaming source — typically OPFS
    /// random-access reads against the original safetensors blob.
    ///
    /// Errors if the model wasn't built with PLE wired
    /// (`hidden_size_per_layer_input` was `None` in the config).
    pub fn set_per_layer_embed_table(
        &mut self,
        table: Arc<dyn PerLayerEmbedTable>,
    ) -> Result<()> {
        let ple = self.per_layer_embedding.as_mut().ok_or_else(|| {
            candle::Error::Msg(
                "TextModel was not constructed with PLE — \
                 hidden_size_per_layer_input is None"
                    .into(),
            )
        })?;
        ple.embed_tokens_per_layer = Some(table);
        Ok(())
    }
}
