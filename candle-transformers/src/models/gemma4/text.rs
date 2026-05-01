//! Gemma 4 text decoder.
//!
//! and following the candle gemma3.rs patterns.

use std::sync::Arc;

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_b as linear_bias, Activation, Linear, VarBuilder};

use super::config::Gemma4TextConfig;

// ── RmsNorm (Gemma-style with +1 offset) ────────────────────────────────────

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
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

/// Pure RMS normalization without learned weight (used for V norm).
fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let original_dtype = v.dtype();
    let v_f32 = v.to_dtype(DType::F32)?;
    let mean_sq = v_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    v_f32.broadcast_div(&rms)?.to_dtype(original_dtype)
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
}

impl MLP {
    fn new(hidden_size: usize, intermediate_size: usize, act: Activation, bias: bool, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("gate_proj"))?;
        let up_proj = linear_bias(hidden_size, intermediate_size, bias, vb.pp("up_proj"))?;
        let down_proj = linear_bias(intermediate_size, hidden_size, bias, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
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

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
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
        let q_proj = linear_bias(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear_bias(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear_bias(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

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
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Q/K norms
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;
        // V norm (RMS without learned weight)
        let v = v_norm(&v, self.rms_norm_eps)?;

        // Apply RoPE
        let (q, k) = if self.is_sliding {
            self.rotary_emb_local
                .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
        } else {
            self.rotary_emb_global
                .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
        };

        let (k, v) = match &mut self.kv_cache {
            KvCache::Normal(cache) => cache.append(&k, &v)?,
            KvCache::Rotating(cache) => cache.append(&k, &v)?,
        };

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

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
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
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
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            is_sliding,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        // Per-layer-input slice `[B, T, hidden_per_layer]` — Gemma 3n's
        // PLE side-channel. Wired through but unused in Phase 2; Phase 3
        // consumes it as part of the AltUp correction tail.
        _per_layer_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, sliding_attention_mask, seqlen_offset)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
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

#[derive(Debug, Clone)]
struct PerLayerEmbedding {
    /// Looks up `[B, T] -> [B, T, num_layers * hidden_per_layer]`.
    embed_tokens_per_layer: candle_nn::Embedding,
    /// Projects the main `inputs_embeds` `[B, T, hidden_size]` into the
    /// same `num_layers * hidden_per_layer` space so the two signals
    /// can be summed.
    per_layer_model_projection: Linear,
    /// RMSNorm applied to the projection before it's merged with the
    /// embed-table lookup.
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
        let embed_tokens_per_layer = candle_nn::embedding(
            vocab_per_layer,
            total,
            vb.pp("embed_tokens_per_layer"),
        )?;
        let per_layer_model_projection =
            candle_nn::linear_no_bias(cfg.hidden_size, total, vb.pp("per_layer_model_projection"))?;
        let per_layer_projection_norm = RmsNorm::new(
            total,
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
        // `embed_tokens_per_layer(ids)` -> [B, T, L*H_per]
        let table = self.embed_tokens_per_layer.forward(input_ids)?;
        let table = table.reshape((b, t, self.num_hidden_layers, self.hidden_per_layer))?;

        // Project inputs_embeds and reshape to match.
        let proj = inputs_embeds.apply(&self.per_layer_model_projection)?;
        // Multiply by 1/√hidden_size before the RMSNorm so the trained
        // scale lines up with HF.
        let proj = (proj * self.per_layer_projection_scale)?;
        let proj = proj.apply(&self.per_layer_projection_norm)?;
        let proj =
            proj.reshape((b, t, self.num_hidden_layers, self.hidden_per_layer))?;

        // (proj + table) * rsqrt(2)
        let merged = (proj.broadcast_add(&table)? * self.per_layer_input_scale)?;
        Ok(merged.contiguous()?)
    }
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
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let per_layer_embedding = if cfg.hidden_size_per_layer_input.is_some() {
            Some(PerLayerEmbedding::new(cfg, vb_m.clone())?)
        } else {
            None
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
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        let mut xs = xs.clone();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let per_layer_slice = match per_layer_inputs {
                Some(table) => Some(table.narrow(2, layer_idx, 1)?.squeeze(2)?),
                None => None,
            };
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
                per_layer_slice.as_ref(),
            )?
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
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
        let (attention_mask, sliding_attention_mask) =
            self.create_attention_masks(batch_size, seq_len, seqlen_offset)?;

        let mut xs = xs.clone();
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let per_layer_slice = match per_layer_inputs {
                Some(table) => Some(table.narrow(2, layer_idx, 1)?.squeeze(2)?),
                None => None,
            };
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
                per_layer_slice.as_ref(),
            )?
        }
        xs.narrow(1, seq_len - 1, 1)?.apply(&self.norm)
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
}
