//! Gemma 4 quantized decoder — the QMatMul/GGUF counterpart to
//! [`gemma4::text::TextModel`].
//!
//! **Scope of this initial port** — basic decoder only:
//! - RMSNorm input + post-attention + pre-feedforward + post-feedforward
//! - GQA self-attention with q_norm / k_norm / v_norm
//! - p-RoPE (partial 25%) on full layers, standard RoPE on sliding
//! - SwiGLU MLP (gate + up + down)
//! - KV-cache (Normal for full layers, Rotating for sliding)
//!
//! **Not yet ported** (gated off via `Gemma4TextConfig::disable_altup`,
//! `disable_laurel`, `disable_per_layer_input_gate` and
//! `num_kv_shared_layers = 0`):
//! - Per-Layer Embeddings (PLE)
//! - AltUp (Alternating Updates)
//! - LAuReL (Learned Augmented Residual Layer)
//! - KV-share donor / receiver attention
//! - layer_scalar
//! - activation sparsity (Gaussian-topk)
//!
//! Output won't match Gemma 4 reference exactly with the auxiliary
//! towers off — this scaffold exists to exercise the QMatMul + GGUF
//! integration end-to-end on the WGPU q4_k.pwgsl kernel. Adding the
//! auxiliary towers is straightforward Linear → QMatMul translation;
//! the holdout is verifying each tower against a reference output,
//! which needs an Ollama-published gemma4:e2b GGUF + matching
//! reference run.

use std::sync::Arc;

use candle::quantized::{gguf_file, QMatMul, QTensor};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, Embedding};

#[derive(Debug, Clone)]
enum KvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

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
}

impl MLP {
    fn forward_inner(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let lhs = gate.apply(&self.act_fn)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * up)?)
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> { self.forward_inner(xs) }
}

// ── Attention ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    is_sliding: bool,
    rotary_emb_global: Arc<ProportionalRotaryEmbedding>,
    rotary_emb_local: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
}

impl Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = v_norm(&v, self.rms_norm_eps)?;

        let (q, k) = if self.is_sliding {
            self.rotary_emb_local.apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
        } else {
            self.rotary_emb_global.apply_rotary_emb_qkv(&q, &k, seqlen_offset)?
        };

        let v = v.contiguous()?;
        let (k, v) = match &mut self.kv_cache {
            KvCache::Normal(c) => c.append(&k.contiguous()?, &v)?,
            KvCache::Rotating(c) => c.append(&k.contiguous()?, &v)?,
        };

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Gemma 4 sets pre-softmax scale to 1.0 — q_norm/k_norm produce
        // unit-magnitude queries/keys so the dot-products are already
        // O(1). No divide here.
        let attn_weights = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;

        let mask = if self.is_sliding { sliding_attention_mask } else { attention_mask };
        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
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
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, sliding_attention_mask, seqlen_offset)?;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let normed = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        residual + mlp_out
    }

    fn clear_kv_cache(&mut self) { self.self_attn.clear_kv_cache(); }
}

// ── ModelWeights (top-level) ────────────────────────────────────────────────

pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
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

        let rotary_global = Arc::new(ProportionalRotaryEmbedding::new(
            dtype,
            cfg.head_dim,
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
        let tok_q = ct.tensor(reader, "token_embd.weight", device)?;
        let embed_tokens = Embedding::new(tok_q.dequantize(device)?, cfg.hidden_size);
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

            let q_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?)?;
            let k_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?)?;
            let v_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?)?;
            let o_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?)?;

            let q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                cfg.rms_norm_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                cfg.rms_norm_eps,
            )?;

            let kv_cache = if is_sliding {
                KvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(2, cfg.effective_sliding_window()))
            } else {
                KvCache::Normal(candle_nn::kv_cache::KvCache::new(2, cfg.max_position_embeddings))
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
                rotary_emb_global: rotary_global.clone(),
                rotary_emb_local: rotary_local.clone(),
                kv_cache,
            };

            let gate_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?)?;
            let up_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?)?;
            let down_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?)?;
            let mlp = MLP { gate_proj, up_proj, down_proj, act_fn: cfg.hidden_activation };

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

            layers.push(DecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            cfg: cfg.clone(),
        })
    }

    /// One forward step. `xs` is `[b, q_len]` token ids; result is
    /// `[b, q_len, vocab_size]` logits.
    pub fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, q_len) = xs.dims2()?;
        let mut hidden = self.embed_tokens.forward(xs)?;
        // Gemma scale: input embeddings are multiplied by sqrt(hidden_size).
        let scale = (self.cfg.hidden_size as f64).sqrt();
        hidden = (hidden * scale)?;

        let attention_mask = if q_len <= 1 {
            None
        } else {
            Some(prepare_attention_mask(b_sz, q_len, seqlen_offset, &self.device, hidden.dtype())?)
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

        for layer in self.layers.iter_mut() {
            hidden = layer.forward(
                &hidden,
                attention_mask.as_ref(),
                sliding_attention_mask.as_ref(),
                seqlen_offset,
            )?;
        }
        let hidden = self.norm.forward(&hidden)?;
        // Take the last token only for autoregressive sampling — same as
        // gemma4/text.rs.
        let hidden = hidden.i((.., q_len - 1, ..))?.unsqueeze(1)?;
        self.lm_head.forward(&hidden)
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
