//! Configuration for the Gemma 4 multimodal model.

use candle_nn::Activation;

// ── Text config defaults ────────────────────────────────────────────────────

fn default_attention_bias() -> bool {
    false
}
fn default_head_dim() -> usize {
    256
}
fn default_hidden_activation() -> Activation {
    Activation::GeluPytorchTanh
}
fn default_num_attention_heads() -> usize {
    8
}
fn default_num_key_value_heads() -> usize {
    4
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.
}
fn default_vocab_size() -> usize {
    262144
}
fn default_query_pre_attn_scalar() -> usize {
    256
}
fn default_max_position_embeddings() -> usize {
    131072
}
fn default_tie_word_embeddings() -> bool {
    true
}
fn default_sliding_window_pattern() -> usize {
    6
}
fn default_global_head_dim() -> usize {
    512
}
fn default_use_flash_attn() -> bool {
    false
}
// ── AltUp / LAuReL defaults (Gemma 3n) ──────────────────────────────────────
fn default_altup_num_inputs() -> usize {
    4
}
fn default_altup_active_idx() -> usize {
    0
}
fn default_altup_correct_scale() -> bool {
    true
}
fn default_altup_coef_clip() -> Option<f64> {
    Some(120.0)
}
fn default_laurel_rank() -> usize {
    64
}

// ── Rope parameters ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4RopeLayerParams {
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4RopeParameters {
    pub full_attention: Option<Gemma4RopeLayerParams>,
    pub sliding_attention: Option<Gemma4RopeLayerParams>,
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
    pub partial_rotary_factor: Option<f64>,
}

// ── Gemma4TextConfig ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4TextConfig {
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Optional per-layer intermediate_size override. When present, layer
    /// `i` uses `intermediate_sizes[i]` for its MLP gate/up/down
    /// projections; otherwise the scalar `intermediate_size` applies to
    /// every layer. Gemma4-E2B (Gemma 3n) ships with elastic MLP widths
    /// — some layers are 6144-wide, others 12288-wide — so loading those
    /// weights without shape mismatches requires the per-layer table.
    /// Vec length must equal `num_hidden_layers` when set.
    #[serde(default)]
    pub intermediate_sizes: Option<Vec<usize>>,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_sliding_window_pattern", alias = "_sliding_window_pattern")]
    pub sliding_window_pattern: usize,
    pub layer_types: Vec<String>,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    pub num_global_key_value_heads: Option<usize>,
    pub rope_parameters: Option<Gemma4RopeParameters>,
    pub use_bidirectional_attention: Option<String>,
    #[serde(default = "default_use_flash_attn")]
    pub use_flash_attn: bool,

    // ── Per-Layer Embeddings (Gemma 3n) ─────────────────────────────────
    /// Width of each per-layer auxiliary embedding (the small dim that
    /// gets added into every decoder layer alongside the main residual
    /// stream). `None` disables PLE entirely; Gemma 3n's E2B sets this
    /// to 256.
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    /// Vocabulary size of the per-layer embedding table. Gemma 3n uses a
    /// dedicated 262_144-entry table that's distinct from the main
    /// `vocab_size` (which adds soft-image / soft-audio token IDs on
    /// top). `None` falls back to the main `vocab_size`.
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,

    // ── AltUp (Alternating Updates) ─────────────────────────────────────
    /// Number of parallel hidden streams Gemma 3n maintains through the
    /// decoder. Each layer runs predict / activate / correct over
    /// `altup_num_inputs` copies of the hidden state.
    #[serde(default = "default_altup_num_inputs")]
    pub altup_num_inputs: usize,
    /// Index into the AltUp stack that carries the "primary" prediction —
    /// the one passed through input_layernorm + attention + MLP every
    /// layer. The other streams are corrected based on the primary's
    /// activated output.
    #[serde(default = "default_altup_active_idx")]
    pub altup_active_idx: usize,
    /// When `true` (the trained default for Gemma 3n) the corrected
    /// active prediction is multiplied by a learnable per-feature scale
    /// before being fed into the per-layer-input gate.
    #[serde(default = "default_altup_correct_scale")]
    pub altup_correct_scale: bool,
    /// During training, AltUp clamps `prediction_coefs` and
    /// `correction_coefs` to `[-altup_coef_clip, +altup_coef_clip]`.
    /// At inference this is informational only (no clamp on read).
    #[serde(default = "default_altup_coef_clip")]
    pub altup_coef_clip: Option<f64>,

    // ── LAuReL (Learned Augmented Residual Layer) ───────────────────────
    /// Rank of the low-rank residual augmentation applied alongside the
    /// attention output (mirrors HF `Gemma3nTextLaurelBlock`).
    #[serde(default = "default_laurel_rank")]
    pub laurel_rank: usize,

    // ── Activation sparsity (Gemma 3n) ──────────────────────────────────
    /// Per-layer sparsity ratio applied to `gate_proj` *before* the
    /// activation function. Sparsity 0.95 zeros out the bottom 95% of
    /// the per-token gate values via a Gaussian top-K threshold; 0.0
    /// disables sparsity for that layer. None ⇒ disabled for all layers.
    /// Vec length must equal `num_hidden_layers` when set.
    #[serde(default)]
    pub activation_sparsity_pattern: Option<Vec<f64>>,
}

impl Gemma4TextConfig {
    pub fn effective_sliding_window(&self) -> usize {
        if self.use_bidirectional_attention.as_deref() == Some("all") {
            (self.sliding_window / 2) + 1
        } else {
            self.sliding_window
        }
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.full_attention.as_ref())
            .and_then(|fa| fa.partial_rotary_factor)
            .unwrap_or(0.25)
    }

    pub fn rope_local_base_freq(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.sliding_attention.as_ref())
            .and_then(|sa| sa.rope_theta)
            .unwrap_or(10000.0)
    }

    pub fn is_sliding(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .map(|s| s == "sliding_attention")
            .unwrap_or(false)
    }

    /// Resolve the MLP intermediate_size for a specific decoder layer,
    /// preferring the per-layer override (`intermediate_sizes`) when set.
    pub fn intermediate_size_at(&self, layer_idx: usize) -> usize {
        self.intermediate_sizes
            .as_ref()
            .and_then(|v| v.get(layer_idx).copied())
            .unwrap_or(self.intermediate_size)
    }

    /// Resolve the activation-sparsity ratio for a given decoder layer.
    /// Returns `0.0` (no sparsity) when the model wasn't trained with
    /// the per-layer pattern.
    pub fn activation_sparsity_at(&self, layer_idx: usize) -> f64 {
        self.activation_sparsity_pattern
            .as_ref()
            .and_then(|v| v.get(layer_idx).copied())
            .unwrap_or(0.0)
    }
}

// ── Vision config defaults ──────────────────────────────────────────────────

fn default_vision_hidden_size() -> usize {
    768
}
fn default_vision_intermediate_size() -> usize {
    3072
}
fn default_vision_num_hidden_layers() -> usize {
    16
}
fn default_vision_num_attention_heads() -> usize {
    12
}
fn default_vision_num_key_value_heads() -> usize {
    12
}
fn default_vision_head_dim() -> usize {
    64
}
fn default_vision_hidden_activation() -> Activation {
    Activation::GeluPytorchTanh
}
fn default_vision_rms_norm_eps() -> f64 {
    1e-6
}
fn default_vision_patch_size() -> usize {
    16
}
fn default_vision_position_embedding_size() -> usize {
    10240
}
fn default_vision_pooling_kernel_size() -> usize {
    3
}
fn default_vision_default_output_length() -> usize {
    280
}
fn default_vision_standardize() -> bool {
    false
}

// ── Gemma4VisionConfig ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_vision_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_vision_hidden_activation")]
    pub hidden_activation: Activation,
    #[serde(default = "default_vision_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_position_embedding_size")]
    pub position_embedding_size: usize,
    #[serde(default = "default_vision_pooling_kernel_size")]
    pub pooling_kernel_size: usize,
    #[serde(default = "default_vision_default_output_length")]
    pub default_output_length: usize,
    #[serde(default = "default_vision_standardize")]
    pub standardize: bool,
    pub rope_parameters: Option<Gemma4RopeParameters>,
}

impl Gemma4VisionConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| {
                rp.full_attention
                    .as_ref()
                    .and_then(|fa| fa.rope_theta)
                    .or(rp.rope_theta)
            })
            .unwrap_or(100.0)
    }
}

// ── Audio config defaults ───────────────────────────────────────────────────

fn default_audio_input_feat_size() -> usize {
    128
}
fn default_audio_hidden_size() -> usize {
    1024
}
fn default_conf_attention_chunk_size() -> usize {
    12
}
fn default_conf_attention_context_left() -> usize {
    13
}
fn default_conf_attention_context_right() -> usize {
    0
}
fn default_conf_attention_invalid_logits_value() -> f64 {
    -1e9
}
fn default_conf_attention_logit_cap() -> f64 {
    50.0
}
fn default_conf_num_attention_heads() -> usize {
    8
}
fn default_conf_num_hidden_layers() -> usize {
    12
}
fn default_conf_conv_kernel_size() -> usize {
    5
}
fn default_conf_reduction_factor() -> usize {
    1
}
fn default_conf_residual_weight() -> f64 {
    0.5
}
fn default_sscp_conv_channel_size() -> Vec<usize> {
    vec![128, 32]
}
fn default_sscp_conv_kernel_size() -> Vec<Vec<usize>> {
    vec![vec![3, 3], vec![3, 3]]
}
fn default_sscp_conv_stride_size() -> Vec<Vec<usize>> {
    vec![vec![2, 2], vec![2, 2]]
}
fn default_audio_vocab_size() -> usize {
    128
}
fn default_sscp_conv_group_norm_eps() -> f64 {
    1e-6
}
fn default_sscp_conv_eps() -> f64 {
    1e-3
}
fn default_audio_rms_norm_eps() -> f64 {
    1e-6
}
fn default_gradient_clipping() -> f64 {
    1e10
}
fn default_output_proj_dims() -> Option<usize> {
    Some(1536)
}

// ── Gemma4AudioConfig ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4AudioConfig {
    #[serde(default = "default_audio_input_feat_size")]
    pub input_feat_size: usize,
    #[serde(default = "default_audio_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_output_proj_dims")]
    pub output_proj_dims: Option<usize>,
    #[serde(default = "default_conf_attention_chunk_size", alias = "attention_chunk_size")]
    pub conf_attention_chunk_size: usize,
    #[serde(
        default = "default_conf_attention_context_left",
        alias = "attention_context_left"
    )]
    pub conf_attention_context_left: usize,
    #[serde(
        default = "default_conf_attention_context_right",
        alias = "attention_context_right"
    )]
    pub conf_attention_context_right: usize,
    #[serde(
        default = "default_conf_attention_invalid_logits_value",
        alias = "attention_invalid_logits_value"
    )]
    pub conf_attention_invalid_logits_value: f64,
    #[serde(default = "default_conf_attention_logit_cap", alias = "attention_logit_cap")]
    pub conf_attention_logit_cap: f64,
    #[serde(default = "default_conf_num_attention_heads", alias = "num_attention_heads")]
    pub conf_num_attention_heads: usize,
    #[serde(default = "default_conf_num_hidden_layers", alias = "num_hidden_layers")]
    pub conf_num_hidden_layers: usize,
    #[serde(default = "default_conf_conv_kernel_size", alias = "conv_kernel_size")]
    pub conf_conv_kernel_size: usize,
    #[serde(default = "default_conf_reduction_factor")]
    pub conf_reduction_factor: usize,
    #[serde(default = "default_conf_residual_weight", alias = "residual_weight")]
    pub conf_residual_weight: f64,
    #[serde(
        default = "default_sscp_conv_channel_size",
        alias = "subsampling_conv_channels"
    )]
    pub sscp_conv_channel_size: Vec<usize>,
    #[serde(default = "default_sscp_conv_kernel_size")]
    pub sscp_conv_kernel_size: Vec<Vec<usize>>,
    #[serde(default = "default_sscp_conv_stride_size")]
    pub sscp_conv_stride_size: Vec<Vec<usize>>,
    #[serde(default = "default_audio_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_sscp_conv_group_norm_eps")]
    pub sscp_conv_group_norm_eps: f64,
    #[serde(default = "default_sscp_conv_eps")]
    pub sscp_conv_eps: f64,
    #[serde(default = "default_audio_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_gradient_clipping")]
    pub gradient_clipping: f64,
}

// ── Top-level config defaults ───────────────────────────────────────────────

fn default_image_token_id() -> usize {
    258880
}
fn default_audio_token_id() -> usize {
    258881
}
fn default_video_token_id() -> usize {
    258884
}

// ── Gemma4Config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
    pub vision_config: Gemma4VisionConfig,
    pub audio_config: Option<Gemma4AudioConfig>,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: usize,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: usize,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: usize,
}
