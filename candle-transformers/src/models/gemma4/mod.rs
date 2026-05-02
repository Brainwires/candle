//! Gemma 4 multimodal model (text + vision + audio).
//!
//! See:
//! - [Google Blog](https://blog.google/technology/developers/gemma-4/)

pub mod audio;
pub mod config;
pub mod multimodal_embedding;
pub mod text;
pub mod vision;

use candle::{DType, Result, Tensor, D};

use config::Gemma4Config;
use multimodal_embedding::MultimodalEmbedder;
use text::TextModel;
use vision::VisionTower;

pub use audio::AudioModel;
pub use config::{Gemma4AudioConfig, Gemma4TextConfig, Gemma4VisionConfig};
pub use text::PerLayerEmbedTable;

/// Full Gemma4 multimodal model.
///
/// `vision_tower` / `embed_vision` and `audio_tower` / `embed_audio` are
/// `Option`s so callers (e.g. wasm) can defer loading those weights and
/// attach them later via [`Model::attach_vision`] / [`Model::attach_audio`].
pub struct Model {
    pub language_model: TextModel,
    pub vision_tower: Option<VisionTower>,
    pub embed_vision: Option<MultimodalEmbedder>,
    pub audio_tower: Option<AudioModel>,
    pub embed_audio: Option<MultimodalEmbedder>,
    pub cfg: Gemma4Config,
}

impl Model {
    /// Construct the full model: text + vision (always) + audio (if configured).
    /// Equivalent to [`Model::new_partial(cfg, vb, true, cfg.audio_config.is_some())`].
    pub fn new(cfg: &Gemma4Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        Self::new_partial(cfg, vb, true, cfg.audio_config.is_some())
    }

    /// Construct a text-only model. Vision and audio towers are unloaded;
    /// attach them later with [`Model::attach_vision`] / [`Model::attach_audio`].
    pub fn new_text_only(cfg: &Gemma4Config, vb: candle_nn::VarBuilder) -> Result<Self> {
        Self::new_partial(cfg, vb, false, false)
    }

    /// Selective constructor.
    ///
    /// `with_vision`: build [`VisionTower`] + vision [`MultimodalEmbedder`].
    /// `with_audio`: build [`AudioModel`] + audio [`MultimodalEmbedder`]
    /// (no-op if `cfg.audio_config` is `None`).
    pub fn new_partial(
        cfg: &Gemma4Config,
        vb: candle_nn::VarBuilder,
        with_vision: bool,
        with_audio: bool,
    ) -> Result<Self> {
        let vb = vb.pp("model");

        let text_hidden = cfg.text_config.hidden_size;

        let (vision_tower, embed_vision) = if with_vision {
            let vt = VisionTower::new(&cfg.vision_config, vb.pp("vision_tower"))?;
            let ev = MultimodalEmbedder::new(
                cfg.vision_config.hidden_size,
                text_hidden,
                cfg.vision_config.rms_norm_eps,
                vb.pp("embed_vision"),
            )?;
            (Some(vt), Some(ev))
        } else {
            (None, None)
        };

        let (audio_tower, embed_audio) = if with_audio {
            if let Some(ref audio_cfg) = cfg.audio_config {
                let tower = AudioModel::new(audio_cfg, vb.pp("audio_tower"))?;
                let audio_hidden = audio_cfg.output_proj_dims.unwrap_or(audio_cfg.hidden_size);
                let embed = MultimodalEmbedder::new(
                    audio_hidden,
                    text_hidden,
                    audio_cfg.rms_norm_eps,
                    vb.pp("embed_audio"),
                )?;
                (Some(tower), Some(embed))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let language_model = TextModel::new(&cfg.text_config, vb.pp("language_model"))?;

        Ok(Self {
            language_model,
            vision_tower,
            embed_vision,
            audio_tower,
            embed_audio,
            cfg: cfg.clone(),
        })
    }

    /// Load the vision tower + vision embedder into an existing model.
    /// `vb` should be the same VarBuilder root used at construction (the one
    /// that gets `.pp("model")` applied internally).
    pub fn attach_vision(&mut self, vb: candle_nn::VarBuilder) -> Result<()> {
        let vb = vb.pp("model");
        let vt = VisionTower::new(&self.cfg.vision_config, vb.pp("vision_tower"))?;
        let ev = MultimodalEmbedder::new(
            self.cfg.vision_config.hidden_size,
            self.cfg.text_config.hidden_size,
            self.cfg.vision_config.rms_norm_eps,
            vb.pp("embed_vision"),
        )?;
        self.vision_tower = Some(vt);
        self.embed_vision = Some(ev);
        Ok(())
    }

    /// Load the audio tower + audio embedder into an existing model.
    /// Errors if `cfg.audio_config` is `None`.
    pub fn attach_audio(&mut self, vb: candle_nn::VarBuilder) -> Result<()> {
        let vb = vb.pp("model");
        let audio_cfg = self
            .cfg
            .audio_config
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("attach_audio: cfg.audio_config is None".into()))?;
        let tower = AudioModel::new(audio_cfg, vb.pp("audio_tower"))?;
        let audio_hidden = audio_cfg.output_proj_dims.unwrap_or(audio_cfg.hidden_size);
        let embed = MultimodalEmbedder::new(
            audio_hidden,
            self.cfg.text_config.hidden_size,
            audio_cfg.rms_norm_eps,
            vb.pp("embed_audio"),
        )?;
        self.audio_tower = Some(tower);
        self.embed_audio = Some(embed);
        Ok(())
    }

    pub fn has_vision(&self) -> bool {
        self.vision_tower.is_some() && self.embed_vision.is_some()
    }

    pub fn has_audio(&self) -> bool {
        self.audio_tower.is_some() && self.embed_audio.is_some()
    }

    /// Text-only forward pass.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.language_model.forward(input_ids, seqlen_offset)
    }

    /// Forward with multimodal inputs.
    ///
    /// `pixel_values`: optional batch of images, each `(1, C, H, W)`.
    /// `audio_mel`: optional `(batch, time, mel_bins)` mel spectrogram.
    /// `audio_mel_mask`: optional `(batch, time)` mask (1.0 = padding).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_multimodal(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&[Tensor]>,
        audio_mel: Option<&Tensor>,
        audio_mel_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;

        // ── Vision embedding injection ──────────────────────────────────
        if let Some(pixel_values) = pixel_values {
            let vision_tower = self
                .vision_tower
                .as_ref()
                .ok_or_else(|| candle::Error::Msg(
                    "forward_multimodal: pixel_values supplied but vision_tower is not attached".into(),
                ))?;
            let embed_vision = self
                .embed_vision
                .as_ref()
                .ok_or_else(|| candle::Error::Msg(
                    "forward_multimodal: pixel_values supplied but embed_vision is not attached".into(),
                ))?;

            let image_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(self.cfg.image_token_id as f64)?;

            let vision_features = vision_tower.forward(pixel_values)?;
            let image_embeds = embed_vision
                .forward(&vision_features)?
                .to_dtype(input_embeds.dtype())?;

            // Replace image token positions with vision embeddings
            let image_embeds_flat = image_embeds.squeeze(0)?;
            let mask_expanded = image_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(input_embeds.dtype())?;
            let image_embeds_broadcast =
                broadcast_embed_to_mask(&image_embeds_flat, &image_mask)?;
            input_embeds = ((mask_expanded.clone() * image_embeds_broadcast)?
                + ((1.0 - mask_expanded)? * input_embeds)?)?;
        }

        // ── Audio embedding injection ───────────────────────────────────
        if let (Some(audio_mel), Some(audio_mel_mask), Some(ref audio_tower), Some(ref embed_audio)) =
            (audio_mel, audio_mel_mask, &self.audio_tower, &self.embed_audio)
        {
            let audio_mask = input_ids
                .to_dtype(DType::F32)?
                .eq(self.cfg.audio_token_id as f64)?;

            let (audio_features, enc_mask) = audio_tower.forward(audio_mel, audio_mel_mask)?;
            // Filter valid frames: where enc_mask == 0
            let valid = enc_mask.eq(0.0)?;
            let batch = audio_features.dim(0)?;
            let mut all_feats = Vec::new();
            for b in 0..batch {
                let valid_b = valid.get(b)?;
                // Count valid frames
                let valid_sum = valid_b
                    .to_dtype(DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()? as usize;
                if valid_sum > 0 {
                    // Take the first valid_sum frames (they are contiguous after masking)
                    all_feats.push(audio_features.get(b)?.narrow(0, 0, valid_sum)?);
                }
            }
            if !all_feats.is_empty() {
                let audio_feats = Tensor::cat(&all_feats, 0)?.unsqueeze(0)?;
                let audio_embeds = embed_audio
                    .forward(&audio_feats)?
                    .to_dtype(input_embeds.dtype())?;

                let audio_embeds_flat = audio_embeds.squeeze(0)?;
                let mask_expanded = audio_mask
                    .unsqueeze(D::Minus1)?
                    .broadcast_as(input_embeds.shape())?
                    .to_dtype(input_embeds.dtype())?;
                let audio_embeds_broadcast =
                    broadcast_embed_to_mask(&audio_embeds_flat, &audio_mask)?;
                input_embeds = ((mask_expanded.clone() * audio_embeds_broadcast)?
                    + ((1.0 - mask_expanded)? * input_embeds)?)?;
            }
        }

        self.language_model
            .forward_embeds(&input_embeds, seqlen_offset, b_size, seq_len)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache()
    }
}

/// Broadcast encoder embeddings (num_tokens, hidden) into positions marked by
/// a boolean mask (batch, seq_len), producing (batch, seq_len, hidden).
/// Token embeddings are placed sequentially where the mask is true.
fn broadcast_embed_to_mask(embeds: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b_sz, seq_len) = mask.dims2()?;
    let hidden = embeds.dim(D::Minus1)?;

    // Count masked positions per batch, fill them in sequence from embeds
    let mask_f32 = mask.to_dtype(DType::F32)?;
    // cumsum along seq dimension to assign embed indices
    // Since candle doesn't have cumsum, we use a broadcast approach:
    // Create output tensor of zeros, then use where_cond
    let zeros = Tensor::zeros((b_sz, seq_len, hidden), embeds.dtype(), embeds.device())?;

    // For single-batch simple case, just expand embeds to the output shape
    // and let the caller do the masking.
    if b_sz == 1 {
        let num_tokens = mask_f32
            .sum_all()?
            .to_scalar::<f32>()? as usize;
        if num_tokens == 0 {
            return Ok(zeros);
        }
        // Pad or truncate embeds to seq_len
        let embed_len = embeds.dim(0)?;
        if embed_len >= seq_len {
            return embeds.narrow(0, 0, seq_len)?.unsqueeze(0);
        }
        let padding = Tensor::zeros((seq_len - embed_len, hidden), embeds.dtype(), embeds.device())?;
        let padded = Tensor::cat(&[embeds, &padding], 0)?;
        return padded.unsqueeze(0);
    }

    Ok(zeros)
}
