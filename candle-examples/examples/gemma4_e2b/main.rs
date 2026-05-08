//! Native CPU validation harness for `quantized_gemma4_e2b`.
//!
//! Greedy decode (no sampling) — emits each token as `[step N] id=X
//! token=Y` so the output can be diffed against a reference Ollama
//! run line-by-line.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{anyhow, Result};
use clap::Parser;
use std::io::Write;

use candle::quantized::gguf_file;
use candle::{DType, Device, Tensor};
use candle_nn::Activation;
use candle_transformers::models::gemma4::config::{
    Gemma4RopeLayerParams, Gemma4RopeParameters, Gemma4TextConfig,
};
use candle_transformers::models::quantized_gemma4_e2b::ModelWeights;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the GGUF weights file.
    #[arg(long)]
    gguf_path: String,

    /// Path to a HuggingFace tokenizer.json.
    #[arg(long)]
    tokenizer_file: String,

    /// User prompt (no chat template applied — feed the raw text).
    #[arg(long)]
    prompt: String,

    /// Number of tokens to generate.
    #[arg(long, default_value_t = 10)]
    max_new_tokens: usize,

    /// Device: "cpu" only is supported here. Accepted for symmetry with
    /// the broader plan; any other value falls back to CPU with a note.
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn build_config_from_gguf(ct: &gguf_file::Content) -> Result<Gemma4TextConfig> {
    let prefix = ["gemma4", "gemma3"]
        .iter()
        .find(|p| {
            ct.metadata
                .contains_key(&format!("{p}.attention.head_count"))
        })
        .copied()
        .unwrap_or("gemma4");

    let md_get = |s: &str| -> Result<&gguf_file::Value> {
        let key = format!("{prefix}.{s}");
        ct.metadata
            .get(&key)
            .ok_or_else(|| anyhow!("cannot find {key} in GGUF metadata"))
    };
    let md_get_opt = |s: &str| -> Option<gguf_file::Value> {
        let key = format!("{prefix}.{s}");
        ct.metadata.get(&key).cloned()
    };

    let num_attention_heads = md_get("attention.head_count")?.to_u32()? as usize;
    let num_key_value_heads = md_get("attention.head_count_kv")?.to_u32()? as usize;
    let num_hidden_layers = md_get("block_count")?.to_u32()? as usize;
    let hidden_size = md_get("embedding_length")?.to_u32()? as usize;

    let (intermediate_size, intermediate_sizes) = match md_get("feed_forward_length")? {
        gguf_file::Value::Array(arr) => {
            let mut sizes = Vec::with_capacity(arr.len());
            for v in arr {
                let n = match v {
                    gguf_file::Value::I32(n) => *n as usize,
                    gguf_file::Value::U32(n) => *n as usize,
                    other => {
                        return Err(anyhow!(
                            "feed_forward_length array entry has unexpected type: {other:?}"
                        ))
                    }
                };
                sizes.push(n);
            }
            let first = sizes.first().copied().unwrap_or(0);
            (first, Some(sizes))
        }
        v => (v.to_u32()? as usize, None),
    };

    let head_dim_swa = md_get_opt("attention.key_length_swa")
        .and_then(|m| m.to_u32().ok())
        .map(|v| v as usize);
    let head_dim_full = md_get("attention.key_length")?.to_u32()? as usize;
    let head_dim = head_dim_swa.unwrap_or(head_dim_full);
    let global_head_dim = head_dim_full;

    let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let max_position_embeddings = md_get("context_length")?.to_u32()? as usize;

    let rope_theta_full = md_get_opt("rope.freq_base")
        .and_then(|m| m.to_f32().ok())
        .unwrap_or(1_000_000.0) as f64;
    let rope_theta_swa = md_get_opt("rope.freq_base_swa")
        .and_then(|m| m.to_f32().ok())
        .unwrap_or(10_000.0) as f64;

    let vocab_size = ct
        .tensor_infos
        .get("token_embd.weight")
        .map(|ti| ti.shape.dims()[0])
        .ok_or_else(|| anyhow!("missing token_embd.weight in GGUF"))?;

    let sliding_window = md_get_opt("attention.sliding_window")
        .and_then(|m| m.to_u32().ok())
        .unwrap_or(512) as usize;

    let layer_types: Vec<String> = match md_get_opt("attention.sliding_window_pattern") {
        Some(gguf_file::Value::Array(arr)) => {
            let mut types = Vec::with_capacity(arr.len());
            for v in &arr {
                let is_sliding = matches!(v, gguf_file::Value::Bool(true));
                types.push(if is_sliding {
                    "sliding_attention".to_string()
                } else {
                    "full_attention".to_string()
                });
            }
            types.resize(num_hidden_layers, "sliding_attention".to_string());
            types
        }
        _ => {
            let period = md_get_opt("attention.sliding_window_type")
                .and_then(|m| m.to_u32().ok())
                .unwrap_or(5) as usize;
            (0..num_hidden_layers)
                .map(|i| {
                    let is_sliding = (i + 1) % period > 0;
                    if is_sliding {
                        "sliding_attention".to_string()
                    } else {
                        "full_attention".to_string()
                    }
                })
                .collect()
        }
    };
    let sliding_window_pattern = layer_types
        .iter()
        .position(|t| t == "full_attention")
        .map(|p| p + 1)
        .unwrap_or(num_hidden_layers);

    let num_kv_shared_layers = md_get_opt("attention.shared_kv_layers")
        .or_else(|| md_get_opt("attention.kv_shared_layers"))
        .and_then(|m| m.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or_else(|| match num_hidden_layers {
            35 => 20,
            30 => 10,
            _ => 0,
        });

    let final_logit_softcapping = md_get_opt("final_logit_softcapping")
        .and_then(|m| m.to_f32().ok())
        .map(|v| v as f64);

    let hidden_size_per_layer_input = md_get_opt("embedding_length_per_layer_input")
        .and_then(|m| m.to_u32().ok())
        .map(|v| v as usize);

    let rope_parameters = Some(Gemma4RopeParameters {
        full_attention: Some(Gemma4RopeLayerParams {
            rope_theta: Some(rope_theta_full),
            rope_type: None,
            partial_rotary_factor: Some(0.25),
        }),
        sliding_attention: Some(Gemma4RopeLayerParams {
            rope_theta: Some(rope_theta_swa),
            rope_type: None,
            partial_rotary_factor: None,
        }),
        rope_theta: Some(rope_theta_full),
        rope_type: None,
        partial_rotary_factor: Some(0.25),
    });

    let _ = num_kv_shared_layers; // currently derived inside the model via cfg.donor_layer_idx_for

    Ok(Gemma4TextConfig {
        attention_bias: false,
        head_dim,
        hidden_activation: Activation::GeluPytorchTanh,
        hidden_size,
        intermediate_size,
        intermediate_sizes,
        num_attention_heads,
        num_hidden_layers,
        num_key_value_heads,
        rms_norm_eps,
        rope_theta: rope_theta_full,
        vocab_size,
        sliding_window,
        final_logit_softcapping,
        query_pre_attn_scalar: 1,
        max_position_embeddings,
        tie_word_embeddings: true,
        sliding_window_pattern,
        layer_types,
        global_head_dim,
        num_global_key_value_heads: None,
        rope_parameters,
        use_bidirectional_attention: None,
        use_flash_attn: false,
        hidden_size_per_layer_input,
        vocab_size_per_layer_input: None,
        altup_num_inputs: 1,
        altup_active_idx: 0,
        altup_correct_scale: false,
        altup_coef_clip: None,
        laurel_rank: 0,
        activation_sparsity_pattern: None,
        disable_altup: true,
        disable_laurel: true,
        disable_per_layer_input_gate: false,
        num_kv_shared_layers,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.device.to_lowercase() != "cpu" {
        eprintln!(
            "[gemma4_e2b] only --device cpu is wired in this validation harness; \
             continuing on CPU regardless of --device={}",
            args.device
        );
    }
    let device = Device::Cpu;

    println!("[gemma4_e2b] loading tokenizer from {}", args.tokenizer_file);
    let tokenizer = Tokenizer::from_file(&args.tokenizer_file).map_err(anyhow::Error::msg)?;

    println!("[gemma4_e2b] opening GGUF at {}", args.gguf_path);
    let mut file = std::fs::File::open(&args.gguf_path)?;
    let content = gguf_file::Content::read(&mut file)?;

    let cfg = build_config_from_gguf(&content)?;
    println!(
        "[gemma4_e2b] config: layers={} hidden={} attn_heads={} kv_heads={} \
         head_dim={} global_head_dim={} sliding_window={} ple_dim={:?} \
         num_kv_shared_layers={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.global_head_dim,
        cfg.sliding_window,
        cfg.hidden_size_per_layer_input,
        cfg.num_kv_shared_layers,
    );

    println!("[gemma4_e2b] loading model weights…");
    let load_start = std::time::Instant::now();
    let mut model = ModelWeights::from_gguf(content, &mut file, &device, &cfg)?;
    println!(
        "[gemma4_e2b] weights loaded in {:.2?}",
        load_start.elapsed()
    );

    let encoded = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(anyhow::Error::msg)?;
    let mut tokens: Vec<u32> = encoded.get_ids().to_vec();
    println!(
        "[gemma4_e2b] prompt tokens ({}): {:?}",
        tokens.len(),
        tokens
    );

    // Prefill — feed the full prompt at offset 0.
    let prefill_start = std::time::Instant::now();
    let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let next_token = greedy_argmax(&logits)?;
    let prefill_dt = prefill_start.elapsed();
    println!(
        "[gemma4_e2b] prefill done in {:.2?} ({} tokens)",
        prefill_dt,
        tokens.len()
    );
    let mut step = 0usize;
    print_step(step, next_token, &tokenizer);
    tokens.push(next_token);
    step += 1;

    for _ in 1..args.max_new_tokens {
        let last = *tokens.last().unwrap();
        let input = Tensor::new(&[last][..], &device)?.unsqueeze(0)?;
        let seqlen_offset = tokens.len() - 1;
        let logits = model.forward(&input, seqlen_offset)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = greedy_argmax(&logits)?;
        print_step(step, next_token, &tokenizer);
        tokens.push(next_token);
        step += 1;
    }

    std::io::stdout().flush()?;
    Ok(())
}

fn greedy_argmax(logits: &Tensor) -> Result<u32> {
    let v: Vec<f32> = logits.to_vec1()?;
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best_id = i as u32;
        }
    }
    Ok(best_id)
}

fn print_step(step: usize, id: u32, tokenizer: &Tokenizer) {
    let token_str = tokenizer
        .decode(&[id], false)
        .unwrap_or_else(|_| "<decode-err>".to_string());
    let escaped = token_str.replace('\\', "\\\\").replace('\n', "\\n");
    println!("[step {step}] id={id} token={escaped}");
}
