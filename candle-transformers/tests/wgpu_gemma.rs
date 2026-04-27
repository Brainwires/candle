//! WebGPU Gemma 3 end-to-end correctness tests (Phase 3.7).
//!
//! Builds a tiny synthetic Gemma 3 config (2 layers, hidden=64, 4 heads,
//! head_dim=16, vocab=32), initializes weights on CPU via VarMap,
//! clones them to Device::Wgpu, runs forward on both backends, and
//! checks that logits match within tolerance.
//!
//! Also tests KV-cache consistency across multiple decoding steps.
//!
//! Skips gracefully when no Wgpu adapter is available.
//!
//! Run:
//!   cargo test -p candle-transformers --features wgpu --test wgpu_gemma
#![cfg(feature = "wgpu")]

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma3;
use std::collections::HashMap;

fn try_wgpu() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_diff = 0f32;
    let mut idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
            idx = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{label}: max abs diff {max_diff} > tol {tol} at index {idx} ({} vs {})",
        a[idx],
        b[idx]
    );
    max_diff
}

/// Tiny Gemma 3 config for testing.
fn tiny_config() -> gemma3::Config {
    gemma3::Config {
        attention_bias: false,
        head_dim: 16,
        hidden_activation: candle_nn::Activation::GeluPytorchTanh,
        hidden_size: 64,
        intermediate_size: 128,
        num_attention_heads: 4,
        num_hidden_layers: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        rope_local_base_freq: 10000.0,
        vocab_size: 32,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: 16,
        sliding_window: 32,
        sliding_window_pattern: 2,
        max_position_embeddings: 64,
    }
}

/// Build a HashMap of random tensors for all Gemma 3 weights on the CPU,
/// then convert to the target device.
fn build_weight_map(cfg: &gemma3::Config, device: &Device) -> HashMap<String, Tensor> {
    let cpu = Device::Cpu;
    let _dtype = DType::F32;
    let hs = cfg.hidden_size;
    let is = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let vs = cfg.vocab_size;

    let mut map = HashMap::new();

    // Embedding.
    let seed_val = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.00123 + 0.01).sin() * 0.1)
            .collect()
    };

    let add = |map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]| {
        let n: usize = shape.iter().product();
        let data = seed_val(n);
        let t_cpu = Tensor::from_slice(&data, shape, &cpu).unwrap();
        let t = if device.is_cpu() {
            t_cpu
        } else {
            t_cpu.to_device(device).unwrap()
        };
        map.insert(name.to_string(), t);
    };

    add(&mut map, "model.embed_tokens.weight", &[vs, hs]);
    add(&mut map, "model.norm.weight", &[hs]);

    for layer_idx in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{layer_idx}");

        // Attention projections.
        add(
            &mut map,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[nh * hd, hs],
        );
        add(
            &mut map,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[nkv * hd, hs],
        );
        add(
            &mut map,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[nkv * hd, hs],
        );
        add(
            &mut map,
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[hs, nh * hd],
        );

        // QK norms.
        add(
            &mut map,
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[hd],
        );
        add(
            &mut map,
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[hd],
        );

        // MLP.
        add(
            &mut map,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[is, hs],
        );
        add(&mut map, &format!("{prefix}.mlp.up_proj.weight"), &[is, hs]);
        add(
            &mut map,
            &format!("{prefix}.mlp.down_proj.weight"),
            &[hs, is],
        );

        // Layer norms.
        add(&mut map, &format!("{prefix}.input_layernorm.weight"), &[hs]);
        add(
            &mut map,
            &format!("{prefix}.pre_feedforward_layernorm.weight"),
            &[hs],
        );
        add(
            &mut map,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            &[hs],
        );
        add(
            &mut map,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[hs],
        );
    }

    map
}

/// Softmax-last-dim on GPU matches CPU.
#[test]
fn softmax_last_dim() {
    let Some(gpu) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;

    let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.3).collect();

    let t_cpu = Tensor::from_slice(&data, (2, 8), &cpu).unwrap();
    let out_cpu = candle_nn::ops::softmax_last_dim(&t_cpu).unwrap();
    let out_cpu_flat: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();

    let t_gpu = Tensor::from_slice(&data, (2, 8), &gpu).unwrap();
    let out_gpu = candle_nn::ops::softmax_last_dim(&t_gpu).unwrap();
    let out_gpu_flat: Vec<f32> = out_gpu.flatten_all().unwrap().to_vec1().unwrap();

    let diff = assert_close(&out_cpu_flat, &out_gpu_flat, 1e-5, "softmax_last_dim");
    eprintln!("[softmax_last_dim] max diff = {diff}");
}

/// Softmax with rows > 64 elements to test strided loop in the kernel.
#[test]
fn softmax_large_row() {
    let Some(gpu) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;

    let n = 256;
    let data: Vec<f32> = (0..(2 * n)).map(|i| ((i as f32) * 0.01).sin()).collect();

    let t_cpu = Tensor::from_slice(&data, (2, n), &cpu).unwrap();
    let out_cpu = candle_nn::ops::softmax_last_dim(&t_cpu).unwrap();
    let out_cpu_flat: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();

    let t_gpu = Tensor::from_slice(&data, (2, n), &gpu).unwrap();
    let out_gpu = candle_nn::ops::softmax_last_dim(&t_gpu).unwrap();
    let out_gpu_flat: Vec<f32> = out_gpu.flatten_all().unwrap().to_vec1().unwrap();

    let diff = assert_close(&out_cpu_flat, &out_gpu_flat, 1e-5, "softmax_large_row");
    eprintln!("[softmax_large_row] max diff = {diff}");
}

/// End-to-end: tiny Gemma 3 forward pass on Wgpu matches CPU.
#[test]
fn gemma_tiny_e2e() {
    let Some(gpu) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;
    let cfg = tiny_config();

    // Build weight maps for both devices.
    let cpu_weights = build_weight_map(&cfg, &cpu);
    let gpu_weights = build_weight_map(&cfg, &gpu);

    let vb_cpu = VarBuilder::from_tensors(cpu_weights, DType::F32, &cpu);
    let vb_gpu = VarBuilder::from_tensors(gpu_weights, DType::F32, &gpu);

    let mut model_cpu = gemma3::Model::new(false, &cfg, vb_cpu).unwrap();
    let mut model_gpu = gemma3::Model::new(false, &cfg, vb_gpu).unwrap();

    // Input token IDs: batch=1, seq_len=4.
    let input_ids_data: Vec<u32> = vec![1, 5, 12, 28];
    let input_cpu = Tensor::from_slice(&input_ids_data, (1, 4), &cpu).unwrap();
    let input_gpu = Tensor::from_slice(&input_ids_data, (1, 4), &gpu).unwrap();

    // Forward pass.
    let logits_cpu = model_cpu.forward(&input_cpu, 0).unwrap();
    let logits_gpu = model_gpu.forward(&input_gpu, 0).unwrap();

    let cpu_flat: Vec<f32> = logits_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let gpu_flat: Vec<f32> = logits_gpu.flatten_all().unwrap().to_vec1().unwrap();

    eprintln!(
        "[gemma_tiny_e2e] CPU logits shape: {:?}",
        logits_cpu.shape()
    );
    eprintln!(
        "[gemma_tiny_e2e] GPU logits shape: {:?}",
        logits_gpu.shape()
    );
    eprintln!(
        "[gemma_tiny_e2e] CPU logits (first 8): {:?}",
        &cpu_flat[..cpu_flat.len().min(8)]
    );
    eprintln!(
        "[gemma_tiny_e2e] GPU logits (first 8): {:?}",
        &gpu_flat[..gpu_flat.len().min(8)]
    );

    let diff = assert_close(&cpu_flat, &gpu_flat, 1e-2, "gemma_tiny_e2e");
    eprintln!("[gemma_tiny_e2e] max diff = {diff}");
}

/// KV cache consistency: initial prompt then token-by-token generation.
#[test]
fn gemma_tiny_kv_cache_steps() {
    let Some(gpu) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;
    let cfg = tiny_config();

    let cpu_weights = build_weight_map(&cfg, &cpu);
    let gpu_weights = build_weight_map(&cfg, &gpu);

    let vb_cpu = VarBuilder::from_tensors(cpu_weights, DType::F32, &cpu);
    let vb_gpu = VarBuilder::from_tensors(gpu_weights, DType::F32, &gpu);

    let mut model_cpu = gemma3::Model::new(false, &cfg, vb_cpu).unwrap();
    let mut model_gpu = gemma3::Model::new(false, &cfg, vb_gpu).unwrap();

    // Initial prompt: 4 tokens.
    let prompt: Vec<u32> = vec![1, 5, 12, 28];
    let prompt_len = prompt.len();

    let input_cpu = Tensor::from_slice(&prompt, (1, prompt_len), &cpu).unwrap();
    let input_gpu = Tensor::from_slice(&prompt, (1, prompt_len), &gpu).unwrap();

    let logits_cpu = model_cpu.forward(&input_cpu, 0).unwrap();
    let logits_gpu = model_gpu.forward(&input_gpu, 0).unwrap();

    let cpu_flat: Vec<f32> = logits_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let gpu_flat: Vec<f32> = logits_gpu.flatten_all().unwrap().to_vec1().unwrap();

    let diff = assert_close(&cpu_flat, &gpu_flat, 1e-2, "kv_cache_step_0");
    eprintln!("[kv_cache_step_0] max diff = {diff}");

    // Generate 4 tokens one at a time using KV cache.
    let mut seqlen_offset = prompt_len;
    let next_tokens: Vec<u32> = vec![3, 15, 7, 22];

    for (step, &tok) in next_tokens.iter().enumerate() {
        let tok_cpu = Tensor::from_slice(&[tok], (1, 1), &cpu).unwrap();
        let tok_gpu = Tensor::from_slice(&[tok], (1, 1), &gpu).unwrap();

        let logits_cpu = model_cpu.forward(&tok_cpu, seqlen_offset).unwrap();
        let logits_gpu = model_gpu.forward(&tok_gpu, seqlen_offset).unwrap();

        let cpu_flat: Vec<f32> = logits_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let gpu_flat: Vec<f32> = logits_gpu.flatten_all().unwrap().to_vec1().unwrap();

        let diff = assert_close(
            &cpu_flat,
            &gpu_flat,
            1e-2,
            &format!("kv_cache_step_{}", step + 1),
        );
        eprintln!("[kv_cache_step_{}] max diff = {diff}", step + 1);

        seqlen_offset += 1;
    }
}
