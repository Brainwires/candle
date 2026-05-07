
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
//! flash_attn_decode WGPU-vs-CPU correctness check.
//!
//! Generates a small random Q/K/V triple, runs the fused
//! flash_attn_decode on the WGPU device, and compares the output
//! against the CPU reference path (which the same op exposes via its
//! `cpu_fwd`). Differences should sit within fp32 rounding noise
//! (~1e-4 absolute) — divergences larger than that signal a real
//! kernel bug.
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{Device, Result, Tensor};
use candle_nn::flash_attn::flash_attn_decode;
#[cfg(feature = "wgpu")]
#[test]
async fn flash_attn_decode_wgpu_matches_cpu_no_sliding() -> Result<()> {
    let wgpu = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            ewasm_bindgen_test::console_log!(
                "skipping flash_attn_decode wgpu test — no wgpu device: {e}"
            );
            return Ok(());
        }
    };
    let cpu = Device::Cpu;
    let b = 1usize;
    let h = 2usize;
    let kv_len = 8usize;
    let d = 64usize;
    let q_data: Vec<f32> = (0..b * h * 1 * d)
        .map(|i| ((i as f32) * 0.013 - 0.5).sin())
        .collect();
    let k_data: Vec<f32> = (0..b * h * kv_len * d)
        .map(|i| ((i as f32) * 0.017 + 0.25).cos())
        .collect();
    let v_data: Vec<f32> = (0..b * h * kv_len * d)
        .map(|i| ((i as f32) * 0.011 - 0.1).sin())
        .collect();
    let q_cpu = Tensor::from_slice(&q_data, (b, h, 1, d), &cpu)?;
    let k_cpu = Tensor::from_slice(&k_data, (b, h, kv_len, d), &cpu)?;
    let v_cpu = Tensor::from_slice(&v_data, (b, h, kv_len, d), &cpu)?;
    let q_gpu = Tensor::from_slice(&q_data, (b, h, 1, d), &wgpu)?;
    let k_gpu = Tensor::from_slice(&k_data, (b, h, kv_len, d), &wgpu)?;
    let v_gpu = Tensor::from_slice(&v_data, (b, h, kv_len, d), &wgpu)?;
    let out_cpu = flash_attn_decode(&q_cpu, &k_cpu, &v_cpu, None)?;
    let out_gpu = flash_attn_decode(&q_gpu, &k_gpu, &v_gpu, None)?;
    let cpu_vec = out_cpu.flatten_all()?.to_vec1_async::<f32>().await?;
    let gpu_vec = out_gpu
        .to_device_async(&cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<f32>()
        .await?;
    assert_eq!(cpu_vec.len(), gpu_vec.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (i, (c, g)) in cpu_vec.iter().zip(gpu_vec.iter()).enumerate() {
        let diff = (c - g).abs();
        let rel = diff / c.abs().max(1e-6);
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(rel);
        assert!(
            diff < 1e-3 || rel < 1e-3,
            "flash_attn_decode CPU/WGPU divergence at {i}: cpu={c} gpu={g} diff={diff}"
        );
    }
    ewasm_bindgen_test::console_log!(
        "flash_attn_decode (no-sliding) CPU vs WGPU OK — max_abs={max_abs:.2e} max_rel={max_rel:.2e}"
    );
    Ok(())
}
#[cfg(feature = "wgpu")]
#[test]
async fn flash_attn_decode_wgpu_matches_cpu_sliding() -> Result<()> {
    let wgpu = match Device::new_wgpu(0) {
        Ok(d) => d,
        Err(e) => {
            ewasm_bindgen_test::console_log!(
                "skipping flash_attn_decode sliding wgpu test — no wgpu device: {e}"
            );
            return Ok(());
        }
    };
    let cpu = Device::Cpu;
    let b = 1usize;
    let h = 2usize;
    let kv_len = 16usize;
    let d = 64usize;
    let sliding = 4u32;
    let q_data: Vec<f32> = (0..b * h * 1 * d)
        .map(|i| ((i as f32) * 0.019 + 0.1).cos())
        .collect();
    let k_data: Vec<f32> = (0..b * h * kv_len * d)
        .map(|i| ((i as f32) * 0.021 - 0.3).sin())
        .collect();
    let v_data: Vec<f32> = (0..b * h * kv_len * d)
        .map(|i| ((i as f32) * 0.023 + 0.7).cos())
        .collect();
    let q_cpu = Tensor::from_slice(&q_data, (b, h, 1, d), &cpu)?;
    let k_cpu = Tensor::from_slice(&k_data, (b, h, kv_len, d), &cpu)?;
    let v_cpu = Tensor::from_slice(&v_data, (b, h, kv_len, d), &cpu)?;
    let q_gpu = Tensor::from_slice(&q_data, (b, h, 1, d), &wgpu)?;
    let k_gpu = Tensor::from_slice(&k_data, (b, h, kv_len, d), &wgpu)?;
    let v_gpu = Tensor::from_slice(&v_data, (b, h, kv_len, d), &wgpu)?;
    let out_cpu = flash_attn_decode(&q_cpu, &k_cpu, &v_cpu, Some(sliding))?;
    let out_gpu = flash_attn_decode(&q_gpu, &k_gpu, &v_gpu, Some(sliding))?;
    let cpu_vec = out_cpu.flatten_all()?.to_vec1_async::<f32>().await?;
    let gpu_vec = out_gpu
        .to_device_async(&cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<f32>()
        .await?;
    let mut max_abs = 0f32;
    for (i, (c, g)) in cpu_vec.iter().zip(gpu_vec.iter()).enumerate() {
        let diff = (c - g).abs();
        let rel = diff / c.abs().max(1e-6);
        max_abs = max_abs.max(diff);
        assert!(
            diff < 1e-3 || rel < 1e-3,
            "flash_attn_decode (sliding={sliding}) CPU/WGPU divergence at {i}: cpu={c} gpu={g} diff={diff}"
        );
    }
    ewasm_bindgen_test::console_log!(
        "flash_attn_decode (sliding={sliding}) CPU vs WGPU OK — max_abs={max_abs:.2e}"
    );
    Ok(())
}
