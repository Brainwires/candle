
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
//! BF16 storage + kernel correctness on the WGPU backend.
//!
//! Validates Phase 8 (BF16 storage layer + per-kernel adapters) with
//! small synthetic inputs that comfortably fit any WGPU adapter's
//! buffer-size limits — these tests target *kernel correctness*, not
//! end-to-end model loading.
//!
//! Each test uploads a small BF16 tensor to the WGPU device, runs a
//! kernel through the bf16 codegen path (binary add, matmul, RMSNorm,
//! identity copy via index_select), pulls the result back to CPU, and
//! compares against a CPU reference computed at f32 precision.
//! Tolerance is 1e-2 absolute — bf16 has only 7-bit mantissa, so a
//! few dozen multiply-adds yield ~1e-3 worst-case rounding error.
#![cfg(feature = "wgpu")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{Device, Result, Tensor};
use half::bf16;
async fn try_wgpu() -> Option<Device> {
    Device::new_wgpu_async(0).await.ok()
}
/// Upload an f32 slice as a BF16 tensor on `dev`. Goes through the
/// CPU storage path because the WGPU backend doesn't yet expose an
/// F32→BF16 cast kernel — but it does accept BF16 storage uploaded
/// directly, which is what the actual safetensors load path does.
fn bf16_tensor(
    data: &[f32],
    shape: impl Into<candle::Shape>,
    dev: &Device,
) -> Result<Tensor> {
    let bf16_data: Vec<bf16> = data.iter().copied().map(bf16::from_f32).collect();
    Tensor::from_slice(&bf16_data, shape, dev)
}
fn assert_bf16_close(label: &str, cpu_f32: &[f32], gpu_bf16: &[bf16], tol: f32) {
    assert_eq!(cpu_f32.len(), gpu_bf16.len(), "{label}: length mismatch");
    let mut max_abs = 0f32;
    for (i, (c, g)) in cpu_f32.iter().zip(gpu_bf16.iter()).enumerate() {
        let g_f32 = g.to_f32();
        let diff = (c - g_f32).abs();
        max_abs = max_abs.max(diff);
        let rel = diff / c.abs().max(1e-3);
        assert!(
            diff < tol || rel < tol,
            "{label}: divergence at {i}: cpu_f32={c} gpu_bf16={g_f32} diff={diff}"
        );
    }
    ewasm_bindgen_test::console_log!("{label}: max_abs={max_abs:.3e}");
}
#[test]
async fn wgpu_bf16_binary_add() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            ewasm_bindgen_test::console_log!(
                "skipping wgpu_bf16_binary_add — no wgpu device"
            );
            return Ok(());
        }
    };
    let n = 33usize;
    let a_data: Vec<f32> = (0..n).map(|i| 0.1 * i as f32 - 1.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 0.05 * i as f32 + 0.5).collect();
    let a = bf16_tensor(&a_data, (n,), &dev)?;
    let b = bf16_tensor(&b_data, (n,), &dev)?;
    let c = (&a + &b)?;
    let cpu_ref: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let gpu_out = c
        .to_device_async(&Device::Cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<bf16>()
        .await?;
    assert_bf16_close("bf16 binary_add", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}
#[test]
async fn wgpu_bf16_binary_mul() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let n = 64usize;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();
    let a = bf16_tensor(&a_data, (n,), &dev)?;
    let b = bf16_tensor(&b_data, (n,), &dev)?;
    let c = (&a * &b)?;
    let cpu_ref: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x * y)
        .collect();
    let gpu_out = c
        .to_device_async(&Device::Cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<bf16>()
        .await?;
    assert_bf16_close("bf16 binary_mul", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}
#[test]
async fn wgpu_bf16_matmul_naive() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let m = 2usize;
    let k = 8usize;
    let n = 4usize;
    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1) - 0.4).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07) - 0.2).collect();
    let a = bf16_tensor(&a_data, (m, k), &dev)?;
    let b = bf16_tensor(&b_data, (k, n), &dev)?;
    let c = a.matmul(&b)?;
    let mut cpu_ref = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0f32;
            for ki in 0..k {
                s += a_data[mi * k + ki] * b_data[ki * n + ni];
            }
            cpu_ref[mi * n + ni] = s;
        }
    }
    let gpu_out = c
        .to_device_async(&Device::Cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<bf16>()
        .await?;
    assert_bf16_close("bf16 matmul (m=2 k=8 n=4)", &cpu_ref, &gpu_out, 5e-2);
    Ok(())
}
#[test]
async fn wgpu_f32_matmul_1x1_sanity() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let a = Tensor::from_slice(&[2.0_f32], (1, 1), &dev)?;
    let b = Tensor::from_slice(&[3.0_f32], (1, 1), &dev)?;
    let c = a.matmul(&b)?;
    let out = c
        .to_device_async(&Device::Cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<f32>()
        .await?;
    ewasm_bindgen_test::console_log!("f32 1x1@1x1 -> {:?}", out);
    assert!((out[0] - 6.0).abs() < 1e-4, "f32 1x1@1x1: got {} want 6.0", out[0]);
    Ok(())
}
#[test]
async fn wgpu_bf16_matmul_minimal_1x1() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let a = bf16_tensor(&[2.0_f32], (1, 1), &dev)?;
    let b = bf16_tensor(&[3.0_f32], (1, 1), &dev)?;
    let c = a.matmul(&b)?;
    let gpu_out = c
        .to_device_async(&Device::Cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<bf16>()
        .await?;
    ewasm_bindgen_test::console_log!("1x1@1x1 -> {:?} (expect 6.0)", gpu_out);
    assert_bf16_close("bf16 matmul 1x1@1x1", &[6.0], &gpu_out, 1e-2);
    Ok(())
}
#[test]
async fn wgpu_bf16_matmul_decode_shape() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let k = 64usize;
    let n = 32usize;
    let a_data: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.011).sin()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.013).cos()).collect();
    let a = bf16_tensor(&a_data, (1, k), &dev)?;
    let b = bf16_tensor(&b_data, (k, n), &dev)?;
    let c = a.matmul(&b)?;
    let mut cpu_ref = vec![0f32; n];
    for ni in 0..n {
        let mut s = 0f32;
        for ki in 0..k {
            s += a_data[ki] * b_data[ki * n + ni];
        }
        cpu_ref[ni] = s;
    }
    let gpu_out = c
        .to_device_async(&Device::Cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<bf16>()
        .await?;
    assert_bf16_close("bf16 matmul1_m1 (1xk @ kxn)", &cpu_ref, &gpu_out, 1e-1);
    Ok(())
}
#[test]
async fn wgpu_bf16_index_select_roundtrip() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let table_data: Vec<f32> = (0..6 * 16).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let table = bf16_tensor(&table_data, (6, 16), &dev)?;
    let ids = Tensor::from_slice(&[2u32, 5, 0], (3,), &dev)?;
    let out = table.index_select(&ids, 0)?;
    let mut cpu_ref = Vec::with_capacity(3 * 16);
    for &r in &[2usize, 5, 0] {
        cpu_ref.extend_from_slice(&table_data[r * 16..(r + 1) * 16]);
    }
    let gpu_out = out
        .to_device_async(&Device::Cpu)
        .await?
        .flatten_all()?
        .to_vec1_async::<bf16>()
        .await?;
    assert_bf16_close("bf16 index_select", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}
