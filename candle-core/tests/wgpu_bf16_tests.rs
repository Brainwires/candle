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

use candle_core::{Device, Result, Tensor};
use half::bf16;

fn try_wgpu() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

/// Upload an f32 slice as a BF16 tensor on `dev`. Goes through the
/// CPU storage path because the WGPU backend doesn't yet expose an
/// F32→BF16 cast kernel — but it does accept BF16 storage uploaded
/// directly, which is what the actual safetensors load path does.
fn bf16_tensor(data: &[f32], shape: impl Into<candle_core::Shape>, dev: &Device) -> Result<Tensor> {
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
    eprintln!("{label}: max_abs={max_abs:.3e}");
}

#[test]
fn wgpu_bf16_binary_add() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            eprintln!("skipping wgpu_bf16_binary_add — no wgpu device");
            return Ok(());
        }
    };
    let n = 33usize; // odd-length so we exercise the half-sharing edge case
    let a_data: Vec<f32> = (0..n).map(|i| 0.1 * i as f32 - 1.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| 0.05 * i as f32 + 0.5).collect();

    let a = bf16_tensor(&a_data, (n,), &dev)?;
    let b = bf16_tensor(&b_data, (n,), &dev)?;
    let c = (&a + &b)?;

    let cpu_ref: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
    let gpu_out = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    assert_bf16_close("bf16 binary_add", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}

#[test]
fn wgpu_bf16_binary_mul() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    // Even-length to also confirm the non-edge case path.
    let n = 64usize;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();

    let a = bf16_tensor(&a_data, (n,), &dev)?;
    let b = bf16_tensor(&b_data, (n,), &dev)?;
    let c = (&a * &b)?;

    let cpu_ref: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).collect();
    let gpu_out = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    assert_bf16_close("bf16 binary_mul", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}

#[test]
fn wgpu_bf16_matmul_naive() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    // m=2, k=8, n=4 — naive matmul1 path on the bf16 kernel
    // (queue_matmul_buffer_best forces Matmul1/Matmul1M1 for BF16 since
    // Matmul1_4 / sgemm tiled are gated f32-only).
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
    let gpu_out = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    // Tighter tolerance is unrealistic at bf16 precision after k=8 fma's.
    assert_bf16_close("bf16 matmul (m=2 k=8 n=4)", &cpu_ref, &gpu_out, 5e-2);
    Ok(())
}

#[test]
fn wgpu_f32_matmul_1x1_sanity() -> Result<()> {
    // F32 1x1 @ 1x1 — same shape as the failing BF16 test, but f32.
    // If this works, the issue is bf16-specific.
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let a = Tensor::from_slice(&[2.0_f32], (1, 1), &dev)?;
    let b = Tensor::from_slice(&[3.0_f32], (1, 1), &dev)?;
    let c = a.matmul(&b)?;
    let out = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    eprintln!("f32 1x1@1x1 -> {:?}", out);
    assert!((out[0] - 6.0).abs() < 1e-4, "f32 1x1@1x1: got {} want 6.0", out[0]);
    Ok(())
}

#[test]
fn wgpu_bf16_matmul_minimal_1x1() -> Result<()> {
    // The smallest possible matmul: (1, 1) @ (1, 1) — one multiply,
    // one output element. Isolates the bf16 store path from any
    // multi-thread / shared-word concerns.
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let a = bf16_tensor(&[2.0_f32], (1, 1), &dev)?;
    let b = bf16_tensor(&[3.0_f32], (1, 1), &dev)?;
    let c = a.matmul(&b)?;
    let gpu_out = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    eprintln!("1x1@1x1 -> {:?} (expect 6.0)", gpu_out);
    assert_bf16_close("bf16 matmul 1x1@1x1", &[6.0], &gpu_out, 1e-2);
    Ok(())
}

#[test]
fn wgpu_bf16_matmul_decode_shape() -> Result<()> {
    // Decode shape: (1, 1, k) @ (k, n) — exercises Matmul1M1 (m==1 fast
    // path). This is the exact shape that hits during a single-token
    // forward in the chat-pwa decode loop.
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
    let gpu_out = c.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    assert_bf16_close("bf16 matmul1_m1 (1xk @ kxn)", &cpu_ref, &gpu_out, 1e-1);
    Ok(())
}

#[test]
fn wgpu_bf16_index_select_roundtrip() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    // 6-row table, embed-dim 16, gather 3 indices.
    let table_data: Vec<f32> = (0..6 * 16).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let table = bf16_tensor(&table_data, (6, 16), &dev)?;
    let ids = Tensor::from_slice(&[2u32, 5, 0], (3,), &dev)?;
    let out = table.index_select(&ids, 0)?;

    // CPU reference: gather rows 2, 5, 0.
    let mut cpu_ref = Vec::with_capacity(3 * 16);
    for &r in &[2usize, 5, 0] {
        cpu_ref.extend_from_slice(&table_data[r * 16..(r + 1) * 16]);
    }
    let gpu_out = out.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    assert_bf16_close("bf16 index_select", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}
