//! BF16 correctness on the WGPU backend for candle-nn ops.
//!
//! Companion to `candle-core/tests/wgpu_bf16_tests.rs`. These cover
//! the higher-level ops that live in candle-nn (rms_norm, softmax,
//! silu, rotary_emb) — each one gets uploaded as BF16, run through
//! the WGPU kernel, and diffed against a CPU f32 reference.

#![cfg(feature = "wgpu")]

use candle::{Device, Result, Shape, Tensor};
use half::bf16;

fn try_wgpu() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

fn bf16_tensor(data: &[f32], shape: impl Into<Shape>, dev: &Device) -> Result<Tensor> {
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
fn wgpu_bf16_softmax_trivial_one_elem() -> Result<()> {
    // softmax([x]) = [1.0] regardless of x. If this fails, the kernel
    // isn't even running its single-element path correctly.
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let x = bf16_tensor(&[5.0_f32], (1, 1, 1, 1), &dev)?;
    let y = candle_nn::ops::softmax_last_dim(&x)?;
    let gpu_out = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    eprintln!("softmax([5]) -> {:?}", gpu_out);
    assert_bf16_close("bf16 softmax(1 elem)", &[1.0], &gpu_out, 1e-2);
    Ok(())
}

#[test]
fn wgpu_bf16_softmax_two_elem() -> Result<()> {
    // softmax([0, 1]): hand-trace
    //   max = 1; exps = [exp(-1), exp(0)] = [0.3679, 1.0]; sum = 1.3679
    //   output = [0.2689, 0.7311]
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let x = bf16_tensor(&[0.0_f32, 1.0_f32], (1, 1, 1, 2), &dev)?;
    let y = candle_nn::ops::softmax_last_dim(&x)?;
    let gpu_out = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    eprintln!("softmax([0,1]) -> {:?}", gpu_out);
    assert_bf16_close("bf16 softmax(2 elem)", &[0.2689414, 0.7310586], &gpu_out, 1e-2);
    Ok(())
}

#[test]
fn wgpu_bf16_softmax_last_dim() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    // 1x1x4x8 — softmax across an 8-wide last dim, 4 rows.
    let n = 8usize;
    let rows = 4usize;
    let mut all_data = Vec::with_capacity(rows * n);
    for r in 0..rows {
        for i in 0..n {
            all_data.push(((r * n + i) as f32) * 0.137 - 1.0);
        }
    }
    let x = bf16_tensor(&all_data, (1, 1, rows, n), &dev)?;
    let y = candle_nn::ops::softmax_last_dim(&x)?;

    let mut cpu_ref = Vec::with_capacity(rows * n);
    for r in 0..rows {
        let row = &all_data[r * n..(r + 1) * n];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for e in exps {
            cpu_ref.push(e / sum);
        }
    }
    let gpu_out = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    assert_bf16_close("bf16 softmax_last_dim", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}

#[test]
fn wgpu_bf16_rms_norm() -> Result<()> {
    use candle_nn::{Module, RmsNorm};
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let n = 16usize;
    let x_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07) - 0.3).collect();
    let w_data: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.01)).collect();
    let x = bf16_tensor(&x_data, (1, n), &dev)?;
    let w = bf16_tensor(&w_data, (n,), &dev)?;
    let rms = RmsNorm::new(w, 1e-6);
    let y = rms.forward(&x)?;

    let mean_sq: f32 = x_data.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + 1e-6).sqrt();
    let cpu_ref: Vec<f32> = x_data
        .iter()
        .zip(w_data.iter())
        .map(|(x, w)| x * inv_rms * w)
        .collect();
    let gpu_out = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    assert_bf16_close("bf16 rms_norm", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}

#[test]
fn wgpu_bf16_unary_silu() -> Result<()> {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let n = 17usize; // odd to exercise edge alignment
    let x_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.21) - 1.5).collect();
    let x = bf16_tensor(&x_data, (n,), &dev)?;
    let y = candle_nn::ops::silu(&x)?;
    let cpu_ref: Vec<f32> = x_data.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    let gpu_out = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    assert_bf16_close("bf16 unary silu", &cpu_ref, &gpu_out, 1e-2);
    Ok(())
}

#[test]
fn wgpu_bf16_rotary_emb_rope() -> Result<()> {
    // candle_nn::rotary_emb::rope (contiguous-half layout — pairs are
    // (idx, idx+half) within each head, not interleaved).
    // Q: (b=1, h=2, t=3, d=8); cos/sin: (t=3, d/2=4).
    let dev = match try_wgpu() {
        Some(d) => d,
        None => return Ok(()),
    };
    let b = 1usize;
    let h = 2usize;
    let t = 3usize;
    let d = 8usize;
    let half = d / 2;

    let xs_data: Vec<f32> = (0..b * h * t * d)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let cos_data: Vec<f32> = (0..t * half).map(|i| ((i as f32) * 0.5).cos()).collect();
    let sin_data: Vec<f32> = (0..t * half).map(|i| ((i as f32) * 0.5).sin()).collect();

    let xs = bf16_tensor(&xs_data, (b, h, t, d), &dev)?;
    let cos = bf16_tensor(&cos_data, (t, half), &dev)?;
    let sin = bf16_tensor(&sin_data, (t, half), &dev)?;
    let y = candle_nn::rotary_emb::rope(&xs, &cos, &sin)?;

    let mut cpu_ref = vec![0f32; b * h * t * d];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                for di in 0..half {
                    let base = ((bi * h + hi) * t + ti) * d;
                    let x0 = xs_data[base + di];
                    let x1 = xs_data[base + di + half];
                    let c = cos_data[ti * half + di];
                    let s = sin_data[ti * half + di];
                    cpu_ref[base + di] = x0 * c - x1 * s;
                    cpu_ref[base + di + half] = x0 * s + x1 * c;
                }
            }
        }
    }
    let gpu_out = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<bf16>()?;
    assert_bf16_close("bf16 rotary_emb (rope)", &cpu_ref, &gpu_out, 5e-2);
    Ok(())
}
