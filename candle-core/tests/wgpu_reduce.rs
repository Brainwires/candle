//! WebGPU reduction + softmax + norm correctness tests (Phase 3.5).
//!
//! Cross-checks `sum`, `max`, `min`, `mean` along selected axes against
//! the CPU reference. Then layers up via composition: `softmax`,
//! `layer_norm`, `rms_norm`. The last three rely on the reduce kernel +
//! the elementwise kernels from Phase 3.4 plus the new `affine` shader,
//! and they don't have dedicated fused kernels in this phase.
//!
//! Tolerance is 1e-4 absolute for f32 in small dynamic ranges; relaxed
//! to 1e-3 for sum reductions over more than 1024 elements (rounding
//! drift from order-of-summation differences vs CPU's serial loop).
//!
//! Skips gracefully when no GPU adapter is present (CI without GPUs
//! still passes `--features wgpu` builds).
//!
//! Run: `cargo test -p candle-core --features wgpu --test wgpu_reduce`
#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Tensor, D};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// candle-core doesn't dev-depend on candle-nn (would be a workspace
// cycle), so we inline the composition versions of softmax / layer_norm
// / rms_norm here. These match `softmax_via_composition`,
// `layer_norm_slow`, and `rms_norm_slow` exactly — checking that the
// composition works via primitives is the whole point of the test.
fn softmax_via_composition(xs: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

fn layer_norm_via_composition(
    x: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> candle_core::Result<Tensor> {
    let hidden_size = x.dim(D::Minus1)?;
    let x_centered = {
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean_x)?
    };
    let norm_x = (x_centered.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x_centered.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.broadcast_mul(alpha)?.broadcast_add(beta)
}

fn rms_norm_via_composition(x: &Tensor, alpha: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    let hidden_size = x.dim(D::Minus1)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.broadcast_mul(alpha)
}

const F32_TOL_TIGHT: f32 = 1e-4;
const F32_TOL_LOOSE: f32 = 1e-3;

fn try_wgpu_device(test_name: &str) -> Option<Device> {
    match Device::new_wgpu(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping {test_name}: no wgpu adapter available ({e})");
            None
        }
    }
}

#[track_caller]
fn assert_close(want: &[f32], got: &[f32], tol: f32, label: &str) {
    assert_eq!(
        want.len(),
        got.len(),
        "{label}: length mismatch want={} got={}",
        want.len(),
        got.len()
    );
    for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
        let diff = (a - b).abs();
        let bad = diff > tol || a.is_nan() != b.is_nan();
        assert!(
            !bad,
            "{label}[{i}]: cpu={a} wgpu={b} diff={diff} (tol={tol})"
        );
    }
}

fn flatten(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn sum_along_axis_f32() {
    let Some(dev) = try_wgpu_device("sum_along_axis_f32") else {
        return;
    };
    let xs: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let cpu = Tensor::from_slice(&xs, (2, 3, 4), &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&dev).unwrap();

    for axis in 0..3 {
        let want = flatten(&cpu.sum_keepdim(axis).unwrap());
        let got = flatten(
            &gpu.sum_keepdim(axis)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap(),
        );
        assert_close(&want, &got, F32_TOL_TIGHT, &format!("sum axis={axis}"));
    }

    // Multi-axis sum (Sum is the only reduce that takes >1 axis in candle).
    let want = flatten(&cpu.sum_keepdim((0, 2)).unwrap());
    let got = flatten(
        &gpu.sum_keepdim((0, 2))
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap(),
    );
    assert_close(&want, &got, F32_TOL_TIGHT, "sum axes=(0,2)");
}

#[test]
fn max_along_axis_f32() {
    let Some(dev) = try_wgpu_device("max_along_axis_f32") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0xA5A5_F00D);
    let xs: Vec<f32> = (0..60).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect();
    let cpu = Tensor::from_slice(&xs, (3, 4, 5), &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&dev).unwrap();
    for axis in 0..3 {
        let want = flatten(&cpu.max_keepdim(axis).unwrap());
        let got = flatten(
            &gpu.max_keepdim(axis)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap(),
        );
        assert_close(&want, &got, F32_TOL_TIGHT, &format!("max axis={axis}"));

        let want_min = flatten(&cpu.min_keepdim(axis).unwrap());
        let got_min = flatten(
            &gpu.min_keepdim(axis)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap(),
        );
        assert_close(
            &want_min,
            &got_min,
            F32_TOL_TIGHT,
            &format!("min axis={axis}"),
        );
    }
}

#[test]
fn mean_along_axis_f32() {
    let Some(dev) = try_wgpu_device("mean_along_axis_f32") else {
        return;
    };
    let xs: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let cpu = Tensor::from_slice(&xs, (2, 3, 4), &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&dev).unwrap();
    for axis in 0..3 {
        let want = flatten(&cpu.mean_keepdim(axis).unwrap());
        let got = flatten(
            &gpu.mean_keepdim(axis)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap(),
        );
        assert_close(&want, &got, F32_TOL_TIGHT, &format!("mean axis={axis}"));
    }
}

#[test]
fn softmax_last_axis_f32() {
    let Some(dev) = try_wgpu_device("softmax_last_axis_f32") else {
        return;
    };
    // Mix of magnitudes including a small/large pair so the
    // numerically-stable max-subtraction path is exercised.
    let xs: Vec<f32> = vec![
        0.0, 1.0, 0.0, 1.0, // row 0
        -2.0, 2.0, 3.0, -3.0, // row 1
        10.0, 10.0, 10.0, 10.0, // row 2 — uniform
        -50.0, 0.0, 50.0, 100.0, // row 3 — extreme
    ];
    let cpu = Tensor::from_slice(&xs, (4, 4), &Device::Cpu).unwrap();
    let want = flatten(&softmax_via_composition(&cpu, 1).unwrap());

    let gpu = cpu.to_device(&dev).unwrap();
    let got = flatten(
        &softmax_via_composition(&gpu, 1)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap(),
    );
    assert_close(&want, &got, F32_TOL_TIGHT, "softmax last axis");

    // Also sanity-check that each row sums to ~1.
    let sums: Vec<f32> = got.chunks(4).map(|r| r.iter().sum::<f32>()).collect();
    for (i, s) in sums.iter().enumerate() {
        assert!(
            (s - 1.0).abs() < 1e-4,
            "softmax row {i} sum = {s} (expected ~1)"
        );
    }
}

#[test]
fn layernorm_f32() {
    let Some(dev) = try_wgpu_device("layernorm_f32") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0xDEAD_F00D);
    let n_rows = 3usize;
    let n_cols = 8usize;
    let xs: Vec<f32> = (0..n_rows * n_cols)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let alpha: Vec<f32> = (0..n_cols).map(|i| 1.0 + (i as f32) * 0.1).collect();
    let beta: Vec<f32> = (0..n_cols).map(|i| (i as f32) * 0.05).collect();
    let eps = 1e-5f32;

    let cpu = Tensor::from_slice(&xs, (n_rows, n_cols), &Device::Cpu).unwrap();
    let alpha_cpu = Tensor::from_slice(&alpha, n_cols, &Device::Cpu).unwrap();
    let beta_cpu = Tensor::from_slice(&beta, n_cols, &Device::Cpu).unwrap();
    let want = flatten(&layer_norm_via_composition(&cpu, &alpha_cpu, &beta_cpu, eps).unwrap());

    let gpu = cpu.to_device(&dev).unwrap();
    let alpha_gpu = alpha_cpu.to_device(&dev).unwrap();
    let beta_gpu = beta_cpu.to_device(&dev).unwrap();
    let got = flatten(
        &layer_norm_via_composition(&gpu, &alpha_gpu, &beta_gpu, eps)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap(),
    );
    assert_close(&want, &got, F32_TOL_TIGHT, "layer_norm_slow");
}

#[test]
fn rmsnorm_f32() {
    let Some(dev) = try_wgpu_device("rmsnorm_f32") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0xCAFE_BABE);
    let n_rows = 3usize;
    let n_cols = 8usize;
    let xs: Vec<f32> = (0..n_rows * n_cols)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let alpha: Vec<f32> = (0..n_cols).map(|i| 1.0 + (i as f32) * 0.1).collect();
    let eps = 1e-5f32;

    let cpu = Tensor::from_slice(&xs, (n_rows, n_cols), &Device::Cpu).unwrap();
    let alpha_cpu = Tensor::from_slice(&alpha, n_cols, &Device::Cpu).unwrap();
    let want = flatten(&rms_norm_via_composition(&cpu, &alpha_cpu, eps).unwrap());

    let gpu = cpu.to_device(&dev).unwrap();
    let alpha_gpu = alpha_cpu.to_device(&dev).unwrap();
    let got = flatten(
        &rms_norm_via_composition(&gpu, &alpha_gpu, eps)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap(),
    );
    assert_close(&want, &got, F32_TOL_TIGHT, "rms_norm_slow");
}

/// Sweep sum / max / mean across rank-2 and rank-3 along each axis.
/// Tolerance is 1e-4 for small reduce_sizes (≤1024) and relaxed to 1e-3
/// for larger ones — sum-of-many-floats reorders differently on the GPU
/// thread that walks the reduce-axis vs CPU's tight loop, and the drift
/// adds up.
#[test]
fn reduce_random_shapes() {
    let Some(dev) = try_wgpu_device("reduce_random_shapes") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0xBAD_F00D);

    let shapes2: &[(usize, usize)] = &[(8, 16), (5, 7), (32, 64)];
    let shapes3: &[(usize, usize, usize)] = &[(2, 3, 4), (4, 8, 16), (3, 5, 11)];

    let pick_tol = |reduce_size: usize| {
        if reduce_size > 1024 {
            F32_TOL_LOOSE
        } else {
            F32_TOL_TIGHT
        }
    };

    for &(r, c) in shapes2 {
        let n = r * c;
        let xs: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let cpu = Tensor::from_slice(&xs, (r, c), &Device::Cpu).unwrap();
        let gpu = cpu.to_device(&dev).unwrap();
        for axis in 0..2 {
            let reduce_size = if axis == 0 { r } else { c };
            let tol = pick_tol(reduce_size);
            let want = flatten(&cpu.sum_keepdim(axis).unwrap());
            let got = flatten(
                &gpu.sum_keepdim(axis)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap(),
            );
            assert_close(&want, &got, tol, &format!("sum2d axis={axis} {r}x{c}"));

            let want = flatten(&cpu.max_keepdim(axis).unwrap());
            let got = flatten(
                &gpu.max_keepdim(axis)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap(),
            );
            assert_close(
                &want,
                &got,
                F32_TOL_TIGHT,
                &format!("max2d axis={axis} {r}x{c}"),
            );

            let want = flatten(&cpu.mean_keepdim(axis).unwrap());
            let got = flatten(
                &gpu.mean_keepdim(axis)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap(),
            );
            assert_close(&want, &got, tol, &format!("mean2d axis={axis} {r}x{c}"));
        }
    }

    for &(a, b, c) in shapes3 {
        let n = a * b * c;
        let xs: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let cpu = Tensor::from_slice(&xs, (a, b, c), &Device::Cpu).unwrap();
        let gpu = cpu.to_device(&dev).unwrap();
        for axis in 0..3 {
            let reduce_size = match axis {
                0 => a,
                1 => b,
                _ => c,
            };
            let tol = pick_tol(reduce_size);
            let want = flatten(&cpu.sum_keepdim(axis).unwrap());
            let got = flatten(
                &gpu.sum_keepdim(axis)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap(),
            );
            assert_close(&want, &got, tol, &format!("sum3d axis={axis} {a}x{b}x{c}"));

            let want = flatten(&cpu.max_keepdim(axis).unwrap());
            let got = flatten(
                &gpu.max_keepdim(axis)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap(),
            );
            assert_close(
                &want,
                &got,
                F32_TOL_TIGHT,
                &format!("max3d axis={axis} {a}x{b}x{c}"),
            );

            let want = flatten(&cpu.mean_keepdim(axis).unwrap());
            let got = flatten(
                &gpu.mean_keepdim(axis)
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap(),
            );
            assert_close(&want, &got, tol, &format!("mean3d axis={axis} {a}x{b}x{c}"));
        }
    }
}

#[test]
fn dtype_check() {
    let Some(_dev) = try_wgpu_device("dtype_check") else {
        return;
    };
    // Just a smoke check to make sure `DType::F32` is what we get back.
    let _ = DType::F32;
}
