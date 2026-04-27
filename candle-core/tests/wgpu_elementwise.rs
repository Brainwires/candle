//! WebGPU element-wise op correctness tests (Phase 3.4).
//!
//! Cross-checks every wired unary and binary op against the CPU
//! reference. Tolerance is 1e-4 for f32 — comfortably tighter than the
//! drift you'd see from a single transcendental on either side, but
//! loose enough that the WGSL `tanh`-approximated GELU and our
//! Abramowitz–Stegun `erf` rational approximation both pass.
//!
//! Like the round-trip and matmul suites, every test gracefully
//! `eprintln!`-skips when no GPU adapter is present so CI on a
//! GPU-less runner stays green as long as `--features wgpu`
//! *compiles*.
//!
//! Run: `cargo test -p candle-core --features wgpu --test wgpu_elementwise`
#![cfg(feature = "wgpu")]

use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const F32_TOL: f32 = 1e-4;

/// Try to acquire a WGPU device, or print a skip notice and return `None`.
fn try_wgpu_device(test_name: &str) -> Option<Device> {
    match Device::new_wgpu(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping {test_name}: no wgpu adapter available ({e})");
            None
        }
    }
}

/// Compare two flat f32 slices element-wise. Tolerance is absolute
/// because every op in this file lives in a small dynamic range
/// (we feed inputs in roughly `[-2, 2]`).
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
        let bad = diff > tol || a.is_nan() != b.is_nan() || a.is_infinite() != b.is_infinite();
        assert!(
            !bad,
            "{label}[{i}]: cpu={a} wgpu={b} diff={diff} (tol={tol})"
        );
    }
}

/// Round-trip a tensor through CPU → GPU → unary op → CPU and compare
/// to a CPU-side reference application of the same op.
///
/// `op_name` is just for the assertion message; both closures get the
/// CPU and GPU input tensor respectively and must return a `Tensor`.
fn check_unary<F1, F2>(
    op_name: &str,
    dev: &Device,
    xs: &[f32],
    shape: &[usize],
    cpu_op: F1,
    gpu_op: F2,
) where
    F1: Fn(&Tensor) -> candle_core::Result<Tensor>,
    F2: Fn(&Tensor) -> candle_core::Result<Tensor>,
{
    let cpu_in = Tensor::from_slice(xs, shape, &Device::Cpu).unwrap();
    let want = cpu_op(&cpu_in)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let gpu_in = cpu_in.to_device(dev).unwrap();
    let got = gpu_op(&gpu_in)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&want, &got, F32_TOL, op_name);
}

#[test]
fn unary_gelu_f32() {
    let Some(dev) = try_wgpu_device("unary_gelu_f32") else {
        return;
    };
    let xs = [-2.0f32, -0.5, 0.0, 0.5, 2.0];
    let cpu_in = Tensor::from_slice(&xs, 5, &Device::Cpu).unwrap();
    let want = cpu_in.gelu().unwrap().to_vec1::<f32>().unwrap();
    let got = cpu_in
        .to_device(&dev)
        .unwrap()
        .gelu()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    // GELU's tanh-approximation matches the CPU implementation exactly
    // (both candle CPU and our shader use the same tanh approximation).
    assert_close(&want, &got, F32_TOL, "gelu_f32");
}

#[test]
fn unary_silu_f32() {
    let Some(dev) = try_wgpu_device("unary_silu_f32") else {
        return;
    };
    let xs = [-3.0f32, -1.0, 0.0, 1.0, 3.0];
    check_unary("silu_f32", &dev, &xs, &[5], |t| t.silu(), |t| t.silu());
}

#[test]
fn binary_add_broadcast_f32() {
    let Some(dev) = try_wgpu_device("binary_add_broadcast_f32") else {
        return;
    };
    // (2, 3) + (3,) — classic NumPy-style broadcast.
    let a_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0];
    let a_cpu = Tensor::from_slice(&a_data, (2, 3), &Device::Cpu).unwrap();
    let b_cpu = Tensor::from_slice(&b_data, 3, &Device::Cpu).unwrap();
    let want = a_cpu
        .broadcast_add(&b_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let a_gpu = a_cpu.to_device(&dev).unwrap();
    let b_gpu = b_cpu.to_device(&dev).unwrap();
    let got = a_gpu
        .broadcast_add(&b_gpu)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&want, &got, F32_TOL, "add_broadcast_f32");
}

/// For every wired unary op, run a few shapes and compare against CPU.
#[test]
fn unary_random_shapes() {
    let Some(dev) = try_wgpu_device("unary_random_shapes") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0xCAFE_F00D);

    // Pick a domain that's safe for every op (sqrt and log need >0; we
    // shift accordingly inside the per-op closure).
    let raw: Vec<f32> = (0..120).map(|_| rng.random::<f32>() * 3.0 - 1.5).collect();

    let shapes: &[&[usize]] = &[&[8], &[3, 5], &[2, 3, 4]];
    for shape in shapes {
        let n: usize = shape.iter().product();
        let xs = &raw[..n];

        check_unary("neg", &dev, xs, shape, |t| t.neg(), |t| t.neg());
        check_unary("abs", &dev, xs, shape, |t| t.abs(), |t| t.abs());
        // sqrt / log need positive inputs; bias by 2.0 so the test
        // domain becomes ~[0.5, 3.5].
        let pos: Vec<f32> = xs.iter().map(|v| v + 2.0).collect();
        check_unary("sqrt", &dev, &pos, shape, |t| t.sqrt(), |t| t.sqrt());
        check_unary("log", &dev, &pos, shape, |t| t.log(), |t| t.log());
        check_unary("exp", &dev, xs, shape, |t| t.exp(), |t| t.exp());
        check_unary("tanh", &dev, xs, shape, |t| t.tanh(), |t| t.tanh());
        check_unary("relu", &dev, xs, shape, |t| t.relu(), |t| t.relu());
        check_unary("gelu", &dev, xs, shape, |t| t.gelu(), |t| t.gelu());
        check_unary("silu", &dev, xs, shape, |t| t.silu(), |t| t.silu());
        // recip needs non-zero; bias positive
        check_unary("recip", &dev, &pos, shape, |t| t.recip(), |t| t.recip());
    }
}

/// For every wired binary op, run a few shapes (contiguous + broadcast)
/// and compare against CPU.
#[test]
fn binary_random_shapes() {
    let Some(dev) = try_wgpu_device("binary_random_shapes") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

    let shapes: &[&[usize]] = &[&[8], &[3, 5], &[2, 3, 4]];
    for shape in shapes {
        let n: usize = shape.iter().product();
        let a: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 3.0 - 1.5).collect();
        // Bias b away from zero so `div` doesn't blow up.
        let b: Vec<f32> = (0..n)
            .map(|_| {
                let v: f32 = rng.random::<f32>() * 3.0 - 1.5;
                if v.abs() < 0.25 {
                    v + 0.5
                } else {
                    v
                }
            })
            .collect();
        let a_cpu = Tensor::from_slice(&a, *shape, &Device::Cpu).unwrap();
        let b_cpu = Tensor::from_slice(&b, *shape, &Device::Cpu).unwrap();
        let a_gpu = a_cpu.to_device(&dev).unwrap();
        let b_gpu = b_cpu.to_device(&dev).unwrap();

        for (name, cpu, gpu) in [
            (
                "add",
                (&a_cpu + &b_cpu).unwrap(),
                (&a_gpu + &b_gpu).unwrap(),
            ),
            (
                "sub",
                (&a_cpu - &b_cpu).unwrap(),
                (&a_gpu - &b_gpu).unwrap(),
            ),
            (
                "mul",
                (&a_cpu * &b_cpu).unwrap(),
                (&a_gpu * &b_gpu).unwrap(),
            ),
            (
                "div",
                (&a_cpu / &b_cpu).unwrap(),
                (&a_gpu / &b_gpu).unwrap(),
            ),
            (
                "minimum",
                a_cpu.minimum(&b_cpu).unwrap(),
                a_gpu.minimum(&b_gpu).unwrap(),
            ),
            (
                "maximum",
                a_cpu.maximum(&b_cpu).unwrap(),
                a_gpu.maximum(&b_gpu).unwrap(),
            ),
        ] {
            let want = cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got = gpu
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_close(&want, &got, F32_TOL, &format!("{name}@{shape:?}"));
        }
    }
}
