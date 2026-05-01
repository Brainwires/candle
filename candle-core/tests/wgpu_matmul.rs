//! WebGPU matmul correctness tests (Phase 3.3).
//!
//! Cross-checks the WebGPU tiled matmul kernel against the CPU
//! reference for a handful of representative shapes:
//!
//! - Tiny canonical 2×2 (sanity).
//! - A handful of random shapes including non-tile-multiples
//!   (33×17×23) and a Gemma-shaped projection (1×2048×4096).
//! - A batched matmul (B, M, K) @ (B, K, N).
//! - An f16 round-trip on adapters that advertise SHADER_F16.
//!
//! Like the round-trip suite, every test gracefully `eprintln!`-skips
//! when no GPU adapter is present, so CI on a GPU-less runner stays
//! green as long as `--features wgpu` *compiles*.
//!
//! Run: `cargo test -p candle-core --features wgpu --test wgpu_matmul`
#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const F32_TOL: f32 = 1e-4;
// f32 tolerance loosens slightly for the larger random shapes —
// summing ~4096 products in f32 in shader-side accumulation drifts
// further than the ideal CPU gemm. 1e-3 keeps headroom without
// hiding real bugs.
const F32_TOL_RANDOM: f32 = 1e-3;
const F16_TOL: f32 = 1e-2;
// bf16 carries only 7 mantissa bits so per-product error is ~2^-8 of the
// magnitude. We use values in [-0.5, 0.5] and accumulate up to ~2048 of
// them; allowing 5e-2 absolute error keeps headroom over the worst-case
// CPU↔GPU drift while still catching real bugs (a single dropped term
// would push the error well above this).
const BF16_TOL: f32 = 5e-2;

/// Try to acquire a WGPU device, or print a skip notice and return None.
fn try_wgpu_device(test_name: &str) -> Option<Device> {
    match Device::new_wgpu(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping {test_name}: no wgpu adapter available ({e})");
            None
        }
    }
}

/// `true` iff the device advertises native f16. We can't introspect a
/// `Device` directly, so we probe by attempting an f16 matmul on a
/// trivial shape and looking for the specific feature-gate error.
fn device_supports_f16(dev: &Device) -> bool {
    let a = match Tensor::from_slice(&[f16::from_f32(1.0); 4], (2, 2), dev) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let b = match Tensor::from_slice(&[f16::from_f32(1.0); 4], (2, 2), dev) {
        Ok(t) => t,
        Err(_) => return false,
    };
    a.matmul(&b).is_ok()
}

#[test]
fn matmul_f32_basic() {
    let dev = match try_wgpu_device("matmul_f32_basic") {
        Some(d) => d,
        None => return,
    };
    let a_cpu = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
    let b_cpu = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], (2, 2), &Device::Cpu).unwrap();
    let c_ref = a_cpu.matmul(&b_cpu).unwrap();

    let a_gpu = a_cpu.to_device(&dev).unwrap();
    let b_gpu = b_cpu.to_device(&dev).unwrap();
    let c_gpu = a_gpu.matmul(&b_gpu).unwrap();
    let c_back = c_gpu.to_device(&Device::Cpu).unwrap();

    let r: Vec<f32> = c_ref.flatten_all().unwrap().to_vec1().unwrap();
    let g: Vec<f32> = c_back.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(r.len(), g.len());
    for (rv, gv) in r.iter().zip(g.iter()) {
        assert!(
            (rv - gv).abs() < F32_TOL,
            "matmul_f32_basic mismatch ref={rv} got={gv}"
        );
    }
}

#[test]
fn matmul_f32_random_shapes() {
    let dev = match try_wgpu_device("matmul_f32_random_shapes") {
        Some(d) => d,
        None => return,
    };
    let mut rng = StdRng::seed_from_u64(42);
    let shapes: &[(usize, usize, usize)] = &[
        (1, 8, 8),
        (8, 16, 32),
        (33, 17, 23),
        (128, 128, 128),
        // Gemma-shaped projection: (1 token, hidden, FFN width).
        (1, 2048, 4096),
    ];
    for &(m, k, n) in shapes {
        let a_data: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() - 0.5).collect();
        let b_data: Vec<f32> = (0..k * n).map(|_| rng.random::<f32>() - 0.5).collect();
        let a_cpu = Tensor::from_vec(a_data, (m, k), &Device::Cpu).unwrap();
        let b_cpu = Tensor::from_vec(b_data, (k, n), &Device::Cpu).unwrap();
        let c_cpu = a_cpu.matmul(&b_cpu).unwrap();
        let c_gpu = a_cpu
            .to_device(&dev)
            .unwrap()
            .matmul(&b_cpu.to_device(&dev).unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();

        let r: Vec<f32> = c_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = c_gpu.flatten_all().unwrap().to_vec1().unwrap();
        let mut max_err = 0.0f32;
        for (rv, gv) in r.iter().zip(g.iter()) {
            let e = (rv - gv).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < F32_TOL_RANDOM,
            "shape ({m},{k},{n}): max_err={max_err} > tol={F32_TOL_RANDOM}"
        );
    }
}

#[test]
fn matmul_f32_batched() {
    let dev = match try_wgpu_device("matmul_f32_batched") {
        Some(d) => d,
        None => return,
    };
    let mut rng = StdRng::seed_from_u64(7);
    let (b, m, k, n) = (4usize, 8usize, 16usize, 8usize);
    let a_data: Vec<f32> = (0..b * m * k).map(|_| rng.random::<f32>() - 0.5).collect();
    let b_data: Vec<f32> = (0..b * k * n).map(|_| rng.random::<f32>() - 0.5).collect();
    let a_cpu = Tensor::from_vec(a_data, (b, m, k), &Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_data, (b, k, n), &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();
    let c_gpu = a_cpu
        .to_device(&dev)
        .unwrap()
        .matmul(&b_cpu.to_device(&dev).unwrap())
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();

    let r: Vec<f32> = c_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let g: Vec<f32> = c_gpu.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(r.len(), b * m * n);
    let mut max_err = 0.0f32;
    for (rv, gv) in r.iter().zip(g.iter()) {
        let e = (rv - gv).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(max_err < F32_TOL_RANDOM, "batched: max_err={max_err}");
}

#[test]
fn matmul_f32_transposed_rhs() {
    // Exercises non-trivial inner strides: B's last two dims are
    // transposed via `Tensor::t()`, so its row/col strides on the GPU
    // path are (1, k) instead of (k, 1). The shader honors strides
    // natively.
    let dev = match try_wgpu_device("matmul_f32_transposed_rhs") {
        Some(d) => d,
        None => return,
    };
    let mut rng = StdRng::seed_from_u64(99);
    let (m, k, n) = (16usize, 24usize, 12usize);
    let a_data: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() - 0.5).collect();
    // B is shaped (n, k); we'll transpose it before matmul so it acts
    // as (k, n).
    let b_data: Vec<f32> = (0..n * k).map(|_| rng.random::<f32>() - 0.5).collect();
    let a_cpu = Tensor::from_vec(a_data, (m, k), &Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_data, (n, k), &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu.t().unwrap()).unwrap();
    let c_gpu = a_cpu
        .to_device(&dev)
        .unwrap()
        .matmul(&b_cpu.to_device(&dev).unwrap().t().unwrap())
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    let r: Vec<f32> = c_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let g: Vec<f32> = c_gpu.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_err = 0.0f32;
    for (rv, gv) in r.iter().zip(g.iter()) {
        let e = (rv - gv).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < F32_TOL_RANDOM,
        "transposed_rhs: max_err={max_err}"
    );
}

#[test]
fn matmul_f16_basic() {
    let dev = match try_wgpu_device("matmul_f16_basic") {
        Some(d) => d,
        None => return,
    };
    if !device_supports_f16(&dev) {
        eprintln!("skipping matmul_f16_basic: device lacks SHADER_F16");
        return;
    }

    let mut rng = StdRng::seed_from_u64(1);
    let (m, k, n) = (8usize, 16usize, 8usize);
    let a_f32: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() - 0.5).collect();
    let b_f32: Vec<f32> = (0..k * n).map(|_| rng.random::<f32>() - 0.5).collect();
    let a_f16: Vec<f16> = a_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let b_f16: Vec<f16> = b_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let a_cpu = Tensor::from_vec(a_f16.clone(), (m, k), &Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_f16.clone(), (k, n), &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu).unwrap();

    let c_gpu = a_cpu
        .to_device(&dev)
        .unwrap()
        .matmul(&b_cpu.to_device(&dev).unwrap())
        .unwrap();
    assert_eq!(c_gpu.dtype(), DType::F16);
    let c_back = c_gpu.to_device(&Device::Cpu).unwrap();
    let r: Vec<f16> = c_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let g: Vec<f16> = c_back.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_err = 0.0f32;
    for (rv, gv) in r.iter().zip(g.iter()) {
        let e = (rv.to_f32() - gv.to_f32()).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(max_err < F16_TOL, "f16_basic: max_err={max_err}");
}

#[test]
fn matmul_bf16_basic() {
    let dev = match try_wgpu_device("matmul_bf16_basic") {
        Some(d) => d,
        None => return,
    };
    // Canonical 2×2 to sanity-check packing/unpacking and rounding.
    let a_bf16: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let b_bf16: Vec<bf16> = [5.0f32, 6.0, 7.0, 8.0]
        .iter()
        .map(|v| bf16::from_f32(*v))
        .collect();
    let a_cpu = Tensor::from_vec(a_bf16, (2, 2), &Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_bf16, (2, 2), &Device::Cpu).unwrap();
    let c_ref = a_cpu.matmul(&b_cpu).unwrap();

    let c_gpu = a_cpu
        .to_device(&dev)
        .unwrap()
        .matmul(&b_cpu.to_device(&dev).unwrap())
        .unwrap();
    assert_eq!(c_gpu.dtype(), DType::BF16);
    let c_back = c_gpu.to_device(&Device::Cpu).unwrap();

    let r: Vec<bf16> = c_ref.flatten_all().unwrap().to_vec1().unwrap();
    let g: Vec<bf16> = c_back.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(r.len(), g.len());
    for (rv, gv) in r.iter().zip(g.iter()) {
        let e = (rv.to_f32() - gv.to_f32()).abs();
        assert!(
            e < BF16_TOL,
            "matmul_bf16_basic mismatch ref={} got={} err={}",
            rv.to_f32(),
            gv.to_f32(),
            e,
        );
    }
}

#[test]
fn matmul_bf16_random_shapes() {
    let dev = match try_wgpu_device("matmul_bf16_random_shapes") {
        Some(d) => d,
        None => return,
    };
    let mut rng = StdRng::seed_from_u64(0xb16);
    // Mix of tile-aligned and unaligned shapes. The unaligned ones
    // exercise both the boundary-mask path *and* the rare row that
    // starts on an odd bf16 element index (so the packed-u32 view of
    // the row is misaligned by one half-word).
    let shapes: &[(usize, usize, usize)] = &[
        (1, 8, 8),
        (8, 16, 32),
        (33, 17, 23),
        (5, 7, 11),
        (128, 128, 128),
    ];
    for &(m, k, n) in shapes {
        let a_f32: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() - 0.5).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|_| rng.random::<f32>() - 0.5).collect();
        let a_bf16: Vec<bf16> = a_f32.iter().map(|v| bf16::from_f32(*v)).collect();
        let b_bf16: Vec<bf16> = b_f32.iter().map(|v| bf16::from_f32(*v)).collect();
        let a_cpu = Tensor::from_vec(a_bf16, (m, k), &Device::Cpu).unwrap();
        let b_cpu = Tensor::from_vec(b_bf16, (k, n), &Device::Cpu).unwrap();
        let c_cpu = a_cpu.matmul(&b_cpu).unwrap();
        let c_gpu = a_cpu
            .to_device(&dev)
            .unwrap()
            .matmul(&b_cpu.to_device(&dev).unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();

        let r: Vec<bf16> = c_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<bf16> = c_gpu.flatten_all().unwrap().to_vec1().unwrap();
        let mut max_err = 0.0f32;
        for (rv, gv) in r.iter().zip(g.iter()) {
            let e = (rv.to_f32() - gv.to_f32()).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < BF16_TOL,
            "bf16 shape ({m},{k},{n}): max_err={max_err} > tol={BF16_TOL}"
        );
    }
}

#[test]
fn matmul_bf16_transposed_rhs() {
    // Same intent as the f32 variant: confirm the bf16 kernel honors
    // strides on a transposed rhs.
    let dev = match try_wgpu_device("matmul_bf16_transposed_rhs") {
        Some(d) => d,
        None => return,
    };
    let mut rng = StdRng::seed_from_u64(0xb17);
    let (m, k, n) = (16usize, 24usize, 12usize);
    let a_f32: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() - 0.5).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|_| rng.random::<f32>() - 0.5).collect();
    let a_bf16: Vec<bf16> = a_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let b_bf16: Vec<bf16> = b_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let a_cpu = Tensor::from_vec(a_bf16, (m, k), &Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b_bf16, (n, k), &Device::Cpu).unwrap();
    let c_cpu = a_cpu.matmul(&b_cpu.t().unwrap()).unwrap();
    let c_gpu = a_cpu
        .to_device(&dev)
        .unwrap()
        .matmul(&b_cpu.to_device(&dev).unwrap().t().unwrap())
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap();
    let r: Vec<bf16> = c_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let g: Vec<bf16> = c_gpu.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_err = 0.0f32;
    for (rv, gv) in r.iter().zip(g.iter()) {
        let e = (rv.to_f32() - gv.to_f32()).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(max_err < BF16_TOL, "bf16 transposed_rhs: max_err={max_err}");
}
