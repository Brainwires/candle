//! Int-domain Q4_K × Q8K matmul correctness test (Step 4 of the
//! `prancy-tickling-puzzle.md` plan).
//!
//! Drives the new WGPU `matmul_q4k_q8k_m1` kernel through `QMatMul::forward`
//! and compares to CPU's `vec_dot_q4k_q8k` reference. The hard contract is
//! `max(|wgpu - cpu|) / max(|cpu|) < 1e-5` across K ∈ {1024, 2048, 4096, 8192}.
//! If this holds, browser-WGPU Gemma 4 matches CPU output bit-for-bit at
//! the matmul level (drift previously ~0.5% with f32-dequant kernel).
//!
//! The kernel auto-selects only when:
//!   - dtype == Q4K
//!   - m == 1
//!   - dev.supports_dp4a() (WGSL `packed_4x8_integer_dot_product`)
//!   - K is a multiple of 256
//!   - activation row is contiguous (stride_k == 1) with offset 0 and b == 1
//! All five hold for the cases in this test.

#![cfg(feature = "wgpu")]

use candle_core::{
    quantized::{self, GgmlDType},
    Device, Module, Tensor,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[allow(deprecated)]
fn rand_f32(rng: &mut StdRng) -> f32 {
    rng.gen::<f32>()
}

fn try_wgpu() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

/// Run a (m=1, k, n) Q4_K matmul on both CPU and WGPU and assert the
/// max relative difference is below `tol`.
fn run_one(label: &str, m: usize, k: usize, n: usize, tol: f32, seed: u64) {
    let cpu = Device::Cpu;
    let wgpu = match try_wgpu() {
        Some(d) => d,
        None => {
            eprintln!("skipping {label} — no wgpu device");
            return;
        }
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let lhs: Vec<f32> = (0..(m * k)).map(|_| rand_f32(&mut rng) - 0.5).collect();
    // Weights distributed similarly to a real attention-projection.
    let rhs: Vec<f32> = (0..(k * n)).map(|_| (rand_f32(&mut rng) - 0.5) * 0.05).collect();

    // CPU reference: quantize weights once, run matmul on CPU.
    let lhs_cpu = Tensor::from_slice(&lhs, (m, k), &cpu).expect("lhs cpu");
    let rhs_cpu = Tensor::from_slice(&rhs, (k, n), &cpu).expect("rhs cpu");
    // QMatMul stores weights pre-transposed to (n, k); we pass (n, k) by
    // taking `rhs_cpu.t()?` (gives (n, k)).
    let qweights_cpu = quantized::QTensor::quantize(&rhs_cpu.t().expect("t"), GgmlDType::Q4K)
        .expect("quantize cpu");
    let qmm_cpu = quantized::QMatMul::from_qtensor(qweights_cpu).expect("qmm cpu");
    let out_cpu = qmm_cpu.forward(&lhs_cpu).expect("forward cpu");
    let out_cpu_v = out_cpu
        .reshape((m * n,))
        .expect("reshape cpu")
        .to_vec1::<f32>()
        .expect("vec cpu");

    // WGPU path: same weights, same input — through the new int-domain
    // kernel.
    let lhs_w = Tensor::from_slice(&lhs, (m, k), &wgpu).expect("lhs wgpu");
    let rhs_w = Tensor::from_slice(&rhs, (k, n), &wgpu).expect("rhs wgpu");
    let qweights_w = quantized::QTensor::quantize(&rhs_w.t().expect("t"), GgmlDType::Q4K)
        .expect("quantize wgpu");
    let qmm_w = quantized::QMatMul::from_qtensor(qweights_w).expect("qmm wgpu");
    let out_w = qmm_w.forward(&lhs_w).expect("forward wgpu");
    let out_w_v = out_w
        .reshape((m * n,))
        .expect("reshape wgpu")
        .to_vec1::<f32>()
        .expect("vec wgpu");

    assert_eq!(out_cpu_v.len(), out_w_v.len());

    // Compute max-abs-rel-diff. Use max(|cpu|) as the denominator (a
    // single normalizer per row, like the bisect harness's L2-style
    // metric — element-wise division would over-weight tiny outputs).
    let max_abs_cpu = out_cpu_v.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mut worst_idx = 0usize;
    let mut worst_abs = 0.0f32;
    for (i, (&c, &w)) in out_cpu_v.iter().zip(out_w_v.iter()).enumerate() {
        let d = (c - w).abs();
        if d > worst_abs {
            worst_abs = d;
            worst_idx = i;
        }
    }
    let max_rel = if max_abs_cpu > 0.0 {
        worst_abs / max_abs_cpu
    } else {
        worst_abs
    };
    eprintln!(
        "{label}: m={m} k={k} n={n} max_abs_diff={worst_abs:.3e} max_rel={max_rel:.3e} (worst at idx {worst_idx}: cpu={} wgpu={})",
        out_cpu_v[worst_idx], out_w_v[worst_idx]
    );
    assert!(
        max_rel < tol,
        "{label}: max_rel {max_rel:.3e} exceeds tol {tol:.0e} (k={k}, n={n})"
    );
}

#[test]
fn q4k_int_matmul_wgpu_k_1024() {
    run_one("k=1024 (decode)", 1, 1024, 256, 1e-5, 0xC4_5DE_8E1_F00D0);
}

#[test]
fn q4k_int_matmul_wgpu_k_2048() {
    // K=2048 is the gemma4:e2b q_proj K dimension — the decode hot path.
    run_one("k=2048 (decode hot)", 1, 2048, 256, 1e-5, 0xC4_5DE_8E1_F00D1);
}

#[test]
fn q4k_int_matmul_wgpu_k_4096() {
    run_one("k=4096", 1, 4096, 256, 1e-5, 0xC4_5DE_8E1_F00D2);
}

#[test]
fn q4k_int_matmul_wgpu_k_8192() {
    run_one("k=8192", 1, 8192, 256, 1e-5, 0xC4_5DE_8E1_F00D3);
}

#[test]
fn q4k_int_matmul_wgpu_n_1() {
    // Edge case: a single output element. Verifies the per-workgroup
    // reduction works when only one workgroup is dispatched.
    run_one("n=1", 1, 2048, 1, 1e-5, 0xC4_5DE_8E1_F00D4);
}

#[test]
fn q4k_int_matmul_wgpu_n_vocab() {
    // N near vocabulary size — verifies the dispatch scales.
    // 256016 is gemma4:e2b's vocab size. Using a smaller representative
    // value to keep the test fast.
    run_one("n=8192 (vocab-ish)", 1, 2048, 8192, 1e-5, 0xC4_5DE_8E1_F00D5);
}
