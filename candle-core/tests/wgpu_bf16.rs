//! WebGPU BF16 op coverage tests.
//!
//! Cross-checks every op the gemma4 forward path needs on bf16 against
//! the CPU reference. The wgpu backend runs bf16 inputs through an
//! auto-promote `bf16 → f32 → bf16` round-trip for the f-typed kernels
//! (unary, binary, reduce, affine, index_select, softmax, rope) and uses
//! a native bf16 kernel for matmul + strided copy. Either way we expect
//! results within bf16 quantisation noise of the CPU answer.
//!
//! Tolerance: 5e-2 absolute is the right scale given bf16's 7-bit
//! mantissa — single-product rounding error is already ~2^-8 of the
//! magnitude, and several of these ops accumulate.
//!
//! Run: `cargo test -p candle-core --features wgpu --test wgpu_bf16`
#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Tensor};
use half::bf16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f32 = 5e-2;

fn try_wgpu_device(test_name: &str) -> Option<Device> {
    match Device::new_wgpu(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping {test_name}: no wgpu adapter available ({e})");
            None
        }
    }
}

fn vec_bf16(rng: &mut StdRng, n: usize) -> Vec<bf16> {
    (0..n)
        .map(|_| bf16::from_f32(rng.random::<f32>() - 0.5))
        .collect()
}

/// CPU↔GPU bf16 result comparison. Both sides flatten to a `Vec<bf16>`
/// and we check element-wise to the shared tolerance.
#[track_caller]
fn assert_close_bf16(cpu: &Tensor, gpu: &Tensor, label: &str) {
    let r: Vec<bf16> = cpu.flatten_all().unwrap().to_vec1().unwrap();
    let g: Vec<bf16> = gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(r.len(), g.len(), "{label}: length mismatch");
    let mut max_err = 0.0f32;
    for (i, (a, b)) in r.iter().zip(g.iter()).enumerate() {
        let af = a.to_f32();
        let bf = b.to_f32();
        let e = (af - bf).abs();
        if e > max_err {
            max_err = e;
        }
        assert!(
            e < TOL,
            "{label}[{i}]: cpu={af} gpu={bf} err={e} > tol={TOL}",
        );
    }
    let _ = max_err; // helpful in panic messages above
}

// ── unary ────────────────────────────────────────────────────────────

#[test]
fn bf16_unary_silu() {
    let Some(dev) = try_wgpu_device("bf16_unary_silu") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x510);
    let data = vec_bf16(&mut rng, 256);
    let cpu_in = Tensor::from_vec(data, (16, 16), &Device::Cpu).unwrap();
    let cpu_out = cpu_in.silu().unwrap();
    let gpu_out = cpu_in.to_device(&dev).unwrap().silu().unwrap();
    assert_close_bf16(&cpu_out, &gpu_out, "silu");
}

#[test]
fn bf16_unary_sin_cos() {
    let Some(dev) = try_wgpu_device("bf16_unary_sin_cos") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x511);
    let data = vec_bf16(&mut rng, 64);
    let cpu_in = Tensor::from_vec(data, (8, 8), &Device::Cpu).unwrap();
    let cpu_sin = cpu_in.sin().unwrap();
    let cpu_cos = cpu_in.cos().unwrap();
    let gpu_in = cpu_in.to_device(&dev).unwrap();
    assert_close_bf16(&cpu_sin, &gpu_in.sin().unwrap(), "sin");
    assert_close_bf16(&cpu_cos, &gpu_in.cos().unwrap(), "cos");
}

// ── binary ───────────────────────────────────────────────────────────

#[test]
fn bf16_binary_add_mul() {
    let Some(dev) = try_wgpu_device("bf16_binary_add_mul") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x512);
    let a = vec_bf16(&mut rng, 64);
    let b = vec_bf16(&mut rng, 64);
    let a_cpu = Tensor::from_vec(a, (8, 8), &Device::Cpu).unwrap();
    let b_cpu = Tensor::from_vec(b, (8, 8), &Device::Cpu).unwrap();
    let a_gpu = a_cpu.to_device(&dev).unwrap();
    let b_gpu = b_cpu.to_device(&dev).unwrap();

    assert_close_bf16(
        &(&a_cpu + &b_cpu).unwrap(),
        &(&a_gpu + &b_gpu).unwrap(),
        "add",
    );
    assert_close_bf16(
        &(&a_cpu * &b_cpu).unwrap(),
        &(&a_gpu * &b_gpu).unwrap(),
        "mul",
    );
}

#[test]
fn bf16_binary_broadcast_add() {
    // RMSNorm hits broadcast-add via `1.0 + weight`.
    let Some(dev) = try_wgpu_device("bf16_binary_broadcast_add") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x513);
    let x = vec_bf16(&mut rng, 32);
    let w = vec_bf16(&mut rng, 4);
    let x_cpu = Tensor::from_vec(x, (8, 4), &Device::Cpu).unwrap();
    let w_cpu = Tensor::from_vec(w, (1, 4), &Device::Cpu).unwrap();
    let cpu_out = x_cpu.broadcast_add(&w_cpu).unwrap();
    let gpu_out = x_cpu
        .to_device(&dev)
        .unwrap()
        .broadcast_add(&w_cpu.to_device(&dev).unwrap())
        .unwrap();
    assert_close_bf16(&cpu_out, &gpu_out, "broadcast_add");
}

// ── reduce ───────────────────────────────────────────────────────────

#[test]
fn bf16_reduce_sum_keepdim() {
    let Some(dev) = try_wgpu_device("bf16_reduce_sum_keepdim") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x514);
    let data = vec_bf16(&mut rng, 64);
    let cpu_in = Tensor::from_vec(data, (8, 8), &Device::Cpu).unwrap();
    let cpu_out = cpu_in.sum_keepdim(1).unwrap();
    let gpu_out = cpu_in.to_device(&dev).unwrap().sum_keepdim(1).unwrap();
    assert_close_bf16(&cpu_out, &gpu_out, "sum_keepdim");
}

// ── affine ───────────────────────────────────────────────────────────

#[test]
fn bf16_affine() {
    let Some(dev) = try_wgpu_device("bf16_affine") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x515);
    let data = vec_bf16(&mut rng, 32);
    let cpu_in = Tensor::from_vec(data, (4, 8), &Device::Cpu).unwrap();
    let cpu_out = cpu_in.affine(1.5, -0.25).unwrap();
    let gpu_out = cpu_in.to_device(&dev).unwrap().affine(1.5, -0.25).unwrap();
    assert_close_bf16(&cpu_out, &gpu_out, "affine");
}

// ── index_select ─────────────────────────────────────────────────────

#[test]
fn bf16_index_select_embedding() {
    // Mirrors the embed_tokens lookup at the start of a Gemma forward.
    let Some(dev) = try_wgpu_device("bf16_index_select_embedding") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x516);
    let weight = vec_bf16(&mut rng, 16 * 4);
    let weight_cpu = Tensor::from_vec(weight, (16, 4), &Device::Cpu).unwrap();
    let ids_cpu = Tensor::from_vec(vec![0u32, 5, 12, 3, 7], 5, &Device::Cpu).unwrap();
    let cpu_out = weight_cpu.index_select(&ids_cpu, 0).unwrap();
    let gpu_out = weight_cpu
        .to_device(&dev)
        .unwrap()
        .index_select(&ids_cpu.to_device(&dev).unwrap(), 0)
        .unwrap();
    assert_close_bf16(&cpu_out, &gpu_out, "index_select");
}

// ── softmax test lives in candle-nn/tests/wgpu_attention.rs ──────────
//
// `softmax_last_dim` is exposed only through `candle_nn::ops`, so its
// bf16 coverage sits in the candle-nn test crate which can pull in that
// dependency. Same story for the rope tests.

// ── strided copy (transpose then materialise via .contiguous()) ──────

#[test]
fn bf16_strided_copy_via_contiguous() {
    // `t.transpose(0, 1).contiguous()` exercises the strided→contiguous
    // bf16 copy kernel: input is non-contiguous, output is a fresh
    // contiguous buffer. The atomicAnd + atomicOr write pattern needs
    // to round-trip the original bf16 bits exactly (no f32 round-trip
    // is involved for copy), so this test should hit zero error.
    let Some(dev) = try_wgpu_device("bf16_strided_copy_via_contiguous") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x518);
    let data = vec_bf16(&mut rng, 7 * 5);
    let cpu_in = Tensor::from_vec(data, (7, 5), &Device::Cpu).unwrap();
    let cpu_out = cpu_in.t().unwrap().contiguous().unwrap();
    let gpu_out = cpu_in
        .to_device(&dev)
        .unwrap()
        .t()
        .unwrap()
        .contiguous()
        .unwrap();
    // No f32 round-trip: copy preserves bits exactly.
    let r: Vec<bf16> = cpu_out.flatten_all().unwrap().to_vec1().unwrap();
    let g: Vec<bf16> = gpu_out
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(r.len(), g.len());
    for (i, (a, b)) in r.iter().zip(g.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "strided_copy[{i}]: cpu={} gpu={}",
            a.to_f32(),
            b.to_f32(),
        );
    }
}

// ── dtype passes through correctly ───────────────────────────────────

#[test]
fn bf16_output_dtype_is_bf16() {
    // After auto-promote, the result tensor should still report bf16
    // (so downstream ops like the next matmul stay on the bf16 path).
    let Some(dev) = try_wgpu_device("bf16_output_dtype_is_bf16") else {
        return;
    };
    let mut rng = StdRng::seed_from_u64(0x519);
    let data = vec_bf16(&mut rng, 16);
    let t = Tensor::from_vec(data, (4, 4), &Device::Cpu)
        .unwrap()
        .to_device(&dev)
        .unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.silu().unwrap().dtype(), DType::BF16);
    assert_eq!(t.sum_keepdim(0).unwrap().dtype(), DType::BF16);
    assert_eq!(t.affine(2.0, 0.0).unwrap().dtype(), DType::BF16);
    assert_eq!((&t + &t).unwrap().dtype(), DType::BF16);
}
