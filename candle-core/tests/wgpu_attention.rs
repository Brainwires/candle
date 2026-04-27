//! WebGPU attention-primitive correctness tests (Phase 3.6).
//!
//! Covers the candle-core-side primitives that Phase 3.6 wires:
//!   - `Tensor::cat` along the seq-axis (KV cache append pattern).
//!   - `copy2d` byte-level path for arbitrary axes.
//!
//! RoPE and the full causal attention block live in
//! `candle-nn/tests/wgpu_attention.rs` because they need
//! `candle_nn::rotary_emb` / `candle_nn::ops::softmax_last_dim`.
//!
//! Skips gracefully when no Wgpu adapter is available (CI lavapipe sandbox
//! is the target).
//!
//! Run: `cargo test -p candle-core --features wgpu --test wgpu_attention`
#![cfg(feature = "wgpu")]

use candle_core::{Device, Tensor};

fn try_wgpu() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

/// Compare two flat f32 buffers.
fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
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
}

/// Concat along the seq axis simulates a KV-cache append:
/// past `(B, H, T_past, D)` ++ new `(B, H, 1, D)` → `(B, H, T_past+1, D)`.
#[test]
fn cat_along_seq_axis_f32() {
    let Some(dev) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;

    // Past KV: (1, 2, 3, 4).
    let past_data: Vec<f32> = (0..(1 * 2 * 3 * 4)).map(|i| i as f32).collect();
    let new_data: Vec<f32> = (100..(100 + 1 * 2 * 1 * 4)).map(|i| i as f32).collect();

    let past_cpu = Tensor::from_slice(&past_data, (1, 2, 3, 4), &cpu).unwrap();
    let new_cpu = Tensor::from_slice(&new_data, (1, 2, 1, 4), &cpu).unwrap();
    let cat_cpu = Tensor::cat(&[&past_cpu, &new_cpu], 2)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let past_g = Tensor::from_slice(&past_data, (1, 2, 3, 4), &dev).unwrap();
    let new_g = Tensor::from_slice(&new_data, (1, 2, 1, 4), &dev).unwrap();
    let cat_g = Tensor::cat(&[&past_g, &new_g], 2)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_eq!(cat_cpu.len(), 1 * 2 * 4 * 4);
    assert_close(&cat_cpu, &cat_g, 0.0, "cat-seq");
}

/// Cat along axis 0 (batch axis) — covers the `cat0` (raw `copy_strided_src`)
/// path, which is distinct from `cat_contiguous` (`copy2d`) above.
#[test]
fn cat_along_batch_axis_f32() {
    let Some(dev) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;

    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (10..16).map(|i| i as f32).collect();

    let a_cpu = Tensor::from_slice(&a_data, (2, 3), &cpu).unwrap();
    let b_cpu = Tensor::from_slice(&b_data, (2, 3), &cpu).unwrap();
    let cat_cpu = Tensor::cat(&[&a_cpu, &b_cpu], 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let a_g = Tensor::from_slice(&a_data, (2, 3), &dev).unwrap();
    let b_g = Tensor::from_slice(&b_data, (2, 3), &dev).unwrap();
    let cat_g = Tensor::cat(&[&a_g, &b_g], 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_close(&cat_cpu, &cat_g, 0.0, "cat-batch");
}

/// Many-step KV-cache append loop: simulate appending one token at a
/// time, never reading the cache back to CPU between appends.
#[test]
fn kv_cache_append_loop_f32() {
    let Some(dev) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;
    let head_dim = 8usize;
    let n_heads = 2usize;
    let steps = 5usize;

    // Build a deterministic per-step (1, n_heads, 1, head_dim) tensor.
    let make_step = |step: usize| -> Vec<f32> {
        (0..n_heads * head_dim)
            .map(|i| (step * 1000 + i) as f32 * 0.01)
            .collect()
    };

    // CPU reference: build per-step tensors and cat along the seq
    // axis (axis 2), exactly the same op the GPU loop does. Comparing
    // the flat byte order of `Tensor::cat` (B,H,T,D) is what we want.
    let mut cpu_cache: Option<Tensor> = None;
    for s in 0..steps {
        let step_t = Tensor::from_slice(&make_step(s), (1, n_heads, 1, head_dim), &cpu).unwrap();
        cpu_cache = Some(match cpu_cache.take() {
            None => step_t,
            Some(c) => Tensor::cat(&[&c, &step_t], 2).unwrap(),
        });
    }
    let ref_cpu = cpu_cache
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // GPU loop: incremental cat without intermediate readback.
    let mut cache: Option<Tensor> = None;
    for s in 0..steps {
        let step = Tensor::from_slice(&make_step(s), (1, n_heads, 1, head_dim), &dev).unwrap();
        cache = Some(match cache.take() {
            None => step,
            Some(c) => Tensor::cat(&[&c, &step], 2).unwrap(),
        });
    }
    let cache = cache.unwrap();
    assert_eq!(cache.dims(), &[1, n_heads, steps, head_dim]);
    let got = cache.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_close(&ref_cpu, &got, 0.0, "kv-cache-loop");
}
