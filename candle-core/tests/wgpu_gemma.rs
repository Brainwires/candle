//! WebGPU index_select correctness tests (Phase 3.7).
//!
//! Tests the index_select kernel used for embedding lookups.
//!
//! Run:
//!   cargo test -p candle-core --features wgpu --test wgpu_gemma
#![cfg(feature = "wgpu")]

use candle_core::{Device, Tensor};

fn try_wgpu() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) -> f32 {
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
    max_diff
}

/// Basic index_select on dim 0: embedding lookup pattern.
#[test]
fn index_select_dim0() {
    let Some(gpu) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;

    // Source: (4, 3) embedding table.
    let src_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
    let ids_data: Vec<u32> = vec![2, 0, 3, 1];

    let src_cpu = Tensor::from_slice(&src_data, (4, 3), &cpu).unwrap();
    let ids_cpu = Tensor::from_slice(&ids_data, (4,), &cpu).unwrap();
    let out_cpu = src_cpu.index_select(&ids_cpu, 0).unwrap();
    let out_cpu_flat: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();

    let src_gpu = Tensor::from_slice(&src_data, (4, 3), &gpu).unwrap();
    let ids_gpu = Tensor::from_slice(&ids_data, (4,), &gpu).unwrap();
    let out_gpu = src_gpu.index_select(&ids_gpu, 0).unwrap();
    let out_gpu_flat: Vec<f32> = out_gpu.flatten_all().unwrap().to_vec1().unwrap();

    let diff = assert_close(&out_cpu_flat, &out_gpu_flat, 1e-6, "index_select_dim0");
    eprintln!("[index_select_dim0] max diff = {diff}");
}

/// Index_select on dim 1 (column selection).
#[test]
fn index_select_dim1() {
    let Some(gpu) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;

    // Source: (3, 5), select 2 columns.
    let src_data: Vec<f32> = (0..15).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let ids_data: Vec<u32> = vec![4, 1];

    let src_cpu = Tensor::from_slice(&src_data, (3, 5), &cpu).unwrap();
    let ids_cpu = Tensor::from_slice(&ids_data, (2,), &cpu).unwrap();
    let out_cpu = src_cpu.index_select(&ids_cpu, 1).unwrap();
    let out_cpu_flat: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();

    let src_gpu = Tensor::from_slice(&src_data, (3, 5), &gpu).unwrap();
    let ids_gpu = Tensor::from_slice(&ids_data, (2,), &gpu).unwrap();
    let out_gpu = src_gpu.index_select(&ids_gpu, 1).unwrap();
    let out_gpu_flat: Vec<f32> = out_gpu.flatten_all().unwrap().to_vec1().unwrap();

    let diff = assert_close(&out_cpu_flat, &out_gpu_flat, 1e-6, "index_select_dim1");
    eprintln!("[index_select_dim1] max diff = {diff}");
}

/// Larger embedding table — simulates real vocab embeddings.
#[test]
fn index_select_large_embedding() {
    let Some(gpu) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return;
    };
    let cpu = Device::Cpu;

    let vocab = 128;
    let hidden = 64;
    let seq_len = 8;

    let src_data: Vec<f32> = (0..(vocab * hidden))
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    let ids_data: Vec<u32> = (0..seq_len).map(|i| ((i * 17 + 3) % vocab) as u32).collect();

    let src_cpu = Tensor::from_slice(&src_data, (vocab, hidden), &cpu).unwrap();
    let ids_cpu = Tensor::from_slice(&ids_data, (seq_len,), &cpu).unwrap();
    let out_cpu = src_cpu.index_select(&ids_cpu, 0).unwrap();
    let out_cpu_flat: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();

    let src_gpu = Tensor::from_slice(&src_data, (vocab, hidden), &gpu).unwrap();
    let ids_gpu = Tensor::from_slice(&ids_data, (seq_len,), &gpu).unwrap();
    let out_gpu = src_gpu.index_select(&ids_gpu, 0).unwrap();
    let out_gpu_flat: Vec<f32> = out_gpu.flatten_all().unwrap().to_vec1().unwrap();

    let diff = assert_close(&out_cpu_flat, &out_gpu_flat, 1e-6, "index_select_large");
    eprintln!("[index_select_large_embedding] max diff = {diff}");
}
