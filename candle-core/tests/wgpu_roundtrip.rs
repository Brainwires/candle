//! WebGPU CPU↔GPU round-trip tests (Phase 3.2).
//!
//! These tests exercise the data-movement portion of `WgpuStorage`:
//! upload (`from_cpu_storage` via `Tensor::to_device`), readback
//! (`to_cpu_storage` via `Tensor::to_device(&Cpu)`), and zero-init
//! (`zeros_impl`).
//!
//! No GPU? No problem. Each test calls a shared helper that tries to
//! create a `WgpuDevice`; if construction fails (no adapter / no
//! Vulkan-Metal-DX12 driver / running in a headless CI sandbox), we
//! print a `skipping` line and exit 0. CI on a GPU-less runner should
//! not turn red because the wgpu feature compiled fine.
//!
//! Run: `cargo test -p candle-core --features wgpu --test wgpu_roundtrip`
#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Tensor};
use half::f16;

/// Try to acquire a WGPU device, or print a skip notice and return None.
///
/// We use `Device::new_wgpu(0)` which is the sync constructor — fine
/// on native test runners. wasm32 isn't a target for these tests
/// because they assume a sync readback path.
fn try_wgpu_device(test_name: &str) -> Option<Device> {
    match Device::new_wgpu(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping {test_name}: no wgpu adapter available ({e})");
            None
        }
    }
}

#[test]
fn f32_roundtrip() {
    let dev = match try_wgpu_device("f32_roundtrip") {
        Some(d) => d,
        None => return,
    };
    let cpu_in =
        Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).expect("cpu tensor");
    let on_gpu = cpu_in.to_device(&dev).expect("upload");
    assert!(on_gpu.device().is_wgpu());
    let back_to_cpu = on_gpu.to_device(&Device::Cpu).expect("readback");
    let out: Vec<f32> = back_to_cpu
        .flatten_all()
        .unwrap()
        .to_vec1()
        .expect("to_vec1");
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn f16_roundtrip() {
    let dev = match try_wgpu_device("f16_roundtrip") {
        Some(d) => d,
        None => return,
    };
    let values: Vec<f16> = [1.0f32, 2.5, -3.0, 0.125]
        .iter()
        .map(|v| f16::from_f32(*v))
        .collect();
    let cpu_in = Tensor::from_slice(&values, (2, 2), &Device::Cpu).expect("cpu f16 tensor");
    let on_gpu = cpu_in.to_device(&dev).expect("upload");
    assert_eq!(on_gpu.dtype(), DType::F16);
    let back_to_cpu = on_gpu.to_device(&Device::Cpu).expect("readback");
    let out: Vec<f16> = back_to_cpu
        .flatten_all()
        .unwrap()
        .to_vec1()
        .expect("to_vec1");
    assert_eq!(out, values);
}

#[test]
fn u32_roundtrip() {
    let dev = match try_wgpu_device("u32_roundtrip") {
        Some(d) => d,
        None => return,
    };
    let values: Vec<u32> = vec![0, 1, 7, 42, 4294967295];
    let cpu_in = Tensor::from_slice(&values, (5,), &Device::Cpu).expect("cpu u32 tensor");
    let on_gpu = cpu_in.to_device(&dev).expect("upload");
    assert_eq!(on_gpu.dtype(), DType::U32);
    let back_to_cpu = on_gpu.to_device(&Device::Cpu).expect("readback");
    let out: Vec<u32> = back_to_cpu.to_vec1().expect("to_vec1");
    assert_eq!(out, values);
}

#[test]
fn zeros_alloc() {
    let dev = match try_wgpu_device("zeros_alloc") {
        Some(d) => d,
        None => return,
    };
    let z = Tensor::zeros((4, 4), DType::F32, &dev).expect("zeros on gpu");
    assert_eq!(z.dtype(), DType::F32);
    assert_eq!(z.dims(), &[4, 4]);
    let cpu = z.to_device(&Device::Cpu).expect("readback");
    let v: Vec<f32> = cpu.flatten_all().unwrap().to_vec1().expect("to_vec1");
    assert_eq!(v.len(), 16);
    assert!(v.iter().all(|x| *x == 0.0), "expected all zeros, got {v:?}");
}

/// Single-element edge case: makes sure the 4-byte alignment fallback
/// doesn't truncate or read past the end for `len = 1`.
#[test]
fn single_element_f32() {
    let dev = match try_wgpu_device("single_element_f32") {
        Some(d) => d,
        None => return,
    };
    let cpu_in = Tensor::from_slice(&[42.0f32], (1,), &Device::Cpu).expect("cpu tensor");
    let on_gpu = cpu_in.to_device(&dev).expect("upload");
    let back: Vec<f32> = on_gpu
        .to_device(&Device::Cpu)
        .expect("readback")
        .to_vec1()
        .expect("to_vec1");
    assert_eq!(back, vec![42.0]);
}

/// Larger buffer to confirm we're not silently capped by some default
/// WebGPU limit (default minStorageBufferBindingSize is 128 MiB which
/// is way more than this — but kicking the tires is cheap).
#[test]
fn larger_f32_buffer() {
    let dev = match try_wgpu_device("larger_f32_buffer") {
        Some(d) => d,
        None => return,
    };
    let n: usize = 4096;
    let values: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let cpu_in = Tensor::from_slice(&values, (n,), &Device::Cpu).expect("cpu tensor");
    let on_gpu = cpu_in.to_device(&dev).expect("upload");
    let back: Vec<f32> = on_gpu
        .to_device(&Device::Cpu)
        .expect("readback")
        .to_vec1()
        .expect("to_vec1");
    assert_eq!(back, values);
}
