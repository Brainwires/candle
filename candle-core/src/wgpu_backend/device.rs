//! WebGPU device handle (Phase 3.1 scaffolding).
//!
//! Phase 3.1 keeps the surface intentionally minimal: a thin `Arc`-wrapped
//! handle plus a [`crate::backend::BackendDevice`] impl whose methods all
//! return `Err(Error::Msg(...))`. The actual `wgpu::Instance` /
//! `wgpu::Adapter` / `wgpu::Device` negotiation lives behind an async
//! `WgpuDevice::new` that we will wire up in Phase 3.2 once we have a
//! reason to allocate a real GPU buffer. For now the synchronous
//! constructor returns a `not yet implemented` error, which is enough to
//! make the type system happy without forcing us to settle the
//! pollster/wasm-bindgen-futures question prematurely.

use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Result, Shape};
use std::sync::Arc;

use super::storage::WgpuStorage;

/// Opaque handle to a WebGPU device.
///
/// Phase 3.1 stub. Cheap to clone (internally `Arc`). The inner data is
/// `()` for now; Phase 3.2 will add `wgpu::Device` and `wgpu::Queue`.
#[derive(Debug, Clone)]
pub struct WgpuDevice {
    inner: Arc<WgpuDeviceInner>,
}

#[derive(Debug)]
struct WgpuDeviceInner {
    // TODO(phase-3.2): wgpu::Device, wgpu::Queue, wgpu::AdapterInfo.
    _phantom: std::marker::PhantomData<()>,
}

impl WgpuDevice {
    fn not_implemented<T>(op: &str) -> Result<T> {
        Err(crate::Error::Msg(format!(
            "wgpu: {op} not yet implemented (Phase 3.2+)"
        )))
    }
}

impl PartialEq for WgpuDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for WgpuDevice {}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(_ordinal: usize) -> Result<Self> {
        Self::not_implemented("device construction")
    }

    fn location(&self) -> crate::DeviceLocation {
        // Phase 3.1: WebGPU does not yet have its own DeviceLocation
        // variant; piggy-back on `Cpu` so the existing same-device checks
        // do not panic. Phase 3.2 introduces `DeviceLocation::Wgpu`.
        crate::DeviceLocation::Cpu
    }

    fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Self::not_implemented("zeros_impl")
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Self::not_implemented("alloc_uninit")
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Self::not_implemented("storage_from_slice")
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Self::not_implemented("storage_from_cpu_storage")
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Self::not_implemented("storage_from_cpu_storage_owned")
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Self::not_implemented("rand_uniform")
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Self::not_implemented("rand_normal")
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Self::not_implemented("set_seed")
    }

    fn get_current_seed(&self) -> Result<u64> {
        Self::not_implemented("get_current_seed")
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
