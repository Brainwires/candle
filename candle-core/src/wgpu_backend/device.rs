//! WebGPU device handle.
//!
//! Phase 3.2 wires real adapter / device negotiation. `WgpuDevice` now
//! holds an `Arc` of the live `wgpu::Device`, `wgpu::Queue`, and
//! `wgpu::AdapterInfo`. Compute kernels still come later (Phase 3.3+);
//! this module is responsible only for owning the GPU handles and
//! implementing the data-movement portion of the [`BackendDevice`] trait.
//!
//! # Constructors
//!
//! - [`WgpuDevice::new_async`] is universal — works on every target. Use
//!   it from async contexts (browsers in particular).
//! - [`WgpuDevice::new`] is the sync constructor and is gated on native
//!   targets. It blocks via `pollster::block_on`. We deliberately do not
//!   expose a sync entry on wasm32 because blocking inside a JS event
//!   loop deadlocks the browser; the PWA wiring lands in Phase 3.8.
//!
//! # Buffer alignment
//!
//! WebGPU's `COPY_BUFFER_ALIGNMENT` is 4 bytes. We round buffer sizes up
//! to the next multiple of 4 — relevant for `f16`, `u8`, and any future
//! sub-byte dtypes. Trailing padding bytes are allocated but never read.

use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, DType, Result, Shape};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::storage::WgpuStorage;

/// Opaque handle to a WebGPU device.
///
/// Cheap to clone (internally `Arc`).
#[derive(Clone)]
pub struct WgpuDevice {
    pub(crate) inner: Arc<WgpuDeviceInner>,
}

pub(crate) struct WgpuDeviceInner {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    /// Compute pipelines compiled lazily and cached by a stable key.
    /// Keys are kernel-name + dtype strings (e.g. `"matmul:f32"`); see
    /// `wgpu_backend::ops` for the compilation paths.
    pub(crate) pipelines: Mutex<HashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
    /// Whether the underlying adapter advertises `Features::SHADER_F16`.
    /// Kernels that need f16 storage / accumulation check this before
    /// dispatching.
    pub(crate) supports_shader_f16: bool,
}

impl std::fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuDevice")
            .field("backend", &self.inner.adapter_info.backend)
            .field("name", &self.inner.adapter_info.name)
            .field("device_type", &self.inner.adapter_info.device_type)
            .finish()
    }
}

impl std::fmt::Debug for WgpuDeviceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuDeviceInner")
            .field("adapter_info", &self.adapter_info)
            .finish()
    }
}

impl PartialEq for WgpuDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for WgpuDevice {}

impl WgpuDevice {
    /// Async constructor — works on every target (native + wasm32).
    ///
    /// Picks a high-performance adapter via [`wgpu::Instance::request_adapter`]
    /// and requests a default device. No surface is required because we
    /// only use the compute pipeline.
    pub async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| crate::Error::Msg(format!("wgpu: no suitable GPU adapter found: {e}")))?;

        let adapter_info = adapter.get_info();

        // Negotiate SHADER_F16 if the adapter supports it. Many software
        // renderers (lavapipe in CI, llvmpipe under Mesa) don't advertise
        // the feature; we degrade gracefully by leaving the f16 matmul
        // path returning `not_implemented` rather than failing at device
        // creation. Higher-level code can fall back to an f32-promoted
        // path or pick a different backend.
        let supports_shader_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let mut required_features = wgpu::Features::empty();
        if supports_shader_f16 {
            required_features |= wgpu::Features::SHADER_F16;
        } else {
            // One-line warning so the user knows which path is gated.
            // Don't spam: only emit when the env var is set, otherwise
            // staying quiet matches the rest of candle's logging style.
            if std::env::var("CANDLE_WGPU_LOG").is_ok() {
                eprintln!(
                    "wgpu: adapter {:?} does not advertise SHADER_F16; \
                     f16 matmul will fall back to not_implemented",
                    adapter_info.name
                );
            }
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("brainwires-candle-wgpu-device"),
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    experimental_features: Default::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: Default::default(),
                },
            )
            .await
            .map_err(|e| crate::Error::Msg(format!("wgpu: request_device failed: {e}")))?;

        Ok(Self {
            inner: Arc::new(WgpuDeviceInner {
                device,
                queue,
                adapter_info,
                pipelines: Mutex::new(HashMap::new()),
                supports_shader_f16,
            }),
        })
    }

    /// Sync constructor — native only. Blocks via `pollster`.
    ///
    /// On wasm32 use [`Self::new_async`] from your async context instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Borrow the underlying `wgpu::Device`. Internal helper for
    /// `WgpuStorage` — `pub(crate)` so it doesn't leak into the public
    /// candle API.
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    /// Borrow the underlying `wgpu::Queue`.
    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    /// `true` when the underlying adapter supports the `SHADER_F16`
    /// feature and the device was created with it enabled. Kernels that
    /// require native f16 must check this before compiling.
    pub(crate) fn supports_shader_f16(&self) -> bool {
        self.inner.supports_shader_f16
    }

    /// Get-or-create a compute pipeline keyed by a stable `&'static str`.
    ///
    /// Compiling WGSL is moderately expensive (tens to hundreds of
    /// milliseconds the first time). Subsequent calls reuse the cached
    /// pipeline. The mutex is held only for the duration of the cache
    /// lookup / insertion — pipeline compilation itself happens *before*
    /// the lock is taken so concurrent first-time callers don't serialize
    /// on each other... but in practice today candle-core is mostly
    /// single-threaded at the op-dispatch layer, so the lock contention
    /// story doesn't matter much.
    pub(crate) fn get_or_create_pipeline(
        &self,
        key: &'static str,
        wgsl_source: &str,
        entry_point: &str,
    ) -> Arc<wgpu::ComputePipeline> {
        // Fast path: pipeline already cached.
        if let Some(p) = self.inner.pipelines.lock().unwrap().get(key) {
            return Arc::clone(p);
        }
        // Slow path: compile + insert. The lock is dropped during
        // compilation so concurrent first-time callers don't serialize
        // unnecessarily — at the cost of possibly compiling the same
        // shader twice on a true race. The wins outweigh the duplication
        // since matmul is a hot dispatch.
        let module = self
            .inner
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(key),
                source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
            });
        let pipeline =
            self.inner
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(key),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
        let pipeline = Arc::new(pipeline);
        let mut guard = self.inner.pipelines.lock().unwrap();
        // Race-safe: another thread may have inserted while we compiled.
        Arc::clone(guard.entry(key).or_insert(pipeline))
    }
}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    /// `ordinal` is currently ignored — we always pick the
    /// high-performance adapter the platform reports. Multi-GPU
    /// selection is out of scope until we have a use case.
    fn new(_ordinal: usize) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::new()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(crate::Error::Msg(
                "wgpu: BackendDevice::new is sync and not available on wasm32; \
                 use `WgpuDevice::new_async().await` from your async context"
                    .to_string(),
            ))
        }
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Wgpu { gpu_id: 0 }
    }

    fn same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        WgpuStorage::alloc_zeros(self.clone(), dtype, shape)
    }

    /// Allocates an uninitialized buffer of the right size.
    ///
    /// WebGPU has no "uninitialized" buffer mode the way CUDA does —
    /// `STORAGE | COPY_SRC | COPY_DST` buffers are zero-filled by the
    /// driver on creation. So `alloc_uninit` is functionally identical
    /// to `zeros_impl`. The cost is paid either way; calling code that
    /// will overwrite the buffer immediately is no slower than CUDA.
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        WgpuStorage::alloc_zeros(self.clone(), dtype, shape)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let cpu = T::to_cpu_storage(data);
        WgpuStorage::from_cpu_storage(self.clone(), &cpu)
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Self::Storage> {
        WgpuStorage::from_cpu_storage(self.clone(), cpu)
    }

    fn storage_from_cpu_storage_owned(&self, cpu: CpuStorage) -> Result<Self::Storage> {
        WgpuStorage::from_cpu_storage(self.clone(), &cpu)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        // Phase 3.4 lands a real RNG kernel.
        Err(crate::Error::Msg(
            "wgpu: rand_uniform not yet implemented (Phase 3.4)".to_string(),
        ))
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(crate::Error::Msg(
            "wgpu: rand_normal not yet implemented (Phase 3.4)".to_string(),
        ))
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        // No on-device RNG yet (Phase 3.4). Accepting + ignoring the seed
        // keeps higher-level code that calls `set_seed` unconditionally
        // from blowing up before any rand_* op is reached.
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Err(crate::Error::Msg(
            "wgpu: get_current_seed not yet implemented (Phase 3.4)".to_string(),
        ))
    }

    fn synchronize(&self) -> Result<()> {
        // `Maintain::Wait` blocks until the GPU has finished all queued
        // work. On wasm32 `device.poll` is a no-op (browsers schedule
        // this themselves) — that matches WebGPU semantics.
        self.inner.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        Ok(())
    }
}

// ----------------------------------------------------------------------
// Sanity: WgpuDevice must satisfy the full BackendStorage::Device contract,
// otherwise `Storage::Wgpu` arms in candle-core won't compile.
//
// (Static check; no runtime cost.)
// ----------------------------------------------------------------------
#[allow(dead_code)]
fn _assert_backend_device() {
    fn is_backend_device<D: BackendDevice<Storage = WgpuStorage>>() {}
    is_backend_device::<WgpuDevice>();
    fn is_backend_storage<S: BackendStorage<Device = WgpuDevice>>() {}
    is_backend_storage::<WgpuStorage>();
}

// wasm32 is single-threaded — there is no real thread to race against.
// wgpu's JS-backed handles are !Send+!Sync because the browser's WebGPU
// objects are tied to the main thread, but on wasm32-unknown-unknown
// there IS only one thread, so the assertion is sound.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for WgpuDevice {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WgpuDevice {}
#[cfg(target_arch = "wasm32")]
unsafe impl Send for WgpuDeviceInner {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WgpuDeviceInner {}
