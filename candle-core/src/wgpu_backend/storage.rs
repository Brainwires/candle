//! WebGPU storage backed by a real `wgpu::Buffer`.
//!
//! Phase 3.2 wires CPU↔GPU data movement (alloc, upload, readback,
//! contiguous clone) but leaves all compute kernels — matmul, elementwise,
//! reductions, etc. — as `not yet implemented`. Those land in
//! Phases 3.3 onward.
//!
//! # Buffer layout
//!
//! Each `WgpuStorage` owns one `wgpu::Buffer` containing `len` elements
//! of `dtype`. The buffer's byte size is `len * dtype.size_in_bytes()`
//! rounded up to a multiple of `wgpu::COPY_BUFFER_ALIGNMENT` (4 bytes).
//! This rounding only affects sub-4-byte dtypes (`f16`, `bf16`, `u8`,
//! `i16`, `f8e4m3`); the trailing padding bytes are allocated but never
//! read by any kernel — kernels iterate by element count, not by buffer
//! size.
//!
//! # Memory pool — DEFERRED
//!
//! For Phase 3.2 every allocation goes straight to
//! `device.create_buffer`. A small pool / slab allocator is on the
//! Phase 3.9 list (perf polish). Marked with TODO(phase-3.9) below so
//! that work has a clear entry point.

#![allow(dead_code)]

use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
use std::sync::Arc;

use super::device::WgpuDevice;

/// Storage backed by a `wgpu::Buffer`.
///
/// The buffer is created with `STORAGE | COPY_SRC | COPY_DST` usage so
/// that it can both feed compute kernels and round-trip via copy
/// commands. Readback uses a separate ephemeral `MAP_READ | COPY_DST`
/// staging buffer — STORAGE buffers are not directly mappable by spec.
#[derive(Debug, Clone)]
pub struct WgpuStorage {
    /// The GPU buffer holding raw bytes.
    pub(crate) buffer: Arc<wgpu::Buffer>,
    /// Length in *elements* (not bytes).
    pub(crate) len: usize,
    /// Element dtype.
    pub(crate) dtype: DType,
    /// Owning device handle.
    pub(crate) device: WgpuDevice,
}

/// Round `n` up to the nearest multiple of `wgpu::COPY_BUFFER_ALIGNMENT`.
/// WebGPU requires copy operations on 4-byte-aligned offsets and sizes.
fn aligned_size(n: u64) -> u64 {
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    n.div_ceil(align) * align
}

/// Convert a `CpuStorage` to a contiguous byte slice for upload. We
/// can't borrow the underlying `Vec<T>` as `&[u8]` for non-byte dtypes
/// in safe code, so use `bytemuck`-style transmute via `slice::from_raw_parts`.
/// Each match arm is sound because `Vec<T>` is contiguous and `T` is
/// plain-old-data for every dtype we handle here.
fn cpu_storage_bytes(cpu: &CpuStorage) -> Vec<u8> {
    fn slice_to_bytes<T: Copy>(s: &[T]) -> Vec<u8> {
        let byte_len = std::mem::size_of_val(s);
        // Safety: T is POD for every dtype we support (integers, half,
        // bf16, f32, f64, F8E4M3 — all `repr(transparent)` over a
        // primitive). The lifetime of the returned Vec is independent
        // of `s` because we copy.
        let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, byte_len) };
        bytes.to_vec()
    }
    match cpu {
        CpuStorage::U8(v) => v.clone(),
        CpuStorage::U32(v) => slice_to_bytes(v),
        CpuStorage::I16(v) => slice_to_bytes(v),
        CpuStorage::I32(v) => slice_to_bytes(v),
        CpuStorage::I64(v) => slice_to_bytes(v),
        CpuStorage::BF16(v) => slice_to_bytes(v),
        CpuStorage::F16(v) => slice_to_bytes(v),
        CpuStorage::F32(v) => slice_to_bytes(v),
        CpuStorage::F64(v) => slice_to_bytes(v),
        CpuStorage::F8E4M3(v) => slice_to_bytes(v),
        // Sub-byte dtypes already store raw bytes.
        CpuStorage::F6E2M3(v) => v.clone(),
        CpuStorage::F6E3M2(v) => v.clone(),
        CpuStorage::F4(v) => v.clone(),
        CpuStorage::F8E8M0(v) => v.clone(),
    }
}

/// Reconstruct a `CpuStorage` of the given `dtype` from a byte buffer
/// containing exactly `len` elements (so byte length is
/// `len * dtype.size_in_bytes()` for full-byte dtypes — sub-byte dtypes
/// are out of scope for Phase 3.2).
fn cpu_storage_from_bytes(dtype: DType, len: usize, bytes: &[u8]) -> Result<CpuStorage> {
    fn bytes_to_vec<T: Copy>(bytes: &[u8], len: usize) -> Vec<T> {
        debug_assert_eq!(bytes.len(), len * std::mem::size_of::<T>());
        let mut out: Vec<T> = Vec::with_capacity(len);
        // Safety: bytes is properly sized for `len` elements of `T`.
        // We copy element-wise via the byte representation. `T` is POD
        // for every dtype handled below.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                out.as_mut_ptr() as *mut u8,
                len * std::mem::size_of::<T>(),
            );
            out.set_len(len);
        }
        out
    }
    Ok(match dtype {
        DType::U8 => CpuStorage::U8(bytes[..len].to_vec()),
        DType::U32 => CpuStorage::U32(bytes_to_vec::<u32>(&bytes[..len * 4], len)),
        DType::I16 => CpuStorage::I16(bytes_to_vec::<i16>(&bytes[..len * 2], len)),
        DType::I32 => CpuStorage::I32(bytes_to_vec::<i32>(&bytes[..len * 4], len)),
        DType::I64 => CpuStorage::I64(bytes_to_vec::<i64>(&bytes[..len * 8], len)),
        DType::BF16 => CpuStorage::BF16(bytes_to_vec::<half::bf16>(&bytes[..len * 2], len)),
        DType::F16 => CpuStorage::F16(bytes_to_vec::<half::f16>(&bytes[..len * 2], len)),
        DType::F32 => CpuStorage::F32(bytes_to_vec::<f32>(&bytes[..len * 4], len)),
        DType::F64 => CpuStorage::F64(bytes_to_vec::<f64>(&bytes[..len * 8], len)),
        DType::F8E4M3 => CpuStorage::F8E4M3(bytes_to_vec::<float8::F8E4M3>(&bytes[..len], len)),
        DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
            return Err(crate::Error::Msg(format!(
                "wgpu: readback for sub-byte dtype {dtype:?} not yet implemented"
            )));
        }
    })
}

impl WgpuStorage {
    fn not_implemented<T>(op: &str) -> Result<T> {
        Err(crate::Error::Msg(format!(
            "wgpu: {op} not yet implemented (Phase 3.3+)"
        )))
    }

    /// Allocate a zeroed buffer for `len = shape.elem_count()` elements
    /// of `dtype`. Backed by `device.create_buffer` with
    /// `MAPPED_AT_CREATION = true` so we can write zeros without a
    /// queue submission. (Newly-created buffers are not guaranteed to
    /// be zero in WebGPU spec; explicit fill is required.)
    pub fn alloc_zeros(device: WgpuDevice, dtype: DType, shape: &Shape) -> Result<Self> {
        let len = shape.elem_count();
        let elem_bytes = dtype.size_in_bytes();
        if elem_bytes == 0 {
            return Err(crate::Error::Msg(format!(
                "wgpu: alloc_zeros for sub-byte dtype {dtype:?} not yet supported"
            )));
        }
        let raw_size = (len * elem_bytes) as u64;
        // `wgpu::Buffer::size` must be > 0; allocate at least one
        // alignment unit to keep the API happy for empty tensors.
        let size = aligned_size(raw_size.max(wgpu::COPY_BUFFER_ALIGNMENT));

        // TODO(phase-3.9): replace direct create_buffer with a pooled
        // allocator that re-uses recently-freed buffers of the same
        // size class. For now we eat the per-alloc cost.
        let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-storage-zeros"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = buffer.slice(..).get_mapped_range_mut();
            view.slice(..).fill(0);
        }
        buffer.unmap();

        Ok(Self {
            buffer: Arc::new(buffer),
            len,
            dtype,
            device,
        })
    }

    /// Upload a CPU storage to a fresh GPU buffer.
    pub fn from_cpu_storage(device: WgpuDevice, cpu: &CpuStorage) -> Result<Self> {
        let dtype = cpu.dtype();
        let len = match cpu {
            CpuStorage::U8(v) => v.len(),
            CpuStorage::U32(v) => v.len(),
            CpuStorage::I16(v) => v.len(),
            CpuStorage::I32(v) => v.len(),
            CpuStorage::I64(v) => v.len(),
            CpuStorage::BF16(v) => v.len(),
            CpuStorage::F16(v) => v.len(),
            CpuStorage::F32(v) => v.len(),
            CpuStorage::F64(v) => v.len(),
            CpuStorage::F8E4M3(v) => v.len(),
            CpuStorage::F6E2M3(v)
            | CpuStorage::F6E3M2(v)
            | CpuStorage::F4(v)
            | CpuStorage::F8E8M0(v) => v.len(),
        };
        let bytes = cpu_storage_bytes(cpu);
        let raw_size = bytes.len() as u64;
        let size = aligned_size(raw_size.max(wgpu::COPY_BUFFER_ALIGNMENT));

        let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-storage-upload"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = buffer.slice(..).get_mapped_range_mut();
            view.slice(..bytes.len()).copy_from_slice(&bytes);
            view.slice(bytes.len()..).fill(0);
        }
        buffer.unmap();

        Ok(Self {
            buffer: Arc::new(buffer),
            len,
            dtype,
            device,
        })
    }

    /// Wrap an existing `wgpu::Buffer` as a `WgpuStorage`.
    ///
    /// The caller is responsible for ensuring the buffer contains valid
    /// data for `len` elements of `dtype`. Used internally by ops modules
    /// and externally for streamed uploads that bypass WASM linear memory.
    pub fn from_raw_buffer(
        buffer: Arc<wgpu::Buffer>,
        len: usize,
        dtype: DType,
        device: WgpuDevice,
    ) -> Self {
        Self {
            buffer,
            len,
            dtype,
            device,
        }
    }

    /// Read the GPU buffer back into a `CpuStorage`.
    ///
    /// Strategy: create a `MAP_READ | COPY_DST` staging buffer the same
    /// size as our storage buffer, encode a copy, submit, then
    /// `map_async` the staging buffer and poll until the map resolves.
    pub fn read_to_cpu(&self) -> Result<CpuStorage> {
        let elem_bytes = self.dtype.size_in_bytes();
        if elem_bytes == 0 {
            return Err(crate::Error::Msg(format!(
                "wgpu: readback for sub-byte dtype {:?} not yet supported",
                self.dtype
            )));
        }
        let payload_bytes = (self.len * elem_bytes) as u64;
        let staging_size = aligned_size(payload_bytes.max(wgpu::COPY_BUFFER_ALIGNMENT));

        let staging = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-storage-readback-staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu-storage-readback"),
                });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, staging_size);
        self.device.queue().submit(Some(encoder.finish()));

        // Map asynchronously and block until the callback fires.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // Send result; ignore failure if receiver is gone (shouldn't happen).
            let _ = tx.send(result);
        });

        // On native, poll(Wait) blocks until queued work + map callbacks resolve.
        // On wasm32 this is a no-op and the surrounding code must instead
        // await a future — which is fine because Phase 3.2's sync read
        // path is exercised only from native tests; wasm32 readback will
        // ride the async story we wire up in Phase 3.8.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device.device().poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            let map_result = rx
                .recv()
                .map_err(|e| crate::Error::Msg(format!("wgpu: map channel closed: {e}")))?;
            map_result
                .map_err(|e| crate::Error::Msg(format!("wgpu: buffer map_async failed: {e}")))?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            // Sync readback isn't possible on wasm — the JS event loop
            // can't make progress while we block. Surface a clear error;
            // the async read path in Phase 3.8 will use
            // `wasm-bindgen-futures` instead.
            let _ = rx; // silence unused warning
            return Err(crate::Error::Msg(
                "wgpu: synchronous readback is unavailable on wasm32; \
                 use the async path that lands in Phase 3.8"
                    .to_string(),
            ));
        }

        let cpu = {
            let view = slice.get_mapped_range();
            cpu_storage_from_bytes(self.dtype, self.len, &view[..payload_bytes as usize])?
        };
        // Drop the view before unmap (BufferView holds the lock).
        staging.unmap();
        Ok(cpu)
    }

    /// Asynchronous readback. Required path on `wasm32`, where the JS
    /// event loop drives `map_async` completion and a synchronous
    /// `recv()` would deadlock the worker. Functionally identical to
    /// [`Self::read_to_cpu`]; uses a hand-rolled `Future` over the
    /// `map_async` callback so the wasm-bindgen-futures executor can
    /// suspend the task while the GPU work and mapping complete.
    pub async fn read_to_cpu_async(&self) -> Result<CpuStorage> {
        let elem_bytes = self.dtype.size_in_bytes();
        if elem_bytes == 0 {
            return Err(crate::Error::Msg(format!(
                "wgpu: readback for sub-byte dtype {:?} not yet supported",
                self.dtype
            )));
        }
        let payload_bytes = (self.len * elem_bytes) as u64;
        let staging_size = aligned_size(payload_bytes.max(wgpu::COPY_BUFFER_ALIGNMENT));

        let staging = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-storage-readback-staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu-storage-readback"),
                });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, staging_size);
        self.device.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);

        // Bridge `map_async`'s fire-and-forget callback to a `Future`. The
        // shared state holds either the result (filled by the callback)
        // or the executor's waker (filled when we `await` before the
        // callback fires). Arc<Mutex<…>> works on both wasm32 (no-op)
        // and native (we may park briefly while `device.poll(Wait)` runs
        // off-thread before the callback resolves the channel).
        struct State {
            result: Option<std::result::Result<(), wgpu::BufferAsyncError>>,
            waker: Option<std::task::Waker>,
        }
        let state = std::sync::Arc::new(std::sync::Mutex::new(State {
            result: None,
            waker: None,
        }));
        let cb_state = std::sync::Arc::clone(&state);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let mut s = cb_state.lock().unwrap();
            s.result = Some(result);
            if let Some(w) = s.waker.take() {
                w.wake();
            }
        });

        // On native we still need to poll the device to drain queued
        // submissions — the callback won't fire otherwise. `Wait` is
        // safe here because we're on a thread that's not the JS event
        // loop. On wasm32 this is a no-op and the browser GPU thread
        // delivers the callback when the mapping is ready.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.device
                .device()
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .ok();
        }

        struct ReadbackFuture(std::sync::Arc<std::sync::Mutex<State>>);
        impl std::future::Future for ReadbackFuture {
            type Output = std::result::Result<(), wgpu::BufferAsyncError>;
            fn poll(
                self: std::pin::Pin<&mut Self>,
                cx: &mut std::task::Context<'_>,
            ) -> std::task::Poll<Self::Output> {
                let mut s = self.0.lock().unwrap();
                if let Some(r) = s.result.take() {
                    std::task::Poll::Ready(r)
                } else {
                    s.waker = Some(cx.waker().clone());
                    std::task::Poll::Pending
                }
            }
        }
        ReadbackFuture(state)
            .await
            .map_err(|e| crate::Error::Msg(format!("wgpu: buffer map_async failed: {e}")))?;

        let cpu = {
            let view = slice.get_mapped_range();
            cpu_storage_from_bytes(self.dtype, self.len, &view[..payload_bytes as usize])?
        };
        staging.unmap();
        Ok(cpu)
    }
}

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    /// Contiguous-only clone for Phase 3.2.
    ///
    /// A real strided clone needs a copy kernel (we have to gather
    /// non-adjacent elements according to layout strides). That kernel
    /// lands in Phase 3.3 alongside the elementwise plumbing. Until
    /// then, fall back to `not_implemented` for the strided case so
    /// that callers get a precise error rather than silently corrupting
    /// data.
    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        if !layout.is_contiguous() || layout.start_offset() != 0 {
            return Self::not_implemented("try_clone (non-contiguous layout)");
        }
        let elem_bytes = self.dtype.size_in_bytes();
        if elem_bytes == 0 {
            return Self::not_implemented("try_clone for sub-byte dtype");
        }
        let bytes = (self.len * elem_bytes) as u64;
        let size = aligned_size(bytes.max(wgpu::COPY_BUFFER_ALIGNMENT));

        let dst = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-storage-clone"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wgpu-storage-clone"),
                });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &dst, 0, size);
        self.device.queue().submit(Some(encoder.finish()));

        Ok(Self {
            buffer: Arc::new(dst),
            len: self.len,
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.read_to_cpu()
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype == DType::BF16 {
            let f32_in = promote_bf16_to_f32(self, layout)?;
            let f32_l = Layout::contiguous(layout.shape());
            let f32_out = super::ops::affine::affine(&f32_in, &f32_l, mul, add)?;
            return demote_f32_to_bf16(&f32_out, layout.shape());
        }
        super::ops::affine::affine(self, layout, mul, add)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Self::not_implemented("powf")
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Self::not_implemented("elu")
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if self.dtype == DType::BF16 {
            let f32_in = promote_bf16_to_f32(self, layout)?;
            let f32_l = Layout::contiguous(layout.shape());
            let f32_out = super::ops::reduce::reduce_op(&f32_in, op, &f32_l, reduce_dims)?;
            // Reduce keeps `reduce_dims` as size-1 axes. The output shape
            // is inferred from the (input shape, reduce_dims) pair.
            let mut out_shape: Vec<usize> = layout.dims().to_vec();
            for &d in reduce_dims {
                out_shape[d] = 1;
            }
            return demote_f32_to_bf16(&f32_out, &Shape::from(out_shape));
        }
        super::ops::reduce::reduce_op(self, op, layout, reduce_dims)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Self::not_implemented("cmp")
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        super::ops::cast::to_dtype(self, layout, dtype)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if self.dtype == DType::BF16 {
            let f32_in = promote_bf16_to_f32(self, layout)?;
            let f32_l = Layout::contiguous(layout.shape());
            let f32_out = super::ops::unary::unary::<B>(&f32_in, &f32_l)?;
            return demote_f32_to_bf16(&f32_out, layout.shape());
        }
        super::ops::unary::unary::<B>(self, layout)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype == DType::BF16 || rhs.dtype == DType::BF16 {
            // Binary requires matching dtype; if either is bf16 both must be.
            let lhs_f32 = promote_bf16_to_f32(self, lhs_layout)?;
            let rhs_f32 = promote_bf16_to_f32(rhs, rhs_layout)?;
            let lhs_l = Layout::contiguous(lhs_layout.shape());
            let rhs_l = Layout::contiguous(rhs_layout.shape());
            let f32_out =
                super::ops::binary::binary::<B>(&lhs_f32, &rhs_f32, &lhs_l, &rhs_l)?;
            // Output shape mirrors the broadcast shape selected by the
            // existing binary kernel — for matching shapes that's the
            // shared shape, for stride-0 broadcasts it's the larger dim.
            let out_shape = broadcast_shape(lhs_layout.shape(), rhs_layout.shape())?;
            return demote_f32_to_bf16(&f32_out, &out_shape);
        }
        super::ops::binary::binary::<B>(self, rhs, lhs_layout, rhs_layout)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Self::not_implemented("where_cond")
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Self::not_implemented("conv1d")
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Self::not_implemented("conv_transpose1d")
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Self::not_implemented("conv2d")
    }

    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Self::not_implemented("conv_transpose2d")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Self::not_implemented("avg_pool2d")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Self::not_implemented("max_pool2d")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Self::not_implemented("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Self::not_implemented("upsample_nearest2d")
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Self::not_implemented("upsample_bilinear2d")
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Self::not_implemented("gather")
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Self::not_implemented("scatter_set")
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Self::not_implemented("scatter_add_set")
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if self.dtype == DType::BF16 {
            let f32_src = promote_bf16_to_f32(self, src_l)?;
            let f32_src_l = Layout::contiguous(src_l.shape());
            let f32_out =
                super::ops::index_select::index_select(&f32_src, ids, &f32_src_l, ids_l, dim)?;
            // Output shape replaces `dim` with the length of `ids`.
            let mut out_shape: Vec<usize> = src_l.dims().to_vec();
            out_shape[dim] = ids_l.shape().elem_count();
            return demote_f32_to_bf16(&f32_out, &Shape::from(out_shape));
        }
        super::ops::index_select::index_select(self, ids, src_l, ids_l, dim)
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Self::not_implemented("index_add")
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        super::ops::matmul::matmul(self, rhs, bmnk, lhs_layout, rhs_layout)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        super::ops::copy::copy_strided_src(self, dst, dst_offset, src_l)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        super::ops::copy::copy2d(
            self,
            dst,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Self::not_implemented("const_set")
    }
}

// ── BF16 auto-promotion helpers ──────────────────────────────────────
//
// The wgpu compute kernels for unary, binary, reduce, affine,
// index_select, softmax, and rope all run in f32 (the f16 paths are
// gated behind a naga `enable f16;` limitation). For models that load
// weights in bf16 — Gemma4 in the chat-pwa being the motivating case —
// we route bf16 operands through a transient f32 round-trip:
//
//   bf16 input  ──cast──▶  f32 contig  ──f32 op──▶  f32 out  ──cast──▶  bf16 out
//
// `cast::to_dtype` requires a contiguous source, so non-contiguous bf16
// layouts (transposes, broadcast views) are first materialised via
// `copy::copy_strided_src` (which now has a bf16 path) before the cast.
//
// Memory cost: ~2× the operand size in scratch f32 storage during the
// op. For Gemma4 forward pass operands (per-token activations of a few
// MB) that's negligible. For matmul we keep the native bf16 kernel
// because rhs weights would be tens of GB and casting them per-call
// would dominate VRAM.

pub(super) fn promote_bf16_to_f32(s: &WgpuStorage, l: &Layout) -> Result<WgpuStorage> {
    debug_assert_eq!(s.dtype, DType::BF16);
    if l.is_contiguous() {
        // `cast::to_dtype` honors `start_offset` so a contiguous-with-
        // offset layout is fine without a materialise step.
        return super::ops::cast::to_dtype(s, l, DType::F32);
    }
    // Materialise a contiguous bf16 copy at offset 0, then cast.
    let shape = l.shape();
    let mut tmp = WgpuStorage::alloc_zeros(s.device.clone(), DType::BF16, shape)?;
    super::ops::copy::copy_strided_src(s, &mut tmp, 0, l)?;
    super::ops::cast::to_dtype(&tmp, &Layout::contiguous(shape), DType::F32)
}

pub(super) fn demote_f32_to_bf16(s: &WgpuStorage, shape: &Shape) -> Result<WgpuStorage> {
    debug_assert_eq!(s.dtype, DType::F32);
    super::ops::cast::to_dtype(s, &Layout::contiguous(shape), DType::BF16)
}

/// Broadcast two shapes per numpy/candle rules. Returns the broadcast
/// shape or an error if the shapes are incompatible.
fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let l = lhs.dims();
    let r = rhs.dims();
    let n = l.len().max(r.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ld = l.get(l.len().wrapping_sub(1).wrapping_sub(i)).copied().unwrap_or(1);
        let rd = r.get(r.len().wrapping_sub(1).wrapping_sub(i)).copied().unwrap_or(1);
        let d = if ld == rd {
            ld
        } else if ld == 1 {
            rd
        } else if rd == 1 {
            ld
        } else {
            return Err(crate::Error::Msg(format!(
                "wgpu: bf16 binary auto-promote: incompatible shapes {l:?} vs {r:?}"
            )));
        };
        out.push(d);
    }
    out.reverse();
    Ok(Shape::from(out))
}

// Silence unused-import warnings on the rare path where neither variant
// of `cpu_storage_bytes` ends up calling a typed slice helper.
#[allow(dead_code)]
fn _silence_cpu_storage_ref(_: CpuStorageRef<'_>) {}

#[cfg(target_arch = "wasm32")]
unsafe impl Send for WgpuStorage {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WgpuStorage {}
