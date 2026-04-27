//! Buffer-to-buffer copy paths on WebGPU (Phase 3.6).
//!
//! Drives `WgpuStorage::copy_strided_src` and `WgpuStorage::copy2d`,
//! which back `Tensor::cat` (KV-cache append, mask construction) and
//! various other layout-shuffling paths.
//!
//! # Strategy
//!
//! - **Contiguous source** (`SingleBlock` per `Layout::strided_blocks`)
//!   → single `copy_buffer_to_buffer`. No kernel; just a queue submit.
//!   Accepts the layout's `start_offset()` as the source start.
//! - **Strided source** → small WGSL "ucopy" kernel that reads from a
//!   strided source layout and writes contiguous output at the
//!   destination offset. Same shape/stride uniform shape as the unary
//!   kernels (rank ≤ 4).
//! - **`copy2d`** (used by `cat_contiguous`) → `d1` strided copies,
//!   each of `d2` contiguous elements. We issue `d1` host-side
//!   `copy_buffer_to_buffer` calls in a single command encoder. The
//!   driver coalesces these efficiently for the typical KV-cache
//!   access pattern (small `d1`, large `d2`).
//!
//! # Limitations (deferred)
//!
//! - Strided rank > 4 — same gate as the elementwise kernel; reshape
//!   first or wait for the rank-6 polish in Phase 3.9.
//! - Sub-byte dtypes — not supported (no kernel reads/writes them yet).
//! - f16 — works for `copy2d` (pure byte copy, dtype-agnostic) but the
//!   strided kernel template specializes by scalar type and only the
//!   f32 pipeline is wired today.

use crate::{DType, Layout, Result, StridedBlocks};

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/copy.wgsl");
const MAX_RANK: usize = 4;
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct CopyMeta {
    n_elements: u32,
    rank: u32,
    src_offset: u32,
    dst_offset: u32,
    shape: [u32; 4],
    src_strides: [u32; 4],
}

fn meta_bytes(m: &CopyMeta) -> &[u8] {
    // SAFETY: `repr(C)` of `u32`s only — no uninit padding.
    unsafe {
        std::slice::from_raw_parts(
            (m as *const CopyMeta) as *const u8,
            std::mem::size_of::<CopyMeta>(),
        )
    }
}

fn render_wgsl(scalar: &str) -> String {
    TEMPLATE.replace("__SCALAR_T__", scalar)
}

/// Element size in bytes for a dtype that survives the copy paths.
/// Matches `DType::size_in_bytes()` for full-byte dtypes.
fn elem_bytes(dtype: DType) -> Result<u64> {
    let sz = dtype.size_in_bytes();
    if sz == 0 {
        return Err(crate::Error::Msg(format!(
            "wgpu: copy on sub-byte dtype {dtype:?} not supported"
        )));
    }
    Ok(sz as u64)
}

/// Pipeline cache key for the f32 strided ucopy kernel.
const UCOPY_KEY_F32: &str = "ucopy:f32";

/// Issue an in-place strided→contiguous copy: writes `src` (interpreted
/// via `src_l`) into `dst.buffer` starting at element index `dst_offset`.
pub(crate) fn copy_strided_src(
    src: &WgpuStorage,
    dst: &mut WgpuStorage,
    dst_offset: usize,
    src_l: &Layout,
) -> Result<()> {
    if src.dtype != dst.dtype {
        return Err(crate::Error::Msg(format!(
            "wgpu: copy_strided_src dtype mismatch ({:?} vs {:?})",
            src.dtype, dst.dtype
        )));
    }
    let n_elements = src_l.shape().elem_count();
    if n_elements == 0 {
        return Ok(());
    }
    let bytes_per = elem_bytes(src.dtype)?;

    // Fast path: contiguous block → single copy_buffer_to_buffer.
    if let StridedBlocks::SingleBlock { start_offset, len } = src_l.strided_blocks() {
        debug_assert_eq!(len, n_elements);
        let device = src.device.device();
        let queue = src.device.queue();
        let copy_bytes = (len as u64) * bytes_per;
        let src_byte_off = (start_offset as u64) * bytes_per;
        let dst_byte_off = (dst_offset as u64) * bytes_per;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-copy-strided-src-contig"),
        });
        encoder.copy_buffer_to_buffer(
            &src.buffer,
            src_byte_off,
            &dst.buffer,
            dst_byte_off,
            copy_bytes,
        );
        queue.submit(Some(encoder.finish()));
        return Ok(());
    }

    // Strided fallback: dispatch the ucopy kernel. f32 only for now;
    // f16 is gated behind the same naga limit as the rest of the
    // f-typed kernels.
    if src.dtype != DType::F32 {
        if src.dtype == DType::F16 {
            return Err(crate::Error::Msg(
                "wgpu: copy_strided_src on f16 strided source deferred \
                 (naga `enable f16;` gate)"
                    .to_string(),
            ));
        }
        return Err(crate::Error::Msg(format!(
            "wgpu: copy_strided_src not implemented for strided dtype {:?}",
            src.dtype
        )));
    }

    let dims = src_l.dims();
    if dims.len() > MAX_RANK {
        return Err(crate::Error::Msg(format!(
            "wgpu: copy_strided_src rank {} > {} (Phase 3.6 limit)",
            dims.len(),
            MAX_RANK
        )));
    }
    let mut shape = [1u32; 4];
    let mut src_strides = [0u32; 4];
    let stride = src_l.stride();
    for (i, &d) in dims.iter().enumerate() {
        shape[i] = u32::try_from(d).map_err(|_| {
            crate::Error::Msg(format!("wgpu: copy shape dim {} = {} too big", i, d))
        })?;
        src_strides[i] = u32::try_from(stride[i]).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: copy stride dim {} = {} (negative or huge stride not supported)",
                i, stride[i]
            ))
        })?;
    }
    let src_off = u32::try_from(src_l.start_offset()).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: copy src start offset {} exceeds u32::MAX",
            src_l.start_offset()
        ))
    })?;
    let dst_off = u32::try_from(dst_offset).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: copy dst offset {} exceeds u32::MAX",
            dst_offset
        ))
    })?;
    let n_elements_u32 = u32::try_from(n_elements).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: copy element count {} exceeds u32::MAX",
            n_elements
        ))
    })?;

    let meta = CopyMeta {
        n_elements: n_elements_u32,
        rank: dims.len() as u32,
        src_offset: src_off,
        dst_offset: dst_off,
        shape,
        src_strides,
    };

    let device = &src.device;
    let wgsl = render_wgsl("f32");
    let pipeline = device.get_or_create_pipeline(UCOPY_KEY_F32, &wgsl, "main");

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-ucopy-meta"),
        size: std::mem::size_of::<CopyMeta>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut view = meta_buffer.slice(..).get_mapped_range_mut();
        view.copy_from_slice(meta_bytes(&meta));
    }
    meta_buffer.unmap();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-ucopy-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-ucopy-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-ucopy-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_elements_u32.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));
    Ok(())
}

/// 2D strided copy: `d1` rows of `d2` contiguous elements, with element
/// strides `src_stride1` / `dst_stride1` between rows. `src_offset` /
/// `dst_offset` are element offsets into the buffers.
///
/// Implemented as `d1` `copy_buffer_to_buffer` calls in a single
/// command encoder. Pure byte-level copy — works for any dtype that
/// has a non-zero `size_in_bytes()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy2d(
    src: &WgpuStorage,
    dst: &mut WgpuStorage,
    d1: usize,
    d2: usize,
    src_stride1: usize,
    dst_stride1: usize,
    src_offset: usize,
    dst_offset: usize,
) -> Result<()> {
    if src.dtype != dst.dtype {
        return Err(crate::Error::Msg(format!(
            "wgpu: copy2d dtype mismatch ({:?} vs {:?})",
            src.dtype, dst.dtype
        )));
    }
    if d1 == 0 || d2 == 0 {
        return Ok(());
    }
    let bytes_per = elem_bytes(src.dtype)?;
    let row_bytes = (d2 as u64) * bytes_per;
    if row_bytes == 0 {
        return Ok(());
    }
    // copy_buffer_to_buffer requires offset and size to be 4-byte aligned.
    // Sub-4-byte dtypes (f16, u8, ...) might have rows whose byte size
    // isn't aligned; reject for now.
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    if row_bytes % align != 0 {
        return Err(crate::Error::Msg(format!(
            "wgpu: copy2d row bytes {} not aligned to {} for dtype {:?}; \
             reshape to a 4-byte-aligned row width or wait for f16 path",
            row_bytes, align, src.dtype
        )));
    }
    let src_stride1_bytes = (src_stride1 as u64) * bytes_per;
    let dst_stride1_bytes = (dst_stride1 as u64) * bytes_per;
    if src_stride1_bytes % align != 0 || dst_stride1_bytes % align != 0 {
        return Err(crate::Error::Msg(format!(
            "wgpu: copy2d row stride not 4-byte aligned (src {}, dst {})",
            src_stride1_bytes, dst_stride1_bytes
        )));
    }
    let src_off_bytes = (src_offset as u64) * bytes_per;
    let dst_off_bytes = (dst_offset as u64) * bytes_per;
    if src_off_bytes % align != 0 || dst_off_bytes % align != 0 {
        return Err(crate::Error::Msg(format!(
            "wgpu: copy2d offset not 4-byte aligned (src {}, dst {})",
            src_off_bytes, dst_off_bytes
        )));
    }

    let device = src.device.device();
    let queue = src.device.queue();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu-copy2d"),
    });
    for i1 in 0..d1 as u64 {
        encoder.copy_buffer_to_buffer(
            &src.buffer,
            src_off_bytes + i1 * src_stride1_bytes,
            &dst.buffer,
            dst_off_bytes + i1 * dst_stride1_bytes,
            row_bytes,
        );
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}
