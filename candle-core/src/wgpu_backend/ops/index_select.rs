//! Index-select on WebGPU (Phase 3.7).
//!
//! Gathers rows from a source tensor along a given dimension using a
//! 1-D index tensor.  Drives `WgpuStorage::index_select` which is the
//! op behind `candle_nn::Embedding::forward`.
//!
//! Limitations:
//!   - Source must be contiguous.
//!   - Indices must be U32.
//!   - f16 deferred (naga `enable f16;` gate).
//!   - Rank ≤ 8 (practical; kernel itself is rank-agnostic since it
//!     works in left/right factorised coordinates).

use crate::{DType, Layout, Result};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/index_select.wgsl");
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct IndexSelectMeta {
    n_elements: u32,
    left_size: u32,
    src_dim: u32,
    n_ids: u32,
    right_size: u32,
    ids_offset: u32,
    ids_stride: u32,
    src_offset: u32,
}

fn meta_bytes(m: &IndexSelectMeta) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (m as *const IndexSelectMeta) as *const u8,
            std::mem::size_of::<IndexSelectMeta>(),
        )
    }
}

fn render_wgsl() -> String {
    TEMPLATE.replace("__SCALAR_T__", "f32")
}

pub(crate) fn index_select(
    src: &WgpuStorage,
    ids: &WgpuStorage,
    src_layout: &Layout,
    ids_layout: &Layout,
    dim: usize,
) -> Result<WgpuStorage> {
    let device = &src.device;

    // ---- dtype checks ----
    if src.dtype != DType::F32 {
        if src.dtype == DType::F16 {
            return Err(crate::Error::Msg(
                "wgpu: index_select on f16 deferred (naga gate)".to_string(),
            ));
        }
        return Err(crate::Error::Msg(format!(
            "wgpu: index_select not implemented for src dtype {:?}",
            src.dtype
        )));
    }
    if ids.dtype != DType::U32 {
        return Err(crate::Error::Msg(format!(
            "wgpu: index_select requires U32 indices, got {:?}",
            ids.dtype
        )));
    }

    // ---- source must be contiguous ----
    if !src_layout.is_contiguous() {
        return Err(crate::Error::Msg(
            "wgpu: index_select requires contiguous source".to_string(),
        ));
    }

    // ---- ids must be 1-D ----
    let ids_dims = ids_layout.dims();
    let n_ids = match ids_dims {
        [n] => *n,
        _ => {
            return Err(crate::Error::Msg(format!(
                "wgpu: index_select requires 1-D ids, got {:?}",
                ids_dims
            )))
        }
    };

    let src_dims = src_layout.dims();
    if dim >= src_dims.len() {
        return Err(crate::Error::Msg(format!(
            "wgpu: index_select dim {} >= rank {}",
            dim,
            src_dims.len()
        )));
    }

    let src_dim = src_dims[dim];
    let left_size: usize = src_dims[..dim].iter().product();
    let right_size: usize = src_dims[dim + 1..].iter().product();

    // Output shape: src_dims with src_dims[dim] replaced by n_ids.
    let mut out_dims = src_dims.to_vec();
    out_dims[dim] = n_ids;
    let n_elements: usize = out_dims.iter().product();

    if n_elements == 0 {
        return WgpuStorage::alloc_zeros(device.clone(), DType::F32, &out_dims.into());
    }

    let n_elements_u32 = u32::try_from(n_elements).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: index_select output size {} exceeds u32::MAX",
            n_elements
        ))
    })?;

    let ids_stride = ids_layout.stride()[0];
    let ids_offset = ids_layout.start_offset();
    let src_offset = src_layout.start_offset();

    let meta = IndexSelectMeta {
        n_elements: n_elements_u32,
        left_size: left_size as u32,
        src_dim: src_dim as u32,
        n_ids: n_ids as u32,
        right_size: right_size as u32,
        ids_offset: ids_offset as u32,
        ids_stride: ids_stride as u32,
        src_offset: src_offset as u32,
    };

    let wgsl = render_wgsl();
    let pipeline = device.get_or_create_pipeline("index_select:f32", &wgsl, "main");

    let elem_bytes: u64 = 4;
    let raw_size = (n_elements as u64) * elem_bytes;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-index-select-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-index-select-meta"),
        size: std::mem::size_of::<IndexSelectMeta>() as u64,
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
            label: Some("wgpu-index-select-bg"),
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
                    resource: ids.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-index-select-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-index-select-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = n_elements_u32.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups_x, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(WgpuStorage::from_raw_buffer(
        Arc::new(out_buffer),
        n_elements,
        DType::F32,
        device.clone(),
    ))
}
