//! Softmax-last-dim on WebGPU (Phase 3.7).
//!
//! Fused numerically-stable softmax along the last dimension.
//! One workgroup (64 threads) handles one row; shared-memory
//! reductions compute max and sum.
//!
//! This is wired into the `candle_nn::ops::SoftmaxLastDim` CustomOp1
//! via the `wgpu_fwd` trait method so that models running on
//! `Device::Wgpu` get GPU-accelerated softmax.

use crate::{DType, Layout, Result, Shape};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/softmax.wgsl");

#[repr(C)]
#[derive(Clone, Copy)]
struct SoftmaxMeta {
    n_rows: u32,
    row_len: u32,
    src_offset: u32,
    _pad0: u32,
}

fn meta_bytes(m: &SoftmaxMeta) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (m as *const SoftmaxMeta) as *const u8,
            std::mem::size_of::<SoftmaxMeta>(),
        )
    }
}

fn render_wgsl() -> String {
    TEMPLATE.replace("__SCALAR_T__", "f32")
}

pub fn softmax_last_dim(
    src: &WgpuStorage,
    layout: &Layout,
) -> Result<(WgpuStorage, Shape)> {
    let device = &src.device;

    if src.dtype != DType::F32 {
        if src.dtype == DType::F16 {
            return Err(crate::Error::Msg(
                "wgpu: softmax on f16 deferred (naga gate)".to_string(),
            ));
        }
        return Err(crate::Error::Msg(format!(
            "wgpu: softmax not implemented for dtype {:?}",
            src.dtype
        )));
    }

    if !layout.is_contiguous() {
        return Err(crate::Error::Msg(
            "wgpu: softmax requires contiguous input".to_string(),
        ));
    }

    let dims = layout.dims();
    if dims.is_empty() {
        return Err(crate::Error::Msg(
            "wgpu: softmax on scalar not supported".to_string(),
        ));
    }

    let row_len = dims[dims.len() - 1];
    let n_rows: usize = dims[..dims.len() - 1].iter().product();
    let n_elements = n_rows * row_len;

    if n_elements == 0 {
        let out = WgpuStorage::alloc_zeros(device.clone(), DType::F32, &layout.shape().clone())?;
        return Ok((out, layout.shape().clone()));
    }

    let meta = SoftmaxMeta {
        n_rows: n_rows as u32,
        row_len: row_len as u32,
        src_offset: layout.start_offset() as u32,
        _pad0: 0,
    };

    let wgsl = render_wgsl();
    let pipeline = device.get_or_create_pipeline("softmax_last_dim:f32", &wgsl, "main");

    let elem_bytes: u64 = 4;
    let raw_size = (n_elements as u64) * elem_bytes;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-softmax-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-softmax-meta"),
        size: std::mem::size_of::<SoftmaxMeta>() as u64,
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
            label: Some("wgpu-softmax-bg"),
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
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-softmax-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-softmax-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per row.
        pass.dispatch_workgroups(n_rows as u32, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    let out = WgpuStorage::from_raw_buffer(
        Arc::new(out_buffer),
        n_elements,
        DType::F32,
        device.clone(),
    );
    Ok((out, layout.shape().clone()))
}
