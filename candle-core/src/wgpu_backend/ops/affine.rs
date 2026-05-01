//! Affine `out = mul * x + add` on WebGPU (Phase 3.5).
//!
//! Drives `WgpuStorage::affine`. Candle lowers all scalar arithmetic
//! (`tensor + 1.0`, `tensor / N`, `&t * 0.5`, ...) into a single
//! `affine(mul, add)` call, so wiring this unblocks the layer/rms_norm
//! compositions that pull in `(norm_x + eps as f64)?.sqrt()?` and
//! `(... / hidden_size as f64)?` patterns.
//!
//! Limitations match the unary kernel: rank ≤ 4, no negative strides,
//! f16 deferred until naga can parse `enable f16;`.

use crate::{DType, Layout, Result};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/affine.wgsl");
const MAX_RANK: usize = 4;
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct AffineMeta {
    n_elements: u32,
    rank: u32,
    a_offset: u32,
    _pad0: u32,
    shape: [u32; 4],
    a_strides: [u32; 4],
    mul: f32,
    add: f32,
    _pad1: u32,
    _pad2: u32,
}

fn meta_bytes(m: &AffineMeta) -> &[u8] {
    // SAFETY: `repr(C)` of `u32`/`f32`s with explicit pads — no uninit holes.
    unsafe {
        std::slice::from_raw_parts(
            (m as *const AffineMeta) as *const u8,
            std::mem::size_of::<AffineMeta>(),
        )
    }
}

fn render_wgsl() -> String {
    TEMPLATE.replace("__SCALAR_T__", "f32")
}

pub(crate) fn affine(
    src: &WgpuStorage,
    layout: &Layout,
    mul: f64,
    add: f64,
) -> Result<WgpuStorage> {
    let device = &src.device;
    if src.dtype != DType::F32 {
        if src.dtype == DType::F16 {
            return Err(crate::Error::Msg(
                "wgpu: affine on f16 deferred (naga `enable f16;` gate)".to_string(),
            ));
        }
        return Err(crate::Error::Msg(format!(
            "wgpu: affine not implemented for dtype {:?} (only f32 in Phase 3.5)",
            src.dtype
        )));
    }
    let dims = layout.dims();
    if dims.len() > MAX_RANK {
        return Err(crate::Error::Msg(format!(
            "wgpu: affine rank {} > {} (Phase 3.5 limit)",
            dims.len(),
            MAX_RANK
        )));
    }
    let n_elements: usize = dims.iter().product();
    if n_elements == 0 {
        return Err(crate::Error::Msg(
            "wgpu: affine on empty tensor not supported".to_string(),
        ));
    }
    let n_elements_u32 = u32::try_from(n_elements).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: affine element count {} exceeds u32::MAX",
            n_elements
        ))
    })?;

    let stride = layout.stride();
    let mut shape = [1u32; 4];
    let mut a_strides = [0u32; 4];
    for (i, &d) in dims.iter().enumerate() {
        shape[i] = u32::try_from(d).map_err(|_| {
            crate::Error::Msg(format!("wgpu: affine shape dim {} = {} too big", i, d))
        })?;
        a_strides[i] = u32::try_from(stride[i]).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: affine stride dim {} = {} too big",
                i, stride[i]
            ))
        })?;
    }
    let a_offset = u32::try_from(layout.start_offset()).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: affine start offset {} exceeds u32::MAX",
            layout.start_offset()
        ))
    })?;

    let meta = AffineMeta {
        n_elements: n_elements_u32,
        rank: dims.len() as u32,
        a_offset,
        _pad0: 0,
        shape,
        a_strides,
        mul: mul as f32,
        add: add as f32,
        _pad1: 0,
        _pad2: 0,
    };

    let wgsl = render_wgsl();
    // Single pipeline for f32 — no per-(mul,add) specialization (those
    // are uniform inputs, not compile-time constants).
    let pipeline = device.get_or_create_pipeline("affine:f32", &wgsl, "main");

    let elem_bytes: u64 = 4;
    let raw_size = (n_elements as u64) * elem_bytes;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-affine-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-affine-meta"),
        size: std::mem::size_of::<AffineMeta>() as u64,
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
            label: Some("wgpu-affine-bg"),
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
            label: Some("wgpu-affine-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-affine-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let total_groups = n_elements_u32.div_ceil(WORKGROUP_SIZE);
        let (gx, gy) = super::split_1d_dispatch(total_groups);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(WgpuStorage::from_raw_buffer(
        Arc::new(out_buffer),
        n_elements,
        DType::F32,
        device.clone(),
    ))
}
