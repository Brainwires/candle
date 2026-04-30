//! Type-cast (to_dtype) on WebGPU (Phase 3.7).
//!
//! Casts elements between scalar types.  Source must be contiguous.
//! Currently supports U32 → F32 (needed for RotaryEmbedding's
//! `Tensor::arange(0u32, ...).to_dtype(f32)`).
//!
//! Additional cast pairs added as needed.

use crate::{DType, Layout, Result};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/cast.wgsl");
const TEMPLATE_TO_BF16: &str = include_str!("../kernels/cast_to_bf16.wgsl");
const TEMPLATE_FROM_BF16: &str = include_str!("../kernels/cast_from_bf16.wgsl");
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct CastMeta {
    n_elements: u32,
    src_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

fn meta_bytes(m: &CastMeta) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            (m as *const CastMeta) as *const u8,
            std::mem::size_of::<CastMeta>(),
        )
    }
}

fn render_wgsl(src_type: &str, dst_type: &str) -> String {
    TEMPLATE
        .replace("__SRC_T__", src_type)
        .replace("__DST_T__", dst_type)
}

fn wgsl_type(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::U32 => Ok("u32"),
        DType::I32 => Ok("i32"),
        _ => Err(crate::Error::Msg(format!(
            "wgpu: cast not implemented for dtype {:?}",
            dtype
        ))),
    }
}

pub(crate) fn to_dtype(
    src: &WgpuStorage,
    layout: &Layout,
    dst_dtype: DType,
) -> Result<WgpuStorage> {
    let device = &src.device;
    let src_dtype = src.dtype;

    if !layout.is_contiguous() {
        return Err(crate::Error::Msg(
            "wgpu: to_dtype requires contiguous source".to_string(),
        ));
    }

    let n_elements = layout.shape().elem_count();
    if n_elements == 0 {
        return WgpuStorage::alloc_zeros(device.clone(), dst_dtype, layout.shape());
    }

    let n_elements_u32 = u32::try_from(n_elements).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: to_dtype element count {} exceeds u32::MAX",
            n_elements
        ))
    })?;

    let meta = CastMeta {
        n_elements: n_elements_u32,
        src_offset: layout.start_offset() as u32,
        _pad0: 0,
        _pad1: 0,
    };

    enum CastKind {
        Direct { wgsl: String, dispatch: u32 },
        ToBf16 { wgsl: String, dispatch: u32 },
        FromBf16 { wgsl: String, dispatch: u32 },
    }

    // BF16 is packed as `array<u32>` (2 BF16s per u32) and uses dedicated
    // kernels — WGSL has no native BF16 type. The other dtype pairs use
    // the simple cast-by-construction template.
    let (pipeline_key, kind) = match (src_dtype, dst_dtype) {
        (DType::U32, DType::F32) => (
            "cast:u32:f32",
            CastKind::Direct {
                wgsl: render_wgsl("u32", "f32"),
                dispatch: n_elements_u32.div_ceil(WORKGROUP_SIZE),
            },
        ),
        (DType::I32, DType::F32) => (
            "cast:i32:f32",
            CastKind::Direct {
                wgsl: render_wgsl("i32", "f32"),
                dispatch: n_elements_u32.div_ceil(WORKGROUP_SIZE),
            },
        ),
        (DType::F32, DType::U32) => (
            "cast:f32:u32",
            CastKind::Direct {
                wgsl: render_wgsl("f32", "u32"),
                dispatch: n_elements_u32.div_ceil(WORKGROUP_SIZE),
            },
        ),
        (DType::F32, DType::I32) => (
            "cast:f32:i32",
            CastKind::Direct {
                wgsl: render_wgsl("f32", "i32"),
                dispatch: n_elements_u32.div_ceil(WORKGROUP_SIZE),
            },
        ),
        // → BF16: one thread per BF16 *pair*, so dispatch = ceil(n / 2 / WG).
        (DType::F32, DType::BF16) => (
            "cast:f32:bf16",
            CastKind::ToBf16 {
                wgsl: TEMPLATE_TO_BF16.replace("__SRC_T__", "f32"),
                dispatch: n_elements_u32.div_ceil(2).div_ceil(WORKGROUP_SIZE),
            },
        ),
        (DType::U32, DType::BF16) => (
            "cast:u32:bf16",
            CastKind::ToBf16 {
                wgsl: TEMPLATE_TO_BF16.replace("__SRC_T__", "u32"),
                dispatch: n_elements_u32.div_ceil(2).div_ceil(WORKGROUP_SIZE),
            },
        ),
        (DType::I32, DType::BF16) => (
            "cast:i32:bf16",
            CastKind::ToBf16 {
                wgsl: TEMPLATE_TO_BF16.replace("__SRC_T__", "i32"),
                dispatch: n_elements_u32.div_ceil(2).div_ceil(WORKGROUP_SIZE),
            },
        ),
        // BF16 →: one thread per BF16 element.
        (DType::BF16, DType::F32) => (
            "cast:bf16:f32",
            CastKind::FromBf16 {
                wgsl: TEMPLATE_FROM_BF16.replace("__DST_T__", "f32"),
                dispatch: n_elements_u32.div_ceil(WORKGROUP_SIZE),
            },
        ),
        (DType::BF16, DType::U32) => (
            "cast:bf16:u32",
            CastKind::FromBf16 {
                wgsl: TEMPLATE_FROM_BF16.replace("__DST_T__", "u32"),
                dispatch: n_elements_u32.div_ceil(WORKGROUP_SIZE),
            },
        ),
        (DType::BF16, DType::I32) => (
            "cast:bf16:i32",
            CastKind::FromBf16 {
                wgsl: TEMPLATE_FROM_BF16.replace("__DST_T__", "i32"),
                dispatch: n_elements_u32.div_ceil(WORKGROUP_SIZE),
            },
        ),
        (a, b) if a == b => {
            // Same-dtype "cast" is a copy; route through the direct template
            // to keep one code path.
            let t = wgsl_type(a)?;
            (
                "cast:identity",
                CastKind::Direct {
                    wgsl: render_wgsl(t, t),
                    dispatch: n_elements_u32.div_ceil(WORKGROUP_SIZE),
                },
            )
        }
        _ => {
            return Err(crate::Error::Msg(format!(
                "wgpu: cast from {:?} to {:?} not implemented",
                src_dtype, dst_dtype
            )))
        }
    };
    let (wgsl, dispatch_groups) = match &kind {
        CastKind::Direct { wgsl, dispatch }
        | CastKind::ToBf16 { wgsl, dispatch }
        | CastKind::FromBf16 { wgsl, dispatch } => (wgsl.as_str(), *dispatch),
    };
    let pipeline = device.get_or_create_pipeline(&pipeline_key, wgsl, "main");

    let dst_elem_bytes = dst_dtype.size_in_bytes() as u64;
    let raw_size = (n_elements as u64) * dst_elem_bytes;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-cast-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-cast-meta"),
        size: std::mem::size_of::<CastMeta>() as u64,
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
            label: Some("wgpu-cast-bg"),
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
            label: Some("wgpu-cast-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-cast-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_groups, 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(WgpuStorage::from_raw_buffer(
        Arc::new(out_buffer),
        n_elements,
        dst_dtype,
        device.clone(),
    ))
}
