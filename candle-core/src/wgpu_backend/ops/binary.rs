//! Element-wise binary ops on WebGPU (Phase 3.4).
//!
//! Drives `WgpuStorage::binary_impl<B: BinaryOpT>`. The kernel template
//! lives in `kernels/elementwise_binary.wgsl`; we substitute the scalar
//! dtype and the op expression at runtime, then memoize the compiled
//! pipeline forever.
//!
//! # Broadcasting
//!
//! Unlike most candle backends — which lower broadcasting via explicit
//! reshape/expand ops on the host before reaching `binary_impl` — we
//! handle it on-device by feeding per-axis input strides relative to the
//! *output* shape. A broadcasted axis on either input carries stride 0,
//! so multiple output indices read the same source element. This makes
//! the kernel correct for any `Layout` candle hands us, including the
//! ones where one operand has a leading 1-axis.
//!
//! # Limitations (deferred)
//!
//! - Rank > 4. Same reason as `unary.rs`.
//! - Negative strides — rejected at host (would need i32 and a sign-aware
//!   index walk).
//! - f16 — same gate as `unary.rs` / `matmul.rs`: WGSL frontend doesn't
//!   parse `enable f16;` in the wgpu version we're pinned to.
//! - Integer dtypes for ops that aren't well-defined on integers (div on
//!   floats vs integer div on ints, NaN on `min`/`max`) are simply not
//!   wired. We ship f32 only; revisit when a Gemma path actually needs
//!   integer arithmetic.

use crate::op::BinaryOpT;
use crate::{DType, Layout, Result};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/elementwise_binary.wgsl");

const MAX_RANK: usize = 4;
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct ElemBinaryMeta {
    n_elements: u32,
    rank: u32,
    a_offset: u32,
    b_offset: u32,
    shape: [u32; 4],
    a_strides: [u32; 4],
    b_strides: [u32; 4],
}

fn meta_bytes(m: &ElemBinaryMeta) -> &[u8] {
    // SAFETY: `ElemBinaryMeta` is `repr(C)` with all-`u32` fields and no
    // padding holes (head fields total 16 bytes, then three `[u32; 4]`s
    // each 16 bytes — naturally aligned).
    unsafe {
        std::slice::from_raw_parts(
            (m as *const ElemBinaryMeta) as *const u8,
            std::mem::size_of::<ElemBinaryMeta>(),
        )
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // F16 variant wired but currently rejected at the dtype gate
enum ScalarDType {
    F32,
    F16,
}

impl ScalarDType {
    fn wgsl_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
    fn elem_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
    fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
        }
    }
}

/// Map a candle `BinaryOpT::NAME` onto a (cache-key, op-body) pair.
fn op_for(name: &str, dtype: ScalarDType) -> Option<(&'static str, &'static str)> {
    let body: &'static str = match name {
        "add" => "return x + y;",
        "sub" => "return x - y;",
        "mul" => "return x * y;",
        "div" => "return x / y;",
        // Note: WGSL `min`/`max` propagate NaN like fmin/fmax-ish, while
        // candle CPU's `Minimum`/`Maximum` use `if v1 > v2` which is
        // false for NaN. Discrepancy on NaN inputs is acceptable for
        // Gemma forward pass; documented for posterity.
        "minimum" => "return min(x, y);",
        "maximum" => "return max(x, y);",
        _ => return None,
    };
    let key: &'static str = match (name, dtype) {
        ("add", ScalarDType::F32) => "binary:add:f32",
        ("add", ScalarDType::F16) => "binary:add:f16",
        ("sub", ScalarDType::F32) => "binary:sub:f32",
        ("sub", ScalarDType::F16) => "binary:sub:f16",
        ("mul", ScalarDType::F32) => "binary:mul:f32",
        ("mul", ScalarDType::F16) => "binary:mul:f16",
        ("div", ScalarDType::F32) => "binary:div:f32",
        ("div", ScalarDType::F16) => "binary:div:f16",
        ("minimum", ScalarDType::F32) => "binary:minimum:f32",
        ("minimum", ScalarDType::F16) => "binary:minimum:f16",
        ("maximum", ScalarDType::F32) => "binary:maximum:f32",
        ("maximum", ScalarDType::F16) => "binary:maximum:f16",
        _ => return None,
    };
    Some((key, body))
}

fn render_wgsl(scalar: ScalarDType, op_body: &str) -> String {
    TEMPLATE
        .replace("__SCALAR_T__", scalar.wgsl_name())
        .replace("__OP_BODY__", op_body)
}

pub(crate) fn binary<B: BinaryOpT>(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<WgpuStorage> {
    if lhs.dtype != rhs.dtype {
        return Err(crate::Error::Msg(format!(
            "wgpu: binary {} dtype mismatch lhs={:?} rhs={:?}",
            B::NAME,
            lhs.dtype,
            rhs.dtype
        )));
    }
    let device = &lhs.device;
    let scalar = match lhs.dtype {
        DType::F32 => ScalarDType::F32,
        DType::F16 => {
            return Err(crate::Error::Msg(format!(
                "wgpu: binary {} on f16 requires a WGSL frontend that \
                 implements `enable f16;` (naga >= 22.0). Tracking: \
                 bump wgpu in Phase 3.9.",
                B::NAME
            )));
        }
        other => {
            return Err(crate::Error::Msg(format!(
                "wgpu: binary {} not implemented for dtype {:?} \
                 (only f32 is wired in Phase 3.4)",
                B::NAME,
                other
            )))
        }
    };

    // Candle hands us layouts with the same broadcast shape on lhs and
    // rhs (callers go through `Tensor::broadcast_*` first). We use lhs's
    // dims as the output shape and feed each input's dims/strides
    // independently — broadcasted axes have stride 0.
    let dims = lhs_layout.dims();
    if dims != rhs_layout.dims() {
        return Err(crate::Error::Msg(format!(
            "wgpu: binary {} requires pre-broadcast layouts to share shape; \
             got lhs={:?} rhs={:?}",
            B::NAME,
            dims,
            rhs_layout.dims()
        )));
    }
    if dims.len() > MAX_RANK {
        return Err(crate::Error::Msg(format!(
            "wgpu: binary {} rank {} > {} (Phase 3.4 limit; \
             reshape/contiguify before dispatch)",
            B::NAME,
            dims.len(),
            MAX_RANK
        )));
    }

    let (key, body) = op_for(B::NAME, scalar).ok_or_else(|| {
        crate::Error::Msg(format!(
            "wgpu: binary op {} not yet wired on Wgpu backend",
            B::NAME
        ))
    })?;

    let wgsl = render_wgsl(scalar, body);
    let pipeline = device.get_or_create_pipeline(key, &wgsl, "main");

    let n_elements = dims.iter().product::<usize>();
    if n_elements == 0 {
        return Err(crate::Error::Msg(format!(
            "wgpu: binary {} on empty tensor (zero-sized dim) not supported",
            B::NAME
        )));
    }
    let n_elements_u32 = u32::try_from(n_elements).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: binary {} element count {} exceeds u32::MAX",
            B::NAME,
            n_elements
        ))
    })?;

    let mut shape = [1u32; 4];
    let mut a_strides = [0u32; 4];
    let mut b_strides = [0u32; 4];
    let lhs_stride = lhs_layout.stride();
    let rhs_stride = rhs_layout.stride();
    for (i, &d) in dims.iter().enumerate() {
        shape[i] = u32::try_from(d).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: binary {} shape dim {} = {} exceeds u32::MAX",
                B::NAME,
                i,
                d
            ))
        })?;
        a_strides[i] = u32::try_from(lhs_stride[i]).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: binary {} lhs stride dim {} = {} exceeds u32::MAX",
                B::NAME,
                i,
                lhs_stride[i]
            ))
        })?;
        b_strides[i] = u32::try_from(rhs_stride[i]).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: binary {} rhs stride dim {} = {} exceeds u32::MAX",
                B::NAME,
                i,
                rhs_stride[i]
            ))
        })?;
    }
    let a_offset = u32::try_from(lhs_layout.start_offset()).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: binary {} lhs start offset exceeds u32::MAX",
            B::NAME
        ))
    })?;
    let b_offset = u32::try_from(rhs_layout.start_offset()).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: binary {} rhs start offset exceeds u32::MAX",
            B::NAME
        ))
    })?;

    let meta = ElemBinaryMeta {
        n_elements: n_elements_u32,
        rank: dims.len() as u32,
        a_offset,
        b_offset,
        shape,
        a_strides,
        b_strides,
    };

    let elem_bytes = scalar.elem_bytes();
    let raw_size = (n_elements as u64) * elem_bytes;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-binary-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-binary-meta"),
        size: std::mem::size_of::<ElemBinaryMeta>() as u64,
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
            label: Some("wgpu-binary-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lhs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rhs.buffer.as_entire_binding(),
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
            label: Some("wgpu-binary-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-binary-pass"),
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
        scalar.dtype(),
        device.clone(),
    ))
}
