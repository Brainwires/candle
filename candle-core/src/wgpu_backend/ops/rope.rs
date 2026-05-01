//! Rotary position embeddings on WebGPU (Phase 3.6).
//!
//! Drives the three RoPE flavors used by candle-nn:
//!   - `rope_i` — interleaved pairs along the head_dim axis.
//!   - `rope`   — split-halves along the head_dim axis.
//!   - `rope_thd` — same as `rope` but the source layout is
//!                  (B, T, H, D) instead of (B, H, T, D).
//!
//! All variants share one WGSL template (`kernels/rope.wgsl`); we
//! distinguish them via a `variant` discriminator in the uniform plus
//! a stable pipeline-cache key. The kernel handles batched/unbatched
//! cos/sin via a flag.
//!
//! # Constraints (host-enforced)
//!
//! - `src` rank 4, `head_dim` even.
//! - `src`, `cos`, `sin` all contiguous (matches CPU/CUDA paths).
//! - `cos`/`sin` shape is either (T, D/2) or (B, T, D/2).
//! - dtype f32 only for now (f16 gated, see other ops).

#![allow(dead_code)]

use crate::backend::BackendStorage;
use crate::{DType, Layout, Result, Shape};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/rope.wgsl");
const WORKGROUP_SIZE: u32 = 64;

/// Variant selector that lines up with the kernel's `variant` field.
#[derive(Clone, Copy, Debug)]
pub enum RopeVariant {
    /// Interleaved pairs (`rope_i`).
    Interleaved = 0,
    /// Split halves on (B, H, T, D) (`rope`).
    SplitHalves = 1,
    /// Split halves on (B, T, H, D) (`rope_thd`).
    SplitHalvesThd = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeMeta {
    n_pairs: u32,
    b: u32,
    h: u32,
    t: u32,
    d: u32,
    unbatched: u32,
    variant: u32,
    _pad0: u32,
}

fn meta_bytes(m: &RopeMeta) -> &[u8] {
    // SAFETY: `repr(C)`, all `u32`, fixed size, no uninit padding.
    unsafe {
        std::slice::from_raw_parts(
            (m as *const RopeMeta) as *const u8,
            std::mem::size_of::<RopeMeta>(),
        )
    }
}

fn render_wgsl(scalar: &str) -> String {
    TEMPLATE.replace("__SCALAR_T__", scalar)
}

fn pipeline_key(variant: RopeVariant) -> &'static str {
    // Same shader code in each cache slot — we keep distinct keys per
    // variant only so `wgpu` reports the right label in error messages.
    // (Variant is selected at runtime via the uniform.)
    match variant {
        RopeVariant::Interleaved => "rope:f32:int",
        RopeVariant::SplitHalves => "rope:f32:split",
        RopeVariant::SplitHalvesThd => "rope:f32:thd",
    }
}

/// Run a RoPE forward pass on the Wgpu backend.
///
/// `src_l` describes a contiguous rank-4 tensor; `cos_l` / `sin_l`
/// describe contiguous rank-2 (T, D/2) or rank-3 (B, T, D/2) tensors.
/// Returns a fresh contiguous output of the same shape as `src`.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    variant: RopeVariant,
    src: &WgpuStorage,
    src_l: &Layout,
    cos: &WgpuStorage,
    cos_l: &Layout,
    sin: &WgpuStorage,
    sin_l: &Layout,
) -> Result<(WgpuStorage, Shape)> {
    // bf16 round-trip: promote all three inputs, run f32 rope, demote
    // the output. cos/sin are typically already f32 (they come from
    // `RotaryEmbedding::new`'s f32 sin/cos table — see the gemma4
    // ProportionalRotaryEmbedding pattern); this branch exists so a
    // bf16 src + bf16 cos/sin combination doesn't fall off the bf16 cliff.
    if src.dtype == DType::BF16 || cos.dtype == DType::BF16 || sin.dtype == DType::BF16 {
        // Lift each operand to f32 (idempotent if already f32).
        let lift =
            |s: &WgpuStorage, l: &Layout| -> Result<WgpuStorage> {
                if s.dtype == DType::F32 {
                    if l.is_contiguous() {
                        // f32 contiguous already — clone the buffer view as-is.
                        // (try_clone copies bytes; cheap for the small cos/sin tables.)
                        s.try_clone(l)
                    } else {
                        // Materialise f32 contiguous via the cast identity path
                        // (which routes through copy_strided for non-contiguous src).
                        let mut tmp = WgpuStorage::alloc_zeros(
                            s.device.clone(),
                            DType::F32,
                            l.shape(),
                        )?;
                        super::copy::copy_strided_src(s, &mut tmp, 0, l)?;
                        Ok(tmp)
                    }
                } else if s.dtype == DType::BF16 {
                    super::super::storage::promote_bf16_to_f32(s, l)
                } else {
                    Err(crate::Error::Msg(format!(
                        "wgpu: rope expected f32 or bf16, got {:?}",
                        s.dtype
                    )))
                }
            };
        let src_f32 = lift(src, src_l)?;
        let cos_f32 = lift(cos, cos_l)?;
        let sin_f32 = lift(sin, sin_l)?;
        let src_l_f32 = Layout::contiguous(src_l.shape());
        let cos_l_f32 = Layout::contiguous(cos_l.shape());
        let sin_l_f32 = Layout::contiguous(sin_l.shape());
        let (f32_out, out_shape) = rope(
            variant,
            &src_f32,
            &src_l_f32,
            &cos_f32,
            &cos_l_f32,
            &sin_f32,
            &sin_l_f32,
        )?;
        // Demote back to whatever dtype `src` was.
        if src.dtype == DType::BF16 {
            let bf16_out = super::super::storage::demote_f32_to_bf16(&f32_out, &out_shape)?;
            return Ok((bf16_out, out_shape));
        }
        return Ok((f32_out, out_shape));
    }
    if src.dtype != DType::F32 || cos.dtype != DType::F32 || sin.dtype != DType::F32 {
        if src.dtype == DType::F16 || cos.dtype == DType::F16 || sin.dtype == DType::F16 {
            return Err(crate::Error::Msg(
                "wgpu: rope on f16 deferred (naga `enable f16;` gate)".to_string(),
            ));
        }
        return Err(crate::Error::Msg(format!(
            "wgpu: rope only implemented for f32 (got src={:?}, cos={:?}, sin={:?})",
            src.dtype, cos.dtype, sin.dtype
        )));
    }
    if !src_l.is_contiguous() || !cos_l.is_contiguous() || !sin_l.is_contiguous() {
        return Err(crate::Error::Msg(
            "wgpu: rope requires contiguous src/cos/sin layouts".to_string(),
        ));
    }
    let dims = src_l.dims();
    if dims.len() != 4 {
        return Err(crate::Error::Msg(format!(
            "wgpu: rope expects rank-4 src (got rank {})",
            dims.len()
        )));
    }
    let (a0, a1, a2, a3) = (dims[0], dims[1], dims[2], dims[3]);
    // For variants 0/1: src is (B, H, T, D). For variant 2: (B, T, H, D).
    let (b, h, t, d) = match variant {
        RopeVariant::Interleaved | RopeVariant::SplitHalves => (a0, a1, a2, a3),
        RopeVariant::SplitHalvesThd => (a0, a2, a1, a3),
    };
    if d % 2 != 0 {
        return Err(crate::Error::Msg(format!(
            "wgpu: rope head_dim {d} must be even"
        )));
    }
    let dh = d / 2;
    let cs_dims = cos_l.dims();
    let unbatched = match cs_dims.len() {
        2 => 0u32,
        3 => 1u32,
        n => {
            return Err(crate::Error::Msg(format!(
                "wgpu: rope expected cos/sin rank 2 or 3 (got {n})"
            )));
        }
    };
    if sin_l.dims() != cs_dims {
        return Err(crate::Error::Msg(
            "wgpu: rope cos and sin must share the same shape".to_string(),
        ));
    }
    let expected_cs_last = dh;
    if *cs_dims.last().unwrap() != expected_cs_last {
        return Err(crate::Error::Msg(format!(
            "wgpu: rope cos/sin last dim {} != head_dim/2 = {}",
            cs_dims.last().unwrap(),
            expected_cs_last
        )));
    }
    if cs_dims.len() == 3 && cs_dims[0] != b {
        return Err(crate::Error::Msg(format!(
            "wgpu: rope unbatched cos/sin batch dim {} != src batch {}",
            cs_dims[0], b
        )));
    }
    let cs_seq = if cs_dims.len() == 3 {
        cs_dims[1]
    } else {
        cs_dims[0]
    };
    if cs_seq != t {
        return Err(crate::Error::Msg(format!(
            "wgpu: rope cos/sin seq dim {cs_seq} != src seq {t}",
        )));
    }

    let n_pairs: usize = b * h * t * dh;
    if n_pairs == 0 {
        return Err(crate::Error::Msg(
            "wgpu: rope on empty tensor not supported".to_string(),
        ));
    }
    let n_pairs_u32 = u32::try_from(n_pairs).map_err(|_| {
        crate::Error::Msg(format!("wgpu: rope pair count {n_pairs} exceeds u32::MAX"))
    })?;
    let to_u32 = |x: usize, name: &str| {
        u32::try_from(x).map_err(|_| crate::Error::Msg(format!("wgpu: rope {name} {x} > u32::MAX")))
    };
    let meta = RopeMeta {
        n_pairs: n_pairs_u32,
        b: to_u32(b, "b")?,
        h: to_u32(h, "h")?,
        t: to_u32(t, "t")?,
        d: to_u32(d, "d")?,
        unbatched,
        variant: variant as u32,
        _pad0: 0,
    };

    let device = &src.device;
    let key = pipeline_key(variant);
    let wgsl = render_wgsl("f32");
    let pipeline = device.get_or_create_pipeline(key, &wgsl, "main");

    let n_elements = b * h * t * d;
    let elem_bytes: u64 = 4;
    let raw_size = (n_elements as u64) * elem_bytes;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-rope-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-rope-meta"),
        size: std::mem::size_of::<RopeMeta>() as u64,
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
            label: Some("wgpu-rope-bg"),
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
                    resource: cos.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sin.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = device
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-rope-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-rope-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n_pairs_u32.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    let shape = Shape::from(dims.to_vec());
    Ok((
        WgpuStorage::from_raw_buffer(Arc::new(out_buffer), n_elements, DType::F32, device.clone()),
        shape,
    ))
}
