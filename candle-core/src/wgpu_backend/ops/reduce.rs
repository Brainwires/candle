//! Reduction ops on WebGPU (Phase 3.5).
//!
//! Implements `WgpuStorage::reduce_op` for `ReduceOp::{Sum, Min, Max}`
//! over a set of reduce axes. The kernel template lives in
//! `kernels/reduce.wgsl`; we substitute the scalar dtype and the
//! accumulator init / combine / finalize expressions at runtime, then
//! memoize the compiled pipeline forever.
//!
//! `mean` is exposed as a separate entry point that drives the same
//! kernel with an extra `divisor` baked into the finalize step. Candle's
//! own `reduce_op` API doesn't include `Mean` (it composes mean as
//! `sum(x) / N` higher up), so this is just an internal helper.
//!
//! # Limitations (deferred)
//!
//! - Rank > 4. Same reason as elementwise; the WGSL uniform packs four
//!   `vec4<u32>`s for shape/stride. Bumping to rank-6 (vision models)
//!   means a longer uniform — Phase 3.9 polish.
//! - `ArgMin` / `ArgMax` aren't supported. They return integer indices,
//!   which would change the output dtype; the reduce kernel currently
//!   produces same-dtype output. Wiring those needs a parallel kernel
//!   variant that writes `u32` indices. Not on the Gemma critical path.
//! - f16 — same naga gate as the elementwise paths. Returns
//!   `not_implemented` until the WGSL frontend supports `enable f16;`.
//! - Negative strides — rejected at host, since we pack into `u32`.

use crate::op::ReduceOp;
use crate::{DType, Layout, Result};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/reduce.wgsl");

const MAX_RANK: usize = 4;
const WORKGROUP_SIZE: u32 = 64;

/// Uniform layout matching `ReduceMeta` in the WGSL template.
///
/// `repr(C)` with explicit padding to satisfy std140-ish alignment:
/// each `vec4<u32>` lands on a 16-byte boundary, and the trailing
/// scalar block (`divisor`, `reduce_total`, two pads) is aligned to 16
/// as well so the next slot would start aligned if we added one.
#[repr(C)]
#[derive(Clone, Copy)]
struct ReduceMeta {
    n_out: u32,
    rank: u32,
    src_offset: u32,
    op_id: u32,
    shape: [u32; 4],
    in_strides: [u32; 4],
    out_shape: [u32; 4],
    reduce_mask: [u32; 4],
    reduce_shape: [u32; 4],
    divisor: f32,
    reduce_total: u32,
    _pad0: u32,
    _pad1: u32,
}

fn meta_bytes(m: &ReduceMeta) -> &[u8] {
    // SAFETY: all `u32`/`f32` fields, fully `repr(C)`, no uninit padding.
    unsafe {
        std::slice::from_raw_parts(
            (m as *const ReduceMeta) as *const u8,
            std::mem::size_of::<ReduceMeta>(),
        )
    }
}

/// Which scalar combine + finalize the kernel should compile in.
///
/// We specialize per (op, scalar) so the WGSL is straight-line for the
/// hot loop rather than carrying a runtime `op_id` switch. The
/// pipeline cache keys reflect the same.
///
/// `Mean` is intentionally absent: candle's `Tensor::mean_keepdim`
/// already composes mean as `sum * (1 / reduce_size)` via the `affine`
/// path (see `tensor.rs::mean_keepdim`), so wiring `Sum` + `affine` is
/// enough. The kernel template still carries a `divisor` field and a
/// `__FINALIZE__` marker for future use (e.g. fused mean) — for now
/// every kind sets divisor = 1.0 and finalize = identity.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ReduceKind {
    Sum,
    Min,
    Max,
}

impl ReduceKind {
    fn cache_key(self) -> &'static str {
        match self {
            Self::Sum => "reduce:sum:f32",
            Self::Min => "reduce:min:f32",
            Self::Max => "reduce:max:f32",
        }
    }
    fn init_expr(self) -> &'static str {
        match self {
            Self::Sum => "0.0",
            // f32::MIN is ~-3.4e38; use -1e38 to avoid edge cases with
            // -inf canceling / propagating. Any input strictly greater
            // than -1e38 wins, which covers every realistic Gemma value.
            Self::Max => "-1.0e38",
            Self::Min => "1.0e38",
        }
    }
    fn combine_body(self) -> &'static str {
        match self {
            Self::Sum => "return acc + v;",
            Self::Max => "return max(acc, v);",
            Self::Min => "return min(acc, v);",
        }
    }
    fn finalize_body(self) -> &'static str {
        // No-op finalize for now; the kernel template carries `divisor`
        // for forward compat with a fused-mean kernel if perf needs it.
        "return acc;"
    }
}

fn render_wgsl(kind: ReduceKind) -> String {
    TEMPLATE
        .replace("__SCALAR_T__", "f32")
        .replace("__INIT__", kind.init_expr())
        .replace("__COMBINE__", kind.combine_body())
        .replace("__FINALIZE__", kind.finalize_body())
}

/// Drive the reduce kernel for `kind` over `reduce_dims` (which must be
/// a subset of `0..layout.rank()`). Output shape matches the input with
/// the reduced axes set to 1 (keepdim semantics — the `Tensor` wrapper
/// does the squeeze separately).
///
/// `kind == Mean` divides the sum by `reduce_size` in the kernel's
/// finalize step; for the other kinds, divisor is 1.0.
pub(crate) fn reduce(
    src: &WgpuStorage,
    layout: &Layout,
    reduce_dims: &[usize],
    kind: ReduceKind,
) -> Result<WgpuStorage> {
    let device = &src.device;
    if src.dtype != DType::F32 {
        if src.dtype == DType::F16 {
            return Err(crate::Error::Msg(format!(
                "wgpu: reduce_op {:?} on f16 requires a WGSL frontend that \
                 implements `enable f16;` (naga >= 22.0). Tracking: \
                 bump wgpu in Phase 3.9.",
                kind
            )));
        }
        return Err(crate::Error::Msg(format!(
            "wgpu: reduce_op {:?} not implemented for dtype {:?} \
             (only f32 is wired in Phase 3.5)",
            kind, src.dtype
        )));
    }

    let dims = layout.dims();
    let rank = dims.len();
    if rank > MAX_RANK {
        return Err(crate::Error::Msg(format!(
            "wgpu: reduce_op rank {} > {} (Phase 3.5 limit)",
            rank, MAX_RANK
        )));
    }
    if rank == 0 {
        // Reducing a scalar: kernel logic still works (n_out=1,
        // reduce_total=1) but n_elements computations are simpler if
        // we just bail. Candle never asks for this in practice.
        return Err(crate::Error::Msg(
            "wgpu: reduce_op on rank-0 tensor not supported".to_string(),
        ));
    }
    for &d in reduce_dims {
        if d >= rank {
            return Err(crate::Error::Msg(format!(
                "wgpu: reduce dim {} out of range for rank {}",
                d, rank
            )));
        }
    }

    // Build out_shape (= dims with reduced axes set to 1) and reduce_shape
    // (= 1 on non-reduced axes, dims[d] on reduced axes). reduce_total
    // is the product of the reduced extents.
    let mut shape = [1u32; 4];
    let mut in_strides = [0u32; 4];
    let mut out_shape = [1u32; 4];
    let mut reduce_mask = [0u32; 4];
    let mut reduce_shape = [1u32; 4];
    let stride = layout.stride();

    let is_reduced = |d: usize| reduce_dims.iter().any(|&rd| rd == d);

    let mut reduce_total: u64 = 1;
    let mut n_out: u64 = 1;
    for (i, &d) in dims.iter().enumerate() {
        let d_u32 = u32::try_from(d).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: reduce shape dim {} = {} exceeds u32::MAX",
                i, d
            ))
        })?;
        let st_u32 = u32::try_from(stride[i]).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: reduce stride dim {} = {} exceeds u32::MAX",
                i, stride[i]
            ))
        })?;
        shape[i] = d_u32;
        in_strides[i] = st_u32;
        if is_reduced(i) {
            reduce_mask[i] = 1;
            reduce_shape[i] = d_u32;
            out_shape[i] = 1;
            reduce_total *= d as u64;
        } else {
            out_shape[i] = d_u32;
            n_out *= d as u64;
        }
    }
    let reduce_total_u32 = u32::try_from(reduce_total).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: reduce total product {} exceeds u32::MAX",
            reduce_total
        ))
    })?;
    let n_out_u32 = u32::try_from(n_out).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: reduce output count {} exceeds u32::MAX",
            n_out
        ))
    })?;

    // Reducing zero elements would divide-by-zero on mean and produce a
    // garbage init value otherwise. Reject explicitly.
    if reduce_total == 0 {
        return Err(crate::Error::Msg(
            "wgpu: reduce over a zero-sized axis is not supported".to_string(),
        ));
    }
    if n_out == 0 {
        return Err(crate::Error::Msg(
            "wgpu: reduce produced zero output elements (zero-sized non-reduced axis)".to_string(),
        ));
    }

    let src_offset = u32::try_from(layout.start_offset()).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: reduce start offset {} exceeds u32::MAX",
            layout.start_offset()
        ))
    })?;

    // Divisor is unused (no fused-mean kernel yet); the kernel multiplies
    // by 1.0 in `__FINALIZE__`. Kept in the meta for forward compat.
    let divisor = 1.0f32;

    let op_id = match kind {
        ReduceKind::Sum => 0,
        ReduceKind::Min => 1,
        ReduceKind::Max => 2,
    };

    let meta = ReduceMeta {
        n_out: n_out_u32,
        rank: rank as u32,
        src_offset,
        op_id,
        shape,
        in_strides,
        out_shape,
        reduce_mask,
        reduce_shape,
        divisor,
        reduce_total: reduce_total_u32,
        _pad0: 0,
        _pad1: 0,
    };

    let wgsl = render_wgsl(kind);
    let pipeline = device.get_or_create_pipeline(kind.cache_key(), &wgsl, "main");

    // Output buffer: contiguous f32 of `n_out` elements.
    let elem_bytes: u64 = 4;
    let raw_size = n_out * elem_bytes;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-reduce-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-reduce-meta"),
        size: std::mem::size_of::<ReduceMeta>() as u64,
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
            label: Some("wgpu-reduce-bg"),
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
            label: Some("wgpu-reduce-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-reduce-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let total_groups = n_out_u32.div_ceil(WORKGROUP_SIZE);
        let (gx, gy) = super::split_1d_dispatch(total_groups);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(WgpuStorage::from_raw_buffer(
        Arc::new(out_buffer),
        n_out as usize,
        DType::F32,
        device.clone(),
    ))
}

/// Entry point for `BackendStorage::reduce_op`. Maps candle's `ReduceOp`
/// onto the local `ReduceKind` (rejecting Arg{Min,Max} which need an
/// integer-output kernel we haven't written).
pub(crate) fn reduce_op(
    src: &WgpuStorage,
    op: ReduceOp,
    layout: &Layout,
    reduce_dims: &[usize],
) -> Result<WgpuStorage> {
    let kind = match op {
        ReduceOp::Sum => ReduceKind::Sum,
        ReduceOp::Min => ReduceKind::Min,
        ReduceOp::Max => ReduceKind::Max,
        ReduceOp::ArgMin | ReduceOp::ArgMax => {
            return Err(crate::Error::Msg(format!(
                "wgpu: reduce_op {:?} not implemented (needs integer-output kernel; \
                 deferred — not on the Gemma critical path)",
                op
            )));
        }
    };
    reduce(src, layout, reduce_dims, kind)
}
