//! Tiled batched matmul on WebGPU.
//!
//! Phase 3.3 — this is the first real compute kernel in the wgpu
//! backend. The shader (see `kernels/matmul.wgsl`) computes a single
//! batched matmul `C = A @ B` where each operand may carry arbitrary
//! row/column/batch strides. Strides come straight from
//! `Layout::stride()`, which means contiguous and many common
//! permuted/transposed inputs go through this path without a CPU-side
//! repack.
//!
//! # Why a single kernel for both contiguous and strided?
//!
//! Strides cost an extra multiply per index but no extra dispatches and
//! no extra memory. For Gemma's matmul shapes (Q · Kᵀ, attention
//! output projections, FFN gates) at least one of the two operands is
//! transposed, so a strided-aware kernel is the right default. Phase 3.9
//! may specialize the contiguous case for perf tuning.
//!
//! # Limitations (deferred)
//!
//! - Negative strides are rejected. They come from reverse views which
//!   no Gemma path uses; we'll add support if a model needs it.
//! - Zero strides on the inner reduce axis (`k`) would imply broadcasting
//!   across the reduction — that's not a meaningful matmul, rejected.
//! - Broadcasting batch dims (e.g. `(1, M, K) @ (B, K, N)`) is supported
//!   transparently because we honor each operand's batch stride
//!   independently — when `lhs` has only one batch matrix its batch
//!   stride is 0 and every workgroup re-reads the same matrix.

use crate::{DType, Layout, Result};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const MATMUL_F32_WGSL: &str = include_str!("../kernels/matmul.wgsl");
const MATMUL_F16_WGSL: &str = include_str!("../kernels/matmul_f16.wgsl");

const TILE: u32 = 16;

/// Uniform layout matching `MatmulMeta` in the WGSL shaders. Field
/// order and types must stay in lockstep with the kernel definitions.
#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulMeta {
    m: u32,
    n: u32,
    k: u32,
    a_stride_row: u32,
    a_stride_col: u32,
    b_stride_row: u32,
    b_stride_col: u32,
    c_stride_row: u32,
    c_stride_col: u32,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    a_offset: u32,
    b_offset: u32,
    _pad: u32,
}

// SAFETY: plain-old-data for a uniform buffer.
fn meta_bytes(meta: &MatmulMeta) -> &[u8] {
    // SAFETY: `MatmulMeta` is `repr(C)` with all-`u32` fields, so its
    // representation is a fixed-size byte buffer with no padding holes.
    unsafe {
        std::slice::from_raw_parts(
            (meta as *const MatmulMeta) as *const u8,
            std::mem::size_of::<MatmulMeta>(),
        )
    }
}

/// Validate that the `lhs_layout` and `rhs_layout` describe a runnable
/// batched matmul matching `bmnk`, and pull out the row/col/batch
/// strides we need to feed to the shader.
struct MatmulPlan {
    b: u32,
    m: u32,
    n: u32,
    k: u32,
    a_stride_row: u32,
    a_stride_col: u32,
    b_stride_row: u32,
    b_stride_col: u32,
    c_stride_row: u32,
    c_stride_col: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    a_offset: u32,
    b_offset: u32,
}

fn plan(
    bmnk: (usize, usize, usize, usize),
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<MatmulPlan> {
    let (b, m, n, k) = bmnk;
    let lhs_stride = lhs_layout.stride();
    let rhs_stride = rhs_layout.stride();
    let lhs_rank = lhs_stride.len();
    let rhs_rank = rhs_stride.len();
    if lhs_rank < 2 || rhs_rank < 2 {
        return Err(crate::Error::Msg(format!(
            "wgpu: matmul requires rank >= 2, got lhs={lhs_rank} rhs={rhs_rank}"
        )));
    }

    let a_stride_row = lhs_stride[lhs_rank - 2];
    let a_stride_col = lhs_stride[lhs_rank - 1];
    let b_stride_row = rhs_stride[rhs_rank - 2];
    let b_stride_col = rhs_stride[rhs_rank - 1];

    // Determine the per-batch stride. If there is no batch dim we use 0
    // (and `b` had better be 1 — candle's tensor.matmul always produces
    // `b == 1` in that case).
    let a_batch_stride = match lhs_stride.get(..lhs_rank - 2) {
        Some([]) | None => 0,
        Some([s]) => *s,
        // Multi-dim batch: candle collapses into a single "b" dimension
        // before reaching us. The CPU path assumes contiguous-ish batch
        // layout (`s1 == stride * dims[1]`); we replicate that
        // assumption here. Anything weirder — fall back to error so the
        // caller knows to pre-contiguify.
        Some(other) => match other {
            &[s1, stride] if s1 == stride * lhs_layout.dims()[1] => stride,
            &[_, stride] if lhs_layout.dims()[0] == 1 => stride,
            &[stride, _] if lhs_layout.dims()[1] == 1 => stride,
            _ => {
                return Err(crate::Error::Msg(format!(
                    "wgpu: matmul lhs batch strides {other:?} not supported \
                     (multi-dim batch must be contiguous)"
                )))
            }
        },
    };
    let b_batch_stride = match rhs_stride.get(..rhs_rank - 2) {
        Some([]) | None => 0,
        Some([s]) => *s,
        Some(other) => match other {
            &[s1, stride] if s1 == stride * rhs_layout.dims()[1] => stride,
            &[_, stride] if rhs_layout.dims()[0] == 1 => stride,
            &[stride, _] if rhs_layout.dims()[1] == 1 => stride,
            _ => {
                return Err(crate::Error::Msg(format!(
                    "wgpu: matmul rhs batch strides {other:?} not supported \
                     (multi-dim batch must be contiguous)"
                )))
            }
        },
    };

    // Output is always contiguous (row-major (m, n)) per matmul contract.
    let c_stride_row = n;
    let c_stride_col = 1;
    let c_batch_stride = m * n;

    // Range checks. We funnel everything through u32 since WGSL uniforms
    // address up to 4 GiB elements — plenty for Gemma. Anything bigger
    // would require a custom large-tensor path.
    fn to_u32(v: usize, what: &str) -> Result<u32> {
        u32::try_from(v).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: matmul {what} = {v} exceeds u32::MAX (large-tensor path not yet supported)"
            ))
        })
    }

    Ok(MatmulPlan {
        b: to_u32(b, "batch")?,
        m: to_u32(m, "m")?,
        n: to_u32(n, "n")?,
        k: to_u32(k, "k")?,
        a_stride_row: to_u32(a_stride_row, "lhs row stride")?,
        a_stride_col: to_u32(a_stride_col, "lhs col stride")?,
        b_stride_row: to_u32(b_stride_row, "rhs row stride")?,
        b_stride_col: to_u32(b_stride_col, "rhs col stride")?,
        c_stride_row: to_u32(c_stride_row, "out row stride")?,
        c_stride_col: to_u32(c_stride_col, "out col stride")?,
        a_batch_stride: to_u32(a_batch_stride, "lhs batch stride")?,
        b_batch_stride: to_u32(b_batch_stride, "rhs batch stride")?,
        c_batch_stride: to_u32(c_batch_stride, "out batch stride")?,
        a_offset: to_u32(lhs_layout.start_offset(), "lhs start offset")?,
        b_offset: to_u32(rhs_layout.start_offset(), "rhs start offset")?,
    })
}

pub(crate) fn matmul(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    bmnk: (usize, usize, usize, usize),
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<WgpuStorage> {
    if lhs.dtype != rhs.dtype {
        return Err(crate::Error::Msg(format!(
            "wgpu: matmul dtype mismatch lhs={:?} rhs={:?}",
            lhs.dtype, rhs.dtype
        )));
    }
    let plan = plan(bmnk, lhs_layout, rhs_layout)?;
    match lhs.dtype {
        DType::F32 => dispatch(lhs, rhs, &plan, MatmulVariant::F32),
        DType::F16 => {
            if !lhs.device.supports_shader_f16() {
                return Err(crate::Error::Msg(
                    "wgpu: f16 matmul requires the SHADER_F16 device feature; \
                     this adapter does not advertise it"
                        .to_string(),
                ));
            }
            // wgpu 0.20 ships naga 0.20 whose WGSL frontend does not
            // implement `enable f16;` (UnimplementedF16 — see naga's
            // tracking issue). The adapter may advertise the feature
            // but the toolchain can't compile the kernel, so creating
            // the shader module would panic on validation. We surface
            // a precise error here instead. Once we bump wgpu to a
            // version backed by naga >= 22.0 (`enable f16;` accepted)
            // the kernel compiles and this guard is removed.
            //
            // Probing via `Adapter::features().contains(SHADER_F16)`
            // is *not* sufficient because feature advertisement only
            // reflects what the *backend* (Vulkan/Metal/DX12) can do,
            // not what the WGSL parser can express.
            if !wgsl_frontend_supports_f16() {
                return Err(crate::Error::Msg(
                    "wgpu: f16 matmul requires a WGSL frontend that implements \
                     `enable f16;` (naga >= 22.0). Current build uses wgpu 0.20 / \
                     naga 0.20 which does not. Tracking: bump wgpu in Phase 3.9."
                        .to_string(),
                ));
            }
            dispatch(lhs, rhs, &plan, MatmulVariant::F16)
        }
        other => Err(crate::Error::Msg(format!(
            "wgpu: matmul not implemented for dtype {other:?}"
        ))),
    }
}

/// `true` once the WGSL frontend bundled with the active `wgpu` crate
/// understands `enable f16;`. naga 0.20 (shipped with wgpu 0.20)
/// returns `UnimplementedF16` for the directive; naga >= 22 accepts
/// it. We compile-time gate on the wgpu version we depend on — the
/// dependency lives in `candle-core/Cargo.toml` so this stays a single
/// place to update when bumping wgpu.
const fn wgsl_frontend_supports_f16() -> bool {
    // Conservative: leave false until we explicitly bump wgpu in a
    // future phase. Flipping this to `true` should be paired with a
    // Cargo.toml update to a wgpu version that ships naga >= 22.
    false
}

#[derive(Clone, Copy)]
enum MatmulVariant {
    F32,
    F16,
}

impl MatmulVariant {
    fn cache_key(self) -> &'static str {
        match self {
            Self::F32 => "matmul:f32",
            Self::F16 => "matmul:f16",
        }
    }
    fn wgsl(self) -> &'static str {
        match self {
            Self::F32 => MATMUL_F32_WGSL,
            Self::F16 => MATMUL_F16_WGSL,
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

fn dispatch(
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    p: &MatmulPlan,
    variant: MatmulVariant,
) -> Result<WgpuStorage> {
    let device = &lhs.device;
    let pipeline = device.get_or_create_pipeline(variant.cache_key(), variant.wgsl(), "main");

    let out_len = (p.b as u64) * (p.m as u64) * (p.n as u64);
    if out_len == 0 {
        return Err(crate::Error::Msg(
            "wgpu: matmul output is empty (zero-sized batch/m/n) — not supported".to_string(),
        ));
    }
    let elem_bytes = variant.elem_bytes();
    let raw_size = out_len * elem_bytes;
    // Round to COPY_BUFFER_ALIGNMENT (4 bytes). Mirrors WgpuStorage's
    // alloc_zeros logic so readback / clone work uniformly.
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-matmul-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniform buffer with the meta struct.
    let meta = MatmulMeta {
        m: p.m,
        n: p.n,
        k: p.k,
        a_stride_row: p.a_stride_row,
        a_stride_col: p.a_stride_col,
        b_stride_row: p.b_stride_row,
        b_stride_col: p.b_stride_col,
        c_stride_row: p.c_stride_row,
        c_stride_col: p.c_stride_col,
        batch: p.b,
        a_batch_stride: p.a_batch_stride,
        b_batch_stride: p.b_batch_stride,
        c_batch_stride: p.c_batch_stride,
        a_offset: p.a_offset,
        b_offset: p.b_offset,
        _pad: 0,
    };
    let meta_buffer = {
        let buf = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-matmul-meta"),
            size: std::mem::size_of::<MatmulMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(meta_bytes(&meta));
        buf.unmap();
        buf
    };

    // Build the bind group from the auto-derived layout. Slot order
    // (uniform, storage-read a, storage-read b, storage-read_write c)
    // mirrors the shader's @binding declarations.
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-matmul-bg"),
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
            label: Some("wgpu-matmul-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-matmul-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = p.n.div_ceil(TILE);
        let groups_y = p.m.div_ceil(TILE);
        let groups_z = p.b;
        pass.dispatch_workgroups(groups_x, groups_y, groups_z);
    }
    device.queue().submit(Some(encoder.finish()));

    Ok(WgpuStorage::from_raw_buffer(
        Arc::new(out_buffer),
        out_len as usize,
        variant.dtype(),
        device.clone(),
    ))
}
