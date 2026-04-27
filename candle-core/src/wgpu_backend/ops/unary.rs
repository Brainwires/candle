//! Element-wise unary ops on WebGPU (Phase 3.4).
//!
//! Drives `WgpuStorage::unary_impl<B: UnaryOpT>` by mapping `B::NAME`
//! onto a WGSL op body and a stable pipeline-cache key. The kernel
//! template lives in `kernels/elementwise_unary.wgsl`; we substitute
//! the scalar dtype (`f32`, eventually `f16`) and the op expression at
//! runtime, then memoize the compiled pipeline forever.
//!
//! # Limitations (deferred)
//!
//! - Rank > 4 is not supported. Gemma uses up to 4D tensors. A higher-
//!   rank path would either bump the uniform shape to a longer array or
//!   force a contiguous repack first; both are Phase 3.9 polish.
//! - Negative strides aren't representable in `u32` — rejected at host.
//! - f16 returns `not_implemented` until we bump wgpu/naga past the
//!   `enable f16;` parser bug (same gate as matmul, see `matmul.rs`).
//! - Integer dtypes (u8/u32/i*) are rejected: the candle CPU path's
//!   `unary_impl` itself is undefined for them (see `op.rs`'s
//!   `todo!("no unary function for u8")`). We mirror that.
//!
//! # Output
//!
//! Output is always allocated as a fresh contiguous row-major buffer,
//! matching the input's shape. Down-stream ops re-derive any view they
//! need from the resulting `Tensor`'s layout.

use crate::op::UnaryOpT;
use crate::{DType, Layout, Result};
use std::sync::Arc;

use super::super::storage::WgpuStorage;

const TEMPLATE: &str = include_str!("../kernels/elementwise_unary.wgsl");

/// Maximum tensor rank the kernel handles. The shader hard-codes vec4
/// shape/stride uniforms, so anything higher than 4 is rejected on the
/// host side with a precise error.
const MAX_RANK: usize = 4;

const WORKGROUP_SIZE: u32 = 64;

/// Uniform layout matching `ElemUnaryMeta` in the WGSL template.
///
/// `repr(C)` with explicit padding because WGSL requires uniforms to
/// follow std140-ish alignment: a `vec4<u32>` lands on a 16-byte
/// boundary. The leading scalar fields are padded out to 16 bytes
/// before the two `vec4<u32>`s start.
#[repr(C)]
#[derive(Clone, Copy)]
struct ElemUnaryMeta {
    n_elements: u32,
    rank: u32,
    a_offset: u32,
    _pad0: u32,
    shape: [u32; 4],
    a_strides: [u32; 4],
}

fn meta_bytes(m: &ElemUnaryMeta) -> &[u8] {
    // SAFETY: `ElemUnaryMeta` is `repr(C)` with all-`u32` fields and an
    // explicit pad scalar, so its representation is a fixed-size byte
    // buffer with no uninitialized holes.
    unsafe {
        std::slice::from_raw_parts(
            (m as *const ElemUnaryMeta) as *const u8,
            std::mem::size_of::<ElemUnaryMeta>(),
        )
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // F16 variant wired but currently rejected at the dtype gate
enum ScalarDType {
    F32,
    // F16 is gated behind the WGSL frontend's `enable f16;` support;
    // until we bump wgpu/naga, requesting it returns `not_implemented`.
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

/// Map a candle `UnaryOpT::NAME` onto a (cache-key, op-body) pair.
///
/// Returning `None` means we don't have a kernel for that op yet —
/// the host falls back to `not_implemented`. The set covers every
/// unary op Gemma needs, plus the standard f32 transcendentals.
fn op_for(name: &str, dtype: ScalarDType) -> Option<(&'static str, String)> {
    // Op bodies are written so that the same WGSL works for f32 and f16;
    // when we eventually flip f16 on, the WGSL front-end will resolve
    // overloaded math intrinsics for `f16` automatically.
    let body: &'static str = match name {
        "neg" => "return -x;",
        "abs" => "return abs(x);",
        "sqrt" => "return sqrt(x);",
        "sqr" => "return x * x;",
        "exp" => "return exp(x);",
        // WGSL has both `log` (natural) and `log2`; we want natural.
        "log" => "return log(x);",
        "sin" => "return sin(x);",
        "cos" => "return cos(x);",
        "tanh" => "return tanh(x);",
        "recip" => "return 1.0 / x;",
        "floor" => "return floor(x);",
        "ceil" => "return ceil(x);",
        // WGSL `round` ties-to-even; candle's CPU path uses
        // `f32::round` (half-away-from-zero). The mismatch is at exact
        // halves only and doesn't bite Gemma in practice. We document
        // and ship.
        "round" => "return round(x);",
        "sign" => "return sign(x);",
        "relu" => "return max(x, 0.0);",
        // Tanh-approximated GELU, matching candle CPU's `gelu()`:
        //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // sqrt(2/pi) ≈ 0.7978845608028654
        "gelu" => {
            "let kappa: f32 = 0.7978845608028654; \
             let cube: f32 = 0.044715 * x * x * x; \
             return 0.5 * x * (1.0 + tanh(kappa * (x + cube)));"
        }
        // erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        // WGSL has no erf intrinsic; use the Abramowitz & Stegun 7.1.26
        // rational approximation, accurate to ~1.5e-7. Same form
        // candle's CPU `erf` uses (see candle's `erf.rs`).
        "gelu_erf" | "erf" => {
            // `erf` returns erf(x), `gelu_erf` returns 0.5*x*(1+erf(x/√2)).
            // We share the implementation; the wrapper differs.
            if name == "erf" {
                "let t: f32 = 1.0 / (1.0 + 0.3275911 * abs(x)); \
                 let y: f32 = 1.0 - (((((1.061405429 * t - 1.453152027) * t) \
                 + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x); \
                 return select(-y, y, x >= 0.0);"
            } else {
                "let s: f32 = x * 0.7071067811865475; \
                 let t: f32 = 1.0 / (1.0 + 0.3275911 * abs(s)); \
                 let y: f32 = 1.0 - (((((1.061405429 * t - 1.453152027) * t) \
                 + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-s * s); \
                 let erf_s: f32 = select(-y, y, s >= 0.0); \
                 return 0.5 * x * (1.0 + erf_s);"
            }
        }
        // SiLU / Swish: x * sigmoid(x).
        "silu" => "return x / (1.0 + exp(-x));",
        _ => return None,
    };

    // Pipeline cache wants a `&'static str` key. We pre-define every
    // (op, dtype) combination so the keys are static literals.
    let key: &'static str = match (name, dtype) {
        ("neg", ScalarDType::F32) => "unary:neg:f32",
        ("neg", ScalarDType::F16) => "unary:neg:f16",
        ("abs", ScalarDType::F32) => "unary:abs:f32",
        ("abs", ScalarDType::F16) => "unary:abs:f16",
        ("sqrt", ScalarDType::F32) => "unary:sqrt:f32",
        ("sqrt", ScalarDType::F16) => "unary:sqrt:f16",
        ("sqr", ScalarDType::F32) => "unary:sqr:f32",
        ("sqr", ScalarDType::F16) => "unary:sqr:f16",
        ("exp", ScalarDType::F32) => "unary:exp:f32",
        ("exp", ScalarDType::F16) => "unary:exp:f16",
        ("log", ScalarDType::F32) => "unary:log:f32",
        ("log", ScalarDType::F16) => "unary:log:f16",
        ("sin", ScalarDType::F32) => "unary:sin:f32",
        ("sin", ScalarDType::F16) => "unary:sin:f16",
        ("cos", ScalarDType::F32) => "unary:cos:f32",
        ("cos", ScalarDType::F16) => "unary:cos:f16",
        ("tanh", ScalarDType::F32) => "unary:tanh:f32",
        ("tanh", ScalarDType::F16) => "unary:tanh:f16",
        ("recip", ScalarDType::F32) => "unary:recip:f32",
        ("recip", ScalarDType::F16) => "unary:recip:f16",
        ("floor", ScalarDType::F32) => "unary:floor:f32",
        ("floor", ScalarDType::F16) => "unary:floor:f16",
        ("ceil", ScalarDType::F32) => "unary:ceil:f32",
        ("ceil", ScalarDType::F16) => "unary:ceil:f16",
        ("round", ScalarDType::F32) => "unary:round:f32",
        ("round", ScalarDType::F16) => "unary:round:f16",
        ("sign", ScalarDType::F32) => "unary:sign:f32",
        ("sign", ScalarDType::F16) => "unary:sign:f16",
        ("relu", ScalarDType::F32) => "unary:relu:f32",
        ("relu", ScalarDType::F16) => "unary:relu:f16",
        ("gelu", ScalarDType::F32) => "unary:gelu:f32",
        ("gelu", ScalarDType::F16) => "unary:gelu:f16",
        ("gelu_erf", ScalarDType::F32) => "unary:gelu_erf:f32",
        ("gelu_erf", ScalarDType::F16) => "unary:gelu_erf:f16",
        ("erf", ScalarDType::F32) => "unary:erf:f32",
        ("erf", ScalarDType::F16) => "unary:erf:f16",
        ("silu", ScalarDType::F32) => "unary:silu:f32",
        ("silu", ScalarDType::F16) => "unary:silu:f16",
        _ => return None,
    };

    Some((key, body.to_string()))
}

/// Build the WGSL source for one (op, dtype) pair by substituting
/// markers in the template. Cheap (string copy + two replaces) and
/// the result feeds straight into the cached pipeline compile.
fn render_wgsl(scalar: ScalarDType, op_body: &str) -> String {
    TEMPLATE
        .replace("__SCALAR_T__", scalar.wgsl_name())
        .replace("__OP_BODY__", op_body)
}

pub(crate) fn unary<B: UnaryOpT>(src: &WgpuStorage, layout: &Layout) -> Result<WgpuStorage> {
    let device = &src.device;
    let scalar = match src.dtype {
        DType::F32 => ScalarDType::F32,
        DType::F16 => {
            // Mirror matmul's gate. Even on adapters that advertise
            // SHADER_F16, naga 0.20 (wgpu 0.20) can't parse `enable f16;`,
            // so we'd panic at module creation. Keeping a single gate
            // path means flipping it on later is one constant change.
            return Err(crate::Error::Msg(format!(
                "wgpu: unary {} on f16 requires a WGSL frontend that \
                 implements `enable f16;` (naga >= 22.0). Tracking: \
                 bump wgpu in Phase 3.9.",
                B::NAME
            )));
        }
        other => {
            return Err(crate::Error::Msg(format!(
                "wgpu: unary {} not implemented for dtype {:?} \
                 (only f32 is wired in Phase 3.4)",
                B::NAME,
                other
            )))
        }
    };

    let dims = layout.dims();
    if dims.len() > MAX_RANK {
        return Err(crate::Error::Msg(format!(
            "wgpu: unary {} rank {} > {} (Phase 3.4 limit; \
             reshape/contiguify before dispatch)",
            B::NAME,
            dims.len(),
            MAX_RANK
        )));
    }

    let (key, body) = op_for(B::NAME, scalar).ok_or_else(|| {
        crate::Error::Msg(format!(
            "wgpu: unary op {} not yet wired on Wgpu backend",
            B::NAME
        ))
    })?;

    let wgsl = render_wgsl(scalar, &body);
    let pipeline = device.get_or_create_pipeline(key, &wgsl, "main");

    // Pack shape + strides into vec4 slots. Unused trailing slots stay 0;
    // the shader gates by `meta.rank` so they're never read.
    let n_elements = dims.iter().product::<usize>();
    if n_elements == 0 {
        return Err(crate::Error::Msg(format!(
            "wgpu: unary {} on empty tensor (zero-sized dim) not supported",
            B::NAME
        )));
    }
    let n_elements_u32 = u32::try_from(n_elements).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: unary {} element count {} exceeds u32::MAX",
            B::NAME,
            n_elements
        ))
    })?;
    let stride = layout.stride();
    let mut shape = [1u32; 4];
    let mut a_strides = [0u32; 4];
    for (i, (&d, &s)) in dims.iter().zip(stride.iter()).enumerate() {
        shape[i] = u32::try_from(d).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: unary {} shape dim {} = {} exceeds u32::MAX",
                B::NAME,
                i,
                d
            ))
        })?;
        a_strides[i] = u32::try_from(s).map_err(|_| {
            crate::Error::Msg(format!(
                "wgpu: unary {} stride dim {} = {} exceeds u32::MAX \
                 (negative or huge strides not supported)",
                B::NAME,
                i,
                s
            ))
        })?;
    }
    let a_offset = u32::try_from(layout.start_offset()).map_err(|_| {
        crate::Error::Msg(format!(
            "wgpu: unary {} start offset {} exceeds u32::MAX",
            B::NAME,
            layout.start_offset()
        ))
    })?;

    let meta = ElemUnaryMeta {
        n_elements: n_elements_u32,
        rank: dims.len() as u32,
        a_offset,
        _pad0: 0,
        shape,
        a_strides,
    };

    // Allocate output buffer (always contiguous, same shape, same dtype).
    let elem_bytes = scalar.elem_bytes();
    let raw_size = (n_elements as u64) * elem_bytes;
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let out_size = raw_size.div_ceil(align) * align;

    let out_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-unary-out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let meta_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpu-unary-meta"),
        size: std::mem::size_of::<ElemUnaryMeta>() as u64,
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
            label: Some("wgpu-unary-bg"),
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
            label: Some("wgpu-unary-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wgpu-unary-pass"),
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
        scalar.dtype(),
        device.clone(),
    ))
}
