// Element-wise unary op kernel template (Phase 3.4).
//
// One shader per (op, dtype). Selection is done host-side: the dispatcher
// substitutes `__SCALAR_T__` and `__OP_BODY__` markers via simple string
// replacement and feeds the result through `WgpuDevice::get_or_create_pipeline`.
//
// Layout:
//   * Up to 4D tensors (rank > 4 is rejected on the host side).
//   * Inputs may be non-contiguous; the host packs `params.shape`,
//     `params.a_strides`, and `params.start_offset` from `Layout`.
//   * Outputs are always contiguous, row-major. Dispatcher allocates a
//     fresh contiguous buffer.
//
// Workgroup is 1D, 64 threads. Each thread processes exactly one output
// element. The grid is sized to `ceil(n_elements / 64)`.

struct ElemUnaryMeta {
    n_elements: u32,
    rank: u32,
    a_offset: u32,
    _pad0: u32,
    shape: vec4<u32>,
    a_strides: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: ElemUnaryMeta;
@group(0) @binding(1) var<storage, read> a: array<__SCALAR_T__>;
@group(0) @binding(2) var<storage, read_write> out: array<__SCALAR_T__>;

fn unravel_a(linear: u32) -> u32 {
    // Convert a contiguous (row-major) linear index in the output tensor
    // into the source tensor's element index by walking the multi-dim
    // coordinate through `a_strides`. Output shape == input shape, so we
    // share `params.shape` for both. Shapes higher than `rank` are 1.
    var rem: u32 = linear;
    var src: u32 = params.a_offset;
    // Loop unrolled with rank guard for ranks 0..=4.
    if (params.rank >= 1u) {
        // Process from last dim to first. We invert by computing the
        // dim-wise index iteratively: for d in (rank-1 .. 0): coord =
        // rem % shape[d]; rem /= shape[d]; src += coord * stride[d].
        // We unroll because WGSL has no dynamic indexing into vec4
        // outside of scalar loops, and we want it readable.
        // Step: dim = rank - 1
        let d0 = params.rank - 1u;
        let s0 = params.shape[d0];
        let st0 = params.a_strides[d0];
        let c0 = rem % s0;
        rem = rem / s0;
        src = src + c0 * st0;
    }
    if (params.rank >= 2u) {
        let d1 = params.rank - 2u;
        let s1 = params.shape[d1];
        let st1 = params.a_strides[d1];
        let c1 = rem % s1;
        rem = rem / s1;
        src = src + c1 * st1;
    }
    if (params.rank >= 3u) {
        let d2 = params.rank - 3u;
        let s2 = params.shape[d2];
        let st2 = params.a_strides[d2];
        let c2 = rem % s2;
        rem = rem / s2;
        src = src + c2 * st2;
    }
    if (params.rank >= 4u) {
        let d3 = params.rank - 4u;
        let s3 = params.shape[d3];
        let st3 = params.a_strides[d3];
        let c3 = rem % s3;
        rem = rem / s3;
        src = src + c3 * st3;
    }
    return src;
}

fn op_apply(x: __SCALAR_T__) -> __SCALAR_T__ {
    __OP_BODY__
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_elements) {
        return;
    }
    let src_idx = unravel_a(i);
    out[i] = op_apply(a[src_idx]);
}
