// Reduction kernel template (Phase 3.5).
//
// One thread per output element. The thread walks the input element-set
// that maps to its output coord by:
//   * Unraveling its linear output index into a multi-dim coord using
//     `out_shape` (which equals input shape with reduced axes set to 1).
//   * Walking the reduced axes' Cartesian product (flat counter from 0
//     to `reduce_total`) and adding the per-axis contribution to a base
//     pointer derived from input strides.
//
// Substitution markers:
//   __SCALAR_T__   — element dtype (`f32` for now; `f16` deferred)
//   __INIT__       — accumulator init expression (e.g. `0.0`, `-1e38`)
//   __COMBINE__    — function body merging accumulator `acc` with `v`
//   __FINALIZE__   — final transform (e.g. mean: multiply by divisor)
//
// Up to rank-4 inputs supported (matches elementwise kernels).

struct ReduceMeta {
    n_out: u32,
    rank: u32,
    src_offset: u32,
    op_id: u32,            // unused at runtime; kernel is specialized per op
    shape: vec4<u32>,      // input shape
    in_strides: vec4<u32>, // input strides
    out_shape: vec4<u32>,  // input shape with reduced axes -> 1
    reduce_mask: vec4<u32>,// 1 if axis is reduced, 0 otherwise
    reduce_shape: vec4<u32>, // shape[d] if axis d is reduced, else 1 (used to walk reduce coords)
    divisor: f32,          // 1.0 for non-mean, 1/reduce_size for mean
    reduce_total: u32,     // product over reduced axes (>=1)
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: ReduceMeta;
@group(0) @binding(1) var<storage, read> a: array<__SCALAR_T__>;
@group(0) @binding(2) var<storage, read_write> out: array<__SCALAR_T__>;

fn combine(acc: __SCALAR_T__, v: __SCALAR_T__) -> __SCALAR_T__ {
    __COMBINE__
}

fn finalize(acc: __SCALAR_T__) -> __SCALAR_T__ {
    __FINALIZE__
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_out) {
        return;
    }

    // Unravel `i` into a multi-dim coord using out_shape (last-dim fastest).
    // Compute the input-base offset by accumulating coord * in_stride for
    // non-reduced axes only (reduced axes are size 1 in out_shape, so the
    // coord is always 0 there — multiplying by stride is harmless).
    var rem: u32 = i;
    var base: u32 = params.src_offset;

    if (params.rank >= 1u) {
        let d = params.rank - 1u;
        let s = params.out_shape[d];
        let st = params.in_strides[d];
        let c = rem % s;
        rem = rem / s;
        base = base + c * st;
    }
    if (params.rank >= 2u) {
        let d = params.rank - 2u;
        let s = params.out_shape[d];
        let st = params.in_strides[d];
        let c = rem % s;
        rem = rem / s;
        base = base + c * st;
    }
    if (params.rank >= 3u) {
        let d = params.rank - 3u;
        let s = params.out_shape[d];
        let st = params.in_strides[d];
        let c = rem % s;
        rem = rem / s;
        base = base + c * st;
    }
    if (params.rank >= 4u) {
        let d = params.rank - 4u;
        let s = params.out_shape[d];
        let st = params.in_strides[d];
        let c = rem % s;
        rem = rem / s;
        base = base + c * st;
    }

    // Walk the reduced axes via a flat counter from 0..reduce_total.
    // For each step: unravel into per-axis coords using reduce_shape (=1
    // on non-reduced axes), then add coord*stride to base. Non-reduced
    // axes contribute 0 because reduce_shape[d] == 1 forces coord = 0.
    var acc: __SCALAR_T__ = __INIT__;
    let total: u32 = params.reduce_total;
    var k: u32 = 0u;
    loop {
        if (k >= total) { break; }

        var off: u32 = base;
        var rk: u32 = k;

        if (params.rank >= 1u) {
            let d = params.rank - 1u;
            let s = params.reduce_shape[d];
            let st = params.in_strides[d];
            let c = rk % s;
            rk = rk / s;
            off = off + c * st;
        }
        if (params.rank >= 2u) {
            let d = params.rank - 2u;
            let s = params.reduce_shape[d];
            let st = params.in_strides[d];
            let c = rk % s;
            rk = rk / s;
            off = off + c * st;
        }
        if (params.rank >= 3u) {
            let d = params.rank - 3u;
            let s = params.reduce_shape[d];
            let st = params.in_strides[d];
            let c = rk % s;
            rk = rk / s;
            off = off + c * st;
        }
        if (params.rank >= 4u) {
            let d = params.rank - 4u;
            let s = params.reduce_shape[d];
            let st = params.in_strides[d];
            let c = rk % s;
            rk = rk / s;
            off = off + c * st;
        }

        let v = a[off];
        acc = combine(acc, v);

        k = k + 1u;
    }

    out[i] = finalize(acc);
}
