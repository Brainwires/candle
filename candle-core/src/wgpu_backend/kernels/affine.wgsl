// Affine kernel (Phase 3.5).
//
// Computes `out[i] = mul * a[unravel(i)] + add` where `mul` and `add`
// are f32 scalars baked into the uniform. Input may be non-contiguous
// (strided); output is always contiguous, row-major. Used to compose
// scalar arithmetic (`tensor + 1.0`, `tensor / N`, ...) which candle's
// `Tensor::affine` lowers into.
//
// Substitution markers:
//   __SCALAR_T__ — input/output dtype (`f32` for now).

struct AffineMeta {
    n_elements: u32,
    rank: u32,
    a_offset: u32,
    _pad0: u32,
    shape: vec4<u32>,
    a_strides: vec4<u32>,
    mul: f32,
    add: f32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: AffineMeta;
@group(0) @binding(1) var<storage, read> a: array<__SCALAR_T__>;
@group(0) @binding(2) var<storage, read_write> out: array<__SCALAR_T__>;

fn unravel_a(linear: u32) -> u32 {
    var rem: u32 = linear;
    var src: u32 = params.a_offset;
    if (params.rank >= 1u) {
        let d = params.rank - 1u;
        let s = params.shape[d];
        let st = params.a_strides[d];
        let c = rem % s;
        rem = rem / s;
        src = src + c * st;
    }
    if (params.rank >= 2u) {
        let d = params.rank - 2u;
        let s = params.shape[d];
        let st = params.a_strides[d];
        let c = rem % s;
        rem = rem / s;
        src = src + c * st;
    }
    if (params.rank >= 3u) {
        let d = params.rank - 3u;
        let s = params.shape[d];
        let st = params.a_strides[d];
        let c = rem % s;
        rem = rem / s;
        src = src + c * st;
    }
    if (params.rank >= 4u) {
        let d = params.rank - 4u;
        let s = params.shape[d];
        let st = params.a_strides[d];
        let c = rem % s;
        rem = rem / s;
        src = src + c * st;
    }
    return src;
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = gid.x + gid.y * num_wg.x * 64u;
    if (i >= params.n_elements) {
        return;
    }
    let src_idx = unravel_a(i);
    out[i] = params.mul * a[src_idx] + params.add;
}
