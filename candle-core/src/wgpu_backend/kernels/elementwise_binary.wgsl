// Element-wise binary op kernel template (Phase 3.4).
//
// One shader per (op, dtype). The host dispatcher substitutes
// `__SCALAR_T__` and `__OP_BODY__` and pipes the result through
// `WgpuDevice::get_or_create_pipeline`.
//
// Layout / broadcasting:
//   * Up to 4D tensors (rank > 4 is rejected on the host side).
//   * Output is always contiguous, row-major, allocated by the host.
//     Output rank/shape is the broadcast result of A and B.
//   * Inputs A and B may be non-contiguous AND broadcast: the host packs
//     each input's per-axis stride relative to the *output* shape. A
//     broadcasted axis carries stride 0, so multiple output indices read
//     the same source element.
//
// Workgroup is 1D, 64 threads.

struct ElemBinaryMeta {
    n_elements: u32,
    rank: u32,
    a_offset: u32,
    b_offset: u32,
    shape: vec4<u32>,
    a_strides: vec4<u32>,
    b_strides: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: ElemBinaryMeta;
@group(0) @binding(1) var<storage, read> a: array<__SCALAR_T__>;
@group(0) @binding(2) var<storage, read> b: array<__SCALAR_T__>;
@group(0) @binding(3) var<storage, read_write> out: array<__SCALAR_T__>;

struct Indices {
    a: u32,
    b: u32,
};

fn unravel(linear: u32) -> Indices {
    var rem: u32 = linear;
    var a_idx: u32 = params.a_offset;
    var b_idx: u32 = params.b_offset;
    if (params.rank >= 1u) {
        let d = params.rank - 1u;
        let s = params.shape[d];
        let c = rem % s;
        rem = rem / s;
        a_idx = a_idx + c * params.a_strides[d];
        b_idx = b_idx + c * params.b_strides[d];
    }
    if (params.rank >= 2u) {
        let d = params.rank - 2u;
        let s = params.shape[d];
        let c = rem % s;
        rem = rem / s;
        a_idx = a_idx + c * params.a_strides[d];
        b_idx = b_idx + c * params.b_strides[d];
    }
    if (params.rank >= 3u) {
        let d = params.rank - 3u;
        let s = params.shape[d];
        let c = rem % s;
        rem = rem / s;
        a_idx = a_idx + c * params.a_strides[d];
        b_idx = b_idx + c * params.b_strides[d];
    }
    if (params.rank >= 4u) {
        let d = params.rank - 4u;
        let s = params.shape[d];
        let c = rem % s;
        rem = rem / s;
        a_idx = a_idx + c * params.a_strides[d];
        b_idx = b_idx + c * params.b_strides[d];
    }
    return Indices(a_idx, b_idx);
}

fn op_apply(x: __SCALAR_T__, y: __SCALAR_T__) -> __SCALAR_T__ {
    __OP_BODY__
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_elements) {
        return;
    }
    let idx = unravel(i);
    out[i] = op_apply(a[idx.a], b[idx.b]);
}
