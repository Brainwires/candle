// Strided copy kernel (Phase 3.6).
//
// Reads from a strided source and writes contiguous output at a fixed
// element offset. This drives `WgpuStorage::copy_strided_src` for the
// non-contiguous case (the contiguous case is a single
// `copy_buffer_to_buffer` issued from the host, no kernel needed).
//
// Substitution markers:
//   __SCALAR_T__ — input/output dtype (`f32` for now).

struct CopyMeta {
    n_elements: u32,
    rank: u32,
    src_offset: u32,
    dst_offset: u32,
    shape: vec4<u32>,
    src_strides: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: CopyMeta;
@group(0) @binding(1) var<storage, read> src: array<__SCALAR_T__>;
@group(0) @binding(2) var<storage, read_write> dst: array<__SCALAR_T__>;

fn unravel_src(linear: u32) -> u32 {
    var rem: u32 = linear;
    var idx: u32 = params.src_offset;
    if (params.rank >= 1u) {
        let d = params.rank - 1u;
        let s = params.shape[d];
        let st = params.src_strides[d];
        let c = rem % s;
        rem = rem / s;
        idx = idx + c * st;
    }
    if (params.rank >= 2u) {
        let d = params.rank - 2u;
        let s = params.shape[d];
        let st = params.src_strides[d];
        let c = rem % s;
        rem = rem / s;
        idx = idx + c * st;
    }
    if (params.rank >= 3u) {
        let d = params.rank - 3u;
        let s = params.shape[d];
        let st = params.src_strides[d];
        let c = rem % s;
        rem = rem / s;
        idx = idx + c * st;
    }
    if (params.rank >= 4u) {
        let d = params.rank - 4u;
        let s = params.shape[d];
        let st = params.src_strides[d];
        let c = rem % s;
        rem = rem / s;
        idx = idx + c * st;
    }
    return idx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_elements) {
        return;
    }
    let s_idx = unravel_src(i);
    dst[params.dst_offset + i] = src[s_idx];
}
