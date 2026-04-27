// Type-cast kernel (Phase 3.7).
//
// Casts elements from one scalar type to another.
// Source must be contiguous.  One thread per output element.
//
// Substitution markers:
//   __SRC_T__  → source type (e.g., u32)
//   __DST_T__  → destination type (e.g., f32)

struct CastMeta {
    n_elements: u32,
    src_offset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: CastMeta;
@group(0) @binding(1) var<storage, read> src: array<__SRC_T__>;
@group(0) @binding(2) var<storage, read_write> out: array<__DST_T__>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_elements {
        return;
    }
    out[idx] = __DST_T__(src[params.src_offset + idx]);
}
