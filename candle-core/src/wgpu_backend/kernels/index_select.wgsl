// Index-select kernel (Phase 3.7).
//
// Given a source tensor of shape (..., src_dim, ...) and a 1-D index
// tensor of shape (n_ids,), produce an output tensor of shape
// (..., n_ids, ...) by gathering rows along `dim`.
//
// One thread per output element.  Thread `gid` maps to output linear
// index `gid`.
//
// Substitution marker: __SCALAR_T__ → f32.

struct IndexSelectMeta {
    n_elements: u32,   // total elements in output
    left_size: u32,    // product of dims before `dim`
    src_dim: u32,      // source size along `dim`
    n_ids: u32,        // number of indices (output size along `dim`)
    right_size: u32,   // product of dims after `dim`
    ids_offset: u32,   // start_offset into ids buffer
    ids_stride: u32,   // stride of the 1-D ids tensor
    src_offset: u32,   // start_offset into source buffer
};

@group(0) @binding(0) var<uniform> params: IndexSelectMeta;
@group(0) @binding(1) var<storage, read> src: array<__SCALAR_T__>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<__SCALAR_T__>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let idx = gid.x + gid.y * num_wg.x * 64u;
    if idx >= params.n_elements {
        return;
    }

    // Decompose linear output index into (left_i, id_i, right_i).
    let right_i = idx % params.right_size;
    let id_i = (idx / params.right_size) % params.n_ids;
    let left_i = idx / (params.right_size * params.n_ids);

    // Look up the actual source index along `dim`.
    let raw_id = ids[params.ids_offset + id_i * params.ids_stride];

    // Compute source linear offset (contiguous source assumed).
    let src_idx = params.src_offset
        + left_i * params.src_dim * params.right_size
        + raw_id * params.right_size
        + right_i;

    out[idx] = src[src_idx];
}
