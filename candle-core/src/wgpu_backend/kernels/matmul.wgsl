// Tiled f32 matmul.
//
// Computes a single batched matmul: C[b, i, j] = sum_k A[b, i, k] * B[b, k, j]
// where each operand may be non-contiguous via per-axis row/col strides plus
// a per-batch stride. Strides are expressed in *elements*, not bytes.
//
// One workgroup computes a TILE x TILE block of one C matrix in the batch
// dimension. The k axis is reduced in TILE-sized chunks loaded into shared
// memory ("workgroup memory" in WGSL).
//
// Workgroup shape: (16, 16, 1). Dispatch:
//   x: ceil(n / 16)
//   y: ceil(m / 16)
//   z: batch
//
// Phase 3.3 prioritizes correctness over perf. Phase 3.9 will tune for the
// specific shapes Gemma needs.

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
};

@group(0) @binding(0) var<uniform> params: MatmulMeta;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    let batch = wid.z;

    let m = params.m;
    let n = params.n;
    let k = params.k;

    if (batch >= params.batch) {
        return;
    }

    let a_base = params.a_offset + batch * params.a_batch_stride;
    let b_base = params.b_offset + batch * params.b_batch_stride;
    let c_base = batch * params.c_batch_stride;

    var acc: f32 = 0.0;

    let num_tiles = (k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        let b_row = t * TILE + lid.y;

        if (row < m && a_col < k) {
            let idx = a_base + row * params.a_stride_row + a_col * params.a_stride_col;
            tile_a[lid.y][lid.x] = a[idx];
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }

        if (b_row < k && col < n) {
            let idx = b_base + b_row * params.b_stride_row + col * params.b_stride_col;
            tile_b[lid.y][lid.x] = b[idx];
        } else {
            tile_b[lid.y][lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            acc = acc + tile_a[lid.y][i] * tile_b[i][lid.x];
        }

        workgroupBarrier();
    }

    if (row < m && col < n) {
        let idx = c_base + row * params.c_stride_row + col * params.c_stride_col;
        c[idx] = acc;
    }
}
