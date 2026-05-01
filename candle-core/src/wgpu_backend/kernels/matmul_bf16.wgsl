// Tiled bf16 matmul.
//
// WGSL has no native bf16 type, so each input/output buffer is bound as
// `array<u32>` with two bf16 elements packed per u32 (low 16 bits hold
// element [2k], high 16 bits hold element [2k+1]). This matches the
// representation used by `cast_from_bf16.wgsl` / `cast_to_bf16.wgsl`.
//
// Accumulation is done in f32 and the result is rounded to bf16 via
// round-to-nearest-even before being written back. The output buffer is
// bound as `array<atomic<u32>>` and must be zero-initialized by the host:
// each output element is written by exactly one thread via `atomicOr`,
// and adjacent threads in a row may target the same u32 (different
// halves), so atomicity is required to avoid lost half-words.
//
// Strides supplied in `MatmulMeta` are in bf16 *elements* (matching the
// f32/f16 kernels and the `Layout` produced on the host). The packed
// u32 layout is purely a shader-side view of the same storage.

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
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> c: array<atomic<u32>>;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

fn bf16_load_a(elem_idx: u32) -> f32 {
    let pair_idx = elem_idx >> 1u;
    let half_idx = elem_idx & 1u;
    let packed = a[pair_idx];
    let bits = (packed >> (half_idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

fn bf16_load_b(elem_idx: u32) -> f32 {
    let pair_idx = elem_idx >> 1u;
    let half_idx = elem_idx & 1u;
    let packed = b[pair_idx];
    let bits = (packed >> (half_idx * 16u)) & 0xFFFFu;
    return bitcast<f32>(bits << 16u);
}

// Round-to-nearest-even f32 → bf16 (returns 16-bit bf16 in low 16 bits of a u32).
// NaNs are preserved as quiet NaNs (top bit of the 7-bit mantissa set) so the
// rounded value never collapses into Inf.
fn f32_to_bf16_bits(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let exp = (bits >> 23u) & 0xFFu;
    let mant = bits & 0x7FFFFFu;
    if (exp == 0xFFu) {
        if (mant != 0u) {
            // NaN — truncate then force the qNaN bit (mantissa bit 6) so
            // the bf16 keeps a non-zero mantissa and doesn't collapse to ±Inf.
            return (bits >> 16u) | 0x40u;
        }
        // ±Inf — direct truncation is correct.
        return bits >> 16u;
    }
    // Round-to-nearest-even: add 0x7FFF + LSB of the result, then truncate.
    let lsb = (bits >> 16u) & 1u;
    let rounding = 0x7FFFu + lsb;
    let rounded = bits + rounding;
    return rounded >> 16u;
}

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
            tile_a[lid.y][lid.x] = bf16_load_a(idx);
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }

        if (b_row < k && col < n) {
            let idx = b_base + b_row * params.b_stride_row + col * params.b_stride_col;
            tile_b[lid.y][lid.x] = bf16_load_b(idx);
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
        let elem_idx = c_base + row * params.c_stride_row + col * params.c_stride_col;
        let pair_idx = elem_idx >> 1u;
        let half_idx = elem_idx & 1u;
        let bf16 = f32_to_bf16_bits(acc);
        let shifted = bf16 << (half_idx * 16u);
        atomicOr(&c[pair_idx], shifted);
    }
}
