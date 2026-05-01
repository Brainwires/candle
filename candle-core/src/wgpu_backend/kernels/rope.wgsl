// Rotary position embedding (Phase 3.6).
//
// Three flavors share this template; the host picks the right entry
// point and pipeline-cache key by substituting `__VARIANT__` and
// `__SCALAR_T__`.
//
// Inputs:
//   - src: shape (B, H, T, D) for `rope` / `rope_i`; (B, T, H, D) for
//     `rope_thd`.
//   - cos / sin: shape (T, D/2) when `unbatched_rope = 0`, or
//     (B, T, D/2) when `unbatched_rope = 1`.
// All inputs are contiguous (the host enforces this), so we can index
// into them as flat arrays.
//
// Output is contiguous, same shape and dtype as src.
//
// Each thread handles one (i_t, i_d) pair, writing two output elements
// per invocation. With workgroup_size 64 this works out to one
// dispatched group per 64 (i_t, i_d) pairs.

struct RopeMeta {
    n_pairs: u32,    // total pairs to process: B * H * T * (D/2)
    b: u32,
    h: u32,
    t: u32,
    d: u32,          // head_dim, must be even
    unbatched: u32,  // 1 if cos/sin are (B, T, D/2), else 0 (T, D/2)
    variant: u32,    // 0 = rope_i (interleaved), 1 = rope (split halves), 2 = rope_thd
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: RopeMeta;
@group(0) @binding(1) var<storage, read> src: array<__SCALAR_T__>;
@group(0) @binding(2) var<storage, read> cos_tab: array<__SCALAR_T__>;
@group(0) @binding(3) var<storage, read> sin_tab: array<__SCALAR_T__>;
@group(0) @binding(4) var<storage, read_write> dst: array<__SCALAR_T__>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let pair = gid.x + gid.y * num_wg.x * 64u;
    if (pair >= params.n_pairs) {
        return;
    }
    let dh = params.d / 2u;
    // Decompose the linear pair index into (b, h_or_t, t_or_h, i_d_half).
    // For variants 0/1 (rope_i, rope): outer loop order is (B, H, T, D/2).
    // For variant 2 (rope_thd):         outer loop order is (B, T, H, D/2).
    // We unify by keeping the "inner-most over T*D/2 within a (B, H) chunk"
    // for the first two, and "inner-most over H*D/2 within a (B, T) chunk"
    // for thd. Same total = B*H*T*(D/2).
    let i_d_half = pair % dh;
    var rest = pair / dh;
    var i_b: u32 = 0u;
    var i_h: u32 = 0u;
    var i_t: u32 = 0u;
    if (params.variant == 2u) {
        // (B, T, H, D/2)
        i_h = rest % params.h;
        rest = rest / params.h;
        i_t = rest % params.t;
        rest = rest / params.t;
        i_b = rest;
    } else {
        // (B, H, T, D/2)
        i_t = rest % params.t;
        rest = rest / params.t;
        i_h = rest % params.h;
        rest = rest / params.h;
        i_b = rest;
    }

    // src indexing: tensors are laid out as (B,H,T,D) for variants 0/1
    // and (B,T,H,D) for variant 2. Compute the two element offsets.
    var i1: u32 = 0u;
    var i2: u32 = 0u;
    if (params.variant == 0u) {
        // rope_i (interleaved pairs over D)
        let base = ((i_b * params.h + i_h) * params.t + i_t) * params.d;
        i1 = base + 2u * i_d_half;
        i2 = i1 + 1u;
    } else if (params.variant == 1u) {
        // rope (split halves over D)
        let base = ((i_b * params.h + i_h) * params.t + i_t) * params.d;
        i1 = base + i_d_half;
        i2 = i1 + dh;
    } else {
        // rope_thd: (B,T,H,D), split halves
        let base = ((i_b * params.t + i_t) * params.h + i_h) * params.d;
        i1 = base + i_d_half;
        i2 = i1 + dh;
    }

    // cos/sin index. Always (T, D/2) flat or (B, T, D/2) flat.
    var cs_idx: u32 = i_t * dh + i_d_half;
    if (params.unbatched == 1u) {
        cs_idx = cs_idx + i_b * params.t * dh;
    }
    let c = cos_tab[cs_idx];
    let s = sin_tab[cs_idx];
    let x1 = src[i1];
    let x2 = src[i2];
    dst[i1] = x1 * c - x2 * s;
    dst[i2] = x1 * s + x2 * c;
}
