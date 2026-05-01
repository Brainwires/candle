// Strided bf16 copy kernel.
//
// Reads from a strided bf16 source view (input bound as `array<u32>`,
// two bf16 elements packed per u32) and writes contiguous bf16 output at
// a fixed element offset.
//
// Output is bound as `array<atomic<u32>>` and updated via a `atomicAnd` +
// `atomicOr` pair: the AND clears the half-word the current thread owns
// (preserving the *other* half-word, which may belong to either a sibling
// thread in this dispatch or to pre-existing buffer content outside the
// [dst_offset, dst_offset + n_elements) write range), the OR fills the
// owned half-word in. This pattern is safe under any interleaving of
// concurrent threads writing different halves of the same u32:
//
//   - A.AND clears low half ; A.OR sets low half
//   - B.AND clears high half; B.OR sets high half
//
// Whether A's pair lands before B's, after B's, or interleaved with B's,
// the final u32 ends up `A_bits | (B_bits << 16)` — the correct packed
// result. No round-trip through f32 is needed: copy preserves bit
// patterns, so we shuffle the raw 16-bit bf16 fields directly.

struct CopyMeta {
    n_elements: u32,
    rank: u32,
    src_offset: u32,
    dst_offset: u32,
    shape: vec4<u32>,
    src_strides: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: CopyMeta;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<atomic<u32>>;

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
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let i = gid.x + gid.y * num_wg.x * 64u;
    if (i >= params.n_elements) {
        return;
    }

    let s_elem = unravel_src(i);
    let s_pair = s_elem >> 1u;
    let s_half = s_elem & 1u;
    let s_packed = src[s_pair];
    let bits = (s_packed >> (s_half * 16u)) & 0xFFFFu;

    let d_elem = params.dst_offset + i;
    let d_pair = d_elem >> 1u;
    let d_half = d_elem & 1u;
    let mask = 0xFFFFu << (d_half * 16u);
    let clear_mask = ~mask;
    let shifted = bits << (d_half * 16u);
    atomicAnd(&dst[d_pair], clear_mask);
    atomicOr(&dst[d_pair], shifted);
}
