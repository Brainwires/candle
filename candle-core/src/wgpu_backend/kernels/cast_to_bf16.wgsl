// F32/U32 → BF16 cast kernel.
//
// WGSL has no native BF16 type, so BF16 storage is packed as `array<u32>`
// where each u32 holds two BF16 values: BF16[2k] in the low 16 bits and
// BF16[2k+1] in the high 16 bits. One thread per BF16 *pair* — total
// dispatch count is `(n_elements + 1) / 2`.
//
// Substitution markers:
//   __SRC_T__  → source scalar type (f32 or u32)

struct CastMeta {
    n_elements: u32,    // BF16 element count
    src_offset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: CastMeta;
@group(0) @binding(1) var<storage, read> src: array<__SRC_T__>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

// Convert one f32 to BF16 bits using round-to-nearest-even.
fn f32_to_bf16_bits(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    // NaN: preserve the NaN flag, set a non-zero mantissa.
    if (bits & 0x7FFFFFFFu) > 0x7F800000u {
        return ((bits >> 16u) | 0x40u) & 0xFFFFu;
    }
    let lsb = (bits >> 16u) & 1u;
    let rounding_bias = 0x7FFFu + lsb;
    return ((bits + rounding_bias) >> 16u) & 0xFFFFu;
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let pair_idx = gid.x + gid.y * num_wg.x * 64u;
    let i = pair_idx * 2u;
    if i >= params.n_elements {
        return;
    }

    let lo_src = src[params.src_offset + i];
    let lo_f32 = f32(lo_src);
    let lo_bits = f32_to_bf16_bits(lo_f32);

    var packed: u32 = lo_bits;
    if (i + 1u) < params.n_elements {
        let hi_src = src[params.src_offset + i + 1u];
        let hi_f32 = f32(hi_src);
        let hi_bits = f32_to_bf16_bits(hi_f32);
        packed = lo_bits | (hi_bits << 16u);
    }
    out[pair_idx] = packed;
}
