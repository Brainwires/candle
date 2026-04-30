// BF16 → F32/U32/I32 cast kernel.
//
// Input is BF16 packed as `array<u32>` (2 BF16s per u32; low 16 bits =
// BF16[2k], high 16 bits = BF16[2k+1]). One thread per BF16 *element*.
//
// Substitution markers:
//   __DST_T__  → destination scalar type (f32, u32, or i32)

struct CastMeta {
    n_elements: u32,    // BF16 element count
    src_offset: u32,    // in BF16 elements (not u32 pairs)
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: CastMeta;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<__DST_T__>;

fn bf16_bits_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_elements {
        return;
    }
    let src_idx = params.src_offset + i;
    let pair_idx = src_idx >> 1u;
    let half_idx = src_idx & 1u;
    let packed = src[pair_idx];
    let bits = (packed >> (half_idx * 16u)) & 0xFFFFu;
    let f = bf16_bits_to_f32(bits);
    out[i] = __DST_T__(f);
}
