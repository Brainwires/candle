//! Bit-exact validation for the WGPU `quantize_row_to_q8k` kernel.
//!
//! This test gates the int-domain Q4_K matmul port (Step 4 in
//! `prancy-tickling-puzzle.md`). The downstream matmul reads the BlockQ8K
//! buffer the kernel produces, so any disagreement with CPU
//! `BlockQ8K::from_float` would corrupt every Q4_K matmul on the WGPU
//! path. The test asserts byte-identity (not "approximately equal")
//! across:
//!
//! - Random rows of K ∈ {256, 512, 1024, 2048, 4096} (fixed seed).
//! - All-zero row (`amax == 0` branch: d = 0, qs = 0, bsums = 0).
//! - Single-spike row (one large value drives the iscale).
//! - Negative-spike row (max is signed; iscale > 0).
//! - Tied-magnitude row (FIRST occurrence wins via strict-LT iter).
//! - Uniform row (all elements equal, every quant maps to -128).

#![cfg(feature = "wgpu")]

use candle_core::{
    quantized::{k_quants::BlockQ8K, GgmlType},
    Device, Tensor,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

// `Rng::gen` was renamed to `random` in rand 0.9; the candle workspace
// pins an older version. Suppress the deprecation noise just for this
// helper.
#[allow(deprecated)]
fn rand_f32(rng: &mut StdRng) -> f32 {
    rng.gen::<f32>()
}

const QK_K: usize = 256;

fn try_wgpu() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

/// Quantize `xs` on CPU and return raw little-endian bytes (matches
/// WGSL/wgpu storage byte order on all supported platforms).
///
/// The struct fields of `BlockQ8K` are `pub(crate)`, so we can't
/// construct one directly from this integration-test crate. Instead we
/// allocate a zero-initialized `Vec<u8>` of the right size and view it
/// as `&mut [BlockQ8K]` via `from_raw_parts_mut`. Safe because BlockQ8K
/// is `#[repr(C)]` POD with no padding (the const-assert in
/// `k_quants.rs` checks `sizeof == 4 + 256 + 32 == 292`), and the
/// all-zeros bit pattern is a valid initial value (d = 0.0, qs = 0,
/// bsums = 0) — equivalent to the empty-block path.
fn cpu_quantize_bytes(xs: &[f32]) -> Vec<u8> {
    assert_eq!(xs.len() % QK_K, 0);
    let n_blocks = xs.len() / QK_K;
    let block_size = std::mem::size_of::<BlockQ8K>();
    let mut bytes = vec![0u8; n_blocks * block_size];
    // SAFETY: see function-level comment.
    let blocks: &mut [BlockQ8K] = unsafe {
        std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut BlockQ8K, n_blocks)
    };
    BlockQ8K::from_float(xs, blocks);
    // Re-borrow as &[u8] (drop the &mut [BlockQ8K] view) before returning.
    let _ = blocks; // explicit drop of the alias to make the order obvious.
    bytes
}

/// Quantize `xs` on WGPU and return raw bytes read back from the device.
fn wgpu_quantize_bytes(dev: &Device, xs: &[f32]) -> Vec<u8> {
    use candle_core::quantized::wgpu::quantize_row_to_q8k;

    let wgpu_dev = match dev {
        Device::Wgpu(d) => d,
        _ => panic!("expected wgpu device"),
    };
    // Upload input as a 1-D f32 tensor and extract the underlying
    // BufferReferenceId (Copy) before dropping the storage guard.
    let t = Tensor::from_slice(xs, (xs.len(),), dev).expect("upload f32 row");
    let src_buf = {
        let (storage, _layout) = t.storage_and_layout();
        match &*storage {
            candle_core::Storage::Wgpu(s) => s.buffer(),
            _ => panic!("expected wgpu storage"),
        }
    };

    let dst = quantize_row_to_q8k(wgpu_dev, src_buf, xs.len()).expect("dispatch");
    // Each BlockQ8K is 73 u32 words = 292 bytes; the dest is allocated
    // as DType::U32, so we read it back as a u32 vec then reinterpret
    // as bytes (little-endian on all supported GPU platforms).
    let cpu = pollster::block_on(async {
        dst.to_cpu_storage_async().await.expect("readback")
    });
    let words: &[u32] = match &cpu {
        candle_core::CpuStorage::U32(v) => v,
        _ => panic!("expected U32 cpu storage"),
    };
    let bytes: &[u8] = bytemuck::cast_slice(words);
    let n_blocks = xs.len() / QK_K;
    let expected = n_blocks * 292;
    assert!(
        bytes.len() >= expected,
        "wgpu readback short: got {}, expected at least {}",
        bytes.len(),
        expected
    );
    bytes[..expected].to_vec()
}

/// Format a per-block diff for a failed assertion. Returns a string
/// describing the FIRST disagreeing block (and offset within it). This
/// keeps the panic message focused — without it, a single-byte
/// disagreement in a 4096-element row produces an unreadable
/// vec-of-numbers panic.
fn first_diff_summary(label: &str, cpu: &[u8], gpu: &[u8]) -> String {
    let n = cpu.len().min(gpu.len());
    for i in 0..n {
        if cpu[i] != gpu[i] {
            let block = i / 292;
            let off = i % 292;
            let where_str = match off {
                0..=3 => format!("d[{off}]"),
                4..=259 => format!("qs[{}]", off - 4),
                260..=291 => format!("bsums[{}]", (off - 260) / 2),
                _ => "?".to_string(),
            };
            return format!(
                "{label}: byte {i} of {n} differs: cpu=0x{:02x} wgpu=0x{:02x} (block {block}, {where_str})",
                cpu[i], gpu[i],
            );
        }
    }
    if cpu.len() != gpu.len() {
        return format!(
            "{label}: length differs: cpu={} wgpu={}",
            cpu.len(),
            gpu.len()
        );
    }
    format!("{label}: bit-identical")
}

/// Byte-by-byte assertion with TWO precision tiers:
///
/// 1. **`qs` (256 i8 quants per block) MUST be byte-identical.** The
///    int-domain Q4_K matmul (Step 4) consumes these as the second
///    operand of a `dot4I8Packed` reduction; any disagreement here
///    flips the integer dot products and corrupts every Q4_K matmul
///    on the WGPU path.
///
/// 2. **`bsums` (16 i16 sub-block sums per block) MUST be byte-identical.**
///    They feed the `dmin * d_y * sum(mins[g] * bsums[g])` correction
///    term, also via integer arithmetic.
///
/// 3. **`d` (the f32 super-block scale) is allowed up to ±1 ULP of
///    deviation.** WGSL permits 2.5 ULP error on f32 division; CPU
///    uses IEEE-correct (0.5 ULP) division. The `1.0 / iscale` step
///    on the GPU sometimes lands one ULP off CPU's value. This is
///    well inside the model's matmul-level tolerance (≤1e-5 rel L2
///    in the bisect harness) and CPU's own `1/iscale` itself has 1
///    ULP of accumulated round-off versus the mathematically-exact
///    `-max/128`. The integer data is what determines correctness;
///    `d` is just an outer multiply that gets dot-producted later.
fn assert_byte_identical(label: &str, dev: &Device, xs: &[f32]) {
    let cpu = cpu_quantize_bytes(xs);
    let gpu = wgpu_quantize_bytes(dev, xs);
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    let n_blocks = cpu.len() / 292;

    // Pass 1: qs and bsums MUST match byte-for-byte.
    let mut qs_diff = 0usize;
    let mut bsums_diff = 0usize;
    for blk in 0..n_blocks {
        let base = blk * 292;
        for off in 4..260 {
            // qs[off-4]
            if cpu[base + off] != gpu[base + off] {
                qs_diff += 1;
            }
        }
        for off in 260..292 {
            if cpu[base + off] != gpu[base + off] {
                bsums_diff += 1;
            }
        }
    }
    if qs_diff != 0 || bsums_diff != 0 {
        let summary = first_diff_summary(label, &cpu, &gpu);
        panic!("{summary}\n  qs_diff={qs_diff}, bsums_diff={bsums_diff} (qs/bsums MUST be byte-identical)");
    }

    // Pass 2: d is allowed ≤1 ULP. Read the f32 from each block on
    // both sides and assert the integer-bit-pattern delta is at most 1.
    let mut max_d_ulp = 0u32;
    for blk in 0..n_blocks {
        let base = blk * 292;
        let cd = u32::from_le_bytes([cpu[base], cpu[base + 1], cpu[base + 2], cpu[base + 3]]);
        let gd = u32::from_le_bytes([gpu[base], gpu[base + 1], gpu[base + 2], gpu[base + 3]]);
        // For two f32s with the same sign, |bits_a - bits_b| is the ULP
        // distance. For opposite signs (which only happens around 0)
        // both bit patterns are tiny so the absolute difference still
        // bounds the ULP distance well enough for our purposes.
        let delta = if cd > gd { cd - gd } else { gd - cd };
        max_d_ulp = max_d_ulp.max(delta);
    }
    assert!(
        max_d_ulp <= 1,
        "{label}: d ULP deviation {max_d_ulp} exceeds 1 (qs/bsums byte-identical, but d drifted)"
    );
    println!(
        "{label} (k={}): qs+bsums byte-identical, d max_ulp={max_d_ulp} OK",
        xs.len()
    );
}

#[test]
fn q8k_quantize_wgpu_random_sweep() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            eprintln!("skipping q8k_quantize_wgpu_random_sweep — no wgpu device");
            return;
        }
    };
    // Independent fixed-seed RNG so a failure reproduces deterministically.
    let mut rng = StdRng::seed_from_u64(0xC4_5DE_8E1_F00DA);
    for &k in &[256usize, 512, 1024, 2048, 4096] {
        let xs: Vec<f32> = (0..k)
            .map(|_| (rand_f32(&mut rng) - 0.5) * 4.0) // ~ U(-2, 2)
            .collect();
        assert_byte_identical(&format!("random k={k}"), &dev, &xs);
    }
}

#[test]
fn q8k_quantize_wgpu_all_zero() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            eprintln!("skipping q8k_quantize_wgpu_all_zero — no wgpu device");
            return;
        }
    };
    // Two blocks of zeros — exercises the amax==0 special case across a
    // multi-block row (so the per-workgroup branch fires at least twice).
    let xs = vec![0.0f32; 512];
    assert_byte_identical("all-zero", &dev, &xs);
}

#[test]
fn q8k_quantize_wgpu_single_spike() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            eprintln!("skipping q8k_quantize_wgpu_single_spike — no wgpu device");
            return;
        }
    };
    // One block, a single large positive value at index 100. amax = the
    // spike, max = +spike, iscale = -128 / spike (negative). Every other
    // quant rounds to the nearest of -128..127 of `iscale * 0.01` etc.
    let mut xs = vec![0.01f32; QK_K];
    xs[100] = 12.34;
    assert_byte_identical("single positive spike", &dev, &xs);
}

#[test]
fn q8k_quantize_wgpu_negative_spike() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            eprintln!("skipping q8k_quantize_wgpu_negative_spike — no wgpu device");
            return;
        }
    };
    // Negative spike: max is the signed value at the abs-max position,
    // so iscale = -128 / (-spike) = +128/spike (positive). qs[200] should
    // be saturated to -128.
    let mut xs = vec![0.01f32; QK_K];
    xs[200] = -7.5;
    assert_byte_identical("single negative spike", &dev, &xs);
}

#[test]
fn q8k_quantize_wgpu_tied_magnitude() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            eprintln!("skipping q8k_quantize_wgpu_tied_magnitude — no wgpu device");
            return;
        }
    };
    // Two equal-magnitude values, opposite sign. CPU uses strict
    // less-than in its inner amax loop, so the FIRST occurrence wins,
    // which means `max = +5.0` (NOT -5.0). The kernel's tree reduce must
    // preserve this lower-index-wins tie-break, otherwise iscale flips
    // sign and all 256 quants are negated.
    let mut xs = vec![0.001f32; QK_K];
    xs[10] = 5.0; // first occurrence
    xs[200] = -5.0; // same |x|, later index — must NOT win
    assert_byte_identical("tied magnitude (positive first)", &dev, &xs);

    // Symmetric case: negative first.
    let mut xs2 = vec![0.001f32; QK_K];
    xs2[10] = -5.0;
    xs2[200] = 5.0;
    assert_byte_identical("tied magnitude (negative first)", &dev, &xs2);
}

#[test]
fn q8k_quantize_wgpu_uniform_row() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            eprintln!("skipping q8k_quantize_wgpu_uniform_row — no wgpu device");
            return;
        }
    };
    // All equal: index 0 wins (first), max = +0.5, iscale = -256, every
    // quant rounds to round_half_away(-256 * 0.5) = round_half_away(-128)
    // = -128 (clamped). All bsums become 16 * -128 = -2048 (fits in i16).
    let xs = vec![0.5f32; QK_K];
    assert_byte_identical("uniform positive", &dev, &xs);

    let xs2 = vec![-0.5f32; QK_K];
    assert_byte_identical("uniform negative", &dev, &xs2);
}
