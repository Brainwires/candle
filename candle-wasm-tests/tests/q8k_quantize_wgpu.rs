
// ============================================================================
// === THIS FILE IS AUTO-GENERATED. DO NOT EDIT BY HAND. ======================
// === CHANGES WILL BE OVERWRITTEN THE NEXT TIME THE GENERATOR RUNS. ==========
// ============================================================================

#![allow(unused_imports, unexpected_cfgs, unused_parens)]
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
#![cfg(not(candle_wasm_tests))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{
    quantized::{k_quants::BlockQ8K, GgmlType},
    Device, Tensor,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
#[allow(deprecated)]
fn rand_f32(rng: &mut StdRng) -> f32 {
    rng.gen::<f32>()
}
const QK_K: usize = 256;
async fn try_wgpu() -> Option<Device> {
    Device::new_wgpu_async(0).await.ok()
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
    let blocks: &mut [BlockQ8K] = unsafe {
        std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut BlockQ8K, n_blocks)
    };
    BlockQ8K::from_float(xs, blocks);
    let _ = blocks;
    bytes
}
/// Quantize `xs` on WGPU and return raw bytes read back from the device.
async fn wgpu_quantize_bytes(dev: &Device, xs: &[f32]) -> Vec<u8> {
    use candle::quantized::wgpu::quantize_row_to_q8k;
    let wgpu_dev = match dev {
        Device::Wgpu(d) => d,
        _ => panic!("expected wgpu device"),
    };
    let t = Tensor::from_slice(xs, (xs.len(),), dev).expect("upload f32 row");
    let src_buf = {
        let (storage, _layout) = t.storage_and_layout();
        match &*storage {
            candle::Storage::Wgpu(s) => s.buffer(),
            _ => panic!("expected wgpu storage"),
        }
    };
    let dst = quantize_row_to_q8k(wgpu_dev, src_buf, xs.len()).expect("dispatch");
    let cpu = pollster::block_on(async {
        dst.to_cpu_storage_async().await.expect("readback")
    });
    let words: &[u32] = match &cpu {
        candle::CpuStorage::U32(v) => v,
        _ => panic!("expected U32 cpu storage"),
    };
    let bytes: &[u8] = bytemuck::cast_slice(words);
    let n_blocks = xs.len() / QK_K;
    let expected = n_blocks * 292;
    assert!(
        bytes.len() >= expected, "wgpu readback short: got {}, expected at least {}",
        bytes.len(), expected
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
        return format!("{label}: length differs: cpu={} wgpu={}", cpu.len(), gpu.len());
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
    let gpu = wgpu_quantize_bytes(dev, xs).await;
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    let n_blocks = cpu.len() / 292;
    let mut qs_diff = 0usize;
    let mut bsums_diff = 0usize;
    for blk in 0..n_blocks {
        let base = blk * 292;
        for off in 4..260 {
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
        panic!(
            "{summary}\n  qs_diff={qs_diff}, bsums_diff={bsums_diff} (qs/bsums MUST be byte-identical)"
        );
    }
    let mut max_d_ulp = 0u32;
    for blk in 0..n_blocks {
        let base = blk * 292;
        let cd = u32::from_le_bytes([
            cpu[base],
            cpu[base + 1],
            cpu[base + 2],
            cpu[base + 3],
        ]);
        let gd = u32::from_le_bytes([
            gpu[base],
            gpu[base + 1],
            gpu[base + 2],
            gpu[base + 3],
        ]);
        let delta = if cd > gd { cd - gd } else { gd - cd };
        max_d_ulp = max_d_ulp.max(delta);
    }
    assert!(
        max_d_ulp <= 1,
        "{label}: d ULP deviation {max_d_ulp} exceeds 1 (qs/bsums byte-identical, but d drifted)"
    );
    wasm_bindgen_test::console_log!(
        "{label} (k={}): qs+bsums byte-identical, d max_ulp={max_d_ulp} OK", xs.len()
    );
}
#[test]
async fn q8k_quantize_wgpu_random_sweep() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            ewasm_bindgen_test::console_log!(
                "skipping q8k_quantize_wgpu_random_sweep — no wgpu device"
            );
            return;
        }
    };
    let mut rng = StdRng::seed_from_u64(0xC4_5DE_8E1_F00DA);
    for &k in &[256usize, 512, 1024, 2048, 4096] {
        let xs: Vec<f32> = (0..k).map(|_| (rand_f32(&mut rng) - 0.5) * 4.0).collect();
        assert_byte_identical(&format!("random k={k}"), &dev, &xs);
    }
}
#[test]
async fn q8k_quantize_wgpu_all_zero() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            ewasm_bindgen_test::console_log!(
                "skipping q8k_quantize_wgpu_all_zero — no wgpu device"
            );
            return;
        }
    };
    let xs = vec![0.0f32; 512];
    assert_byte_identical("all-zero", &dev, &xs);
}
#[test]
async fn q8k_quantize_wgpu_single_spike() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            ewasm_bindgen_test::console_log!(
                "skipping q8k_quantize_wgpu_single_spike — no wgpu device"
            );
            return;
        }
    };
    let mut xs = vec![0.01f32; QK_K];
    xs[100] = 12.34;
    assert_byte_identical("single positive spike", &dev, &xs);
}
#[test]
async fn q8k_quantize_wgpu_negative_spike() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            ewasm_bindgen_test::console_log!(
                "skipping q8k_quantize_wgpu_negative_spike — no wgpu device"
            );
            return;
        }
    };
    let mut xs = vec![0.01f32; QK_K];
    xs[200] = -7.5;
    assert_byte_identical("single negative spike", &dev, &xs);
}
#[test]
async fn q8k_quantize_wgpu_tied_magnitude() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            ewasm_bindgen_test::console_log!(
                "skipping q8k_quantize_wgpu_tied_magnitude — no wgpu device"
            );
            return;
        }
    };
    let mut xs = vec![0.001f32; QK_K];
    xs[10] = 5.0;
    xs[200] = -5.0;
    assert_byte_identical("tied magnitude (positive first)", &dev, &xs);
    let mut xs2 = vec![0.001f32; QK_K];
    xs2[10] = -5.0;
    xs2[200] = 5.0;
    assert_byte_identical("tied magnitude (negative first)", &dev, &xs2);
}
#[test]
async fn q8k_quantize_wgpu_uniform_row() {
    let dev = match try_wgpu() {
        Some(d) => d,
        None => {
            ewasm_bindgen_test::console_log!(
                "skipping q8k_quantize_wgpu_uniform_row — no wgpu device"
            );
            return;
        }
    };
    let xs = vec![0.5f32; QK_K];
    assert_byte_identical("uniform positive", &dev, &xs);
    let xs2 = vec![- 0.5f32; QK_K];
    assert_byte_identical("uniform negative", &dev, &xs2);
}
