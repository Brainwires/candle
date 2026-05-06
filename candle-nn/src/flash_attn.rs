//! Fused decode-attention (q_seq == 1) — Phase 2 of the chat-pwa
//! local Gemma 4 perf overhaul plan.
//!
//! Replaces the 3-dispatch attention path of:
//!
//! ```text
//! attn_weights = Q @ K.T              # [B, H, 1, kv_len]
//! attn_weights += mask                 # causal / sliding
//! attn_weights = softmax(attn_weights, dim=-1)
//! attn_output  = attn_weights @ V      # [B, H, 1, head_dim]
//! ```
//!
//! with a single workgroup-per-(batch, head) WGSL kernel that uses
//! online softmax (Welford-style m / l / O accumulators) — the
//! attn-weights tensor never materializes in global memory, the
//! mask is applied implicitly, and we save 2 dispatches plus an
//! intermediate global-memory write.
//!
//! This is decode-only: `q_len == 1`. The prefill case
//! (`q_len > 1`) keeps using the standard 3-step path.
//!
//! CPU fallback below is the 3-step path implemented in pure Rust;
//! it exists so models that build on this primitive remain runnable
//! on non-WGPU devices and so unit tests on CPU can validate the
//! WGPU kernel by diff.

use candle::{CpuStorage, Layout, Result, Shape, Tensor};

/// Online-softmax fused attention for decode (q_len == 1).
///
/// Mask is implicit: causal + optional sliding window. `q_pos` is
/// taken to be `kv_len - 1` (the latest token), which is the only
/// case where this op is correct.
#[derive(Debug, Clone, Copy)]
struct FlashAttnDecode {
    sliding_window: Option<u32>,
}

impl candle::CustomOp3 for FlashAttnDecode {
    fn name(&self) -> &'static str {
        "flash-attn-decode"
    }

    fn cpu_fwd(
        &self,
        sq: &CpuStorage,
        lq: &Layout,
        sk: &CpuStorage,
        lk: &Layout,
        sv: &CpuStorage,
        lv: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // Reference implementation: full softmax in f32, no online
        // accumulator trickery. Used only as a CPU fallback / golden
        // reference for the WGPU kernel.
        fn inner(
            q: &[f32],
            lq: &Layout,
            k: &[f32],
            lk: &Layout,
            v: &[f32],
            lv: &Layout,
            sliding_window: Option<u32>,
        ) -> Result<(CpuStorage, Shape)> {
            let (b, h, q_len, d) = lq.shape().dims4()?;
            if q_len != 1 {
                candle::bail!("flash_attn_decode: q_len must be 1, got {q_len}");
            }
            let (bk, hk, kv_len, dk) = lk.shape().dims4()?;
            let (bv, hv, kvv, dv) = lv.shape().dims4()?;
            if (b, h, d) != (bk, hk, dk) || (b, h, d) != (bv, hv, dv) || kv_len != kvv {
                candle::bail!(
                    "flash_attn_decode: shape mismatch q={:?} k={:?} v={:?}",
                    lq.shape(),
                    lk.shape(),
                    lv.shape()
                );
            }
            let (oq, _) = lq
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::Msg("Q must be contiguous".into()))?;
            let (ok, _) = lk
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::Msg("K must be contiguous".into()))?;
            let (ovv, _) = lv
                .contiguous_offsets()
                .ok_or_else(|| candle::Error::Msg("V must be contiguous".into()))?;

            let q_pos = (kv_len - 1) as u32;
            let mut out = vec![0f32; b * h * d];
            for bi in 0..b {
                for hi in 0..h {
                    let q_base = oq + (bi * h + hi) * d;
                    let kv_head_base = (bi * h + hi) * kv_len * d;
                    let k_head = &k[ok + kv_head_base..ok + kv_head_base + kv_len * d];
                    let v_head = &v[ovv + kv_head_base..ovv + kv_head_base + kv_len * d];
                    let q_vec = &q[q_base..q_base + d];

                    // Compute scores[i] = Q · K[i]
                    let mut scores = vec![0f32; kv_len];
                    for kvi in 0..kv_len {
                        let causal = (kvi as u32) > q_pos;
                        let sliding = matches!(sliding_window, Some(w) if kvi as u32 + w <= q_pos);
                        if causal || sliding {
                            scores[kvi] = f32::NEG_INFINITY;
                            continue;
                        }
                        let mut s = 0f32;
                        for c in 0..d {
                            s += q_vec[c] * k_head[kvi * d + c];
                        }
                        scores[kvi] = s;
                    }

                    // Stable softmax.
                    let mut m = f32::NEG_INFINITY;
                    for &s in &scores {
                        if s > m {
                            m = s;
                        }
                    }
                    let mut sum = 0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    let inv_sum = 1.0 / sum;

                    // out[hi, c] = sum_kvi softmax(scores)[kvi] * V[kvi, c]
                    let out_base = (bi * h + hi) * d;
                    for c in 0..d {
                        let mut acc = 0f32;
                        for kvi in 0..kv_len {
                            acc += scores[kvi] * inv_sum * v_head[kvi * d + c];
                        }
                        out[out_base + c] = acc;
                    }
                }
            }
            let storage = candle::WithDType::to_cpu_storage_owned(out);
            Ok((storage, (b, h, q_len, d).into()))
        }

        use candle::backend::BackendStorage;
        match (sq, sk, sv) {
            (CpuStorage::F32(q), CpuStorage::F32(k), CpuStorage::F32(v)) => {
                inner(q, lq, k, lk, v, lv, self.sliding_window)
            }
            _ => candle::bail!(
                "flash_attn_decode: only f32 is supported for now (got {:?} / {:?} / {:?})",
                sq.dtype(),
                sk.dtype(),
                sv.dtype()
            ),
        }
    }

    #[cfg(feature = "wgpu")]
    fn wgpu_fwd(
        &self,
        sq: &candle::WgpuStorage,
        lq: &Layout,
        sk: &candle::WgpuStorage,
        lk: &Layout,
        sv: &candle::WgpuStorage,
        lv: &Layout,
    ) -> Result<(candle::WgpuStorage, Shape)> {
        use candle::wgpu::wgpu_functions;

        if sq.dtype() != sk.dtype() || sq.dtype() != sv.dtype() {
            candle::bail!(
                "flash_attn_decode: dtype mismatch q={:?} k={:?} v={:?}",
                sq.dtype(),
                sk.dtype(),
                sv.dtype()
            );
        }
        if !lq.is_contiguous() || !lk.is_contiguous() || !lv.is_contiguous() {
            candle::bail!("flash_attn_decode: Q/K/V must all be contiguous on WGPU");
        }

        let (b, h, q_len, d) = lq.shape().dims4()?;
        if q_len != 1 {
            candle::bail!("flash_attn_decode: q_len must be 1 (got {q_len})");
        }
        let (_, _, kv_len, dk) = lk.shape().dims4()?;
        let (_, _, kv_len_v, dv) = lv.shape().dims4()?;
        if dk != d || dv != d || kv_len_v != kv_len {
            candle::bail!(
                "flash_attn_decode: shape mismatch q={:?} k={:?} v={:?}",
                lq.shape(),
                lk.shape(),
                lv.shape()
            );
        }

        let el_out = b * h * q_len * d;
        let output = sq.device().alloc_uninit_size(sq.dtype(), el_out);

        wgpu_functions::queue_flash_attn_decode(
            sq.device(),
            output.buffer(),
            (sq.buffer(), lq.start_offset() as u32),
            (sk.buffer(), lk.start_offset() as u32),
            (sv.buffer(), lv.start_offset() as u32),
            sq.dtype(),
            b as u32,
            h as u32,
            h as u32, // num_kv_heads — caller is expected to pre-expand K/V
            d as u32,
            kv_len as u32,
            self.sliding_window,
        )?;

        Ok((output, (b, h, q_len, d).into()))
    }
}

/// Fused decode-attention (q_len == 1).
///
/// `q.shape == [B, H, 1, D]`, `k.shape == v.shape == [B, H, kv_len, D]`
/// (K and V must already be expanded to `H` heads — callers using GQA
/// must apply `repeat_kv` first, since the kernel doesn't repeat).
///
/// Returns `[B, H, 1, D]`.
///
/// `sliding_window`:
///   - `None` → full attention (causal-only).
///   - `Some(w)` → sliding window: position `kvi` is masked when
///     `kvi + w <= kv_len - 1`.
pub fn flash_attn_decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sliding_window: Option<u32>,
) -> Result<Tensor> {
    let (b, h, q_len, d) = q.dims4()?;
    if q_len != 1 {
        candle::bail!(
            "flash_attn_decode: requires q_len == 1, got {:?}",
            q.shape()
        );
    }
    let (bk, hk, _, dk) = k.dims4()?;
    let (bv, hv, _, dv) = v.dims4()?;
    if (b, h, d) != (bk, hk, dk) || (b, h, d) != (bv, hv, dv) {
        candle::bail!(
            "flash_attn_decode: shape mismatch q={:?} k={:?} v={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
        candle::bail!("flash_attn_decode: Q/K/V must all be contiguous");
    }
    q.apply_op3_no_bwd(k, v, &FlashAttnDecode { sliding_window })
}
