# Flash Attention Decode Kernel — Design Notes

Phase 2 of the chat-pwa local Gemma 4 perf overhaul plan. **Status:**
not yet implemented; this file lays out the kernel design + integration
plan so a follow-up commit can land it cleanly.

## Goal

For decode (`q_seq == 1`, the common case in chat), replace the existing
3-dispatch attention path:

```text
Q.matmul(K.transpose(2,3))   → attn_weights [B, H, 1, kv_len]
attn_weights.broadcast_add(mask)
softmax_last_dim(attn_weights)
attn_weights.matmul(V)        → output [B, H, 1, head_dim]
```

with a single fused kernel that materializes `attn_weights` only in
shared memory (online softmax, Welford-style `m, l, O` accumulators) and
writes `[B, H, 1, head_dim]` directly. Saves 2 `queue.write_buffer` round
trips + the broadcast_add mask buffer + intermediate global-memory writes.

Projected win on chat-pwa (AMD GCN-4 / Vulkan WebGPU): **~2× decode**.

## Kernel layout

`flash_attn_decode.pwgsl`:

```wgsl
#include "util.pwgsl"

LOAD_INFO(op_flash_attn, 8)
#define op_flash_attn_b              op_flash_attn[0]
#define op_flash_attn_num_heads      op_flash_attn[1]
#define op_flash_attn_num_kv_heads   op_flash_attn[2]
#define op_flash_attn_head_dim       op_flash_attn[3]
#define op_flash_attn_kv_len         op_flash_attn[4]
#define op_flash_attn_sliding_window op_flash_attn[5]   // u32::MAX = no sliding
#define op_flash_attn_q_offset       op_flash_attn[6]
#define op_flash_attn_kv_offset      op_flash_attn[7]

// Inputs:
//   v_input1 = Q [B, num_heads, 1, head_dim]
//   v_input2 = K [B, num_kv_heads, kv_len, head_dim]
//   v_input3 = V [B, num_kv_heads, kv_len, head_dim]
// Output:
//   v_dest   = O [B, num_heads, 1, head_dim]

// One workgroup per (batch, head). Workgroup size 64; each thread
// handles head_dim/64 channels (= 4 for Gemma 4 head_dim=256).
//
// Online softmax accumulators: shared across the workgroup.
//   sharedM: running max of scores seen so far
//   sharedL: running sum of exp(score - sharedM)
//   sharedO[head_dim]: running output O = sum(exp(score - m) * V[i])
//
// Loop over kv_len in tiles. Per tile:
//   1. Each thread loads its channel of Q (broadcasts across kv tile).
//   2. For each kv position in tile:
//      a. Each thread computes partial dot product Q[c] * K[i, c].
//      b. Workgroup reduce (parallel sum across 64 threads) → score[i].
//      c. Apply causal / sliding mask. Sliding: kv_idx + sliding_window <
//         q_offset → score = -inf.
//      d. new_m = max(sharedM, score[i]).
//      e. l_scale = exp(sharedM - new_m).
//      f. sharedL = sharedL * l_scale + exp(score[i] - new_m).
//      g. Each thread updates sharedO[c] = sharedO[c] * l_scale +
//         exp(score[i] - new_m) * V[i, c].
//      h. sharedM = new_m.
// 3. Final: each thread writes O[c] / sharedL to v_dest.

@compute
@workgroup_size(64, 1, 1)
fn flash_attn_decode(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // … see algorithm above, ~250 LOC
}
```

## Integration touchpoints

1. **Kernel source:**
   `candle-wgpu-kernels/src/kernels/flash_attn_decode.pwgsl` (new file,
   ~250 lines).
2. **Pipeline enum:** add `Pipelines::FlashAttnDecode(...)` to
   `candle-wgpu-kernels/src/lib.rs` so the dispatch glue can address it.
   The codegen pipeline needs to emit `flash_attn_decode.pwgsl_generated_*.wgsl`
   variants for f32 / f16 / bf16 (the `DTYPE` macro).
3. **Dispatch wrapper:**
   `candle-core/src/wgpu_backend/wgpu_functions/flash_attn.rs` (new
   file, ~150 lines). Exposes a `flash_attn_decode_fwd(dev, q, k, v,
   mask_window, output)` that:
   - Builds the meta buffer (8 u32 fields).
   - Looks up the FlashAttnDecode pipeline.
   - Creates a 3-input bind group.
   - Enqueues a `dispatch_workgroups(b * num_heads, 1, 1)` call.
4. **Routing in `Attention::forward`:** in
   `candle-transformers/src/models/gemma4/text.rs` (and the new
   `quantized_gemma4.rs`), replace the 3-dispatch decode path with
   the fused call **only when `q_len == 1`**. Prefill (`q_len > 1`)
   keeps the existing 3-dispatch path — the fused kernel is
   vector-form only (per candle PR #3479's lesson).
5. **Storage layout invariants:** `q.shape == [b, num_heads, 1, head_dim]`,
   `k.shape == [b, num_kv_heads, kv_len, head_dim]` (after `repeat_kv`
   to expand to num_heads, since the kernel can't do repeat itself),
   `v` matching `k`. Q/K/V must all be contiguous; the wgpu_backend
   to_dtype kernel's strided-source restriction applies.

## Verification plan

The intra-self_attn diag scaffold already in place
(`gemma4/text.rs::Attention::forward` with the `intra_hook` parameter)
emits `attn_weights_pre_mask`, `_post_mask`, `_post_softmax`,
`attn_after_v_matmul`. The fused kernel won't produce those intermediate
tensors directly, but the final `attn_after_v_matmul` output is what
flows downstream. Compare bit-for-bit to the 3-dispatch path on Mac
Metal first (deterministic adapter, no AMD-specific quirks); only after
that holds compare to AMD/Vulkan.

The chat-pwa diag protocol (set `globalThis.__bw_diag = true;`,
`__bw_diag_layer = 15;` then run a known prompt) gives us per-layer
checkpoints to bisect any divergence.

## Why this isn't shipped yet

- WGSL kernel work has no good test rig short of running on hardware
  + diffing against a reference.
- Without an AMD GCN-4 box in CI we can't validate the wave-64
  alignment assumptions. Existing M=1 matmul kernels in PR #3379 are
  already a strong baseline; flash-attn improves on that with one less
  dispatch + zero intermediate global-memory writes for the
  attention block.
- Online softmax is correct but bf16 underflow at large `kv_len` is a
  real risk. The kernel needs explicit f32 accumulators for `sharedM`,
  `sharedL`, and `sharedO`, with a final f32 → bf16 cast on output.
- Subgroup ops (`subgroupAdd`) for the reduction would significantly
  speed up the dot-product reduction step but require gating on the
  `subgroups` extension being available, which not every adapter
  supports.

## Out of scope here

- Prefill flash attention (q_len > 1) — different memory access
  pattern, larger workgroups, separate kernel.
- Tile sizes / `@workgroup_size` tuning per architecture — needs real
  profiling on AMD GCN, NVIDIA SM, Apple GPU.
- Causal mask materialization — the kernel checks `kv_idx > q_offset` /
  `kv_idx + sliding_window < q_offset` inline, so the upstream
  `prepare_attention_mask` / `prepare_sliding_attention_mask` calls
  can be skipped for q_len == 1.
