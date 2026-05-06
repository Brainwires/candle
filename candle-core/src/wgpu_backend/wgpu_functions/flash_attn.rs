use super::*;

/// Fused decode-attention dispatch (`q_seq == 1` only).
///
/// Replaces the 3-dispatch decode path of `Q @ K.T → softmax(+ mask) → @ V`
/// with a single workgroup-per-(batch, head) kernel that computes online
/// softmax on the fly and never materializes the `[B, H, 1, kv_len]`
/// attn-weights tensor in global memory.
///
/// Layout invariants:
///   q.shape == [B, num_heads,    1,      head_dim]
///   k.shape == [B, num_heads,    kv_len, head_dim]   (post repeat_kv)
///   v.shape == [B, num_heads,    kv_len, head_dim]   (post repeat_kv)
///   out shape [B, num_heads,    1,      head_dim]
///
/// All four buffers must be contiguous and contain f32 data. K and V must
/// already be expanded to `num_heads` along the head axis (the kernel
/// does not handle GQA repeat itself).
///
/// `sliding_window`:
///   - `None` → full attention (causal only).
///   - `Some(w)` → sliding window: kv_idx is masked when
///     `kv_idx + w <= q_pos`.
#[allow(clippy::too_many_arguments)]
pub fn queue_flash_attn_decode(
    dev: &WgpuDevice,
    buffer_dest: BufferReferenceId,
    buffer_q: (BufferReferenceId, u32), // (buffer, start_offset)
    buffer_k: (BufferReferenceId, u32),
    buffer_v: (BufferReferenceId, u32),
    dtype: crate::DType,
    b: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_len: u32,
    sliding_window: Option<u32>,
) -> crate::Result<()> {
    debug_assert!(
        head_dim % 64 == 0,
        "flash_attn_decode requires head_dim divisible by 64 (got {})",
        head_dim
    );
    debug_assert!(
        head_dim <= 1024,
        "flash_attn_decode kernel statically caps head_dim at 1024 (got {})",
        head_dim
    );
    debug_assert!(
        kv_len > 0,
        "flash_attn_decode requires kv_len > 0 (got 0; nothing to attend to)"
    );

    let (buffer_q, q_offset) = buffer_q;
    let (buffer_k, k_offset) = buffer_k;
    let (buffer_v, v_offset) = buffer_v;
    debug_assert_eq!(
        k_offset, v_offset,
        "flash_attn_decode expects K and V to share the same start_offset \
         within their (separate) buffers; got k={k_offset} v={v_offset}"
    );

    let mut queue = dev.get_queue();

    // op_flash_attn meta layout (matches the kernel #defines):
    //   [0] b                  [4] kv_len
    //   [1] num_heads          [5] sliding_window (u32::MAX = none)
    //   [2] num_kv_heads       [6] q_offset
    //   [3] head_dim           [7] kv_offset (k & v share offset)
    queue.add(b);
    queue.add(num_heads);
    queue.add(num_kv_heads);
    queue.add(head_dim);
    queue.add(kv_len);
    queue.add(sliding_window.unwrap_or(u32::MAX));
    queue.add(q_offset);
    queue.add(k_offset);

    let pipeline = queue.get_pipeline(Pipelines::FlashAttnDecode(
        dev.get_dtype(dtype)?,
        candle_wgpu_kernels::flash_attn_decode::Functions::FlashAttnDecode,
    ));

    // 4-buffer bind group: dest + Q (input1) + K (input2) + V (input3).
    let bind_group = dev.create_bind_group_input3(
        buffer_dest,
        buffer_q,
        buffer_k,
        buffer_v,
        dtype.into(),
    );

    // One workgroup per (batch, head).
    queue.enqueue_workgroups(
        pipeline,
        bind_group,
        b * num_heads,
        1,
        1,
        (b * num_heads * head_dim) as usize,
    );

    Ok(())
}
