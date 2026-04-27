//! WebGPU rotary-embedding + causal-attention correctness (Phase 3.6).
//!
//! Cross-checks the three RoPE variants and a small composed
//! scaled-dot-product-attention block (q@k.T → scale → mask add → softmax
//! → @v) against the candle CPU reference. Composition reuses the
//! Phase 3.3–3.5 primitives (matmul, affine, softmax) with the new
//! Phase 3.6 RoPE kernel.
//!
//! Tolerance: 1e-5 for RoPE alone, 1e-3 for the full attention block
//! (composition accumulates float drift through three matmuls and a
//! softmax).
//!
//! Skips when no Wgpu adapter is available.
//!
//! Run: `cargo test -p candle-nn --features wgpu --test wgpu_attention`
#![cfg(feature = "wgpu")]

use candle::{DType, Device, Result, Tensor, D};

fn try_wgpu() -> Option<Device> {
    Device::new_wgpu(0).ok()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_diff = 0f32;
    let mut idx = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
            idx = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{label}: max abs diff {max_diff} > tol {tol} at index {idx} ({} vs {})",
        a[idx],
        b[idx]
    );
}

/// Build a deterministic (B, H, T, D) f32 tensor for `xs` and the
/// matching (T, D/2) cos/sin tables.
fn make_qcs(
    b: usize,
    h: usize,
    t: usize,
    d: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let xs_data: Vec<f32> = (0..b * h * t * d)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let cos_data: Vec<f32> = (0..t * d / 2).map(|i| ((i as f32) * 0.05).cos()).collect();
    let sin_data: Vec<f32> = (0..t * d / 2).map(|i| ((i as f32) * 0.05).sin()).collect();
    let xs = Tensor::from_slice(&xs_data, (b, h, t, d), dev)?;
    let cos = Tensor::from_slice(&cos_data, (t, d / 2), dev)?;
    let sin = Tensor::from_slice(&sin_data, (t, d / 2), dev)?;
    Ok((xs, cos, sin))
}

#[test]
fn rope_interleaved_f32() -> Result<()> {
    let Some(dev) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return Ok(());
    };
    let (b, h, t, d) = (1, 2, 4, 8);
    let cpu = Device::Cpu;
    let (xs_c, cos_c, sin_c) = make_qcs(b, h, t, d, &cpu)?;
    let (xs_g, cos_g, sin_g) = make_qcs(b, h, t, d, &dev)?;

    let want = candle_nn::rotary_emb::rope_i(&xs_c, &cos_c, &sin_c)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let got = candle_nn::rotary_emb::rope_i(&xs_g, &cos_g, &sin_g)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_close(&want, &got, 1e-5, "rope_i");
    Ok(())
}

#[test]
fn rope_non_interleaved_f32() -> Result<()> {
    let Some(dev) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return Ok(());
    };
    let (b, h, t, d) = (1, 2, 4, 8);
    let cpu = Device::Cpu;
    let (xs_c, cos_c, sin_c) = make_qcs(b, h, t, d, &cpu)?;
    let (xs_g, cos_g, sin_g) = make_qcs(b, h, t, d, &dev)?;

    let want = candle_nn::rotary_emb::rope(&xs_c, &cos_c, &sin_c)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let got = candle_nn::rotary_emb::rope(&xs_g, &cos_g, &sin_g)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_close(&want, &got, 1e-5, "rope");
    Ok(())
}

#[test]
fn rope_thd_f32() -> Result<()> {
    let Some(dev) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return Ok(());
    };
    // rope_thd expects (B, T, H, D)
    let (b, t, h, d) = (1, 4, 2, 8);
    let cpu = Device::Cpu;
    let make = |dev: &Device| -> Result<(Tensor, Tensor, Tensor)> {
        let xs_data: Vec<f32> = (0..b * t * h * d)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let cos_data: Vec<f32> = (0..t * d / 2).map(|i| ((i as f32) * 0.05).cos()).collect();
        let sin_data: Vec<f32> = (0..t * d / 2).map(|i| ((i as f32) * 0.05).sin()).collect();
        Ok((
            Tensor::from_slice(&xs_data, (b, t, h, d), dev)?,
            Tensor::from_slice(&cos_data, (t, d / 2), dev)?,
            Tensor::from_slice(&sin_data, (t, d / 2), dev)?,
        ))
    };
    let (xs_c, cos_c, sin_c) = make(&cpu)?;
    let (xs_g, cos_g, sin_g) = make(&dev)?;

    let want = candle_nn::rotary_emb::rope_thd(&xs_c, &cos_c, &sin_c)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let got = candle_nn::rotary_emb::rope_thd(&xs_g, &cos_g, &sin_g)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_close(&want, &got, 1e-5, "rope_thd");
    Ok(())
}

/// Composed softmax over the last dim (no fused kernel on Wgpu yet —
/// per Phase 3.5's plan, softmax composes from sum/exp/broadcast). We
/// use the same composition the Phase 3.5 reduce tests verified:
///   y = exp(x - max) / sum(exp(x - max))
fn softmax_last_dim_via_composition(xs: &Tensor) -> Result<Tensor> {
    let last = xs.rank() - 1;
    let max = xs.max_keepdim(last)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(last)?;
    num.broadcast_div(&den)
}

/// Composed scaled-dot-product-attention with a causal mask, executed
/// on Wgpu via existing primitives + Phase 3.6 ops. No fused attention
/// kernel: matmul + affine (scale) + broadcast_add (mask) + softmax-via-composition + matmul.
fn causal_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let mut attn = (q.matmul(&k.t()?)? * scale)?;
    if let Some(m) = mask {
        attn = attn.broadcast_add(m)?;
    }
    let attn = softmax_last_dim_via_composition(&attn)?;
    attn.matmul(v)
}

#[test]
fn causal_attention_block_f32() -> Result<()> {
    let Some(dev) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return Ok(());
    };
    let cpu = Device::Cpu;
    let (b, h, t, dh) = (1, 2, 4, 8);

    let make_qkv = |dev: &Device| -> Result<(Tensor, Tensor, Tensor)> {
        let q: Vec<f32> = (0..b * h * t * dh)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * t * dh)
            .map(|i| ((i as f32) * 0.017).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * t * dh)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();
        Ok((
            Tensor::from_slice(&q, (b, h, t, dh), dev)?,
            Tensor::from_slice(&k, (b, h, t, dh), dev)?,
            Tensor::from_slice(&v, (b, h, t, dh), dev)?,
        ))
    };
    // Build a triangular causal mask on CPU; transfer to whichever
    // device. Same code path Gemma's `prepare_decoder_attention_mask`
    // uses (CPU build → upload).
    let make_mask = |dev: &Device| -> Result<Tensor> {
        let m: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let m = Tensor::from_slice(&m, (t, t), dev)?;
        m.expand((b, 1, t, t))?.to_dtype(DType::F32)
    };

    let (q_c, k_c, v_c) = make_qkv(&cpu)?;
    let mask_c = make_mask(&cpu)?;
    let want = causal_sdpa(&q_c, &k_c, &v_c, Some(&mask_c))?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let (q_g, k_g, v_g) = make_qkv(&dev)?;
    let mask_g = make_mask(&dev)?;
    let got = causal_sdpa(&q_g, &k_g, &v_g, Some(&mask_g))?
        .flatten_all()?
        .to_vec1::<f32>()?;

    assert_close(&want, &got, 1e-3, "causal-sdpa");
    Ok(())
}

/// The combined RoPE + causal-attention path: apply RoPE to q and k
/// before the dot product. Matches what Gemma's per-layer attention
/// runs at decode time when seq_len > 1.
#[test]
fn rope_then_causal_attention_f32() -> Result<()> {
    let Some(dev) = try_wgpu() else {
        eprintln!("[skip] no wgpu adapter");
        return Ok(());
    };
    let cpu = Device::Cpu;
    let (b, h, t, dh) = (1, 2, 4, 8);

    let build = |dev: &Device| -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let q: Vec<f32> = (0..b * h * t * dh)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let k: Vec<f32> = (0..b * h * t * dh)
            .map(|i| ((i as f32) * 0.017).cos())
            .collect();
        let v: Vec<f32> = (0..b * h * t * dh)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();
        let cos: Vec<f32> = (0..t * dh / 2).map(|i| ((i as f32) * 0.05).cos()).collect();
        let sin: Vec<f32> = (0..t * dh / 2).map(|i| ((i as f32) * 0.05).sin()).collect();
        let mask_data: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask_data, (t, t), dev)?
            .expand((b, 1, t, t))?
            .to_dtype(DType::F32)?;
        Ok((
            Tensor::from_slice(&q, (b, h, t, dh), dev)?,
            Tensor::from_slice(&k, (b, h, t, dh), dev)?,
            Tensor::from_slice(&v, (b, h, t, dh), dev)?,
            Tensor::from_slice(&cos, (t, dh / 2), dev)?,
            Tensor::from_slice(&sin, (t, dh / 2), dev)?,
            mask,
        ))
    };

    let run = |q: &Tensor,
               k: &Tensor,
               v: &Tensor,
               cos: &Tensor,
               sin: &Tensor,
               mask: &Tensor|
     -> Result<Vec<f32>> {
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        causal_sdpa(&q, &k, v, Some(mask))?
            .flatten_all()?
            .to_vec1::<f32>()
    };

    let (qc, kc, vc, cosc, sinc, mc) = build(&cpu)?;
    let want = run(&qc, &kc, &vc, &cosc, &sinc, &mc)?;
    let (qg, kg, vg, cosg, sing, mg) = build(&dev)?;
    let got = run(&qg, &kg, &vg, &cosg, &sing, &mg)?;
    assert_close(&want, &got, 1e-3, "rope+causal-sdpa");
    Ok(())
}
