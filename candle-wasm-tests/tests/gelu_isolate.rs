use candle::{Device, Result, Tensor};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;

#[test]
async fn gelu_wgpu_at_11p5() -> Result<()> {
    let device = Device::new_wgpu(0).expect("wgpu device");
    let inputs = &[-6.7f32, 0.0, 1.0, 5.0, 11.0, 11.509001, 12.0, 20.0, -20.0];
    let t = Tensor::new(inputs, &device)?;
    let y = t.gelu()?;
    let v: Vec<f32> = y.to_vec1_async().await?;
    eprintln!("gelu(input):");
    for (i, (x, gx)) in inputs.iter().zip(v.iter()).enumerate() {
        eprintln!("  [{i}] gelu({x}) = {gx} (finite={})", gx.is_finite());
    }
    Ok(())
}
