use candle::{Device, Result, Tensor};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;

#[cfg(not(target_arch = "wasm32"))]
async fn build_wgpu() -> Device {
    Device::new_wgpu(0).expect("wgpu device")
}

#[test]
async fn softmax_last_dim_isolate_wgpu() -> Result<()> {
    let device = build_wgpu().await;

    let inp = &[1f32, 2f32, 3f32];
    let t = Tensor::new(inp, &device)?;
    let y = candle_nn::ops::softmax_last_dim(&t)?;
    let v: Vec<f32> = y.to_vec1_async().await?;
    eprintln!("softmax_last_dim([1,2,3]) = {:?}", v);

    let inp2 = &[[1f32, 2., 3.], [3., 2., 1.]];
    let t2 = Tensor::new(inp2, &device)?;
    let y2 = candle_nn::ops::softmax_last_dim(&t2)?;
    let v2: Vec<Vec<f32>> = y2.to_vec2_async().await?;
    eprintln!("softmax_last_dim([[1,2,3],[3,2,1]]) = {:?}", v2);

    let inp3: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
    let t3 = Tensor::new(inp3.as_slice(), &device)?;
    let y3 = candle_nn::ops::softmax_last_dim(&t3)?;
    let v3: Vec<f32> = y3.to_vec1_async().await?;
    let sum: f32 = v3.iter().sum();
    eprintln!(
        "softmax_last_dim(1024 elem) sum={sum:.6} first3={:?} last3={:?}",
        &v3[..3],
        &v3[v3.len() - 3..]
    );

    Ok(())
}
