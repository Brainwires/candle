//! WebGPU (wgpu) backend for candle.
//!
//! This is the scaffolding for Phase 3.1 of the brainwires bright-scroll
//! plan. The device, storage, and ops are stubs that return
//! [`Error::Msg`] with `"wgpu: not yet implemented"`. Subsequent phases
//! fill in real implementations:
//!
//! - 3.2: Tensor storage on GPU buffers (`WgpuStorage`, alloc/copy).
//! - 3.3: Matmul kernel.
//! - 3.4: Element-wise ops.
//! - 3.5: Reduction ops (sum, max, mean, softmax, layer/rms norm).
//! - 3.6: Attention-specific ops (RoPE, masked attention, KV cache).
//! - 3.7: Gemma model integration.
//!
//! On `wasm32-unknown-unknown` the wgpu backend uses the browser's
//! WebGPU. Native targets use the platform's wgpu backend
//! (Vulkan/Metal/DX12). See `extras/brainwires-chat-pwa/` for the
//! browser wiring that lands in Phase 3.8.
//!
//! Building for browsers requires `RUSTFLAGS=--cfg=web_sys_unstable_apis`
//! once Phase 3.8 wires `web-sys` GPU bindings.

mod device;
mod storage;

pub use device::WgpuDevice;
pub use storage::WgpuStorage;
