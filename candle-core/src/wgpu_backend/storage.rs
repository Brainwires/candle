//! WebGPU storage (Phase 3.1 scaffolding).
//!
//! All [`crate::backend::BackendStorage`] methods return
//! `Err(Error::Msg("wgpu: <op> not yet implemented"))`. The shape of
//! each method mirrors the existing `dummy_metal_backend` so adding
//! real implementations in Phase 3.2+ is purely additive — we just
//! replace the stub bodies one by one.

#![allow(dead_code)]

use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};

use super::device::WgpuDevice;

/// Storage backed by a `wgpu::Buffer`. Phase 3.1 stub; real allocation
/// arrives in Phase 3.2.
///
/// The struct holds a [`WgpuDevice`] handle so that [`BackendStorage::device`]
/// can return a real `&Self::Device` reference. Phase 3.2 will add the
/// actual `wgpu::Buffer`, dtype, and element count fields here.
#[derive(Debug)]
pub struct WgpuStorage {
    device: WgpuDevice,
    dtype: DType,
    // TODO(phase-3.2): wgpu::Buffer, length in elements.
}

impl WgpuStorage {
    fn not_implemented<T>(op: &str) -> Result<T> {
        Err(crate::Error::Msg(format!(
            "wgpu: {op} not yet implemented (Phase 3.2+)"
        )))
    }

    /// Internal constructor used by stubs that need to materialize a
    /// dummy `WgpuStorage` for the purpose of returning an error from a
    /// `Result<Self>` arm. Phase 3.2 replaces this with real allocation.
    #[allow(dead_code)]
    pub(crate) fn empty(device: WgpuDevice, dtype: DType) -> Self {
        Self { device, dtype }
    }
}

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Self::not_implemented("try_clone")
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Self::not_implemented("to_cpu_storage")
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Self::not_implemented("affine")
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Self::not_implemented("powf")
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Self::not_implemented("elu")
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Self::not_implemented("reduce_op")
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Self::not_implemented("cmp")
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Self::not_implemented("to_dtype")
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Self::not_implemented("unary_impl")
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Self::not_implemented("binary_impl")
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Self::not_implemented("where_cond")
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Self::not_implemented("conv1d")
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Self::not_implemented("conv_transpose1d")
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Self::not_implemented("conv2d")
    }

    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Self::not_implemented("conv_transpose2d")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Self::not_implemented("avg_pool2d")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Self::not_implemented("max_pool2d")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Self::not_implemented("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Self::not_implemented("upsample_nearest2d")
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Self::not_implemented("upsample_bilinear2d")
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Self::not_implemented("gather")
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Self::not_implemented("scatter_set")
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Self::not_implemented("scatter_add_set")
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Self::not_implemented("index_select")
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Self::not_implemented("index_add")
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Self::not_implemented("matmul")
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Self::not_implemented("copy_strided_src")
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        Self::not_implemented("copy2d")
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Self::not_implemented("const_set")
    }
}
