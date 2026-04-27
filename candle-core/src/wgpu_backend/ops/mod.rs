//! WebGPU compute-op dispatchers.
//!
//! Each module here owns the host-side glue for one (or a small family
//! of) WGSL shader(s): pipeline lookup, bind-group creation, dispatch
//! geometry, and shape/stride uniform packing. Phase 3.3 ships `matmul`;
//! Phases 3.4+ add elementwise / reductions / attention ops alongside.

pub(crate) mod affine;
pub(crate) mod index_select;
pub(crate) mod binary;
pub(crate) mod copy;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub mod rope;
pub(crate) mod unary;
pub mod softmax;
