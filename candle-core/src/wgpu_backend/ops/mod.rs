//! WebGPU compute-op dispatchers.
//!
//! Each module here owns the host-side glue for one (or a small family
//! of) WGSL shader(s): pipeline lookup, bind-group creation, dispatch
//! geometry, and shape/stride uniform packing. Phase 3.3 ships `matmul`;
//! Phases 3.4+ add elementwise / reductions / attention ops alongside.

pub(crate) mod affine;
pub(crate) mod binary;
pub(crate) mod cast;
pub(crate) mod copy;
pub(crate) mod index_select;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub mod rope;
pub mod softmax;
pub(crate) mod unary;

/// WebGPU's spec-mandated per-dimension `dispatchWorkgroups` limit. Every
/// conformant adapter supports at least this many workgroups along each
/// of x/y/z; going past it is a validation error and the command encoder
/// is permanently invalidated. For Gemma4 the lm-head logits ([batch,
/// seq, 262144 vocab]) trip the 1-D path easily, so any kernel that
/// indexes by element count needs to split into 2 dimensions.
pub(super) const MAX_GROUPS_PER_DIM: u32 = 65535;

/// Split a logical 1-D workgroup count into a 2-D (x, y) dispatch shape
/// where both dims fit `MAX_GROUPS_PER_DIM`. Caller-side WGSL must then
/// recover the linear thread index as
///
/// ```wgsl
/// let i = gid.x + gid.y * num_workgroups.x * WORKGROUP_SIZE_X;
/// ```
///
/// `gx * gy ≥ groups` is the only invariant; shaders early-return on
/// `i >= n_elements`, so any over-dispatch is harmless.
pub(super) fn split_1d_dispatch(groups: u32) -> (u32, u32) {
    if groups <= MAX_GROUPS_PER_DIM {
        (groups.max(1), 1)
    } else {
        let gy = groups.div_ceil(MAX_GROUPS_PER_DIM);
        (MAX_GROUPS_PER_DIM, gy)
    }
}
