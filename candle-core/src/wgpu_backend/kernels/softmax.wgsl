// Softmax-last-dim kernel (Phase 3.7).
//
// For a tensor of shape (..., D), apply softmax along the last
// dimension. Each workgroup handles one row of length D.
//
// Algorithm per row:
//   1. Find max value (numerical stability).
//   2. Compute exp(x - max) for each element.
//   3. Sum the exponentials.
//   4. Divide each by the sum.
//
// Workgroup size: 64 threads cooperate on one row via shared memory
// reduction.  If D > 64, each thread handles a strided slice.
//
// Substitution marker: __SCALAR_T__ → f32.

struct SoftmaxMeta {
    n_rows: u32,
    row_len: u32,       // last dim size D
    src_offset: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: SoftmaxMeta;
@group(0) @binding(1) var<storage, read> src: array<__SCALAR_T__>;
@group(0) @binding(2) var<storage, read_write> out: array<__SCALAR_T__>;

var<workgroup> shared_max: array<__SCALAR_T__, 64>;
var<workgroup> shared_sum: array<__SCALAR_T__, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    // 2-D row dispatch: row = wg_id.x + wg_id.y * num_wg.x. Lets us
    // softmax over more than 65535 rows (e.g. long-context attention).
    let row = wg_id.x + wg_id.y * num_wg.x;
    if row >= params.n_rows {
        return;
    }
    let tid = lid.x;
    let row_start = params.src_offset + row * params.row_len;
    let d = params.row_len;

    // Step 1: Compute local max across strided elements.
    var local_max: __SCALAR_T__ = -3.402823466e+38; // -FLT_MAX
    var i: u32 = tid;
    loop {
        if i >= d {
            break;
        }
        let val = src[row_start + i];
        if val > local_max {
            local_max = val;
        }
        i = i + 64u;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Parallel reduction for max (log2(64) = 6 steps).
    if tid < 32u { shared_max[tid] = max(shared_max[tid], shared_max[tid + 32u]); }
    workgroupBarrier();
    if tid < 16u { shared_max[tid] = max(shared_max[tid], shared_max[tid + 16u]); }
    workgroupBarrier();
    if tid < 8u { shared_max[tid] = max(shared_max[tid], shared_max[tid + 8u]); }
    workgroupBarrier();
    if tid < 4u { shared_max[tid] = max(shared_max[tid], shared_max[tid + 4u]); }
    workgroupBarrier();
    if tid < 2u { shared_max[tid] = max(shared_max[tid], shared_max[tid + 2u]); }
    workgroupBarrier();
    if tid < 1u { shared_max[tid] = max(shared_max[tid], shared_max[tid + 1u]); }
    workgroupBarrier();

    let row_max = shared_max[0];

    // Step 2 + 3: exp(x - max) and partial sum.
    var local_sum: __SCALAR_T__ = 0.0;
    i = tid;
    loop {
        if i >= d {
            break;
        }
        let e = exp(src[row_start + i] - row_max);
        out[row * d + i] = e;
        local_sum = local_sum + e;
        i = i + 64u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Parallel reduction for sum.
    if tid < 32u { shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { shared_sum[tid] = shared_sum[tid] + shared_sum[tid + 1u]; }
    workgroupBarrier();

    let row_sum = shared_sum[0];

    // Step 4: Divide by sum.
    i = tid;
    loop {
        if i >= d {
            break;
        }
        out[row * d + i] = out[row * d + i] / row_sum;
        i = i + 64u;
    }
}
