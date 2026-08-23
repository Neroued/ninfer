#include "core/arena.h"
#include "core/device.h"
#include "ops/common/volta_mma_splits.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_volta_mma_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Q5VoltaMmaSchedule;
} // namespace

// The first pass swept only 4/8/16/32/48 splits and so never saw the actual optimum, which for
// every registered Q5 shape is 2 or 3. Shared with the Q4 route; see ops/common/volta_mma_splits.h
// for the sweep and the L2 evidence behind it.
int q5_volta_mma_splits(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    return volta_mma_split_count(n, k, t);
}

std::size_t q5_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                         std::int32_t t) noexcept {
    if (q5_volta_mma_splits(n, k, t) == 1) { return 0; }
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(t) * sizeof(float);
}

bool q5_volta_mma_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // k % kGroupK keeps whole Q5 groups inside a split (the high plane is indexed per group);
    // k % 8 keeps the uint4 activation staging 16-byte aligned.
    if (k % Q5RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    if ((k / q5_volta_mma_splits(n, k, t)) < S::kKStep) { return false; }
    return true;
}

// `weight_row_offset` selects a contiguous row band of a parent weight, for ops whose one
// stored parent feeds several outputs -- gdn_input_proj's value_z is [12288,5120] with rows
// 0..6143 going to qkv[4096:10240] and 6144..12287 to z. Row-major with a fixed per-row stride,
// so this is pointer arithmetic and the kernel itself is untouched.
void launch_q5_volta_mma(const Tensor& x, const Weight& w, Tensor& out, bool add_residual,
                         std::int32_t weight_row_offset, WorkspaceArena& ws, cudaStream_t stream,
                         int splits_override) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q5RowSplitStorage::kGroupK;
    const int splits =
        splits_override > 0 ? splits_override : q5_volta_mma_splits(n, k, t);

    const dim3 grid(static_cast<unsigned>((n + S::kRowsPerCta - 1) / S::kRowsPerCta),
                    static_cast<unsigned>(splits),
                    static_cast<unsigned>((t + S::kTTile - 1) / S::kTTile));
    const std::int64_t roff = static_cast<std::int64_t>(weight_row_offset) * padded_groups;
    const auto* codes =
        static_cast<const std::uint8_t*>(w.qdata) + roff * Q5RowSplitStorage::kCodeBytesPerGroup;
    const auto* high =
        static_cast<const std::uint8_t*>(w.qhigh) + roff * Q5RowSplitStorage::kHighBytesPerGroup;
    const auto* scales =
        static_cast<const std::uint8_t*>(w.scales) + roff * Q5RowSplitStorage::kScaleBytesPerGroup;
    const auto* xd = static_cast<const __nv_bfloat16*>(x.data);
    auto* out_data = static_cast<__nv_bfloat16*>(out.data);

    // One split: store BF16 straight out, folding the residual into the same store, and skip the
    // workspace, its memset, the atomics and the narrowing pass entirely.
    if (splits == 1) {
        if (add_residual) {
            q5_volta_mma_gemm_kernel<true, true><<<grid, S::kThreads, 0, stream>>>(
                codes, high, scales, xd, nullptr, out_data, out_ld, n, k, t, padded_groups,
                splits);
        } else {
            q5_volta_mma_gemm_kernel<true, false><<<grid, S::kThreads, 0, stream>>>(
                codes, high, scales, xd, nullptr, out_data, out_ld, n, k, t, padded_groups,
                splits);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    auto scope           = ws.scope();
    const DeviceSpan buf = ws.alloc_bytes(q5_volta_mma_workspace_bytes(n, k, t));
    auto* partial        = static_cast<float*>(buf.data);
    CUDA_CHECK(cudaMemsetAsync(partial, 0, q5_volta_mma_workspace_bytes(n, k, t), stream));

    q5_volta_mma_gemm_kernel<false, false><<<grid, S::kThreads, 0, stream>>>(
        codes, high, scales, xd, partial, out_data, out_ld, n, k, t, padded_groups, splits);
    CUDA_CHECK(cudaGetLastError());

    const std::int64_t count     = static_cast<std::int64_t>(n) * t;
    constexpr int kNarrowThreads = 256;
    const unsigned blocks =
        static_cast<unsigned>((count + kNarrowThreads - 1) / kNarrowThreads);
    if (add_residual) {
        q5_volta_mma_narrow_kernel<true><<<blocks, kNarrowThreads, 0, stream>>>(
            partial, out_data, n, t, out_ld);
    } else {
        q5_volta_mma_narrow_kernel<false><<<blocks, kNarrowThreads, 0, stream>>>(
            partial, out_data, n, t, out_ld);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
