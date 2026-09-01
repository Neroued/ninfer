#include "core/arena.h"
#include "core/device.h"
#include "ops/common/volta_mma_splits.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_volta_mma_gemm.cuh"

#include <cuda_bf16.h>

#include <algorithm>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

using S = Q4VoltaMmaSchedule;

// Narrowing writes through out's real column stride, which need not equal n.
__global__ void q4_volta_mma_narrow_strided_kernel(const float* __restrict__ partial,
                                                   __nv_bfloat16* __restrict__ out, int n, int t,
                                                   int out_ld) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(n) * t) { return; }
    const int col = static_cast<int>(i / n);
    const int row = static_cast<int>(i % n);
    out[static_cast<std::int64_t>(col) * out_ld + row] = __float2bfloat16(partial[i]);
}

} // namespace

// Shared with the Q5 route, which has the identical schedule and so the identical tradeoff;
// see ops/common/volta_mma_splits.h for the sweep and the L2 evidence behind it.
int q4_volta_mma_splits(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    return volta_mma_split_count(n, k, t);
}

std::size_t q4_volta_mma_workspace_bytes(std::int32_t n, std::int32_t k,
                                         std::int32_t t) noexcept {
    if (q4_volta_mma_splits(n, k, t) == 1) { return 0; }
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(t) * sizeof(float);
}

bool q4_volta_mma_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    // k % 8 keeps the uint4 activation staging 16-byte aligned; k % kGroupK keeps whole Q4
    // groups inside a split. Both hold for every registered Qwen3.6/3.8 Q4 shape.
    if (k % Q4RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    if (n <= 0 || t <= 0) { return false; }
    // Each split needs at least one kKStep run of K to itself.
    if ((k / q4_volta_mma_splits(n, k, t)) < S::kKStep) { return false; }
    return true;
}

// `weight_row_offset` selects a contiguous row band of a parent weight, for ops whose one stored
// parent feeds several outputs -- attn_input_proj's query_key is [7168,5120] with rows 0..6143
// going to q and 6144..7167 to k. Row-major with a fixed per-row stride, so this is pointer
// arithmetic and the kernel itself is untouched. Mirrors the Q5 sibling.
void launch_q4_volta_mma(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                         cudaStream_t stream, std::int32_t weight_row_offset,
                         int splits_override) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const std::int32_t out_ld =
        static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q4RowSplitStorage::kGroupK;
    const int splits = splits_override > 0 ? splits_override : q4_volta_mma_splits(n, k, t);

    const dim3 grid(static_cast<unsigned>((n + S::kRowsPerCta - 1) / S::kRowsPerCta),
                    static_cast<unsigned>(splits),
                    static_cast<unsigned>((t + S::kTTile - 1) / S::kTTile));
    const std::int64_t roff = static_cast<std::int64_t>(weight_row_offset) * padded_groups;
    const auto* codes =
        static_cast<const std::uint8_t*>(w.qdata) + roff * Q4RowSplitStorage::kCodeBytesPerGroup;
    const auto* scales =
        static_cast<const std::uint8_t*>(w.scales) + roff * Q4RowSplitStorage::kScaleBytesPerGroup;
    auto* out_data = static_cast<__nv_bfloat16*>(out.data);

    // One split means each CTA owns its output tile outright, so it can store BF16 straight out
    // and the fp32 workspace, its memset, the atomics and the narrowing pass all disappear.
    if (splits == 1) {
        q4_volta_mma_gemm_kernel<true><<<grid, S::kThreads, 0, stream>>>(
            codes, scales, static_cast<const __nv_bfloat16*>(x.data), nullptr, out_data, out_ld, n,
            k, t, padded_groups, splits);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    auto scope           = ws.scope();
    const DeviceSpan buf = ws.alloc_bytes(q4_volta_mma_workspace_bytes(n, k, t));
    auto* partial        = static_cast<float*>(buf.data);

    // Partials are accumulated with atomicAdd, so the buffer has to start at zero.
    CUDA_CHECK(cudaMemsetAsync(partial, 0, q4_volta_mma_workspace_bytes(n, k, t), stream));

    q4_volta_mma_gemm_kernel<false><<<grid, S::kThreads, 0, stream>>>(
        codes, scales, static_cast<const __nv_bfloat16*>(x.data), partial, out_data, out_ld, n, k,
        t, padded_groups, splits);
    CUDA_CHECK(cudaGetLastError());

    const std::int64_t count = static_cast<std::int64_t>(n) * t;
    constexpr int kNarrowThreads = 256;
    q4_volta_mma_narrow_strided_kernel<<<
        static_cast<unsigned>((count + kNarrowThreads - 1) / kNarrowThreads), kNarrowThreads, 0,
        stream>>>(partial, out_data, n, t, out_ld);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
