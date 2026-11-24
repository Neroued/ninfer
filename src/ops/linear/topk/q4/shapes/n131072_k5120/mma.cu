#include "ops/common/numeric/math.h"
#include "ops/linear/topk/q4/kernel/mma.cuh"

namespace ninfer::ops::detail {
namespace {
template <int TileColumns, int kBlockK = 128>
void launch_tile(const Tensor& hidden, const Weight& head, const Tensor* row_to_global_ids,
                 const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    constexpr int shared_bytes = sizeof(Q4MmaReusableStorage<TileColumns, kBlockK>);
    if constexpr (shared_bytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(q4_mma_linear_topk_kernel<TileColumns, kBlockK>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    }
    q4_mma_linear_topk_kernel<TileColumns, kBlockK>
        <<<dim3(workspace.producer_groups, common::div_up(hidden.ne[1], TileColumns)),
           MmaSchedule<TileColumns, kBlockK>::kThreads, shared_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const std::uint8_t*>(head.qdata),
            static_cast<const std::uint8_t*>(head.scales),
            static_cast<const std::int32_t*>(row_to_global_ids->data),
            static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups,
            hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

LinearTopKPlan select_topk_q4_n131072_k5120_mma(int columns) {
    if (columns <= 32) return {launch_tile<32, 128>, 64};
    if (columns <= 48) return {launch_tile<48, 128>, 64};
    if (columns <= 64) return {launch_tile<64, 128>, 64};
    if (columns <= 80) return {launch_tile<80, 128>, 64};
    if (columns <= 96) return {launch_tile<96, 128>, 64};
    if (columns <= 112) return {launch_tile<112, 64>, 64};
    if (columns <= 120) return {launch_tile<120, 64>, 64};
    return {launch_tile<128, 64>, 64};
}
} // namespace ninfer::ops::detail
