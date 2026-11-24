#include "ops/common/numeric/math.h"
#include "ops/linear/topk/q8/kernel/mma.cuh"

namespace ninfer::ops::detail {
namespace {

template <int TileColumns, int kBlockK = 128, int kRowsPerProducer = 128>
void launch_tile(const Tensor& hidden, const Weight& head, const Tensor*,
                 const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    const int valid_rows = head.n == kLinearTopKCompactRows
                               ? kLinearTopKCompactValidRows
                               : kLinearTopKFullValidRows;
    constexpr int shared_bytes =
        sizeof(MmaReusableStorage<TileColumns, kBlockK, kRowsPerProducer>) +
        (kRowsPerProducer > 64 ? TileColumns * kLinearTopK * sizeof(std::uint64_t) : 0);
    if constexpr (shared_bytes > 48 * 1024) {
        CUDA_CHECK(
            cudaFuncSetAttribute(q8_mma_linear_topk_kernel<TileColumns, kBlockK, kRowsPerProducer>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    }
    q8_mma_linear_topk_kernel<TileColumns, kBlockK, kRowsPerProducer>
        <<<dim3(workspace.producer_groups, common::div_up(hidden.ne[1], TileColumns)),
           MmaSchedule<TileColumns, kBlockK, kRowsPerProducer>::kThreads, shared_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const std::uint8_t*>(head.qdata),
            static_cast<const std::uint8_t*>(head.scales), valid_rows,
            static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups,
            hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

LinearTopKPlan select_topk_q8_n248320_k5120_mma(int columns) {
    if (columns <= 32) return {launch_tile<32, 128, 128>, 128};
    if (columns <= 40) return {launch_tile<40, 128, 64>, 64};
    if (columns <= 48) return {launch_tile<48, 128, 64>, 64};
    if (columns <= 64) return {launch_tile<64, 128, 64>, 64};
    if (columns <= 80) return {launch_tile<80, 128, 64>, 64};
    if (columns <= 96) return {launch_tile<96, 128, 64>, 64};
    if (columns <= 112) return {launch_tile<112, 64, 64>, 64};
    if (columns <= 120) return {launch_tile<120, 64, 64>, 64};
    return {launch_tile<128, 64, 64>, 64};
}
LinearTopKPlan select_topk_q8_n122944_k5120_mma(int columns) {
    return select_topk_q8_n248320_k5120_mma(columns);
}

} // namespace ninfer::ops::detail
