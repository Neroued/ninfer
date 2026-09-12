#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <cstdint>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

struct Q4KSplitTopKOutput {
    std::uint64_t* partial_keys;
    const std::int32_t* row_to_global_ids;
    std::int32_t producer_groups;
    std::int32_t columns;

    template <int Capacity>
    __device__ __forceinline__ void store(std::int32_t row0, std::int32_t column0,
                                          float4 values) const {
        const std::int32_t group = row0 / kLinearTopK;
        const std::int32_t rank0 = row0 % kLinearTopK;
        const auto put           = [&](std::int32_t row, std::int32_t rank, std::int32_t column,
                             float value) {
            if (column >= columns) { return; }
            const std::int64_t offset =
                (static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK + rank;
            partial_keys[offset] = score_id_order_key(value, row_to_global_ids[row]);
        };
        put(row0, rank0, column0, values.x);
        put(row0, rank0, column0 + 1, values.y);
        put(row0 + 8, rank0 + 8, column0, values.z);
        put(row0 + 8, rank0 + 8, column0 + 1, values.w);
    }
};

template <int Capacity>
void launch_ksplit(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                   const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    using Geometry             = Q4DraftHeadGeometry<kLinearTopKHidden>;
    using Schedule             = Q4DraftSmallTSchedule;
    constexpr int kTileColumns = ((Capacity + 7) / 8) * 8;
    constexpr int kBlocks      = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Q4KSplitTopKOutput output{static_cast<std::uint64_t*>(workspace.partial_keys.data),
                                    static_cast<const std::int32_t*>(row_to_global_ids.data),
                                    workspace.producer_groups, hidden.ne[1]};
    q4_small_t_mma_kernel<Geometry, kTileColumns, Capacity, Q4KSplitTopKOutput,
                          Q4SmallTMmaIdentityRows, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(hidden.data),
                                                     static_cast<const std::uint8_t*>(head.qdata),
                                                     static_cast<const std::uint8_t*>(head.scales),
                                                     nullptr, output, Q4SmallTMmaIdentityRows{}, hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

using Launch = void (*)(const Tensor&, const Weight&, const Tensor&, const LinearTopKWorkspace&,
                        cudaStream_t);

template <std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{&launch_ksplit<8 * (1 + I)>...};
}

constexpr auto launchers = make_launchers(std::make_index_sequence<2>{});

} // namespace

void linear_topk_q4_launch(const Tensor& hidden, const Weight& head,
                           const Tensor& row_to_global_ids, const LinearTopKWorkspace& workspace,
                           cudaStream_t stream) {
    if (workspace.rows_per_producer == kLinearTopKDirectRows) {
        launchers[(hidden.ne[1] - 1) / 8](hidden, head, row_to_global_ids, workspace, stream);
    } else {
        linear_topk_q4_m64_launch(hidden, head, row_to_global_ids, workspace, stream);
    }
}
} // namespace ninfer::ops::detail
