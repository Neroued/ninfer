#include "core/weight.h"
#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"

#include <cstdint>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

struct Q4TopKOutput {
    std::uint64_t* partial_keys;
    const std::int32_t* row_to_global_ids;
    std::int32_t producer_groups;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const int group = row / kLinearTopK;
        const int rank  = row % kLinearTopK;
        const std::int64_t offset =
            (static_cast<std::int64_t>(token) * producer_groups + group) * kLinearTopK + rank;
        partial_keys[offset] = score_id_order_key(value, row_to_global_ids[row]);
    }
};

template <int Capacity>
void launch_sliced(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                   const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, kLinearTopKHidden, Capacity>;
    const Q4TopKOutput output{static_cast<std::uint64_t*>(workspace.partial_keys.data),
                              static_cast<const std::int32_t*>(row_to_global_ids.data),
                              workspace.producer_groups};
    launch_q4_a16_sliced_k_mma<Schedule>(q4_linear_operands(hidden, head), output,
                                         LinearIdentityEpilogue{}, stream);
}

using Launch = void (*)(const Tensor&, const Weight&, const Tensor&, const LinearTopKWorkspace&,
                        cudaStream_t);

template <std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{&launch_sliced<8 * (1 + I)>...};
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
