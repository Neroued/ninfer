#include "ops/linear/topk/q4/kernel/sliced.cuh"

namespace ninfer::ops::detail {
namespace {
template <int Capacity>
void launch_sliced(const Tensor& hidden, const Weight& head, const Tensor* row_to_global_ids,
                   const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, kLinearTopKHidden, Capacity>;
    const Q4TopKOutput output{static_cast<std::uint64_t*>(workspace.partial_keys.data),
                              static_cast<const std::int32_t*>(row_to_global_ids->data),
                              workspace.producer_groups};
    launch_q4_a16_sliced_k_mma<Schedule>(q4_linear_operands(hidden, head), output,
                                         LinearIdentityEpilogue{}, stream);
}

template <std::size_t... I>
constexpr auto make_launchers(std::index_sequence<I...>) {
    return std::array<LinearTopKProducer, sizeof...(I)>{&launch_sliced<8 * (1 + I)>...};
}

constexpr auto launchers = make_launchers(std::make_index_sequence<2>{});

} // namespace

LinearTopKProducer select_topk_q4_n131072_k5120_sliced(int columns) {
    return launchers[(columns - 1) / 8];
}
} // namespace ninfer::ops::detail
