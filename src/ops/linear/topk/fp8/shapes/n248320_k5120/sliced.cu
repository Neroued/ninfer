#include "ops/linear/topk/fp8/kernel/sliced.cuh"

namespace ninfer::ops::detail {
namespace {

template <int Capacity>
void launch_tile(const Tensor& hidden, const Weight& head, const Tensor*,
                 const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    constexpr int valid_rows = kLinearTopKFullValidRows;
    using Schedule           = Fp8A16SlicedKMmaSchedule<8, Capacity, 2>;
    fp8_grouped_ksplit_topk_kernel<Capacity, Schedule>
        <<<workspace.producer_groups, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const std::uint8_t*>(head.qdata),
            static_cast<const __nv_bfloat16*>(head.scales), valid_rows,
            static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups,
            hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Indices>
constexpr auto make_launchers(std::index_sequence<Indices...>) {
    return std::array<LinearTopKProducer, sizeof...(Indices)>{&launch_tile<8 * (1 + Indices)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<3>{});

} // namespace

LinearTopKProducer select_topk_fp8_n248320_k5120_sliced(int columns) {
    return kLaunchers[(columns - 1) / 8];
}
} // namespace ninfer::ops::detail
