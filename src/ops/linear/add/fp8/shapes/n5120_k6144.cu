#include "ops/linear/add/fp8/launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S, int T = 0, bool Exact = false>
using Instance = Fp8ScheduleInstance<S, 6144, T, Exact>;

LinearAddLaunch select_a16(int tokens) {
    if (tokens == 1)
        return launch_fp8_add_gemv<
            Instance<Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>>>;
    if (tokens == 2)
        return launch_fp8_add_simt<Instance<
            Fp8A16SimtSchedule<8, 1, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>,
            2, true>>;
    if (tokens == 3)
        return launch_fp8_add_simt<Instance<
            Fp8A16SimtSchedule<8, 1, 16, 3, 1, Fp8SimtActivationAccess::TokenPacked,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>,
            3, true>>;
    if (tokens == 4)
        return launch_fp8_add_simt<Instance<
            Fp8A16SimtSchedule<8, 1, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                               Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>,
            4, true>>;
    if (tokens <= 16) return launch_fp8_add_sliced_k_mma<Instance<Fp8SlicedInstance<16, 8, 2>>>;
    if (tokens <= 32) return launch_fp8_add_sliced_k_mma<Instance<Fp8SlicedInstance<16, 4, 2>>>;
    if (tokens <= 64) return launch_fp8_add_sliced_k_mma<Instance<Fp8SlicedInstance<32, 4, 1>>>;
    if (tokens <= 128)
        return launch_fp8_add_mma<Instance<Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>>>;
    return launch_fp8_add_mma<Instance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>>;
}

using SmallTile = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using MidTile   = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
using MidBulk   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
} // namespace

Fp8LinearAddPlan select_fp8_add_n5120_k6144(LinearPolicy policy, std::int32_t tokens) {
    if (!allows_a8(policy) || tokens < 17) return {select_a16(tokens)};
    const std::size_t partial_bytes = tokens > 768   ? Bulk::kPartialBytes
                                      : tokens > 192 ? MidBulk::kPartialBytes
                                                     : 0;
    if (tokens <= 64) return {nullptr, launch_fp8_add_a8<Instance<SmallTile>>, partial_bytes};
    if (tokens <= 128)
        return {nullptr, launch_fp8_add_a8<Instance<Fp8A8T64R64K128>>, partial_bytes};
    if (tokens <= 192) return {nullptr, launch_fp8_add_a8<Instance<MidTile>>, partial_bytes};
    if (tokens <= 768) return {nullptr, launch_fp8_add_a8<Instance<MidBulk>>, partial_bytes};
    return {nullptr, launch_fp8_add_a8<Instance<Bulk>>, partial_bytes};
}
} // namespace ninfer::ops::detail
