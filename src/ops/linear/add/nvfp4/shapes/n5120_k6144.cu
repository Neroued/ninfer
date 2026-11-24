#include "ops/linear/add/nvfp4/launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S, int T = 0, bool Exact = false>
using Instance = Nvfp4ScheduleInstance<S, 6144, T, Exact>;

LinearAddLaunch select_a16(int tokens) {
    if (tokens == 1)
        return launch_nvfp4_add_gemv<Instance<Nvfp4A16GemvSchedule<
            8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>>>;
    if (tokens == 2)
        return launch_nvfp4_add_simt<
            Instance<Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                          Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                          Nvfp4SimtBlockOrder::RowsContiguous, 1>,
                     2, true>>;
    if (tokens <= 8) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<8, 8, 2>>>;
    if (tokens <= 16) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<16, 8, 2>>>;
    if (tokens <= 24) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 8, 1>>>;
    if (tokens <= 32) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 4, 2>>>;
    if (tokens <= 48) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 2, 2>>>;
    if (tokens <= 64) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<64, 2, 2>>>;
    if (tokens <= 128)
        return launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>>>;
    return launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>>;
}
} // namespace

Nvfp4LinearAddPlan select_nvfp4_add_n5120_k6144(LinearPolicy policy, std::int32_t tokens) {
    if (!allows_a4(policy) || tokens < 17) return {select_a16(tokens)};
    if (tokens >= 1024)
        return {nullptr, launch_nvfp4_add_n5120_k6144_tma, Nvfp4ScaleLayout::Tiled256};
    if (tokens <= 64)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>>>};
    if (tokens <= 128)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>>>};
    if (tokens <= 192)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>>>};
    if (tokens <= 384)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>>};
    if (tokens <= 512)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>>>};
    return {nullptr, launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>>>};
}
} // namespace ninfer::ops::detail
