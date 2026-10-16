#include "ops/linear/add/nvfp4/launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class S, int T = 0, bool Exact = false>
using Instance = Nvfp4ScheduleInstance<S, 17408, T, Exact>;

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
    if (tokens == 3)
        return launch_nvfp4_add_simt<
            Instance<Nvfp4A16SimtSchedule<4, 1, 2, 16, 3, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                          Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                          Nvfp4SimtBlockOrder::RowsContiguous, 1>,
                     3, true>>;
    if (tokens == 4)
        return launch_nvfp4_add_simt<
            Instance<Nvfp4A16SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                          Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                          Nvfp4SimtBlockOrder::RowsContiguous, 1>,
                     4, true>>;
    if (tokens == 5)
        return launch_nvfp4_add_simt<
            Instance<Nvfp4A16SimtSchedule<4, 1, 2, 16, 5, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                          Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                          Nvfp4SimtBlockOrder::RowsContiguous, 1>,
                     5, true>>;
    if (tokens <= 8) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<8, 4, 2>>>;
    if (tokens <= 16) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<16, 4, 2>>>;
    if (tokens <= 24) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 4, 2>>>;
    if (tokens <= 32) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 4, 1>>>;
    if (tokens <= 48) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<32, 4, 1>>>;
    if (tokens <= 64) return launch_nvfp4_add_sliced_k_mma<Instance<Nvfp4SlicedInstance<64, 2, 2>>>;
    if (tokens <= 128)
        return launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>>;
    return launch_nvfp4_add_mma<Instance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>>;
}
} // namespace

Nvfp4LinearAddPlan select_nvfp4_add_n5120_k17408(LinearPolicy policy, std::int32_t tokens) {
    if (!allows_a4(policy) || tokens < 8) return {select_a16(tokens)};
    if (tokens >= 1024)
        return {nullptr, launch_nvfp4_add_n5120_k17408_tma, Nvfp4ScaleLayout::Tiled256};
    // At 64 rows per CTA the 5120 output rows are 80 CTAs on 170 SMs, so a speculative verify
    // streams its 50 MB of weights from fewer than half the SMs: 46 us at T=8, about 1.1 TB/s.
    // Halving the tile fills the device, and the long K loop then wants a deeper pipeline: 40 us
    // at T=8 and T=16. Same per-row K order, so the bytes do not change. From 33 the narrower
    // tile re-reads the weights for each token tile and loses, so 33..64 keep M32N64.
    if (tokens <= 32)
        return {nullptr,
                launch_nvfp4_add_a4<Instance<Nvfp4A4MmaSchedule<32, 32, 256, 2, 4, 4, 2>>>};
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
