#include "ops/linear/projection/fp8_block/launch.cuh"
#include <array>
#include <limits>
#include <algorithm>

namespace ninfer::ops::detail {
namespace {
template <class S>
using Static = NativeFp8ScheduleInstance<S, 16384>;

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

template <class S>
struct TokenPanels : S {
    static constexpr int kRasterGroupTokens = 6;
};

using TailGeometry = Fp8BlockA16TailPlan<128, 128, 48, 3>;

struct TailSchedule : TokenPanels<Static<Fp8BlockA16TmaSchedule<128, 128, 128, 32, 64, 2, 2>>> {
    using TailPlan = TailGeometry;
};

constexpr int kFirstSplitTail = 513;

std::size_t workspace_capacity(int first, int last) {
    return fp8_block_a16_tail_capacity<TailGeometry>(4096, std::max(first, kFirstSplitTail), last);
}

Fp8BlockLinearRoute select(int tokens, LinearPolicy) {
    if (tokens >= kFirstSplitTail &&
        fp8_block_a16_tail_plan<TailGeometry>(4096, tokens).split_tail())
        return {fp8_block_linear_tma<TailSchedule>, true, fp8_block_linear_split_tail<TailSchedule>,
                TailGeometry::kWorkspaceBytes, 2 * TailGeometry::kPartialBytes};
    if (tokens <= 2)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<8, 2, 16, 128, 2>>>,
                true};
    if (tokens <= 16)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<8, 8, 8, 256, 1>>>,
                true};
    if (tokens <= 32)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 1>>>,
                true};
    if (tokens <= 64)
        return {fp8_block_linear_sliced<Static<NativeFp8A16SlicedKSchedule<32, 16, 4, 128, 1>>>,
                true};
    if (tokens <= 96)
        return {
            fp8_block_linear_mma<
                RegisterWeights<Static<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2, 2, 2>>>, true>,
            true};
    if (tokens <= 103)
        return {
            fp8_block_linear_mma<
                RegisterWeights<Static<NativeFp8MmaSchedule<48, 64, 128, 16, 32, 2, 2, 2>>>, true>,
            true};
    if (tokens <= 128)
        return {fp8_block_linear_mma<
                    RegisterWeights<Static<NativeFp8MmaSchedule<64, 64, 128, 16, 32, 2, 2>>>>,
                true};
    if (tokens <= 192)
        return {
            fp8_block_linear_mma<
                RegisterWeights<Static<NativeFp8MmaSchedule<128, 64, 128, 32, 32, 1, 2>>>, true>,
            true};
    if (tokens <= 256)
        return {fp8_block_linear_tma<Static<Fp8BlockA16TmaSchedule<64, 64, 128, 32, 32, 2, 8>>>,
                true};
    if (tokens > 384 && tokens <= 1024)
        return {fp8_block_linear_tma<Static<Fp8BlockA16TmaSchedule<128, 128, 128, 32, 64, 2, 2>>>,
                true};
    return {fp8_block_linear_tma<Static<Fp8BlockA16TmaSchedule<128, 128, 128, 32, 64, 2, 8>>>,
            true};
}
} // namespace

const Fp8BlockLinearShape kFp8BlockN4096K16384{4096, 16384, select, workspace_capacity};
} // namespace ninfer::ops::detail
