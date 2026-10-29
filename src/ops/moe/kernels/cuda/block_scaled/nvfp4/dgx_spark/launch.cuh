#pragma once

#include "ops/moe/kernels/cuda/block_scaled/nvfp4/schedule.h"
#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled {

template <int Columns, bool IncludeK256 = false, bool IncludeSingleStage = false, class Function>
void with_nvfp4_tuning(Nvfp4Pipeline pipeline, Nvfp4CachePolicy cache, Function&& function) {
    const auto memory = [&]<int RequestedStages, int RequestedK>() {
        // Keep one complete stage within the Spark shared-memory budget. Wider
        // column tiles trade pipeline depth for accumulator and activation reuse.
        constexpr int k           = RequestedK;
        constexpr int stage_bytes = (128 + Columns) * k * 9 / 16;
        constexpr int limit       = (96 * 1024 - Columns * 4) / stage_bytes;
        constexpr int stages      = RequestedStages < limit ? RequestedStages : limit;
        static_assert(stages >= 1);
        switch (static_cast<int>(cache) & 12) {
        case 0:
            function.template operator()<stages, k, 0>();
            break;
        case 4:
            function.template operator()<stages, k, 4>();
            break;
        case 12:
            function.template operator()<stages, k, 12>();
            break;
        default:
            throw std::logic_error("invalid NVFP4 MoE cache policy");
        }
    };
    switch (pipeline) {
    case Nvfp4Pipeline::K128:
        memory.template operator()<2, 128>();
        break;
    case Nvfp4Pipeline::K256:
        if constexpr (IncludeK256) {
            memory.template operator()<2, 256>();
            break;
        }
        throw std::logic_error("unregistered NVFP4 MoE pipeline");
    case Nvfp4Pipeline::K256SingleStage:
        if constexpr (IncludeSingleStage) {
            memory.template operator()<1, 256>();
            break;
        }
        throw std::logic_error("unregistered NVFP4 MoE pipeline");
    case Nvfp4Pipeline::K256Deep:
        memory.template operator()<3, 256>();
        break;
    case Nvfp4Pipeline::K512:
        memory.template operator()<1, 512>();
        break;
    default:
        throw std::logic_error("invalid NVFP4 MoE pipeline");
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
