#pragma once
#include "core/layout.h"
#include "ops/common/numeric/math.h"
#include <cstdint>

namespace ninfer::ops::sampling_detail {
inline constexpr int kSamplerCandidateCap   = 20;
inline constexpr int kSamplerFastCandidates = kSamplerCandidateCap;

template <int Threads, int Items, int GroupThreads, int GroupItems, int Partials>
struct TopKSchedule {
    static constexpr int kSamplerBlock               = Threads;
    static constexpr int kSamplerTileItems           = Threads;
    static constexpr int kSamplerItemsPerThread      = Items;
    static constexpr int kSamplerPartialTileItems    = Threads * Items;
    static constexpr int kSamplerGroupBlock          = GroupThreads;
    static constexpr int kSamplerGroupItemsPerThread = GroupItems;
    static constexpr int kSamplerGroupTileItems      = GroupThreads * GroupItems;
    static constexpr int kSamplerPartialsPerGroup    = Partials;
    static constexpr int kSamplingTileWarps          = Threads / 32;
    static constexpr int kSamplingTileIndexBits      = 9;
    static constexpr unsigned kSamplingTileIndexMask = (1u << kSamplingTileIndexBits) - 1u;
    static_assert(Threads * Items == 512, "compact BF16 key uses a 9-bit tile index");
    static_assert(Threads == GroupThreads && Items == GroupItems);
    static_assert(Partials * kSamplerCandidateCap <= kSamplerGroupTileItems);
};

struct Workspace {
    unsigned long long* partial_keys         = nullptr;
    std::int32_t* dist_idx                   = nullptr;
    float* dist_prob                         = nullptr;
    std::int32_t* dist_support               = nullptr;
    std::int32_t* group_done                 = nullptr;
    std::int32_t* speculative_finalize_count = nullptr;
    std::int32_t partial_stride              = 0;
};

struct WorkspaceLayout {
    TensorRegion partial_keys;
    TensorRegion dist_idx;
    TensorRegion dist_prob;
    TensorRegion dist_support;
    TensorRegion group_done;
    TensorRegion speculative_finalize_count;
    std::size_t bytes           = 0;
    std::int32_t partial_stride = 0;
    bool acceptance             = false;

    [[nodiscard]] Workspace bind(MemorySpan backing) const {
        if (bytes == 0) { return {}; }
        return {
            static_cast<unsigned long long*>(partial_keys.bind(backing).data),
            acceptance ? static_cast<std::int32_t*>(dist_idx.bind(backing).data) : nullptr,
            acceptance ? static_cast<float*>(dist_prob.bind(backing).data) : nullptr,
            acceptance ? static_cast<std::int32_t*>(dist_support.bind(backing).data) : nullptr,
            static_cast<std::int32_t*>(group_done.bind(backing).data),
            acceptance ? static_cast<std::int32_t*>(speculative_finalize_count.bind(backing).data)
                       : nullptr,
            partial_stride,
        };
    }
};

inline WorkspaceLayout make_workspace(int columns, int partial_blocks, int groups,
                                      bool acceptance) {
    WorkspaceLayout out;
    out.acceptance     = acceptance;
    out.partial_stride = partial_blocks + groups;
    LayoutBuilder layout;
    out.partial_keys =
        layout.add_tensor(DType::I64, {kSamplerCandidateCap, out.partial_stride, columns}, 256,
                          "sampling partial keys");
    if (acceptance) {
        out.dist_idx  = layout.add_tensor(DType::I32, {kSamplerCandidateCap, columns}, 256,
                                          "sampling speculative distribution indices");
        out.dist_prob = layout.add_tensor(DType::FP32, {kSamplerCandidateCap, columns}, 256,
                                          "sampling speculative distribution probabilities");
        out.dist_support =
            layout.add_tensor(DType::I32, {columns}, 256, "sampling speculative support sizes");
    }
    out.group_done = layout.add_tensor(DType::I32, {columns}, 256, "sampling group counters");
    if (acceptance)
        out.speculative_finalize_count =
            layout.add_tensor(DType::I32, {1}, 256, "sampling speculative finalize counter");
    // Speculative acceptance replicates this complete layout once per batch row. Preserve the
    // strongest member alignment in the row stride so every replicated partial-key array remains
    // correctly aligned.
    out.bytes = layout.finish(256, "sampling workspace");
    return out;
}

} // namespace ninfer::ops::sampling_detail
