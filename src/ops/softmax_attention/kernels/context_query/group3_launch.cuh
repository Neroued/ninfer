#pragma once

#include "core/device.h"
#include "ops/softmax_attention/kernels/context_query/group3.cuh"
#include "ops/softmax_attention/kernels/context_query/group3_schedule.h"

namespace ninfer::ops::detail {

template <bool Paged, int Keys, bool Aligned>
void launch_group3_reduce(const Group3Arguments& args, int batch, int warps, cudaStream_t stream) {
    const auto launch = [&]<int Warps>() {
        const dim3 grid((args.tokens * kGroup3Heads + Warps - 1) / Warps, 1, batch);
        group3_reduce<Paged, Keys, Warps, Aligned><<<grid, Warps * 32, 0, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    };
    if (warps == 1)
        launch.template operator()<1>();
    else if (warps == 2)
        launch.template operator()<2>();
    else
        launch.template operator()<4>();
}

template <bool Paged, int Tokens, int MWarps, int KeyBlock, int WarpsK, bool StageQ, bool Aligned,
          bool Direct>
void launch_group3_tiles(const Group3Arguments& args, int batch, int reduce_warps,
                         cudaStream_t stream) {
    using P = Group3Pipeline<Paged, Tokens, MWarps, KeyBlock, WarpsK, StageQ, Aligned, Direct>;
    const dim3 grid(kGroup3KVHeads, args.splits, batch);
    group3_partial<Paged, Tokens, MWarps, KeyBlock, WarpsK, StageQ, Aligned, Direct>
        <<<grid, P::Threads, P::ScratchBytes, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
    if constexpr (Direct) {
        group3_idle<<<1, 1, 0, stream>>>();
        CUDA_CHECK(cudaGetLastError());
    } else {
        launch_group3_reduce<Paged, P::BlockKeys, Aligned>(args, batch, reduce_warps, stream);
    }
}

template <bool Paged, bool Aligned, bool Direct>
void launch_group3_standard(const Group3Arguments& args, int batch, int reduce_warps,
                            cudaStream_t stream) {
    const auto launch = [&]<int MWarps>() {
        launch_group3_tiles<Paged, 0, MWarps, 32, 1, false, Aligned, Direct>(args, batch,
                                                                             reduce_warps, stream);
    };
    const int query_warps = (args.tokens * 3 + 15) / 16;
    if (query_warps == 1)
        launch.template operator()<1>();
    else if (query_warps == 2)
        launch.template operator()<2>();
    else
        launch.template operator()<3>();
}

template <bool Paged>
void launch_group3(const Group3Arguments& args, int batch, Group3Schedule schedule,
                   int reduce_warps, cudaStream_t stream) {
    // The pointed-to tensor views are fixed for the lifetime of a graph capture.
    const auto address_bits = reinterpret_cast<std::uintptr_t>(args.q) |
                              reinterpret_cast<std::uintptr_t>(args.query_k) |
                              reinterpret_cast<std::uintptr_t>(args.query_v) |
                              reinterpret_cast<std::uintptr_t>(args.output);
    const bool aligned = (address_bits & 15) == 0;
    if (args.splits == 1) {
        if (aligned)
            launch_group3_standard<Paged, true, true>(args, batch, reduce_warps, stream);
        else
            launch_group3_standard<Paged, false, true>(args, batch, reduce_warps, stream);
        return;
    }
    if (!aligned) {
        launch_group3_standard<Paged, false, false>(args, batch, reduce_warps, stream);
        return;
    }

    switch (schedule) {
    case Group3Schedule::NarrowKeys:
        return launch_group3_tiles<Paged, 8, 2, 16, 1, false, true, false>(args, batch,
                                                                           reduce_warps, stream);
    case Group3Schedule::ParallelKeys:
        return launch_group3_tiles<Paged, 8, 2, 16, 2, false, true, false>(args, batch,
                                                                           reduce_warps, stream);
    case Group3Schedule::ParallelKeysStagedQuery:
        if constexpr (!Paged) {
            return launch_group3_tiles<Paged, 8, 2, 16, 2, true, true, false>(args, batch,
                                                                              reduce_warps, stream);
        }
        break;
    case Group3Schedule::Standard:
        break;
    }
    launch_group3_standard<Paged, true, false>(args, batch, reduce_warps, stream);
}

} // namespace ninfer::ops::detail
