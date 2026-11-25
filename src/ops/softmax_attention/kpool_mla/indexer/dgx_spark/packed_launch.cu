// Keep the packed kernel ABI in its own translation unit so CUDA code generation
// for the rectangular Graph kernels retains the original shared-memory layout.
#define NINFER_KPOOL_PACKED
#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/indexer/score_bf16.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/score_fp8.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/query_fp8.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/select.cuh"
#include "ops/common/host/launch.h"

#include <type_traits>

namespace ninfer::ops::detail::kpool {
namespace {
__global__ void indexer_idle_kernel() {}

__global__ void packed_indexer_all_kernel(const __grid_constant__ IndexerArguments a) {
    for (std::int64_t column = blockIdx.x; column < a.width; column += gridDim.x) {
        const int end = a.positions[column] + 1, count = end / 4;
        if (threadIdx.x == 0) {
            a.ends[column]   = end;
            a.counts[column] = count;
        }
        auto* ids = a.ids + column * kIndexSelected;
        if (reinterpret_cast<std::uintptr_t>(ids) % alignof(int4) == 0) {
            auto* vectors = reinterpret_cast<int4*>(ids);
            for (int v = threadIdx.x; v < count / 4; v += blockDim.x) {
                const int i = 4 * v;
                vectors[v]  = make_int4(i, i + 1, i + 2, i + 3);
            }
            const int i = count / 4 * 4 + threadIdx.x;
            if (i < count) ids[i] = i;
        } else {
            for (int i = threadIdx.x; i < count; i += blockDim.x) ids[i] = i;
        }
    }
}

template <int Q, int G, int Stages>
void score(PackedIndexerArguments a, IndexerSchedule s, cudaStream_t stream) {
    constexpr auto kernel = packed_indexer_score_kernel<Q, G, Stages>;
    const int shared      = common::prepare_shared<Stages * G * 128 * 2, kernel, true>();
    kernel<<<s.score_workers, (Q < 4 ? 4 : Q) * 32, shared, stream>>>(a);
}

void launch_selection(IndexerArguments a, int workers, bool resident, cudaStream_t stream) {
    const auto partition = [&]<int Items>() {
        if (resident)
            packed_indexer_partition_words_kernel<4, Items><<<workers, 256, 0, stream>>>(a);
        else
            packed_indexer_partition_words_kernel<1, Items><<<workers, 256, 0, stream>>>(a);
    };
    switch (indexer_selection_items(a.columns)) {
    case 4:
        partition.template operator()<4>();
        break;
    case 8:
        partition.template operator()<8>();
        break;
    default:
        partition.template operator()<16>();
        break;
    }
    if (a.partitions > kIndexFanIn)
        packed_indexer_merge_kernel<false>
            <<<a.width * a.batch * a.merge_groups, 256, 0, stream>>>(a);
    else
        indexer_idle_kernel<<<1, 1, 0, stream>>>();
    if (a.partitions > 1)
        packed_indexer_merge_kernel<true><<<a.width * a.batch, 256, 0, stream>>>(a);
    else
        indexer_idle_kernel<<<1, 1, 0, stream>>>();
}

void launch_indexer_window(PackedIndexerArguments a, IndexerSchedule s, cudaStream_t stream) {
    const auto submit = [&]<int Q, int G>() {
        if (s.stages == 2)
            score<Q, G, 2>(a, s, stream);
        else
            score<Q, G, 1>(a, s, stream);
    };
    switch (s.queries) {
    case 1:
        submit.template operator()<1, 64>();
        break;
    case 2:
        submit.template operator()<2, 64>();
        break;
    default:
        submit.template operator()<4, 32>();
        break;
    }
    launch_selection(a, s.select_workers, s.resident_selection, stream);
}

template <Fp8IndexerRoute Route>
void score_fp8(PackedFp8IndexerArguments a, Fp8IndexerSchedule s, cudaStream_t stream) {
    constexpr auto c      = fp8_indexer_config(Route);
    constexpr auto kernel = packed_indexer_score_fp8_kernel<c.queries, c.keys, c.stages, c.blocks,
                                                            c.warp_queries, c.epilogue_prefetch>;
    constexpr int warps   = c.queries / c.warp_queries;
    const int shared      = common::prepare_shared<c.stages * c.keys * 129, kernel, true>();
    kernel<<<s.score_workers, (warps < 4 ? 4 : warps) * 32, shared, stream>>>(a);
}

void launch_fp8_window(PackedFp8IndexerArguments a, Fp8IndexerSchedule s, cudaStream_t stream) {
    switch (s.route) {
    case Fp8IndexerRoute::Q1K64S2:
        score_fp8<Fp8IndexerRoute::Q1K64S2>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q2K64S2:
        score_fp8<Fp8IndexerRoute::Q2K64S2>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q4K32S1:
        score_fp8<Fp8IndexerRoute::Q4K32S1>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q4K32S2:
        score_fp8<Fp8IndexerRoute::Q4K32S2>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q4K64S3:
        score_fp8<Fp8IndexerRoute::Q4K64S3>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q8K16Prefetch:
        score_fp8<Fp8IndexerRoute::Q8K16Prefetch>(a, s, stream);
        break;
    }
    launch_selection(a.index, s.select_workers, s.resident_selection, stream);
}

IndexerArguments segment_window(IndexerArguments a, const RaggedSequenceView& seq, int segment) {
    const int offset = seq.segment_begin(segment);
    a.query += std::int64_t(offset) * 4096;
    a.coefficients += std::int64_t(offset) * 32;
    a.positions += offset;
    a.valid += segment;
    a.table_rows += segment;
    a.ids += std::int64_t(offset) * kIndexSelected;
    a.counts += offset;
    a.ends += offset;
    a.tokens = seq.segment_length(segment);
    a.begin -= offset;
    return a;
}

template <class Arguments>
void launch_packed_windows(Arguments input, const RaggedSequenceView& seq, int window, int sms,
                           cudaStream_t stream) {
    constexpr bool Fp8 = std::is_same_v<Arguments, Fp8IndexerArguments>;
    auto& a            = [&]() -> IndexerArguments& {
        if constexpr (Fp8)
            return input.index;
        else
            return input;
    }();
    if (a.columns <= kIndexSelected) {
        a.width = a.tokens;
        packed_indexer_all_kernel<<<std::min(a.width, 4 * sms), 128, 0, stream>>>(a);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    int segment = 0;
    for (int begin = 0; begin < a.tokens;) {
        a.begin       = begin;
        a.width       = std::min(window, a.tokens - begin);
        const int end = begin + a.width;
        while (seq.segment_begin(segment + 1) <= begin) ++segment;
        // A contained window uses the original key-reuse schedule directly.
        if (end <= seq.segment_begin(segment + 1)) {
            const auto rect = segment_window(a, seq, segment);
            if constexpr (Fp8) {
                auto f  = input;
                f.index = rect;
                launch_fp8_indexer_segment(f, sms, stream);
            } else {
                launch_indexer_segment(rect, sms, stream);
            }
        } else {
            int largest = 0;
            int last    = segment;
            for (int col = begin; col < end; ++last) {
                const int stop = std::min(end, seq.segment_begin(last + 1));
                largest        = std::max(largest, stop - col);
                col            = stop;
            }
            // Select a tile size from the largest contiguous span. Use the
            // actual packed width for the selection worker count below.
            auto shape  = a;
            shape.width = largest;
            PackedIndexerArguments packed{a, static_cast<const int*>(seq.token_to_seq.data), {}, 0};
            const auto populate_tiles = [&](int queries) {
                for (int s = segment, col = begin; s < last; ++s) {
                    const int stop = std::min(end, seq.segment_begin(s + 1));
                    while (col < stop) {
                        const int take                    = std::min(queries, stop - col);
                        packed.tiles[packed.tile_count++] = (take << 8) | (col - begin);
                        col += take;
                    }
                }
            };
            const int workers = std::min(4 * sms, a.width * a.partitions);
            if constexpr (Fp8) {
                auto schedule     = select_fp8_indexer(shape, sms);
                const auto config = fp8_indexer_config(schedule.route);
                populate_tiles(config.queries);
                schedule.score_workers      = std::max(config.blocks * sms, packed.tile_count);
                schedule.select_workers     = workers;
                schedule.resident_selection = workers > 2 * sms;
                PackedFp8IndexerArguments f{packed, input.keys, input.scales, input.query,
                                            input.coefficients};
                packed_indexer_query_fp8_kernel<<<a.width * 8, 128, 0, stream>>>(f);
                launch_fp8_window(f, schedule, stream);
            } else {
                auto schedule = select_indexer(shape, sms);
                populate_tiles(schedule.queries);
                const int blocks            = schedule.queries < 4 && schedule.stages == 2 ? 3 : 4;
                schedule.score_workers      = std::max(blocks * sms, packed.tile_count);
                schedule.select_workers     = workers;
                schedule.resident_selection = workers > 2 * sms;
                launch_indexer_window(packed, schedule, stream);
            }
        }
        begin = end;
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void launch_packed_indexer(IndexerArguments a, const RaggedSequenceView& seq, int window, int sms,
                           cudaStream_t stream) {
    launch_packed_windows(a, seq, window, sms, stream);
}

void launch_packed_fp8_indexer(Fp8IndexerArguments a, const RaggedSequenceView& seq, int window,
                               int sms, cudaStream_t stream) {
    launch_packed_windows(a, seq, window, sms, stream);
}
} // namespace ninfer::ops::detail::kpool
