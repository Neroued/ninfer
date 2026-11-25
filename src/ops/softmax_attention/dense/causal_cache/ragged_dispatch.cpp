#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "core/platform.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/internal.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/ragged_launch.h"
#include "ops/softmax_attention/dense/causal_cache/ragged_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::ops {
namespace {

struct RaggedCausalImplementation {
    void (*short_attention)(const detail::CausalShortLaunch&);
    void (*tiled_attention)(const detail::CausalAttentionOperands&, ConstPagedKVBatchLayerView,
                            detail::CausalTiledMetadata, detail::CausalKvPartition,
                            detail::CausalPartialView, cudaStream_t);
    int (*short_splits)(int heads, int width, int rows, ContextBucket bucket, int sms);
    int tile_rows;
    int fused_width;
    bool tiled_partials;
};

template <auto Select>
int ragged_short_splits(int heads, int width, int rows, ContextBucket bucket, int sms) {
    return Select(heads, width, rows, bucket, sms).partition.capacity;
}

RaggedCausalImplementation ragged_implementation(KvCacheStorage storage) {
    using namespace detail;
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {launch_bf16_ragged_short,
                launch_bf16_ragged_tiled,
                ragged_short_splits<select_bf16_kv_causal_plan>,
                bf16_kv_instance_description(Bf16KvInstance::Tiled64).query_rows,
                kRaggedShortWidth,
                false};
    case KvCacheStorage::Int8Group64:
        return {launch_int8_ragged_short,
                launch_int8_ragged_tiled,
                ragged_short_splits<select_int8_kv_causal_plan>,
                Int8KvTiledInstance::kQueryRows,
                Int8KvCausalPlan::kTokenTile,
                false};
    case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        return {launch_fp8_ragged_short,
                launch_fp8_ragged_tiled,
                ragged_short_splits<select_fp8_kv_causal_plan>,
                Fp8KvTiledInstance::kQueryRows,
                Fp8KvCausalPlan::kTokenTile,
                true};
    case KvCacheStorage::Nvfp4Group16:
        return {launch_nvfp4_ragged_short,
                launch_nvfp4_ragged_tiled,
                ragged_short_splits<select_nvfp4_kv_causal_plan>,
                Nvfp4KvTiledInstance::kQueryRows,
                Nvfp4KvCausalPlan::kTokenTile,
                false};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {launch_k8v4_ragged_short,
                launch_k8v4_ragged_tiled,
                ragged_short_splits<select_k8v4_kv_causal_plan>,
                K8V4KvTiledInstance::kQueryRows,
                K8V4KvCausalPlan::kTokenTile,
                true};
    }
    throw std::invalid_argument("causal_softmax_attention: unsupported storage");
}

template <class Entry>
detail::CausalSegmentMap<Entry> bind_causal_segments(std::span<const Entry> entries,
                                                     MemorySpan storage,
                                                     CudaExecutionView execution) {
    detail::CausalSegmentMap<Entry> map;
    map.count     = static_cast<int>(entries.size());
    map.device_id = execution.device().index();
    if (storage.data) {
        auto* device = static_cast<Entry*>(storage.data);
        detail::stage_causal_segments(entries, device, execution.stream());
        map.device_entries = device;
    } else {
        std::copy(entries.begin(), entries.end(), map.inline_entries);
    }
    return map;
}

struct RaggedEnvelope {
    std::int64_t min_tokens, max_tokens, min_segments, max_segments;
};

// ceil(n/d) for d > 0, n may be negative.
std::int64_t ceil_div(std::int64_t n, std::int64_t d) { return n / d + (n % d > 0); }

std::size_t short_ragged_capacity(RaggedCausalImplementation impl, RaggedEnvelope e, int heads,
                                  ContextBucket bucket, int sms) {
    const std::int64_t visible = context_bucket_tokens(bucket);
    std::size_t maximum        = 0;
    for (int width = 1; width <= detail::kRaggedShortWidth; ++width) {
        // With C short segments of maximum width W their token range is [C + W - 1, C*W].
        // The fewest longs needed is max(0,min_segments-C); the most is max_segments-C.
        const auto all_short_first = std::max(e.min_segments, ceil_div(e.min_tokens, width));
        const auto all_short_last  = std::min(e.max_segments, e.max_tokens - width + 1);
        const auto mixed_first     = std::max<std::int64_t>(
            1, ceil_div((detail::kRaggedShortWidth + 1) * e.min_segments + width - 1 - e.max_tokens,
                            detail::kRaggedShortWidth));
        const auto mixed_last =
            std::min({e.max_segments - 1, e.max_tokens - width - detail::kRaggedShortWidth,
                      -ceil_div(e.min_tokens - visible * e.max_segments, visible - width)});
        const auto last = std::max(all_short_last, mixed_last);
        const int slice = detail::causal_rows_per_launch(std::numeric_limits<int>::max(), width);
        for (int rows = 1; rows <= std::min<std::int64_t>(slice, last); ++rows) {
            // A full slice needs C >= slice. A tail of length rows requires C = rows + k*slice.
            const auto feasible = [&](std::int64_t low, std::int64_t high) {
                if (low > high) return false;
                if (rows == slice) return high >= slice;
                return rows + std::max<std::int64_t>(0, ceil_div(low - rows, slice)) * slice <=
                       high;
            };
            if (!feasible(all_short_first, all_short_last) && !feasible(mixed_first, mixed_last))
                continue;
            WorkspaceLayoutBuilder layout;
            (void)detail::allocate_causal_segment_map<int>(layout, rows);
            const int splits = impl.short_splits(heads, width, rows, bucket, sms);
            (void)detail::allocate_causal_short_partials(layout, heads, width, rows, splits);
            maximum = std::max(maximum, layout.peak_bytes(1));
        }
    }
    return maximum;
}

struct AffineCount {
    std::int64_t slope, intercept;

    std::int64_t at(std::int64_t x) const { return slope * x + intercept; }

    AffineCount operator+(AffineCount rhs) const {
        return {slope + rhs.slope, intercept + rhs.intercept};
    }

    AffineCount operator-(AffineCount rhs) const {
        return {slope - rhs.slope, intercept - rhs.intercept};
    }

    AffineCount operator*(std::int64_t k) const { return {slope * k, intercept * k}; }
};

struct TileInterval {
    std::int64_t first, last;

    void constrain(AffineCount expr, std::int64_t limit) {
        // Intersect with expr(Q) <= limit using exact integer rounding.
        if (expr.slope > 0)
            last = std::min(last, -ceil_div(expr.intercept - limit, expr.slope));
        else if (expr.slope < 0)
            first = std::max(first, ceil_div(expr.intercept - limit, -expr.slope));
        else if (expr.intercept > limit)
            last = 0;
    }
};

int ragged_tiled_direct_waves(int visible, int query_rows) {
    // Once the grid has several waves, small tiles and short contexts get more from
    // reducing partial traffic than from near-perfect wave packing.
    const bool low_work_per_tile =
        query_rows == detail::kRaggedNarrowTile ||
        visible <= static_cast<int>(context_bucket_tokens(ContextBucket::K8));
    if (low_work_per_tile) return 4;
    // On Spark, writing split partials costs more than filling the last wave of
    // an already wide grid. Preserve deeper splitting on the higher-bandwidth target.
    return NINFER_TARGET_SPARK ? 8 : std::numeric_limits<int>::max();
}

int ragged_tiled_split_limit(std::int64_t ctas, int sms, int visible, int query_rows) {
    if (ctas >= ragged_tiled_direct_waves(visible, query_rows) * std::int64_t(sms)) return 1;
    if (ctas >= sms) return 4;
    return detail::kCausalTiledMaxSplits;
}

detail::CausalKvPartition ragged_tiled_partition(int heads, int tiles, int visible, int sms,
                                                 int query_rows) {
    return detail::select_causal_tiled_partition(
        heads, tiles, visible, sms,
        ragged_tiled_split_limit(std::int64_t(heads) * tiles, sms, visible, query_rows));
}

std::size_t tiled_ragged_route_capacity(RaggedEnvelope e, int heads, int visible, int sms, int br,
                                        int maximum_length, int minimum_longest, bool partials) {
    constexpr int short_width = detail::kRaggedShortWidth;
    // Bucket sizes are multiples of both tile heights, which are multiples of short_width.
    const AffineCount tiles{1, 0}, tile_tokens{br, 0};
    const auto short_slack = ceil_div(e.min_tokens, short_width);
    const auto demand_cut  = ceil_div(short_slack, br / short_width);
    const auto period      = std::int64_t(unsigned(sms) / std::gcd(unsigned(sms), unsigned(heads)));

    std::size_t maximum = 0;
    for (bool needs_short_tokens : {false, true}) {
        // The largest possible number of longs for these Q tiles consumes the fewest
        // tokens at the low end and leaves the fewest required shorts at the high end.
        // Its upper bound comes from retaining enough short rows to reach min_tokens.
        const AffineCount available =
            needs_short_tokens ? AffineCount{br / short_width, e.max_segments - short_slack}
                               : AffineCount{0, e.max_segments};
        for (bool all_one_tile : {false, true}) {
            const AffineCount longs = all_one_tile ? tiles : available;
            TileInterval interval{2, std::min(e.max_tokens / (short_width + 1),
                                              e.max_segments * (maximum_length / br))};
            if (needs_short_tokens)
                interval.last = std::min(interval.last, demand_cut - 1);
            else
                interval.first = std::max(interval.first, demand_cut);
            interval.constrain(longs - (all_one_tile ? available : tiles), all_one_tile ? 0 : -1);
            interval.constrain(AffineCount{} - longs, -2);
            interval.constrain(tiles - longs * (maximum_length / br), 0);

            // Every extra tile after a long's first adds Br keys, except the first
            // extra tile replaces the 17-key minimum with Br+1. At least 2*L-Q
            // longs have only one tile. Short rows required by min_segments also cost one
            // token each. Apply both max(0,x) terms as intersections of linear bounds.
            // The wider MXFP8 route requires at least one row longer than the narrow
            // tile. A second tile in any segment already implies that condition.
            const auto extra         = all_one_tile ? minimum_longest - (short_width + 1) : 0;
            const auto base          = (tiles - longs) * br + longs + AffineCount{0, extra};
            const auto one_tile_cost = (longs * 2 - tiles) * short_width;
            const auto short_cost    = AffineCount{0, e.min_segments} - longs;
            interval.constrain(base, e.max_tokens);
            interval.constrain(base + one_tile_cost, e.max_tokens);
            interval.constrain(base + short_cost, e.max_tokens);
            interval.constrain(base + one_tile_cost + short_cost, e.max_tokens);
            if (interval.first > interval.last) continue;

            // Split limits can change at one wave and at the direct-output threshold.
            // Adding a period to Q adds the same number of waves per split to every
            // candidate. Long count and usable tokens only increase, so test the
            // greatest Q of each residue within each region.
            const std::array cuts{
                interval.first, ceil_div(sms, heads),
                ceil_div(ragged_tiled_direct_waves(visible, br) * std::int64_t(sms), heads),
                interval.last + 1};
            for (std::size_t region = 1; region < cuts.size(); ++region) {
                const auto last  = std::min(interval.last, cuts[region] - 1);
                const auto first = std::max({interval.first, cuts[region - 1], last - period + 1});
                if (first > last) continue;
                const int limit =
                    partials ? ragged_tiled_split_limit(last * heads, sms, visible, br) : 0;
                for (auto q = last; q >= first;) {
                    const auto count = longs.at(q);
                    const auto tokens =
                        std::min(tile_tokens.at(q),
                                 e.max_tokens - std::max<std::int64_t>(0, e.min_segments - count));
                    const int splits =
                        partials
                            ? detail::select_causal_tiled_partition(heads, q, visible, sms, limit)
                                  .capacity
                            : 0;
                    WorkspaceLayoutBuilder layout;
                    (void)detail::allocate_causal_segment_map<detail::CausalTiledSegment>(layout,
                                                                                          count);
                    (void)detail::allocate_causal_tiled_partials(layout, heads, tokens, splits);
                    maximum = std::max(maximum, layout.peak_bytes(1));
                    if (limit <= 1) break;

                    // Also skip wave plateaus for hypothetical devices with very large SM counts.
                    std::int64_t same_waves_from = 1;
                    for (int s = 1; s <= limit; ++s) {
                        const auto waves = ceil_div(q * heads * s, sms);
                        same_waves_from =
                            std::max(same_waves_from, (waves - 1) * sms / (heads * s) + 1);
                    }
                    q = same_waves_from - 1;
                }
            }
        }
    }
    return maximum;
}

std::size_t tiled_ragged_capacity(RaggedCausalImplementation impl, RaggedEnvelope e, int heads,
                                  ContextBucket bucket, int sms) {
    const int visible       = context_bucket_tokens(bucket);
    const int shortest_long = detail::kRaggedShortWidth + 1;
    if (!impl.tiled_partials)
        return tiled_ragged_route_capacity(e, heads, visible, sms, impl.tile_rows, visible,
                                           shortest_long, false);

    constexpr int narrow = detail::kRaggedNarrowTile;
    return std::max(
        tiled_ragged_route_capacity(e, heads, visible, sms, narrow, narrow, shortest_long, true),
        tiled_ragged_route_capacity(e, heads, visible, sms, impl.tile_rows, visible, narrow + 1,
                                    true));
}

} // namespace

std::size_t causal_softmax_attention_ragged_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, ContextBucketRange buckets,
    int min_tokens, int max_tokens, int min_segments, int max_segments, int sm_count) {
    constexpr const char* op = "causal_softmax_attention ragged workspace";
    detail::require_causal_geometry(geometry, op);
    buckets.validate(ContextBucket::K256, op);
    (void)common::operand::ragged_envelope(min_tokens, max_tokens, min_segments, max_segments, op);
    if (sm_count <= 0) throw std::invalid_argument("causal_softmax_attention: invalid SM count");

    const auto impl = ragged_implementation(storage);
    const RaggedEnvelope multi{
        min_tokens, max_tokens, std::max(2, min_segments),
        std::min({max_segments, max_tokens, std::numeric_limits<int>::max() - 1})};
    std::size_t maximum = 0;
    buckets.for_each([&](ContextBucket bucket) {
        const int visible = context_bucket_tokens(bucket);
        const auto direct = [&](int first, int last) {
            if (first <= last)
                maximum = std::max(maximum, causal_softmax_attention_workspace_capacity_bytes(
                                                geometry, storage, ContextBucketRange{bucket}, 1,
                                                first, last, sm_count));
        };
        if (min_segments == 1) direct(min_tokens, std::min(max_tokens, visible));
        if (multi.min_segments > multi.max_segments) return;
        const auto min_shorts = multi.min_segments - 1, max_shorts = multi.max_segments - 1;
        const int first_long = std::max<std::int64_t>(
            detail::kRaggedShortWidth + 1, min_tokens - detail::kRaggedShortWidth * max_shorts);
        const int last_long = std::min<std::int64_t>(visible, max_tokens - min_shorts);
        direct(first_long, last_long);
        maximum = std::max(
            maximum, short_ragged_capacity(impl, multi, geometry.query_heads, bucket, sm_count));
        maximum = std::max(
            maximum, tiled_ragged_capacity(impl, multi, geometry.query_heads, bucket, sm_count));
    });
    return maximum;
}

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const RaggedSequenceView& ragged, const Tensor& positions,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    constexpr const char* op = "causal_softmax_attention";
    namespace operand        = common::operand;
    if (workspace.device() != execution.device())
        throw std::invalid_argument(
            "causal_softmax_attention: workspace belongs to another device");

    const auto device = execution.device();
    operand::ragged_batch(ragged, positions, kv_table_rows, op, device);
    detail::require_causal_geometry(geometry, op);
    ContextBucketRange{bucket}.validate(ContextBucket::K256, op);
    if (!std::isfinite(scale) || std::abs(scale - detail::kCausalExpectedScale) > 1.0e-6f)
        throw std::invalid_argument("causal_softmax_attention: scale must be 1/sqrt(256)");

    const int n = ragged.total_tokens(), segments = ragged.num_seqs();
    const int visible           = context_bucket_tokens(bucket);
    const std::size_t alignment = cache.storage == KvCacheStorage::BFloat16 ? 16 : 2;
    operand::tensor(q, DType::BF16, {detail::kCausalHeadDim, geometry.query_heads, n}, alignment,
                    op, "query", device);
    operand::tensor(k, DType::BF16, {detail::kCausalHeadDim, geometry.kv_heads, n}, alignment, op,
                    "key", device);
    operand::tensor(v, DType::BF16, {detail::kCausalHeadDim, geometry.kv_heads, n}, alignment, op,
                    "value", device);
    operand::tensor(out, DType::BF16, {detail::kCausalHeadDim, geometry.query_heads, n}, alignment,
                    op, "output", device);
    const std::array inputs{operand::range(q, op),
                            operand::range(k, op),
                            operand::range(v, op),
                            operand::range(positions, op),
                            operand::range(kv_table_rows, op),
                            operand::range(ragged.cu_seqlens, op),
                            operand::range(ragged.seq_lengths, op),
                            operand::range(ragged.token_to_seq, op)};
    const auto output = operand::range(out, op);
    for (const auto input : inputs) operand::disjoint(output, input, op);
    const auto check_scratch = [&](MemorySpan bytes, const detail::CausalPartialStorage& partial) {
        const auto check = [&](operand::Range scratch) {
            operand::disjoint(output, scratch, op);
            for (const auto input : inputs) operand::disjoint(input, scratch, op);
        };
        if (bytes.data) check(operand::range(bytes.data, bytes.bytes, op));
        for (const auto* tensor : {&partial.acc, &partial.maximum, &partial.sum})
            if (tensor->data) check(operand::range(*tensor, op));
    };
    const int sms                  = execution.multiprocessor_count();
    const auto check_dense_scratch = [&](int width) {
        const auto plan =
            detail::causal_slice_plan(geometry, cache.storage, bucket, 1, width, sms, true);
        if (plan.partial_slots > 0) {
            auto scope = workspace.scope();
            check_scratch({}, detail::allocate_causal_partials(workspace, geometry.query_heads,
                                                               width, plan.partial_slots, 1));
        }
    };
    // The native INT8 and FP8 epilogues handle BF16 output at an odd element offset.
    const bool use_dense = (cache.storage != KvCacheStorage::Int8Group64 &&
                            cache.storage != KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) ||
                           operand::aligned(out.data, 4);
    if (segments == 1 && use_dense) {
        check_dense_scratch(n);
        causal_softmax_attention(q, k, v, positions, Tensor{}, kv_table_rows, geometry, scale,
                                 cache, bucket, workspace, out, execution);
        return;
    }
    detail::validate_causal_batch_cache(cache, geometry.kv_heads, op, device);

    const auto impl           = ragged_implementation(cache.storage);
    constexpr int short_limit = detail::kRaggedShortWidth;
    int short_width = 0, tiled_tiles = 0, tiled_tokens = 0, longest = 0;
    // Avoid host allocation for the common inline case. Keep only one prefix entry per segment.
    std::array<int, detail::kInlineSeqs> short_inline;
    std::array<detail::CausalTiledSegment, detail::kInlineSeqs> tiled_inline;
    std::vector<int> short_overflow;
    std::vector<detail::CausalTiledSegment> tiled_overflow;
    int short_count = 0, tiled_count = 0;
    const auto push = [](auto entry, auto& inline_entries, auto& overflow, int& count) {
        const int capacity = inline_entries.size();
        if (count < capacity) {
            inline_entries[count] = entry;
        } else {
            if (count == capacity) overflow.assign(inline_entries.begin(), inline_entries.end());
            overflow.push_back(entry);
        }
        ++count;
    };
    for (int s = 0; s < segments; ++s) {
        const int len = ragged.segment_length(s);
        if (len > visible)
            throw std::invalid_argument(
                "causal_softmax_attention: context bucket is shorter than a segment");
        if (len <= short_limit) {
            short_width = std::max(short_width, len);
            push(s, short_inline, short_overflow, short_count);
        } else {
            longest = std::max(longest, len);
            tiled_tiles += (len + impl.tile_rows - 1) / impl.tile_rows;
            tiled_tokens += len;
            push(detail::CausalTiledSegment{s, tiled_tiles, tiled_tokens}, tiled_inline,
                 tiled_overflow, tiled_count);
        }
    }

    const auto stream   = execution.stream();
    const auto* prefix  = static_cast<const int*>(ragged.cu_seqlens.data);
    const auto* lengths = static_cast<const int*>(ragged.seq_lengths.data);
    const auto write    = [&](int max_length) {
        detail::kv::launch_paged_write_ragged(k, v, positions, ragged.token_to_seq,
                                                 ragged.seq_lengths, kv_table_rows, cache, max_length,
                                                 stream);
    };

    const bool dense_long = tiled_count == 1 && use_dense;
    if (dense_long) {
        const int segment = tiled_inline[0].segment;
        const int begin = ragged.segment_begin(segment), len = ragged.segment_length(segment);
        auto result = out.slice(2, begin, len);
        check_dense_scratch(len);
        causal_softmax_attention(q.slice(2, begin, len), k.slice(2, begin, len),
                                 v.slice(2, begin, len), positions.slice(0, begin, len), Tensor{},
                                 kv_table_rows.slice(0, segment, 1), geometry, scale, cache, bucket,
                                 workspace, result, execution);
    }
    const bool native_long = tiled_count > 0 && !dense_long;
    if (native_long) {
        auto scope         = workspace.scope();
        const auto entries = tiled_overflow.empty() ? std::span(tiled_inline.data(), tiled_count)
                                                    : std::span(tiled_overflow);
        const auto& tables = cache.k_pages.records.page_offsets;
        // With every long segment within the narrow tile, the wider schedule's
        // prefix sums already count one Q tile per segment.
        const int tile_rows = impl.tiled_partials && longest <= detail::kRaggedNarrowTile
                                  ? detail::kRaggedNarrowTile
                                  : impl.tile_rows;
        // Adjacent CTAs for one tile's GQA heads reuse the tile's KV pages in L2.
        // Short-context BF16 batches with multiple tiles can instead reuse KV
        // across a head's consecutive tiles.
        const bool heads_first =
            impl.tiled_partials
                ? tile_rows == detail::kRaggedNarrowTile || sms >= 4 * geometry.query_heads ||
                      visible >= static_cast<int>(context_bucket_tokens(ContextBucket::K64))
                : cache.storage == KvCacheStorage::BFloat16 &&
                      (bucket != ContextBucket::K2 ||
                       (tiled_tiles == tiled_count &&
                        std::int64_t(tiled_tiles) * geometry.query_heads >= 4 * std::int64_t(sms)));
        const auto segment_map =
            detail::allocate_causal_segment_map<detail::CausalTiledSegment>(workspace, tiled_count);
        const auto partition =
            impl.tiled_partials && tiled_count > 1
                ? ragged_tiled_partition(geometry.query_heads, tiled_tiles, visible, sms, tile_rows)
                : detail::CausalKvPartition{};
        const auto partial = detail::allocate_causal_tiled_partials(
            workspace, geometry.query_heads, tiled_tokens, partition.capacity);
        check_scratch(segment_map, partial);
        const detail::CausalTiledMetadata metadata{
            bind_causal_segments<detail::CausalTiledSegment>(entries, segment_map, execution),
            prefix,
            lengths,
            static_cast<const std::int64_t*>(tables.data),
            static_cast<const int*>(kv_table_rows.data),
            tables.ne[0],
            tiled_tiles,
            tile_rows,
            heads_first};
        auto p  = detail::make_causal_operands(q, positions, out, scale, visible);
        p.width = tiled_tokens;
        write(0);
        impl.tiled_attention(p, cache, metadata, partition, partial.view(), stream);
    }
    if (short_count) {
        const bool fused = !native_long && short_width <= impl.fused_width;
        const detail::CausalAppendInput input{
            fused ? static_cast<const __nv_bfloat16*>(k.data) : nullptr,
            fused ? static_cast<const __nv_bfloat16*>(v.data) : nullptr};
        const auto entries = short_overflow.empty() ? std::span(short_inline.data(), short_count)
                                                    : std::span(short_overflow);
        const int rows     = detail::causal_rows_per_launch(short_count, short_width);
        auto p             = detail::make_causal_operands(q, positions, out, scale, visible);
        p.width            = short_width;
        for (std::size_t begin = 0; begin < entries.size(); begin += rows) {
            auto scope = workspace.scope();
            const auto slice =
                entries.subspan(begin, std::min<std::size_t>(rows, entries.size() - begin));
            p.batch                = static_cast<int>(slice.size());
            const auto segment_map = detail::allocate_causal_segment_map<int>(workspace, p.batch);
            const auto partial     = detail::allocate_causal_short_partials(
                workspace, p.query_heads, p.width, p.batch,
                impl.short_splits(p.query_heads, p.width, p.batch, bucket, sms));
            check_scratch(segment_map, partial);
            if (begin == 0 && !fused && !native_long) write(dense_long ? short_limit : 0);
            const detail::CausalShortMetadata metadata{
                bind_causal_segments<int>(slice, segment_map, execution), prefix, lengths, visible};
            impl.short_attention(
                {p, metadata, input, cache, kv_table_rows, partial.view(), bucket, sms, stream});
        }
    }
}

} // namespace ninfer::ops
