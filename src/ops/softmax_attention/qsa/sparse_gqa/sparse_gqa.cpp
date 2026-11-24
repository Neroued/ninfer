#include "ninfer/ops/qsa.h"
#include "core/layout.h"
#include "ops/common/host/graph_profile.h"
#include "ops/softmax_attention/qsa/geometry.h"
#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/softmax_attention/qsa/validation.h"

#include <vector>

namespace ninfer::ops {
namespace {
using namespace detail::qsa;

void validate_profile(KvCacheStorage storage, int query_heads, int kv_heads, int batch, int lo,
                      int hi, int sms) {
    constexpr ViewValidation v{"qsa_sparse_gqa"};
    v.require(query_heads == registered_query_heads(kv_heads) && query_heads > 0, "query/KV heads");
    v.require(
        storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0 ||
            storage == KvCacheStorage::Int8Group64 || storage == KvCacheStorage::Nvfp4Group16 ||
            storage == KvCacheStorage::Fp8KeyNvfp4Value,
        "cache storage");
    v.require(lo > 0 && hi >= lo && batch >= 1, "T/B interval");
    v.require(sms > 0, "SM count");
}

struct WorkspaceLayout {
    TensorRegion numerator, maximum, sum;
    std::size_t bytes = 0;
};

WorkspaceLayout workspace_layout(int tokens, int batch, int query_heads, SparseGqaPlan plan) {
    if (!plan.partial) return {};
    const int parts = batch * std::max(plan.splits, plan.natural_splits);
    LayoutBuilder builder;
    WorkspaceLayout layout{
        builder.add_tensor(DType::FP32, {256, query_heads, tokens, parts}, 256, "QSA numerator"),
        builder.add_tensor(DType::FP32, {query_heads, tokens, parts}, 256, "QSA maximum"),
        builder.add_tensor(DType::FP32, {query_heads, tokens, parts}, 256, "QSA sum")};
    layout.bytes = builder.finish(1);
    return layout;
}

struct PackedWindow {
    SparseGqaPlan plan{};
    WorkspaceLayout layout{};
};

struct PackedRun {
    int begin = 0, tokens = 0;
    int direct_segment = -1;
    int width          = 0;
    PackedWindow full, tail;
};

PackedWindow packed_window(KvCacheStorage storage, int query_heads, int kv_heads,
                           ContextBucket bucket, int width, int total_tokens, int segments,
                           int sms) {
    if (!width) return {};
    const auto plan = select_sparse_gqa_packed_window_plan(storage, width, total_tokens, segments,
                                                           kv_heads, bucket, sms);
    return {plan, workspace_layout(width, 1, query_heads, plan)};
}

SparseGqaWorkspace bind_workspace(MemorySpan storage, const WorkspaceLayout& layout) {
    if (!layout.bytes) return {};
    return {layout.numerator.bind(storage), layout.maximum.bind(storage), layout.sum.bind(storage)};
}
} // namespace

std::size_t qsa_sparse_gqa_workspace_capacity_bytes(KvCacheStorage storage, int query_heads,
                                                    int kv_heads, ContextBucketRange buckets,
                                                    int batch, int lo, int hi, int sms) {
    buckets.validate(ContextBucket::K256, "qsa_sparse_gqa");
    validate_profile(storage, query_heads, kv_heads, batch, lo, hi, sms);
    std::size_t maximum = 0;
    // Above this bound every extent uses direct output. Below it, allocation is not monotonic
    // because the split count decreases with query parallelism.
    const auto end =
        std::min<std::int64_t>(hi, (sparse_gqa_direct_queries(kv_heads, sms) - 1) / batch);
    buckets.for_each([&](ContextBucket bucket) {
        for (std::int64_t t = lo; t <= end; ++t) {
            const auto plan = select_sparse_gqa_plan(storage, int(t), batch, kv_heads, bucket, sms);
            maximum = std::max(maximum, workspace_layout(int(t), batch, query_heads, plan).bytes);
        }
    });
    return maximum;
}

OpGraphProfile qsa_sparse_gqa_graph_profile(KvCacheStorage storage, int query_heads, int kv_heads,
                                            ContextBucket bucket, int batch, int tokens, int sms) {
    ContextBucketRange{bucket}.validate(ContextBucket::K256, "qsa_sparse_gqa");
    validate_profile(storage, query_heads, kv_heads, batch, tokens, tokens, sms);
    const auto plan   = select_sparse_gqa_plan(storage, tokens, batch, kv_heads, bucket, sms);
    const auto layout = workspace_layout(tokens, batch, query_heads, plan);
    const auto flags =
        std::uint64_t(plan.instance) | (std::uint64_t(plan.partial) << 8) |
        (std::uint64_t(plan.direct_single) << 9) | (std::uint64_t(plan.delay_softmax) << 10) |
        (std::uint64_t(plan.merge_node) << 11) | (std::uint64_t(plan.reuse_buffer) << 12);
    std::array<std::uint64_t, 22> words{std::uint64_t(storage),
                                        std::uint64_t(query_heads),
                                        std::uint64_t(kv_heads),
                                        std::uint64_t(batch),
                                        std::uint64_t(tokens),
                                        std::uint64_t(sms),
                                        std::uint64_t(plan.key_rows),
                                        std::uint64_t(plan.splits),
                                        std::uint64_t(plan.natural_splits),
                                        std::uint64_t(plan.min_split_tiles),
                                        flags,
                                        layout.bytes,
                                        layout.numerator.region.offset,
                                        layout.numerator.region.bytes,
                                        layout.maximum.region.offset,
                                        layout.maximum.region.bytes,
                                        layout.sum.region.offset,
                                        layout.sum.region.bytes};
    std::size_t at             = 18;
    const auto append_launches = [&](int width) {
        for (const bool vector_cache : {false, true}) {
            const auto launch = select_sparse_gqa_launch(storage, kv_heads, bucket, plan, width,
                                                         batch, sms, vector_cache);
            words[at++]       = std::uint64_t(launch.splits) | (std::uint64_t(launch.load) << 32) |
                          (std::uint64_t(launch.kernel) << 40) |
                          (std::uint64_t(launch.vector_cache) << 48);
        }
    };
    const auto queries = std::int64_t(tokens) * batch;
    append_launches(int(std::min<std::int64_t>(queries, kSparseGqaSlab)));
    if (queries > kSparseGqaSlab && queries % kSparseGqaSlab)
        append_launches(int(queries % kSparseGqaSlab));
    return detail::make_op_graph_profile(words);
}

namespace {
void sparse_gqa(const Tensor& query, const Tensor& gate, const ConstPagedKVBatchLayerView& cache,
                const QsaBatchView& batch, const QsaBlockSelection& selected, ContextBucket bucket,
                MemoryArena& workspace, Tensor& out, CudaExecutionView execution,
                const RaggedSequenceView* ragged = nullptr) {
    ContextBucketRange{bucket}.validate(ContextBucket::K256, "qsa_sparse_gqa");
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("QSA workspace belongs to another device");
    }
    const detail::qsa::ViewValidation v{"qsa_sparse_gqa", execution.device()};
    const int tokens = query.ne[2], rows = query.ne[3];
    const int query_heads = query.ne[1], kv_heads = cache.num_kv_heads;
    const int sms = execution.multiprocessor_count();
    validate_profile(cache.storage, query_heads, kv_heads, rows, tokens, tokens, sms);
    v.dense(query, DType::BF16, {256, query_heads, tokens, rows}, 16, "query");
    v.dense(gate, DType::BF16, {256, query_heads, tokens, rows}, 16, "gate");
    v.dense(out, DType::BF16, {256, query_heads, tokens, rows}, 16, "output");
    v.require(cache.head_dim == 256, "cache geometry");
    const auto layout = paged_kv_storage_layout(cache.storage, 256);
    const auto plane  = [&](const ConstPagedPlaneView& data, const ConstPagedPlaneView& scales,
                           PagedKVVectorLayout format) {
        common::operand::paged_plane(
            data, format.data_dtype, {format.data_leading_extent, 64, kv_heads},
            dtype_size(format.data_dtype), "qsa_sparse_gqa", "cache data", execution.device());
        common::operand::same_paged_records(cache.k_pages, data, "qsa_sparse_gqa", "cache");
        v.require(data.records.page_tokens == 64 && data.plane.geometry.tokens_per_row == 1,
                   "record geometry");
        if (format.has_scale()) {
            common::operand::paged_plane(scales, format.scale_dtype,
                                          {format.scale_leading_extent, 64, kv_heads},
                                          dtype_size(format.scale_dtype), "qsa_sparse_gqa",
                                          "cache scales", execution.device());
            common::operand::same_paged_records(data, scales, "qsa_sparse_gqa", "scales");
        } else {
            v.require(!scales.data(), "absent scales");
        }
    };
    plane(cache.k_pages, cache.k_scale_pages, layout.key);
    plane(cache.v_pages, cache.v_scale_pages, layout.value);
    v.dense(batch.positions, DType::I32, {tokens, rows, 1, 1}, 4, "positions");
    v.metadata(batch.valid_columns, ragged ? ragged->num_seqs() : rows, "valid columns");
    v.metadata(batch.table_rows, ragged ? ragged->num_seqs() : rows, "table rows");
    v.dense(selected.block_ids, DType::I32, {512, tokens, rows, 1}, 4, "block ids");
    v.dense(selected.block_counts, DType::I32, {tokens, rows, 1, 1}, 4, "block counts");
    const Tensor* inputs[]{&query,
                           &gate,
                           &cache.k_pages.records.page_offsets,
                           &batch.positions,
                           &batch.valid_columns,
                           &batch.table_rows,
                           &selected.block_ids,
                           &selected.block_counts,
                           ragged ? &ragged->cu_seqlens : nullptr,
                           ragged ? &ragged->token_to_seq : nullptr};
    for (const auto* input : inputs)
        if (input && input->data) v.require(!v.overlap(out, *input), "output/input overlap");
    const auto check_scratch = [&](const SparseGqaWorkspace& scratch) {
        for (const auto* work : {&scratch.numerator, &scratch.maximum, &scratch.sum}) {
            if (!work->data) continue;
            v.require(!v.overlap(*work, out), "workspace/output overlap");
            for (const auto* input : inputs)
                if (input && input->data)
                    v.require(!v.overlap(*work, *input), "workspace/input overlap");
        }
    };
    auto scope = workspace.scope();
    if (!ragged || ragged->num_seqs() == 1) {
        const auto plan =
            select_sparse_gqa_plan(cache.storage, tokens, rows, kv_heads, bucket, sms);
        const auto layout  = workspace_layout(tokens, rows, query_heads, plan);
        const auto storage = layout.bytes ? workspace.alloc_bytes(layout.bytes, 256) : MemorySpan{};
        const auto scratch = bind_workspace(storage, layout);
        check_scratch(scratch);
        launch_sparse_gqa(query, gate, cache, batch, selected, bucket, plan, scratch, out,
                          execution);
        return;
    }

    const auto offsets = ragged->host_cu_seqlens;
    const int cutoff   = sparse_gqa_long_direct_tokens(bucket);
    PackedRun first;
    std::vector<PackedRun> remaining;
    std::size_t peak  = 0;
    const auto append = [&](PackedRun run) {
        peak = std::max({peak, run.full.layout.bytes, run.tail.layout.bytes});
        if (!first.tokens)
            first = run;
        else
            remaining.push_back(run);
    };
    const auto append_short = [&](int first_segment, int end_segment, bool peeled) {
        if (first_segment == end_segment) return;
        const auto windows = sparse_gqa_packed_windows(cache.storage, kv_heads, bucket, peeled);
        const int count    = end_segment - first_segment;
        const int n        = offsets[end_segment] - offsets[first_segment];
        const int width    = std::min(n, windows.limit(n, count));
        append({offsets[first_segment], n, -1, width,
                packed_window(cache.storage, query_heads, kv_heads, bucket, width, n, count, sms),
                packed_window(cache.storage, query_heads, kv_heads, bucket, n % width, n, count,
                              sms)});
    };
    int short_begin = 0;
    if (cutoff) {
        for (int s = 0; s < ragged->num_seqs(); ++s) {
            const int n = offsets[s + 1] - offsets[s];
            if (n < cutoff) continue;
            append_short(short_begin, s, true);
            append({offsets[s], n, s});
            short_begin = s + 1;
        }
    }
    append_short(short_begin, ragged->num_seqs(), short_begin != 0);

    const auto storage      = peak ? workspace.alloc_bytes(peak, 256) : MemorySpan{};
    const auto validate_run = [&](const PackedRun& run) {
        check_scratch(bind_workspace(storage, run.full.layout));
        check_scratch(bind_workspace(storage, run.tail.layout));
    };
    validate_run(first);
    for (const auto& run : remaining) validate_run(run);

    const auto launch_run = [&](const PackedRun& run) {
        if (run.direct_segment >= 0) {
            const int s = run.direct_segment, begin = run.begin, n = run.tokens;
            auto plan   = select_sparse_gqa_plan(cache.storage, n, 1, kv_heads, bucket, sms);
            plan.splits = 1;
            plan.natural_splits = 0;
            plan.partial = plan.merge_node = false;
            const QsaBatchView single{batch.positions.slice(0, begin, n),
                                      batch.valid_columns.slice(0, s, 1),
                                      batch.table_rows.slice(0, s, 1)};
            const QsaBlockSelection blocks{selected.block_ids.slice(1, begin, n),
                                           selected.block_counts.slice(0, begin, n)};
            auto destination = out.slice(2, begin, n);
            launch_sparse_gqa(query.slice(2, begin, n), gate.slice(2, begin, n), cache, single,
                              blocks, bucket, plan, {}, destination, execution);
            return;
        }
        const auto full = bind_workspace(storage, run.full.layout);
        const auto tail = bind_workspace(storage, run.tail.layout);
        for (std::int64_t offset = 0; offset < run.tokens; offset += run.width) {
            const auto n        = int(std::min<std::int64_t>(run.width, run.tokens - offset));
            const bool complete = n == run.width;
            launch_sparse_gqa(query, gate, cache, batch, selected, bucket,
                              complete ? run.full.plan : run.tail.plan, complete ? full : tail, out,
                              execution, &ragged->token_to_seq, run.begin + offset, n);
        }
    };
    launch_run(first);
    for (const auto& run : remaining) launch_run(run);
}
} // namespace

void qsa_sparse_gqa(const Tensor& query, const Tensor& gate,
                    const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                    const QsaBlockSelection& selected, ContextBucket bucket, MemoryArena& workspace,
                    Tensor& out, CudaExecutionView execution) {
    sparse_gqa(query, gate, cache, batch, selected, bucket, workspace, out, execution);
}

std::size_t qsa_sparse_gqa_ragged_workspace_capacity_bytes(KvCacheStorage storage, int query_heads,
                                                           int kv_heads, ContextBucketRange buckets,
                                                           int min_tokens, int max_tokens,
                                                           int min_segments, int max_segments,
                                                           int sm_count) {
    (void)common::operand::ragged_envelope(min_tokens, max_tokens, min_segments, max_segments,
                                           "qsa_sparse_gqa");
    buckets.validate(ContextBucket::K256, "qsa_sparse_gqa");
    validate_profile(storage, query_heads, kv_heads, 1, min_tokens, max_tokens, sm_count);
    std::size_t peak =
        min_segments == 1
            ? qsa_sparse_gqa_workspace_capacity_bytes(storage, query_heads, kv_heads, buckets, 1,
                                                      min_tokens, max_tokens, sm_count)
            : 0;
    if (max_segments == 1 || max_tokens < 2) return peak;

    buckets.for_each([&](ContextBucket bucket) {
        const auto windows   = sparse_gqa_packed_windows(storage, kv_heads, bucket);
        const int cutoff     = sparse_gqa_long_direct_tokens(bucket);
        const auto max_short = std::int64_t(cutoff ? cutoff - 1 : std::numeric_limits<int>::max());
        constexpr auto pivot = SparseGqaPackedWindows::kLongSegment;
        const auto take      = [&](int width, int tokens, int segments) {
            peak = std::max(peak, packed_window(storage, query_heads, kv_heads, bucket, width,
                                                     tokens, segments, sm_count)
                                           .layout.bytes);
        };
        const auto window_period = [&](int period, std::int64_t lo, std::int64_t hi) {
            lo = std::max<std::int64_t>(lo, kSparseGqaPackedSkewQueries + 1);
            if (hi < lo) return;
            const auto last_partial = sparse_gqa_direct_queries(kv_heads, sm_count) - 1;
            const auto end          = std::min<std::int64_t>({period, hi, last_partial});
            for (int width = 1; width <= end; ++width) {
                // A full window is witnessed by any total at least as large as
                // the period. For a remainder use the greatest matching total.
                const auto n = width == period ? hi : hi - (hi - width) % period;
                // Above the small-run cutoff, the split plan depends only on width.
                if (n >= lo) take(width, static_cast<int>(n), 1);
            }
        };

        // Unpeeled calls require S <= N <= S * max_short. For each mean-width
        // band the possible token counts form a continuous interval.
        const int lo_segments = std::max(2, min_segments);
        const int lo_tokens   = std::max(min_tokens, lo_segments);
        const auto hi_tokens  = std::min<std::int64_t>(max_tokens, max_segments * max_short);
        for (int n = lo_tokens; n <= std::min<std::int64_t>(hi_tokens, kSparseGqaPackedSkewQueries);
             ++n)
            for (int s = lo_segments; s <= std::min(n, max_segments); ++s) take(n, n, s);

        if (windows.short_width == windows.long_width)
            window_period(windows.short_width, lo_tokens, hi_tokens);
        else {
            window_period(windows.short_width, lo_tokens,
                          std::min(hi_tokens, std::int64_t(max_segments) * pivot - 1));
            window_period(windows.long_width,
                          std::max<std::int64_t>(lo_tokens, std::int64_t(lo_segments) * pivot),
                          hi_tokens);
        }

        // A peeled segment absorbs any extra tokens. Reserve one token for each
        // other segment required by min_segments. One peeled segment suffices
        // to witness every feasible short run, including runs of one segment.
        const int budget    = max_tokens - cutoff;
        const int min_other = min_segments - 1, max_run_segments = max_segments - 1;
        if (!cutoff || budget < std::max(1, min_other)) return;
        const auto max_run = std::min<std::int64_t>(budget, max_run_segments * max_short);
        for (int n = 1; n <= std::min<std::int64_t>(max_run, kSparseGqaPackedSkewQueries); ++n) {
            const int min_run_segments = std::max(1, min_other - (budget - n));
            for (int s = min_run_segments; s <= std::min(n, max_run_segments); ++s) take(n, n, s);
        }
        const auto peeled = sparse_gqa_packed_windows(storage, kv_heads, bucket, true);
        if (peeled.short_width == peeled.long_width)
            window_period(peeled.short_width, 1, max_run);
        else {
            window_period(peeled.short_width, 1,
                          std::min(max_run, std::int64_t(max_run_segments) * pivot - 1));
            // A long-mean run can have at most floor(n / pivot) segments. With
            // the other segments reserved, n - floor(n / pivot) <= spare.
            const auto spare     = std::int64_t(budget - min_other);
            const auto last_long = spare + spare / (pivot - 1);
            window_period(peeled.long_width, pivot, std::min(max_run, last_long));
        }
    });
    return peak;
}

void qsa_sparse_gqa(const Tensor& query, const Tensor& gate,
                    const ConstPagedKVBatchLayerView& cache, const QsaRaggedBatchView& batch,
                    const QsaBlockSelection& selected, ContextBucket bucket, MemoryArena& workspace,
                    Tensor& out, CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "qsa_sparse_gqa";
    const auto& seq          = batch.ragged;
    const auto device        = execution.device();
    operand::ragged_batch(seq, batch.positions, batch.table_rows, op, device);
    const int n = seq.total_tokens(), h = query.ne[1];
    for (const auto* t : std::array<const Tensor*, 3>{&query, &gate, &out})
        operand::tensor(*t, DType::BF16, {256, h, n}, 16, op, "activation", device);
    operand::tensor(selected.block_ids, DType::I32, {512, n}, 4, op, "block ids", device);
    operand::tensor(selected.block_counts, DType::I32, {n}, 4, op, "block counts", device);
    const QsaBatchView flat{batch.positions, seq.seq_lengths, batch.table_rows};
    sparse_gqa(query, gate, cache, flat, selected, bucket, workspace, out, execution, &seq);
}

} // namespace ninfer::ops
