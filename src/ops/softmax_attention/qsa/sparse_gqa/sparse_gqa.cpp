#include "ninfer/ops/qsa.h"
#include "core/layout.h"
#include "ops/common/host/graph_profile.h"
#include "ops/softmax_attention/qsa/geometry.h"
#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/softmax_attention/qsa/validation.h"

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

void qsa_sparse_gqa(const Tensor& query, const Tensor& gate,
                    const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                    const QsaBlockSelection& selected, ContextBucket bucket, MemoryArena& workspace,
                    Tensor& out, CudaExecutionView execution) {
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
    v.metadata(batch.valid_columns, rows, "valid columns");
    v.metadata(batch.table_rows, rows, "table rows");
    v.dense(selected.block_ids, DType::I32, {512, tokens, rows, 1}, 4, "block ids");
    v.dense(selected.block_counts, DType::I32, {tokens, rows, 1, 1}, 4, "block counts");
    const Tensor* inputs[]{&query,
                           &gate,
                           &cache.k_pages.records.page_offsets,
                           &batch.positions,
                           &batch.valid_columns,
                           &batch.table_rows,
                           &selected.block_ids,
                           &selected.block_counts};
    for (const auto* input : inputs)
        if (input->data) v.require(!v.overlap(out, *input), "output/input overlap");
    const auto plan = select_sparse_gqa_plan(cache.storage, tokens, rows, kv_heads, bucket, sms);
    const auto work_layout = workspace_layout(tokens, rows, query_heads, plan);
    auto scope             = workspace.scope();
    SparseGqaWorkspace scratch{};
    if (work_layout.bytes) {
        const auto storage = workspace.alloc_bytes(work_layout.bytes, 256);
        scratch = {work_layout.numerator.bind(storage), work_layout.maximum.bind(storage),
                   work_layout.sum.bind(storage)};
    }
    for (const auto* work : {&scratch.numerator, &scratch.maximum, &scratch.sum}) {
        if (!work->data) continue;
        v.require(!v.overlap(*work, out), "workspace/output overlap");
        for (const auto* input : inputs)
            if (input->data) v.require(!v.overlap(*work, *input), "workspace/input overlap");
    }
    launch_sparse_gqa(query, gate, cache, batch, selected, bucket, plan, scratch, out, execution);
}
} // namespace ninfer::ops
