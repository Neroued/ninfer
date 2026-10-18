#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/common/host/operands.h"

namespace ninfer::ops::detail::qsa {
namespace {
int partition_head_groups(KvCacheStorage storage, int kv_heads, std::int64_t queries) {
    // A one-query 64-way merge adds latency. The BF16 2/6-query cases also retain
    // the original partition count; other local short calls benefit from more CTAs.
    if (kv_heads == 1 &&
        (queries == 1 || (storage == KvCacheStorage::BFloat16 && (queries == 2 || queries == 6))))
        return 2;
    return kv_heads;
}

bool streaming_history(KvCacheStorage storage, int batch, int kv_heads, std::uint32_t visible) {
    // GB10 has 24 MiB of L2; the bound accounts for the actual KV heads and every batch row.
    const auto bytes = std::uint64_t(visible) * batch * kv_heads *
                       paged_kv_storage_layout(storage, 256).physical_bytes_per_token_head();
    return bytes > 24 * 1024 * 1024;
}
} // namespace

int sparse_gqa_direct_queries(int kv_heads, int sms) {
    // A grouped CTA owns eight queries. One local KV group needs one full SM wave.
    return kv_heads == 1 ? 8 * sms : 256;
}

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage storage, int tokens, int batch, int kv_heads,
                                     QsaSparseGqaExecutionEnvelope envelope, int sms) {
    const auto queries = std::int64_t(tokens) * batch;
    const auto visible = envelope.max_visible_tokens;
    if (queries >= sparse_gqa_direct_queries(kv_heads, sms))
        return {16, 1, SparseGqaInstance::Grouped, false, visible};
    const int partition_heads = partition_head_groups(storage, kv_heads, queries);
    const auto target         = std::max<std::int64_t>(
        1, (std::int64_t(sms) * 3 / (2 * partition_heads) + queries - 1) / queries);
    int splits = 1;
    while (splits < target && splits < (partition_heads == 1 ? 64 : 32)) splits *= 2;
    const bool simt = (visible <= 8 && queries <= 16) || (visible <= 64 && queries == 1);
    if (simt) splits = std::max(1, splits / 12);
    // A 16-key output tile admits four CTAs per SM. Fill those slots when a large
    // history otherwise leaves only two CTAs, keeping the same core/merge node pair.
    const bool expand = !simt && storage != KvCacheStorage::BFloat16 && queries > 6 &&
                        queries * kv_heads * splits <= 2 * sms &&
                        streaming_history(storage, batch, kv_heads, visible);
    const int key_rows = expand ? 16 : 32;
    if (expand) splits = std::min(partition_heads == 1 ? 64 : 32, splits * 2);
    const int tiles = std::max(1, (int(std::min(visible, 2051u)) + key_rows - 1) / key_rows);
    return {key_rows, std::min(splits, tiles),
            simt ? SparseGqaInstance::Simt : SparseGqaInstance::Mma,
            simt || kv_heads != 1 || storage != KvCacheStorage::BFloat16 ||
                std::min(splits, tiles) != 1,
            visible};
}

SparseGqaLaunchPlan select_sparse_gqa_launch(const PagedKVBatchLayerView& cache, SparseGqaPlan plan,
                                             int queries, int batch, int sms) {
    const auto aligned = [](const Tensor& t) {
        return common::operand::aligned(t.data, 16) && t.nb[1] % 16 == 0 && t.nb[2] % 16 == 0 &&
               t.nb[3] % 16 == 0;
    };
    const bool vector    = aligned(cache.k_pages) && aligned(cache.v_pages);
    const bool quantized = cache.storage != KvCacheStorage::BFloat16;
    const auto staged    = quantized && vector ? SparseGqaLoad::Staged : SparseGqaLoad::Direct;
    if (plan.instance == SparseGqaInstance::Grouped)
        return {staged, vector, 1, SparseGqaKernel::Columns16};
    if (plan.instance == SparseGqaInstance::Simt)
        return {SparseGqaLoad::Direct, vector, plan.splits,
                queries > 8 ? SparseGqaKernel::Simt4 : SparseGqaKernel::Simt8};
    if (!vector) return {SparseGqaLoad::Direct, false, plan.splits};
    if (plan.visible_tokens <= 64)
        return {staged, true, plan.splits, SparseGqaKernel::Columns32Wide};

    if (plan.key_rows == 16) return {staged, true, plan.splits, SparseGqaKernel::Columns16};
    const bool streaming =
        streaming_history(cache.storage, batch, cache.num_kv_heads, plan.visible_tokens);
    if (queries == 4) return {staged, true, plan.splits, SparseGqaKernel::Columns16};
    if (queries <= 6) {
        int splits = 1;
        const int partition_heads =
            partition_head_groups(cache.storage, cache.num_kv_heads, queries);
        const int target = std::max(1, 2 * sms / (partition_heads * queries));
        while (splits * 2 <= target && splits < (partition_heads == 1 ? 64 : 32)) splits *= 2;
        return {staged, true, std::min(splits, plan.splits), SparseGqaKernel::Columns32Wide};
    }
    const auto blocks = std::int64_t(queries) * cache.num_kv_heads * plan.splits;
    // The in-place column tile needs a third CTA to hide its extra barriers/QK work.
    if (!quantized || (streaming && blocks > 2 * sms))
        return {quantized ? SparseGqaLoad::InPlace : SparseGqaLoad::Direct, true, plan.splits,
                SparseGqaKernel::Columns32};
    const bool inplace = (blocks + 3 * sms - 1) / (3 * sms) < (blocks + 2 * sms - 1) / (2 * sms);
    return {inplace ? SparseGqaLoad::InPlace : SparseGqaLoad::Staged, true, plan.splits};
}
} // namespace ninfer::ops::detail::qsa
