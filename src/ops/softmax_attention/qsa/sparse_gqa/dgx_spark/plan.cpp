#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"

#include <algorithm>

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

std::int64_t sparse_gqa_direct_queries(int kv_heads, int sms) {
    // A grouped CTA owns eight queries. One local KV group needs one full SM wave.
    return kv_heads == 1 ? std::int64_t(8) * sms : 256;
}

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage storage, int tokens, int batch, int kv_heads,
                                     ContextBucket bucket, int sms) {
    const auto queries = std::int64_t(tokens) * batch;
    if (queries >= sparse_gqa_direct_queries(kv_heads, sms)) {
        if (tokens < 8) {
            // An eight-query prefix tile cannot fit inside one physical request.
            // Keep the same two graph nodes, with one direct attention pass.
            const bool rotated_value = storage == KvCacheStorage::Nvfp4Group16 ||
                                       storage == KvCacheStorage::Fp8KeyNvfp4Value;
            const int key_rows = rotated_value ? 16 : 32;
            auto plan          = SparseGqaPlan{key_rows, 1, SparseGqaInstance::Mma, false};
            plan.merge_node    = true;
            return plan;
        }
        return {16, 1, SparseGqaInstance::Grouped, false};
    }
    const int partition_heads = partition_head_groups(storage, kv_heads, queries);
    const auto target         = std::max<std::int64_t>(
        1, (std::int64_t(sms) * 3 / (2 * partition_heads) + queries - 1) / queries);
    int splits = 1;
    while (splits < target && splits < (partition_heads == 1 ? 64 : 32)) splits *= 2;
    // A 16-key output tile admits four CTAs per SM. Fill those slots when a large
    // history otherwise leaves only two CTAs, keeping the same core/merge node pair.
    const bool expand = storage != KvCacheStorage::BFloat16 && queries > 6 &&
                        queries * kv_heads * splits <= std::int64_t(2) * sms &&
                        streaming_history(storage, batch, kv_heads, context_bucket_tokens(bucket));
    const int key_rows = expand ? 16 : 32;
    if (expand) splits = std::min(partition_heads == 1 ? 64 : 32, splits * 2);
    // Even K2 has at least 64 key tiles. The device caps the chosen split count to live work.
    auto plan = SparseGqaPlan{key_rows, splits, SparseGqaInstance::Mma, splits > 1};
    // The MMA epilogue can normalize any cache format directly when unsplit.
    // Fixed geometry retains attention + merge across buckets.
    // An unsplit MMA writes output directly and leaves the merge node idle.
    plan.merge_node = true;
    return plan;
}

int sparse_gqa_long_direct_tokens(ContextBucket bucket) {
    return bucket == ContextBucket::K2 ? 0 : 512;
}

SparseGqaPackedWindows sparse_gqa_packed_windows(KvCacheStorage storage, int, ContextBucket bucket,
                                                 bool) {
    // The two-split Columns32 path at width 64 recovers parallelism for short
    // deep-history segments. At longer segment lengths the wider window retains reuse.
    if (bucket >= ContextBucket::K64) {
        if (storage == KvCacheStorage::BFloat16) return {64, 256};
        // Keep short FP8 histories resident while preserving the large-segment direct path.
        if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) return {64, kSparseGqaSlab};
    }
    return {kSparseGqaSlab};
}

SparseGqaPlan select_sparse_gqa_packed_window_plan(KvCacheStorage storage, int width,
                                                   int total_tokens, int segments, int kv_heads,
                                                   ContextBucket bucket, int sms) {
    auto plan = select_sparse_gqa_plan(storage, width, 1, kv_heads, bucket, sms);
    if (storage == KvCacheStorage::BFloat16 && bucket > ContextBucket::K2 &&
        total_tokens <= kSparseGqaPackedSkewQueries && plan.instance == SparseGqaInstance::Mma) {
        const int mean   = 1 + (width - 1) / segments;
        const int target = select_sparse_gqa_plan(storage, mean, 1, kv_heads, bucket, sms).splits;
        plan.splits      = std::max(plan.splits, target);
        plan.min_split_tiles = 8;
        plan.partial         = plan.splits > 1;
    }
    return plan;
}

SparseGqaLaunchPlan select_sparse_gqa_launch(KvCacheStorage storage, int kv_heads,
                                             ContextBucket bucket, SparseGqaPlan plan, int queries,
                                             int batch, int sms, bool vector) {
    const bool quantized = storage != KvCacheStorage::BFloat16;
    const auto staged    = quantized && vector ? SparseGqaLoad::Staged : SparseGqaLoad::Direct;
    if (plan.instance == SparseGqaInstance::Grouped)
        return {staged, vector, 1, SparseGqaKernel::Columns16};
    if (queries >= sparse_gqa_direct_queries(kv_heads, sms)) {
        const auto load = plan.key_rows == 16   ? staged
                          : quantized && vector ? SparseGqaLoad::InPlace
                                                : SparseGqaLoad::Direct;
        return {load, vector, 1,
                plan.key_rows == 16 ? SparseGqaKernel::Columns16 : SparseGqaKernel::Columns32};
    }
    if (!vector) return {SparseGqaLoad::Direct, false, plan.splits};
    if (plan.key_rows == 16) return {staged, true, plan.splits, SparseGqaKernel::Columns16};
    const bool streaming =
        streaming_history(storage, batch, kv_heads, context_bucket_tokens(bucket));
    if (queries == 4) return {staged, true, plan.splits, SparseGqaKernel::Columns16};
    if (queries <= 6) {
        int splits                = 1;
        const int partition_heads = partition_head_groups(storage, kv_heads, queries);
        const auto target =
            std::max<std::int64_t>(1, std::int64_t(2) * sms / (partition_heads * queries));
        while (splits * 2 <= target && splits < (partition_heads == 1 ? 64 : 32)) splits *= 2;
        return {staged, true, std::min(splits, plan.splits), SparseGqaKernel::Columns32Wide};
    }
    const auto blocks = std::int64_t(queries) * kv_heads * plan.splits;
    // The in-place column tile needs a third CTA to hide its extra barriers/QK work.
    const auto wave2 = std::int64_t(2) * sms;
    if (!quantized || (streaming && blocks > wave2))
        return {quantized ? SparseGqaLoad::InPlace : SparseGqaLoad::Direct, true, plan.splits,
                SparseGqaKernel::Columns32};
    const auto wave3   = std::int64_t(3) * sms;
    const bool inplace = (blocks + wave3 - 1) / wave3 < (blocks + wave2 - 1) / wave2;
    return {inplace ? SparseGqaLoad::InPlace : SparseGqaLoad::Staged, true, plan.splits};
}
} // namespace ninfer::ops::detail::qsa
