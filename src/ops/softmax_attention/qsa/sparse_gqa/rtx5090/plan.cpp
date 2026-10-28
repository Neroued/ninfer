#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/common/host/operands.h"
#include <limits>

namespace ninfer::ops::detail::qsa {
int sparse_gqa_direct_queries(int, int) { return 256; }

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage storage, int tokens, int batch, int kv_heads,
                                     QsaSparseGqaExecutionEnvelope envelope, int sms) {
    const auto queries = std::int64_t(tokens) * batch;
    const auto visible = envelope.max_visible_tokens;
    if (kv_heads == 4) {
        const bool bf16    = storage == KvCacheStorage::BFloat16;
        const bool partial = queries < sparse_gqa_direct_queries(kv_heads, sms);
        if (bf16 && visible <= 64 && (queries < 16 || (visible <= 16 && queries < 32)))
            return {32, 1, SparseGqaInstance::Simt, true};
        const bool prefix =
            visible <= 2051 && (bf16 || (queries >= 2 && (visible > 64 || queries >= 32)));
        if (!prefix && visible <= 64) return {32, 1, SparseGqaInstance::Simt, true};
        const int rows     = bf16 && prefix && visible <= 32 ? 32
                             : prefix || queries <= 8        ? 64
                             : bf16                          ? 32
                                                             : 16;
        const int packet   = prefix ? rows / 16 : 1;
        const auto groups  = (queries + packet - 1) / packet * 4;
        const int resident = bf16 ? (rows == 32               ? (prefix ? 3 : 5)
                                     : prefix && queries <= 8 ? 1
                                                              : 2)
                                  : (rows == 16 ? 7 : 2);
        const int tiles    = std::max(1, (int(std::min(visible, 2051u)) + rows - 1) / rows);
        int splits         = 1;
        if (partial) {
            // Compare complete waves, including the Q prologue. A fixed CTA target can
            // spill one partition into another wave and create a latency inversion.
            std::int64_t best = std::numeric_limits<std::int64_t>::max();
            for (int candidate = 1; candidate <= std::min(64, tiles); ++candidate) {
                const int count = std::min(candidate, tiles);
                const auto waves =
                    (std::int64_t(groups) * count + resident * sms - 1) / (resident * sms);
                const auto cost = waves * ((tiles + count - 1) / count + 2);
                if (cost < best) {
                    best   = cost;
                    splits = count;
                }
            }
        }
        return {rows, splits, prefix ? SparseGqaInstance::FullPrefix : SparseGqaInstance::Mma,
                partial && (!bf16 || splits > 1)};
    }
    if (queries >= sparse_gqa_direct_queries(kv_heads, sms))
        return visible <= 2051 ? SparseGqaPlan{64, 1, SparseGqaInstance::FullPrefix, false}
                               : SparseGqaPlan{32, 1, SparseGqaInstance::Mma, false};
    // Budget parallel partitions independently of history. The envelope only bounds the
    // selected support and admits a short-context arithmetic schedule.
    const auto target =
        std::max<std::int64_t>(1, (std::int64_t(sms) * 3 / 4 + queries - 1) / queries);
    int splits = 1;
    while (splits < target && splits < 32) splits *= 2;
    const bool simt = visible <= 64;
    if (simt) splits = std::max(1, splits / 12);
    const int tiles = std::max(1, (int(std::min(visible, 2051u)) + 31) / 32);
    return {32, std::min(splits, tiles), simt ? SparseGqaInstance::Simt : SparseGqaInstance::Mma,
            true};
}

SparseGqaLaunchPlan select_sparse_gqa_launch(const ConstPagedKVBatchLayerView& cache,
                                             SparseGqaPlan plan, int queries, int, int sms) {
    const auto aligned = [](const ConstPagedPlaneView& t) {
        return common::operand::aligned(t.data(), 16) && t.row_stride_bytes() % 16 == 0 &&
               t.head_stride_bytes() % 16 == 0 && t.plane.tensor.region.alignment % 16 == 0;
    };
    const bool vector = aligned(cache.k_pages) && aligned(cache.v_pages);
    if (cache.num_kv_heads == 4) {
        const auto load = !vector || plan.instance == SparseGqaInstance::Simt
                              ? SparseGqaLoad::Direct
                          : cache.storage == KvCacheStorage::BFloat16
                              ? (plan.instance == SparseGqaInstance::FullPrefix &&
                                         (queries <= 8 || plan.key_rows == 32)
                                     ? SparseGqaLoad::Direct
                                     : SparseGqaLoad::Reused)
                              : SparseGqaLoad::Aliased;
        return {load, vector, plan.splits};
    }
    if (cache.storage == KvCacheStorage::BFloat16 || !vector ||
        plan.instance == SparseGqaInstance::Simt)
        return {SparseGqaLoad::Direct, vector, plan.splits};
    if (plan.instance == SparseGqaInstance::FullPrefix)
        return {SparseGqaLoad::InPlace, vector, plan.splits};
    // In-place expansion adds barriers but retains three CTAs/SM. Choose it only when
    // that residency removes a wave; otherwise use a separate packed staging slot.
    const auto blocks  = std::int64_t(queries) * cache.num_kv_heads * plan.splits;
    const bool inplace = (blocks + 3 * sms - 1) / (3 * sms) < (blocks + 2 * sms - 1) / (2 * sms);
    return {inplace ? SparseGqaLoad::InPlace : SparseGqaLoad::Staged, vector, plan.splits};
}

} // namespace ninfer::ops::detail::qsa
