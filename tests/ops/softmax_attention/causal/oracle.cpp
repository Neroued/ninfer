#include "fixture.h"

namespace ninfer::test::causal_attention {

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    const auto query_index   = [&](int head, int d, int token) {
        return std::size_t(d) + kHeadDim * (head + std::size_t(geometry.q_heads) * token);
    };
    const int tokens = positions.size(), visible = positions.back() + 1;
    const bool rotate_q = cache.storage != KvCacheStorage::BFloat16;
    const bool rotate_v = cache.storage == KvCacheStorage::Nvfp4Group16 ||
                          cache.storage == KvCacheStorage::Fp8KeyNvfp4Value;
    std::vector<double> query(q.begin(), q.end()), output(q.size());
    if (rotate_q)
        parallel_rows(tokens * geometry.q_heads, q.size() * 8, [&](int first, int last) {
            for (int i = first; i < last; ++i) {
                const int token = i / geometry.q_heads, head = i % geometry.q_heads;
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d) row[d] = q[query_index(head, d, token)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d) query[query_index(head, d, token)] = row[d];
            }
        });
    // Decode the persistent public representation once. No private Q quantization, staging
    // casts, partial rounding or inverse-transform materialization enters this oracle.
    const auto index = [&](int d, int head, int pos) {
        return std::size_t(d) + kHeadDim * (std::size_t(pos) + std::size_t(visible) * head);
    };
    std::vector<double> keys(std::size_t(visible) * geometry.kv_heads * kHeadDim),
        values(keys.size());
    parallel_rows(visible * geometry.kv_heads, keys.size(), [&](int first, int last) {
        for (int row = first; row < last; ++row) {
            const int head = row / visible, pos = row % visible;
            decode_cache_row(
                cache, true, head, pos,
                std::span<double, kHeadDim>(keys.data() + index(0, head, pos), kHeadDim));
            decode_cache_row(
                cache, false, head, pos,
                std::span<double, kHeadDim>(values.data() + index(0, head, pos), kHeadDim));
        }
    });
    const auto products = std::int64_t(tokens) * geometry.q_heads * visible * kHeadDim;
    naive_dense_softmax_attention(
        op_geometry(geometry), tokens, visible, double(kAttentionScale),
        [&](int d, int head, int token) { return query[query_index(head, d, token)]; },
        [&](int d, int head, int pos) { return keys[index(d, head, pos)]; },
        [&](int d, int head, int pos) { return values[index(d, head, pos)]; },
        [&](int token, int pos) { return pos <= positions[token]; },
        [&](int d, int head, int token, double value) {
            output[query_index(head, d, token)] = value;
        },
        products >= (1 << 16));
    if (rotate_v)
        parallel_rows(tokens * geometry.q_heads, q.size() * 8, [&](int first, int last) {
            for (int i = first; i < last; ++i) {
                const int token = i / geometry.q_heads, head = i % geometry.q_heads;
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d) row[d] = output[query_index(head, d, token)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d) output[query_index(head, d, token)] = row[d];
            }
        });
    return output;
}

ReductionCriterion attention_criterion(KvCacheStorage storage) {
    if (storage == KvCacheStorage::BFloat16) return kAttentionBf16Criterion;
    if (storage == KvCacheStorage::Int8Group64) return kAttentionInt8Criterion;
    if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) return kAttentionFp8Criterion;
    if (storage == KvCacheStorage::Nvfp4Group16) return kAttentionNvfp4Criterion;
    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) return kAttentionK8V4Criterion;
    throw std::logic_error("unregistered causal-attention test storage");
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

} // namespace ninfer::test::causal_attention
