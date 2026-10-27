#include "fixture.h"

namespace ninfer::test::causal_attention {

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    const int tokens = positions.size(), visible = positions.back() + 1;
    const bool rotate_q = cache.storage != KvCacheStorage::BFloat16;
    const bool rotate_v = cache.storage == KvCacheStorage::Nvfp4Group16 ||
                          cache.storage == KvCacheStorage::Fp8KeyNvfp4Value;
    std::vector<double> query(q.begin(), q.end()), output(q.size());
    if (rotate_q)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < geometry.q_heads; ++head) {
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d) row[d] = q[q_index(geometry, head, d, token)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d)
                    query[q_index(geometry, head, d, token)] = row[d];
            }
    // Decode the persistent public representation once. No private Q quantization, staging
    // casts, partial rounding or inverse-transform materialization enters this oracle.
    const auto index = [&](int d, int head, int pos) {
        return std::size_t(d) + kHeadDim * (std::size_t(pos) + std::size_t(visible) * head);
    };
    std::vector<double> keys(std::size_t(visible) * geometry.kv_heads * kHeadDim),
        values(keys.size());
    for (int head = 0; head < geometry.kv_heads; ++head)
        for (int pos = 0; pos < visible; ++pos)
            for (int d = 0; d < kHeadDim; ++d) {
                keys[index(d, head, pos)]   = cache_value(cache, true, head, pos, d);
                values[index(d, head, pos)] = cache_value(cache, false, head, pos, d);
            }
    const auto products = std::int64_t(tokens) * geometry.q_heads * visible * kHeadDim;
    naive_dense_softmax_attention(
        op_geometry(geometry), tokens, visible, double(kAttentionScale),
        [&](int d, int head, int token) { return query[q_index(geometry, head, d, token)]; },
        [&](int d, int head, int pos) { return keys[index(d, head, pos)]; },
        [&](int d, int head, int pos) { return values[index(d, head, pos)]; },
        [&](int token, int pos) { return pos <= positions[token]; },
        [&](int d, int head, int token, double value) {
            output[q_index(geometry, head, d, token)] = value;
        },
        products >= (1 << 21));
    if (rotate_v)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < geometry.q_heads; ++head) {
                std::array<double, kHeadDim> row{};
                for (int d = 0; d < kHeadDim; ++d)
                    row[d] = output[q_index(geometry, head, d, token)];
                normalized_hadamard_d256(row);
                for (int d = 0; d < kHeadDim; ++d)
                    output[q_index(geometry, head, d, token)] = row[d];
            }
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
