#pragma once

#include "models/common/parallel_execution.h"

#include "core/layout.h"
#include "models/qwen4_exp/execution/profile.h"
#include "models/common/moe/schedule.h"
#include "models/common/qwen/vision/vision.h"

namespace ninfer::models::qwen4_exp::execution {

[[nodiscard]] SequenceParallelPolicy sequence_parallel_policy(std::uint32_t ranks);

struct WorkspacePlan {
    std::size_t prefill = 0, ordinary = 0, verify = 0, mtp = 0, history = 0, finish = 0;
    std::size_t general_bytes = 0, bytes = 0;
    std::optional<vision::Capacity> vision;
};

[[nodiscard]] WorkspacePlan plan_workspace(const ModelProfile&, const EngineOptions&,
                                           std::uint32_t chunk, std::int32_t sm_count,
                                           const moe::Plan*);
} // namespace ninfer::models::qwen4_exp::execution

namespace ninfer::models::qwen4_exp::execution::workspace {
struct Roots {
    Tensor residual, normalized, input, output, injection;
    Tensor exchange_input, exchange_output;
};

template <class Arena>
Roots roots(Arena& a, const TextConfig& c, std::int32_t m, std::int32_t padded = 0) {
    const auto h = dimension(c.hidden), streams = dimension(c.hc.streams);
    auto residual   = a.alloc(DType::BF16, {h, streams, m});
    auto normalized = a.alloc(DType::BF16, {h, streams, m});
    auto input      = a.alloc(DType::BF16, {h, padded ? padded : m});
    auto output     = a.alloc(DType::BF16, {h, padded ? padded : m});
    return {residual,
            normalized,
            input.slice(1, 0, m),
            output.slice(1, 0, m),
            a.alloc(DType::FP32, {streams, m}),
            input,
            output};
}

struct HistoryRoots {
    Tensor residual, normalized, input;
};

template <class Arena>
HistoryRoots history_roots(Arena& a, const TextConfig& c, std::int32_t m,
                           std::int32_t input_columns = 0) {
    const auto h = dimension(c.hidden), streams = dimension(c.hc.streams);
    return {a.alloc(DType::BF16, {h, streams, m}), a.alloc(DType::BF16, {h, streams, m}),
            a.alloc(DType::BF16, {h, input_columns ? input_columns : m})};
}

struct GdnBuffers {
    Tensor qkv, z, query, key, value, g, beta, recurrent, normalized;
};

template <class Arena>
GdnBuffers gdn(Arena& a, const GdnConfig& c, std::int32_t m, bool prefill) {
    const auto k = dimension(c.key_dim), v = dimension(c.value_dim);
    const auto kh = dimension(c.key_heads), vh = dimension(c.value_heads);
    const auto value = dimension(c.value_width());
    return {prefill ? a.alloc(DType::BF16, {dimension(c.conv_channels()), m}) : Tensor{},
            a.alloc(DType::BF16, {value, m}),
            a.alloc(DType::BF16, {k, kh, m}),
            a.alloc(DType::BF16, {k, kh, m}),
            a.alloc(DType::BF16, {v, vh, m}),
            a.alloc(DType::FP32, {vh, m}),
            a.alloc(DType::FP32, {vh, m}),
            a.alloc(DType::BF16, {value, m}),
            a.alloc(DType::BF16, {value, m})};
}

struct QsaBuffers {
    Tensor query, gate, key, value, index_query, index_key, selection, counts, output;
    Tensor raw_records, coordinate_records;
};

template <class Arena>
QsaBuffers qsa(Arena& a, const QsaConfig& c, std::int32_t m, bool allocate_records) {
    const auto d = dimension(c.head_dim), qh = dimension(c.query_heads), kh = dimension(c.kv_heads);
    const auto id = dimension(c.index_dim), ih = dimension(c.index_heads);
    const auto ik = dimension(c.index_key_width());
    return {a.alloc(DType::BF16, {d, qh, m}),
            a.alloc(DType::BF16, {d, qh, m}),
            a.alloc(DType::BF16, {d, kh, m}),
            a.alloc(DType::BF16, {d, kh, m}),
            a.alloc(DType::BF16, {id, ih, m}),
            a.alloc(DType::BF16, {ik, m}),
            a.alloc(DType::I32, {dimension(c.block_budget), m}),
            a.alloc(DType::I32, {m}),
            a.alloc(DType::BF16, {d, qh, m}),
            allocate_records ? a.alloc(DType::BF16, {ik, m}) : Tensor{},
            allocate_records ? a.alloc(DType::I32, {3, m}) : Tensor{}};
}

struct QsaHistoryBuffers {
    Tensor key, value, index_key, raw_records, coordinate_records;
};

template <class Arena>
QsaHistoryBuffers qsa_history(Arena& a, const QsaConfig& c, std::int32_t m, bool records = true) {
    const auto d = dimension(c.head_dim), kh = dimension(c.kv_heads);
    const auto ik = dimension(c.index_key_width());
    return {a.alloc(DType::BF16, {d, kh, m}), a.alloc(DType::BF16, {d, kh, m}),
            a.alloc(DType::BF16, {ik, m}), records ? a.alloc(DType::BF16, {ik, m}) : Tensor{},
            records ? a.alloc(DType::I32, {3, m}) : Tensor{}};
}

struct PleBuffers {
    Tensor projection, gated, records;
};

template <class Arena>
PleBuffers ple(Arena& a, const TextConfig& c, std::int32_t m, bool prefill) {
    const auto h = dimension(c.hidden), streams = dimension(c.hc.streams);
    const auto channels = dimension(c.residual_width());
    return {a.alloc(DType::BF16, {dimension(c.residual_width() + c.hidden), m}),
            a.alloc(DType::BF16, {h, streams, m}),
            prefill ? a.alloc(DType::BF16, {channels, m}) : Tensor{}};
}

struct ScoreBuffers {
    Tensor logits, targets, logprobs;
};

template <class Arena>
ScoreBuffers score(Arena& a, std::int32_t vocabulary, std::int32_t columns) {
    return {a.alloc(DType::BF16, {vocabulary, columns}), a.alloc(DType::I32, {columns}),
            a.alloc(DType::FP32, {columns})};
}

struct MtpStemBuffers {
    Tensor embedding, normalized_embedding, normalized_hidden, projected_embedding;
};

template <class Arena>
MtpStemBuffers mtp_stem(Arena& a, const TextConfig& c, std::int32_t m,
                        std::int32_t embedding_columns = 0) {
    const auto h = dimension(c.hidden), channels = dimension(c.residual_width());
    return {a.alloc(DType::BF16, {h, embedding_columns ? embedding_columns : m}),
            a.alloc(DType::BF16, {h, m}), a.alloc(DType::BF16, {channels, m}),
            a.alloc(DType::BF16, {h, m})};
}
} // namespace ninfer::models::qwen4_exp::execution::workspace
