#pragma once

#include "common/roofline.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <span>
#include <sstream>

namespace ninfer::bench::qsa {

inline std::string projection_metrics_csv(std::uint64_t weight_bytes, int n, int k, int tokens,
                                          double elapsed_us) {
    const auto bytes   = weight_bytes + 2ULL * (n + k) * tokens;
    const double flops = 2.0 * n * k * tokens;
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << bytes << ',' << flops << ','
        << bytes / (elapsed_us * 1000) << ',' << flops / (elapsed_us * 1e6) << ','
        << bytes / (kRoofline.memory_gbs * 1000) << ','
        << flops / (kRoofline.bf16_f32_tflops * 1e6);
    return out.str();
}

// Unique semantic payloads per call, independent of CTA tiling and workspace representation.
// The per-query cache sum exposes reuse; neither byte count is measured DRAM traffic.
struct Metrics {
    std::uint64_t live_queries = 0, scoring_queries = 0, candidate_blocks = 0, selected_tokens = 0;
    std::uint64_t qk_flops = 0, pv_flops = 0;
    std::uint64_t read_bytes = 0, write_bytes = 0;
    std::uint64_t kv_read = 0, kv_write = 0, index_read = 0, index_write = 0, record_write = 0;
    std::uint64_t per_query_cache_bytes = 0;

    static constexpr const char* columns =
        "live_queries,scoring_queries,candidate_blocks,selected_tokens,qk_flops,pv_flops,"
        "logical_read_bytes,logical_write_bytes,logical_bytes,kv_read_bytes,kv_write_bytes,"
        "index_read_bytes,index_write_bytes,record_write_bytes,per_query_cache_bytes,"
        "logical_GBps,logical_TFLOps,memory_reference_GBps,bf16_f32_reference_TFLOps,"
        "memory_reference_us,compute_reference_us,roofline_reference_us";

    std::string csv(double elapsed_us) const {
        const auto bytes = read_bytes + write_bytes, flops = qk_flops + pv_flops;
        const double memory_us  = bytes / (kRoofline.memory_gbs * 1000.0);
        const double compute_us = flops / (kRoofline.bf16_f32_tflops * 1.0e6);
        std::ostringstream out;
        out << std::fixed << std::setprecision(6) << live_queries << ',' << scoring_queries << ','
            << candidate_blocks << ',' << selected_tokens << ',' << qk_flops << ',' << pv_flops
            << ',' << read_bytes << ',' << write_bytes << ',' << bytes << ',' << kv_read << ','
            << kv_write << ',' << index_read << ',' << index_write << ',' << record_write << ','
            << per_query_cache_bytes << ',' << bytes / (elapsed_us * 1000.0) << ','
            << flops / (elapsed_us * 1.0e6) << ',' << kRoofline.memory_gbs << ','
            << kRoofline.bf16_f32_tflops << ',' << memory_us << ',' << compute_us << ','
            << std::max(memory_us, compute_us);
        return out.str();
    }
};

inline Metrics indexer_metrics(int tokens, std::span<const int> contexts,
                               std::span<const int> lengths) {
    Metrics m;
    m.read_bytes  = lengths.size() * sizeof(std::int32_t);
    m.write_bytes = std::uint64_t(tokens) * lengths.size() * sizeof(std::int32_t);
    for (std::size_t b = 0; b < lengths.size(); ++b) {
        int maximum_scored = 0;
        m.live_queries += lengths[b];
        m.read_bytes += 4ULL * lengths[b];
        for (int t = 0; t < lengths[b]; ++t) {
            const int blocks = (contexts[b] - lengths[b] + t + 1) / 4;
            m.write_bytes += 4ULL * std::min(512, blocks);
            if (blocks <= 512) continue;
            ++m.scoring_queries;
            m.candidate_blocks += blocks;
            maximum_scored = blocks;
        }
        if (maximum_scored) m.read_bytes += 4ULL * (1 + (maximum_scored + 15) / 16);
        m.index_read += 256ULL * maximum_scored;
    }
    m.qk_flops              = m.candidate_blocks * 2 * 4 * 128;
    m.per_query_cache_bytes = m.candidate_blocks * 256;
    m.read_bytes += m.index_read + m.scoring_queries * 4 * 128 * 2;
    return m;
}

inline Metrics prepare_metrics(int tokens, int axes, int kv_heads, std::span<const int> bases,
                               std::span<const int> lengths, std::uint64_t kv_token_bytes,
                               bool queries = true) {
    Metrics m;
    m.read_bytes              = lengths.size() * sizeof(std::int32_t);
    const auto query_elements = queries ? 3072 * kv_heads + 512 : 0;
    m.write_bytes             = std::uint64_t(tokens) * lengths.size() * query_elements * 2;
    std::uint64_t complete    = 0;
    for (std::size_t b = 0; b < lengths.size(); ++b) {
        const int n = lengths[b], first = bases[b];
        if (!n) continue;
        m.live_queries += n;
        const int blocks = (first + n) / 4 - first / 4;
        complete += blocks;
        m.read_bytes +=
            std::uint64_t(n) * ((query_elements + 512 * kv_heads + 128) * 2 + 4 * (axes + 1));
        m.read_bytes += 8 + 4ULL * ((first + n - 1) / 64 - first / 64 + 1);
        if (blocks && first % 4) m.read_bytes += 256ULL * (first % 4) + 12 + 4;
    }
    if (m.live_queries) m.read_bytes += (256 + (queries ? 256 + 128 : 0)) * 2;
    if (complete) m.read_bytes += 128 * 2;
    m.kv_write     = m.live_queries * kv_token_bytes;
    m.index_write  = complete * 128 * 2;
    m.record_write = m.live_queries * (128 * 2 + 3 * 4);
    m.write_bytes += m.kv_write + m.index_write + m.record_write;
    return m;
}

} // namespace ninfer::bench::qsa
