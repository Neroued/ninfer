#include "ninfer/ops/bidirectional_gqa_attention.h"

#include "core/arena.h"
#include "core/kv_cache.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD       = 128;
constexpr int kQHeads  = 32;
constexpr int kKVHeads = 8;
constexpr int kGroup   = 4;
constexpr float kScale = 0.08838834764831844055f;

constexpr ReductionCriterion kBidirectionalGqaBf16Criterion{
    .relative_l2                     = 2.95e-3,
    .gross_absolute                  = 3e-4,
    .gross_relative_to_max_reference = 5.7e-3,
};

std::size_t q_index(int d, int q_head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(q_head) +
                static_cast<std::size_t>(kQHeads) * static_cast<std::size_t>(token));
}

std::size_t query_kv_index(int d, int kv_head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(kv_head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t context_index(int d, int kv_head, int position, int padded_context) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(kv_head));
}

int align_context(int value) { return ((std::max(value, 1) + 127) / 128) * 128; }

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

void bidirectional_gqa_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                              const std::vector<float>& query_v,
                              const std::vector<float>& context_k,
                              const std::vector<float>& context_v, int tokens, int context_length,
                              int padded_context, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);
    const int key_count = context_length + tokens;
    std::vector<double> scores(static_cast<std::size_t>(key_count));

    for (int token = 0; token < tokens; ++token) {
        for (int q_head = 0; q_head < kQHeads; ++q_head) {
            const int kv_head = q_head / kGroup;
            double max_score  = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < key_count; ++key) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const double k_value =
                        key < context_length
                            ? static_cast<double>(
                                  context_k[context_index(d, kv_head, key, padded_context)])
                            : static_cast<double>(
                                  query_k[query_kv_index(d, kv_head, key - context_length)]);
                    dot += static_cast<double>(q[q_index(d, q_head, token)]) * k_value;
                }
                const double score                    = dot * static_cast<double>(kScale);
                scores[static_cast<std::size_t>(key)] = score;
                max_score                             = std::max(max_score, score);
            }

            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - max_score);
                denominator += score;
            }
            for (int d = 0; d < kD; ++d) {
                double numerator = 0.0;
                for (int key = 0; key < key_count; ++key) {
                    const double v_value =
                        key < context_length
                            ? static_cast<double>(
                                  context_v[context_index(d, kv_head, key, padded_context)])
                            : static_cast<double>(
                                  query_v[query_kv_index(d, kv_head, key - context_length)]);
                    numerator += scores[static_cast<std::size_t>(key)] * v_value;
                }
                out[q_index(d, q_head, token)] = numerator / denominator;
            }
        }
    }
}

// DFlash's full companion cache is fixed BF16; mixed target KV never reaches this Op.
KVCacheLayerView make_context_view(DeviceBuffer& k, DeviceBuffer& v, int max_context,
                                   int padded_context) {
    return {
        .k              = Tensor(k.p, DType::BF16, {kD, padded_context, kKVHeads}),
        .v              = Tensor(v.p, DType::BF16, {kD, padded_context, kKVHeads}),
        .k_scale        = Tensor(),
        .v_scale        = Tensor(),
        .max_context    = static_cast<std::uint32_t>(max_context),
        .padded_context = static_cast<std::uint32_t>(padded_context),
        .num_kv_heads   = kKVHeads,
        .head_dim       = kD,
        .k_dtype        = DType::BF16,
        .v_dtype        = DType::BF16,
        .k_quant_group  = 0,
        .v_quant_group  = 0,
    };
}

enum class InputProfile {
    Random,
    QueryVisibility,
};

int run_case(int tokens, int context_length, InputProfile profile = InputProfile::Random,
             int envelope_max = -1) {
    if (envelope_max < 0) envelope_max = context_length;
    const int max_context            = std::max({context_length, envelope_max, 1});
    const int padded_context         = align_context(max_context);
    const std::size_t q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_kv_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t context_count  = static_cast<std::size_t>(kD) * padded_context * kKVHeads;

    std::vector<float> q(q_count);
    std::vector<float> query_k(query_kv_count);
    std::vector<float> query_v(query_kv_count);
    std::vector<float> context_k(context_count);
    std::vector<float> context_v(context_count);

    const auto seed = static_cast<unsigned>(tokens * 131 + context_length * 17);
    fill_uniform(q, 1001u + seed, -0.35f, 0.35f);
    fill_uniform(query_k, 2003u + seed, -0.4f, 0.4f);
    fill_uniform(query_v, 3001u + seed, -0.8f, 0.8f);
    fill_uniform(context_k, 4001u + seed, -0.4f, 0.4f);
    fill_uniform(context_v, 5003u + seed, -0.8f, 0.8f);

    if (profile == InputProfile::QueryVisibility) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(query_k.begin(), query_k.end(), 0.0f);
        std::fill(query_v.begin(), query_v.end(), 0.0f);
        std::fill(context_k.begin(), context_k.end(), 0.0f);
        std::fill(context_v.begin(), context_v.end(), 0.0f);
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) { query_v[query_kv_index(d, kv_head, tokens - 1)] = 1.0f; }
        }
    }

    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    std::vector<double> reference;
    bidirectional_gqa_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length,
                             padded_context, reference);

    const auto q_expected         = bf16_bits(q);
    const auto query_k_expected   = bf16_bits(query_k);
    const auto query_v_expected   = bf16_bits(query_v);
    const auto context_k_expected = bf16_bits(context_k);
    const auto context_v_expected = bf16_bits(context_v);
    const std::vector<int> length_expected{context_length};

    DeviceBuffer d_q         = to_device(q_expected);
    DeviceBuffer d_query_k   = to_device(query_k_expected);
    DeviceBuffer d_query_v   = to_device(query_v_expected);
    DeviceBuffer d_context_k = to_device(context_k_expected);
    DeviceBuffer d_context_v = to_device(context_v_expected);
    DeviceBuffer d_length    = to_device_i32(length_expected);
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor length_tensor(d_length.p, DType::I32, {1});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens});
    KVCacheLayerView context =
        make_context_view(d_context_k, d_context_v, max_context, padded_context);
    DeviceArena workspace(ops::bidirectional_gqa_attention_workspace_bytes(tokens));

    ops::bidirectional_gqa_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                     kScale, context, {0, static_cast<std::uint32_t>(envelope_max)},
                                     workspace, out_tensor, nullptr);
    cuda_synchronize();

    std::string label = "bidirectional_gqa_attention T=" + std::to_string(tokens) +
                        " L=" + std::to_string(context_length);
    if (envelope_max != context_length) {
        label += " envelope=[0," + std::to_string(envelope_max) + "]";
    }
    if (profile == InputProfile::QueryVisibility) label += " query-visibility";

    int failures = verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count),
                                    reference, kBidirectionalGqaBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(d_q, q_count), q_expected);
    failures +=
        verify_exact((label + " query k unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_k, query_kv_count), query_k_expected);
    failures +=
        verify_exact((label + " query v unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_v, query_kv_count), query_v_expected);
    failures +=
        verify_exact((label + " context k unchanged").c_str(),
                     from_device<std::uint16_t>(d_context_k, context_count), context_k_expected);
    failures +=
        verify_exact((label + " context v unchanged").c_str(),
                     from_device<std::uint16_t>(d_context_v, context_count), context_v_expected);
    failures += verify_exact((label + " context length unchanged").c_str(),
                             from_device<int>(d_length, 1), length_expected);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 0;
    }

    int failures = 0;
    failures += run_case(1, 0);
    failures += run_case(2, 1);
    failures += run_case(8, 95, InputProfile::Random, 4096);
    failures += run_case(16, 257);
    failures += run_case(1, 4096);
    failures += run_case(4, 0, InputProfile::QueryVisibility);

    if (failures != 0) {
        std::cerr << "bidirectional_gqa_attention failures=" << failures << '\n';
        return 1;
    }
    std::cout << "bidirectional_gqa_attention: PASS\n";
    return 0;
}
