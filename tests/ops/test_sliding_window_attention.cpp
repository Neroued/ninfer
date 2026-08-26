#include "ninfer/ops/sliding_window_attention.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

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
constexpr int kWindow  = 4096;
constexpr float kScale = 0.08838834764831844055f;
constexpr ops::AttentionHeadGeometry kGeometry{kD, kQHeads, kKVHeads};

constexpr ReductionCriterion kSlidingWindowBf16Criterion{
    .relative_l2                     = 3.95e-3,
    .gross_absolute                  = 3e-4,
    .gross_relative_to_max_reference = 3.0e-3,
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

std::size_t context_index(int d, int kv_head, int slot) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(kWindow) * static_cast<std::size_t>(kv_head));
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

void sliding_window_attention_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                                     const std::vector<float>& query_v,
                                     const std::vector<float>& context_k,
                                     const std::vector<float>& context_v,
                                     const std::vector<int>& positions, int context_length,
                                     int valid_columns, std::vector<double>& out) {
    const int tokens = static_cast<int>(positions.size());
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);

    for (int token = 0; token < valid_columns; ++token) {
        const int query_position = positions[static_cast<std::size_t>(token)];
        const int context_begin  = std::max(0, query_position - (kWindow - 1));
        const int context_keys   = context_length - context_begin;
        const int key_count      = context_keys + valid_columns;
        naive_dense_softmax_attention(
            kGeometry, 1, key_count, static_cast<double>(kScale),
            [&](int d, int head, int) { return static_cast<double>(q[q_index(d, head, token)]); },
            [&](int d, int head, int key) {
                return key < context_keys
                           ? static_cast<double>(context_k[context_index(
                                 d, head, (context_begin + key) & (kWindow - 1))])
                           : static_cast<double>(
                                 query_k[query_kv_index(d, head, key - context_keys)]);
            },
            [&](int d, int head, int key) {
                return key < context_keys
                           ? static_cast<double>(context_v[context_index(
                                 d, head, (context_begin + key) & (kWindow - 1))])
                           : static_cast<double>(
                                 query_v[query_kv_index(d, head, key - context_keys)]);
            },
            [](int, int) { return true; },
            [&](int d, int head, int, double value) { out[q_index(d, head, token)] = value; });
    }
}

CyclicKVCacheLayerView make_context_view(DeviceBuffer& k, DeviceBuffer& v, int lane_capacity = 1) {
    return {
        .k               = Tensor(k.p, DType::BF16, {kD, kWindow, kKVHeads, lane_capacity}),
        .v               = Tensor(v.p, DType::BF16, {kD, kWindow, kKVHeads, lane_capacity}),
        .capacity        = kWindow,
        .padded_capacity = kWindow,
        .num_kv_heads    = kKVHeads,
        .head_dim        = kD,
        .lane_capacity   = lane_capacity,
    };
}

enum class InputProfile {
    Random,
    WindowBoundary,
};

int run_case(int tokens, int context_length, InputProfile profile = InputProfile::Random,
             int envelope_max = -1, int valid_columns = -1) {
    if (envelope_max < 0) envelope_max = context_length;
    if (valid_columns < 0) valid_columns = tokens;
    const std::size_t q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_kv_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t context_count  = static_cast<std::size_t>(kD) * kWindow * kKVHeads;

    std::vector<float> q(q_count);
    std::vector<float> query_k(query_kv_count);
    std::vector<float> query_v(query_kv_count);
    std::vector<float> context_k(context_count);
    std::vector<float> context_v(context_count);
    const auto seed = static_cast<unsigned>(tokens * 131 + context_length * 17);
    fill_uniform(q, 101u + seed, -0.35f, 0.35f);
    fill_uniform(query_k, 211u + seed, -0.4f, 0.4f);
    fill_uniform(query_v, 307u + seed, -0.8f, 0.8f);
    fill_uniform(context_k, 401u + seed, -0.4f, 0.4f);
    fill_uniform(context_v, 503u + seed, -0.8f, 0.8f);

    if (profile == InputProfile::WindowBoundary) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(query_k.begin(), query_k.end(), 0.0f);
        std::fill(query_v.begin(), query_v.end(), 0.0f);
        std::fill(context_k.begin(), context_k.end(), 0.0f);
        std::fill(context_v.begin(), context_v.end(), 0.0f);
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) {
                context_v[context_index(d, kv_head, 0)]         = 512.0f;
                context_v[context_index(d, kv_head, 1)]         = 256.0f;
                query_v[query_kv_index(d, kv_head, tokens - 1)] = 1.0f;
            }
        }
    }

    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = context_length + token;
    }
    std::vector<double> reference;
    sliding_window_attention_oracle(q, query_k, query_v, context_k, context_v, positions,
                                    context_length, valid_columns, reference);

    const auto q_expected         = bf16_bits(q);
    const auto query_k_expected   = bf16_bits(query_k);
    const auto query_v_expected   = bf16_bits(query_v);
    const auto context_k_expected = bf16_bits(context_k);
    const auto context_v_expected = bf16_bits(context_v);
    const std::vector<int> valid_expected{valid_columns};
    const std::vector<int> lane_expected{0};

    DeviceBuffer d_q         = to_device(q_expected);
    DeviceBuffer d_query_k   = to_device(query_k_expected);
    DeviceBuffer d_query_v   = to_device(query_v_expected);
    DeviceBuffer d_context_k = to_device(context_k_expected);
    DeviceBuffer d_context_v = to_device(context_v_expected);
    DeviceBuffer d_positions = to_device_i32(positions);
    DeviceBuffer d_valid     = to_device(valid_expected);
    DeviceBuffer d_lane      = to_device(lane_expected);
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor positions_tensor(d_positions.p, DType::I32, {tokens, 1});
    Tensor valid_tensor(d_valid.p, DType::I32, {1});
    Tensor lane_tensor(d_lane.p, DType::I32, {1});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
    CyclicKVCacheLayerView context = make_context_view(d_context_k, d_context_v);
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{
        0, static_cast<std::uint32_t>(envelope_max)};
    const std::size_t workspace_bytes = ops::sliding_window_attention_workspace_capacity_bytes(
        kGeometry, kWindow, envelope, tokens, tokens, 1);
    DeviceArena workspace(workspace_bytes);

    ops::sliding_window_attention(q_tensor, query_k_tensor, query_v_tensor, positions_tensor,
                                  valid_tensor, lane_tensor, kGeometry, kWindow, kScale, context,
                                  envelope, workspace, out_tensor, nullptr);
    cuda_synchronize();

    std::string label = "sliding_window_attention T=" + std::to_string(tokens) +
                        " V=" + std::to_string(valid_columns) +
                        " L=" + std::to_string(context_length);
    if (envelope_max != context_length) {
        label += " envelope=[0," + std::to_string(envelope_max) + "]";
    }
    if (profile == InputProfile::WindowBoundary) label += " window-boundary";

    int failures =
        valid_columns == 0
            ? verify_exact(label.c_str(), from_device<std::uint16_t>(d_out.data(), q_count),
                           std::vector<std::uint16_t>(q_count, 0))
            : verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count), reference,
                               kSlidingWindowBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(d_q, q_count), q_expected);
    failures +=
        verify_exact((label + " query k unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_k, query_kv_count), query_k_expected);
    failures +=
        verify_exact((label + " query v unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_v, query_kv_count), query_v_expected);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<int>(d_positions, positions.size()), positions);
    failures += verify_exact((label + " valid columns unchanged").c_str(),
                             from_device<int>(d_valid, 1), valid_expected);
    failures += verify_exact((label + " lane unchanged").c_str(), from_device<int>(d_lane, 1),
                             lane_expected);
    failures +=
        verify_exact((label + " context k unchanged").c_str(),
                     from_device<std::uint16_t>(d_context_k, context_count), context_k_expected);
    failures +=
        verify_exact((label + " context v unchanged").c_str(),
                     from_device<std::uint16_t>(d_context_v, context_count), context_v_expected);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_case() {
    constexpr int tokens                 = 2;
    constexpr int batch                  = 2;
    const std::size_t row_q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t row_kv_count       = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t lane_context_count = static_cast<std::size_t>(kD) * kWindow * kKVHeads;
    std::vector<float> q(row_q_count * batch);
    std::vector<float> query_k(row_kv_count * batch);
    std::vector<float> query_v(row_kv_count * batch);
    std::vector<float> context_k(lane_context_count * batch);
    std::vector<float> context_v(lane_context_count * batch);
    std::fill(q.begin(), q.end(), 0.0f);
    std::fill(query_k.begin(), query_k.end(), 0.0f);
    std::fill(context_k.begin(), context_k.end(), 0.0f);
    std::fill(context_v.begin(), context_v.begin() + lane_context_count, 0.25f);
    std::fill(context_v.begin() + lane_context_count, context_v.end(), -0.5f);
    std::fill(query_v.begin(), query_v.begin() + row_kv_count, -0.5f);
    std::fill(query_v.begin() + row_kv_count, query_v.end(), 0.25f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    const std::vector<int> positions{4096, 4097, 65, 65};
    const std::vector<std::int32_t> valid{2, 1};
    const std::vector<std::int32_t> lanes{1, 0};

    DeviceBuffer d_q         = to_device(bf16_bits(q));
    DeviceBuffer d_query_k   = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v   = to_device(bf16_bits(query_v));
    DeviceBuffer d_context_k = to_device(bf16_bits(context_k));
    DeviceBuffer d_context_v = to_device(bf16_bits(context_v));
    DeviceBuffer d_positions = to_device_i32(positions);
    DeviceBuffer d_valid     = to_device(valid);
    DeviceBuffer d_lanes     = to_device(lanes);
    GuardedDeviceBuffer d_out(row_q_count * batch * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, batch});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor positions_tensor(d_positions.p, DType::I32, {tokens, batch});
    Tensor valid_tensor(d_valid.p, DType::I32, {batch});
    Tensor lanes_tensor(d_lanes.p, DType::I32, {batch});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, batch});
    constexpr ops::SlidingWindowAttentionExecutionEnvelope envelope{0, 4096};
    DeviceArena workspace(ops::sliding_window_attention_workspace_capacity_bytes(
        kGeometry, kWindow, envelope, tokens, tokens, batch));
    auto context = make_context_view(d_context_k, d_context_v, batch);

    std::vector<double> expected(row_q_count * batch);
    for (int b = 0; b < batch; ++b) {
        const auto q_begin       = q.begin() + static_cast<std::ptrdiff_t>(b * row_q_count);
        const auto query_k_begin = query_k.begin() + static_cast<std::ptrdiff_t>(b * row_kv_count);
        const auto query_v_begin = query_v.begin() + static_cast<std::ptrdiff_t>(b * row_kv_count);
        const int lane           = lanes[static_cast<std::size_t>(b)];
        const auto context_k_begin =
            context_k.begin() + static_cast<std::ptrdiff_t>(lane * lane_context_count);
        const auto context_v_begin =
            context_v.begin() + static_cast<std::ptrdiff_t>(lane * lane_context_count);
        const auto positions_begin = positions.begin() + static_cast<std::ptrdiff_t>(b * tokens);
        const std::vector<float> row_q(q_begin, q_begin + row_q_count);
        const std::vector<float> row_query_k(query_k_begin, query_k_begin + row_kv_count);
        const std::vector<float> row_query_v(query_v_begin, query_v_begin + row_kv_count);
        const std::vector<float> row_context_k(context_k_begin,
                                               context_k_begin + lane_context_count);
        const std::vector<float> row_context_v(context_v_begin,
                                               context_v_begin + lane_context_count);
        const std::vector<int> row_positions(positions_begin, positions_begin + tokens);
        std::vector<double> row_expected;
        sliding_window_attention_oracle(row_q, row_query_k, row_query_v, row_context_k,
                                        row_context_v, row_positions, row_positions.front(),
                                        valid[static_cast<std::size_t>(b)], row_expected);
        std::copy(row_expected.begin(), row_expected.end(),
                  expected.begin() + static_cast<std::ptrdiff_t>(b * row_q_count));
    }

    ops::sliding_window_attention(q_tensor, query_k_tensor, query_v_tensor, positions_tensor,
                                  valid_tensor, lanes_tensor, kGeometry, kWindow, kScale, context,
                                  envelope, workspace, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction("sliding_window_attention B=2 mixed lengths and lanes",
                                    from_device_bf16(d_out.data(), row_q_count * batch), expected,
                                    kSlidingWindowBf16Criterion);
    failures += d_out.verify_guards("sliding_window_attention B=2 output guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    int failures = 0;
    constexpr ops::SlidingWindowAttentionExecutionEnvelope capacity_envelope{0, 8194};
    const std::size_t interval = ops::sliding_window_attention_workspace_capacity_bytes(
        kGeometry, kWindow, capacity_envelope, 1, 16, 1);
    const std::size_t endpoint = ops::sliding_window_attention_workspace_capacity_bytes(
        kGeometry, kWindow, capacity_envelope, 16, 16, 1);
    if (interval != endpoint) {
        std::cerr << "sliding_window_attention interval capacity did not resolve to its monotonic "
                     "endpoint\n";
        ++failures;
    }
    try {
        (void)ops::sliding_window_attention_workspace_capacity_bytes(kGeometry, kWindow,
                                                                     capacity_envelope, 0, 16, 1);
        std::cerr << "sliding_window_attention accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += run_case(1, 0);
    failures += run_case(16, 1);
    failures += run_case(8, 96, InputProfile::Random, 4096);
    failures += run_case(16, 4096);
    failures += run_case(2, 4096, InputProfile::WindowBoundary);
    failures += run_case(2, 8194);
    failures += run_case(8, 65, InputProfile::Random, 96);
    failures += run_case(8, 65, InputProfile::Random, 96, 0);
    failures += run_case(8, 4096, InputProfile::Random, 4096, 0);
    failures += run_batch_case();

    if (failures != 0) {
        std::cerr << "sliding_window_attention failures=" << failures << '\n';
        return 1;
    }
    std::cout << "sliding_window_attention: PASS\n";
    return 0;
}
