#include "ninfer/ops/bidirectional_gqa_attention.h"

#include "core/arena.h"
#include "core/kv_cache.h"
#include "ops/launcher/bidirectional_gqa_attention.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

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

void fp64_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                 const std::vector<float>& query_v, const std::vector<float>& context_k,
                 const std::vector<float>& context_v, int tokens, int context_length,
                 int padded_context, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);
    const int keys = context_length + tokens;
    std::vector<double> scores(static_cast<std::size_t>(keys));
    std::vector<double> probability(static_cast<std::size_t>(keys));

    for (int token = 0; token < tokens; ++token) {
        for (int q_head = 0; q_head < kQHeads; ++q_head) {
            const int kv_head = q_head / kGroup;
            double maximum    = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < keys; ++key) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const float kval =
                        key < context_length
                            ? context_k[context_index(d, kv_head, key, padded_context)]
                            : query_k[query_kv_index(d, kv_head, key - context_length)];
                    dot += static_cast<double>(q[q_index(d, q_head, token)]) *
                           static_cast<double>(kval);
                }
                const double score                    = dot * static_cast<double>(kScale);
                scores[static_cast<std::size_t>(key)] = score;
                maximum                               = std::max(maximum, score);
            }

            double denominator = 0.0;
            for (int key = 0; key < keys; ++key) {
                const double value = std::exp(scores[static_cast<std::size_t>(key)] - maximum);
                probability[static_cast<std::size_t>(key)] = value;
                denominator += value;
            }
            for (int d = 0; d < kD; ++d) {
                double numerator = 0.0;
                for (int key = 0; key < keys; ++key) {
                    const float vval =
                        key < context_length
                            ? context_v[context_index(d, kv_head, key, padded_context)]
                            : query_v[query_kv_index(d, kv_head, key - context_length)];
                    numerator +=
                        probability[static_cast<std::size_t>(key)] * static_cast<double>(vval);
                }
                out[q_index(d, q_head, token)] = numerator / denominator;
            }
        }
    }
}

KVCacheLayerView make_context_view(DBuf& dk, DBuf& dv, int max_context, int padded_context) {
    return {
        .k              = Tensor(dk.p, DType::BF16, {kD, padded_context, kKVHeads}),
        .v              = Tensor(dv.p, DType::BF16, {kD, padded_context, kKVHeads}),
        .k_scale        = Tensor(),
        .v_scale        = Tensor(),
        .max_context    = static_cast<std::uint32_t>(max_context),
        .padded_context = static_cast<std::uint32_t>(padded_context),
        .num_kv_heads   = kKVHeads,
        .head_dim       = kD,
        .dtype          = DType::BF16,
        .quant_group    = 0,
    };
}

int run_case(int tokens, int context_length, bool capture_graph, bool check_immutability = true,
             int forced_key_block = 0) {
    const int max_context     = std::max(context_length, 1);
    const int padded_context  = align_context(max_context);
    const std::size_t qn      = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t qkvn    = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t cache_n = static_cast<std::size_t>(kD) * padded_context * kKVHeads;

    std::vector<float> q(qn);
    std::vector<float> query_k(qkvn);
    std::vector<float> query_v(qkvn);
    std::vector<float> context_k(cache_n);
    std::vector<float> context_v(cache_n);
    fill_uniform(q, 1100u + static_cast<unsigned>(tokens * 31 + context_length), -0.35f, 0.35f);
    fill_uniform(query_k, 2200u + static_cast<unsigned>(tokens * 37 + context_length), -0.4f, 0.4f);
    fill_uniform(query_v, 3300u + static_cast<unsigned>(tokens * 41 + context_length), -0.8f, 0.8f);
    fill_uniform(context_k, 4400u + static_cast<unsigned>(tokens * 43 + context_length), -0.4f,
                 0.4f);
    fill_uniform(context_v, 5500u + static_cast<unsigned>(tokens * 47 + context_length), -0.8f,
                 0.8f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    std::vector<double> reference;
    fp64_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length, padded_context,
                reference);

    DBuf dq      = to_device_bf16(q);
    DBuf dqk     = to_device_bf16(query_k);
    DBuf dqv     = to_device_bf16(query_v);
    DBuf dck     = to_device_bf16(context_k);
    DBuf dcv     = to_device_bf16(context_v);
    DBuf dlength = to_device_i32(std::vector<int>{context_length});
    GuardedDBuf dout(qn * sizeof(std::uint16_t));
    dout.fill(0x7f);

    Tensor tq(dq.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tqk(dqk.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tqv(dqv.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tl(dlength.p, DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kD, kQHeads, tokens});
    KVCacheLayerView context = make_context_view(dck, dcv, max_context, padded_context);
    DeviceArena workspace(ops::bidirectional_gqa_attention_workspace_bytes(tokens));
    const ops::GqaContextExecutionEnvelope envelope{
        static_cast<std::uint32_t>(context_length),
        static_cast<std::uint32_t>(context_length),
    };
    auto workspace_scope = workspace.scope();
    ops::detail::BidirectionalGqaPlan forced_plan;
    Tensor partial_acc;
    Tensor partial_m;
    Tensor partial_l;
    if (forced_key_block != 0) {
        forced_plan           = ops::detail::bidirectional_gqa_resolve_plan(tokens, envelope);
        forced_plan.key_block = forced_key_block;
        partial_acc =
            workspace.alloc(DType::BF16, {kD, kQHeads, tokens, forced_plan.split_capacity});
        partial_m = workspace.alloc(DType::FP32, {kQHeads, tokens, forced_plan.split_capacity});
        partial_l = workspace.alloc(DType::FP32, {kQHeads, tokens, forced_plan.split_capacity});
    }
    const auto launch = [&](cudaStream_t stream) {
        if (forced_key_block == 0) {
            ops::bidirectional_gqa_attention(tq, tqk, tqv, tl, kScale, context, envelope, workspace,
                                             tout, stream);
        } else {
            ops::detail::bidirectional_gqa_attention_launch(tq, tqk, tqv, tl, kScale, context,
                                                            forced_plan, partial_acc, partial_m,
                                                            partial_l, tout, stream);
        }
    };

    if (capture_graph) {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                   "cudaStreamBeginCapture");
        launch(stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
        cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
        cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch first");
        cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch second");
        cuda_synchronize(stream);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    } else {
        launch(nullptr);
        cuda_synchronize();
    }

    const std::string label =
        "bidirectional gqa T=" + std::to_string(tokens) + " L=" + std::to_string(context_length) +
        (capture_graph ? " graph" : "") +
        (forced_key_block == 0 ? "" : " key_block=" + std::to_string(forced_key_block));
    std::vector<std::uint16_t> got_bits(qn);
    dout.copy_to_host(got_bits.data(), got_bits.size() * sizeof(std::uint16_t));
    std::vector<double> got(got_bits.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        got[i] = static_cast<double>(bf16_to_f32(got_bits[i]));
    }

    int failures = verify(label.c_str(), got, reference, Tolerance::attention_bf16());
    failures += dout.verify_guards((label + " output guards").c_str());
    if (check_immutability) {
        const auto q_before  = from_device<std::uint16_t>(dq, qn);
        const auto qk_before = from_device<std::uint16_t>(dqk, qkvn);
        const auto qv_before = from_device<std::uint16_t>(dqv, qkvn);
        const auto ck_before = from_device<std::uint16_t>(dck, cache_n);
        const auto cv_before = from_device<std::uint16_t>(dcv, cache_n);
        std::vector<std::uint16_t> q_expected(qn), qk_expected(qkvn), qv_expected(qkvn),
            ck_expected(cache_n), cv_expected(cache_n);
        for (std::size_t i = 0; i < qn; ++i) q_expected[i] = f32_to_bf16(q[i]);
        for (std::size_t i = 0; i < qkvn; ++i) {
            qk_expected[i] = f32_to_bf16(query_k[i]);
            qv_expected[i] = f32_to_bf16(query_v[i]);
        }
        for (std::size_t i = 0; i < cache_n; ++i) {
            ck_expected[i] = f32_to_bf16(context_k[i]);
            cv_expected[i] = f32_to_bf16(context_v[i]);
        }
        failures += verify_exact((label + " q unchanged").c_str(), q_before, q_expected);
        failures += verify_exact((label + " query k unchanged").c_str(), qk_before, qk_expected);
        failures += verify_exact((label + " query v unchanged").c_str(), qv_before, qv_expected);
        failures += verify_exact((label + " context k unchanged").c_str(), ck_before, ck_expected);
        failures += verify_exact((label + " context v unchanged").c_str(), cv_before, cv_expected);
    }
    return failures;
}

int all_query_rows_visible_case() {
    constexpr int tokens         = 4;
    constexpr int context_length = 0;
    constexpr int padded_context = 128;
    constexpr int max_context    = 1;
    const std::size_t qn         = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t qkvn       = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t cache_n    = static_cast<std::size_t>(kD) * padded_context * kKVHeads;
    std::vector<float> q(qn, 0.0f), query_k(qkvn, 0.0f), query_v(qkvn, 0.0f);
    std::vector<float> context_k(cache_n, 0.0f), context_v(cache_n, 0.0f);
    for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (int d = 0; d < kD; ++d) { query_v[query_kv_index(d, kv_head, tokens - 1)] = 1.0f; }
    }

    DBuf dq      = to_device_bf16(q);
    DBuf dqk     = to_device_bf16(query_k);
    DBuf dqv     = to_device_bf16(query_v);
    DBuf dck     = to_device_bf16(context_k);
    DBuf dcv     = to_device_bf16(context_v);
    DBuf dlength = to_device_i32(std::vector<int>{context_length});
    DBuf dout(qn * sizeof(std::uint16_t));
    dout.fill(0x55);
    Tensor tq(dq.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tqk(dqk.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tqv(dqv.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tl(dlength.p, DType::I32, {1});
    Tensor tout(dout.p, DType::BF16, {kD, kQHeads, tokens});
    KVCacheLayerView context = make_context_view(dck, dcv, max_context, padded_context);
    DeviceArena workspace(ops::bidirectional_gqa_attention_workspace_bytes(tokens));
    ops::bidirectional_gqa_attention(tq, tqk, tqv, tl, kScale, context, {0, 0}, workspace, tout,
                                     nullptr);
    cuda_synchronize();

    const auto got = from_device_bf16(dout, qn);
    std::vector<double> expected(qn, 0.25);
    int failures = verify("bidirectional gqa all query rows visible", got, expected,
                          Tolerance::attention_bf16());
    if (got[q_index(0, 0, 0)] <= 0.2) {
        std::cerr << "bidirectional gqa: first query did not observe the final query value\n";
        ++failures;
    }
    return failures;
}

int maximum_envelope_case(int tokens) {
    constexpr int context_length = 262144;
    constexpr int padded_context = 262144;
    const std::size_t qn         = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t qkvn       = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t cache_n    = static_cast<std::size_t>(kD) * padded_context * kKVHeads;
    std::vector<float> q(qn, 0.0f);
    std::vector<float> query_k(qkvn, 0.0f);
    std::vector<float> query_v(qkvn, 1.0f);
    DBuf dq  = to_device_bf16(q);
    DBuf dqk = to_device_bf16(query_k);
    DBuf dqv = to_device_bf16(query_v);
    DBuf dck(cache_n * sizeof(std::uint16_t));
    DBuf dcv(cache_n * sizeof(std::uint16_t));
    dck.fill();
    dcv.fill();
    DBuf dlength = to_device_i32(std::vector<int>{context_length});
    GuardedDBuf dout(qn * sizeof(std::uint16_t));
    dout.fill(0x5a);

    Tensor tq(dq.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tqk(dqk.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tqv(dqv.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tl(dlength.p, DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kD, kQHeads, tokens});
    KVCacheLayerView context = make_context_view(dck, dcv, context_length, padded_context);
    DeviceArena workspace(ops::bidirectional_gqa_attention_workspace_bytes(tokens));
    cudaStream_t stream  = nullptr;
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate maximum envelope");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCapture maximum envelope");
    ops::bidirectional_gqa_attention(
        tq, tqk, tqv, tl, kScale, context,
        {static_cast<std::uint32_t>(context_length), static_cast<std::uint32_t>(context_length)},
        workspace, tout, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture maximum envelope");
    cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
               "cudaGraphInstantiate maximum envelope");
    cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch maximum envelope first");
    cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch maximum envelope second");
    cuda_synchronize(stream);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    const auto got = from_device_bf16(dout.data(), qn);
    const double expected_value =
        static_cast<double>(tokens) / static_cast<double>(context_length + tokens);
    std::vector<double> expected(qn, expected_value);
    const std::string label =
        "bidirectional gqa maximum envelope T=" + std::to_string(tokens) + " graph";
    int failures = verify(label.c_str(), got, expected, Tolerance::attention_bf16());
    failures += dout.verify_guards((label + " guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 0;
    }

    int failures = 0;
    for (int tokens = 1; tokens <= 16; ++tokens) {
        failures += run_case(tokens, 17 + tokens * 7, false, tokens == 1 || tokens == 16);
    }
    failures += run_case(1, 0, false);
    failures += run_case(16, 0, false);
    failures += run_case(2, 1, false);
    for (int tokens : {1, 4, 8, 12, 16}) {
        failures += run_case(tokens, 4096, false, tokens == 16);
    }
    failures += run_case(12, 257, false, false, 64);
    for (int tokens : {1, 2, 16}) { failures += run_case(tokens, 128, true, false); }
    failures += all_query_rows_visible_case();
    failures += maximum_envelope_case(2);
    failures += maximum_envelope_case(16);

    if (failures != 0) {
        std::cerr << "bidirectional_gqa_attention failures=" << failures << '\n';
        return 1;
    }
    std::cout << "bidirectional_gqa_attention: PASS\n";
    return 0;
}
