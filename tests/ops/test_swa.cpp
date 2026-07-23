#include "ninfer/ops/swa.h"

#include "core/arena.h"
#include "core/kv_cache.h"
#include "ops/launcher/swa.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD       = 128;
constexpr int kQHeads  = 32;
constexpr int kKVHeads = 8;
constexpr int kGroup   = 4;
constexpr int kWindow  = 4096;
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

std::size_t context_index(int d, int kv_head, int slot) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(kWindow) * static_cast<std::size_t>(kv_head));
}

void fp64_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                 const std::vector<float>& query_v, const std::vector<float>& context_k,
                 const std::vector<float>& context_v, int tokens, int context_length,
                 std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);
    std::vector<double> scores(static_cast<std::size_t>(kWindow - 1 + tokens));
    std::vector<double> probability(scores.size());

    for (int token = 0; token < tokens; ++token) {
        const int query_position = context_length + token;
        const int context_begin  = std::max(0, query_position - (kWindow - 1));
        const int context_keys   = context_length - context_begin;
        const int keys           = context_keys + tokens;
        for (int q_head = 0; q_head < kQHeads; ++q_head) {
            const int kv_head = q_head / kGroup;
            double maximum    = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < keys; ++key) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const float kval =
                        key < context_keys
                            ? context_k[context_index(d, kv_head,
                                                      (context_begin + key) & (kWindow - 1))]
                            : query_k[query_kv_index(d, kv_head, key - context_keys)];
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
                        key < context_keys
                            ? context_v[context_index(d, kv_head,
                                                      (context_begin + key) & (kWindow - 1))]
                            : query_v[query_kv_index(d, kv_head, key - context_keys)];
                    numerator +=
                        probability[static_cast<std::size_t>(key)] * static_cast<double>(vval);
                }
                out[q_index(d, q_head, token)] = numerator / denominator;
            }
        }
    }
}

CyclicKVCacheLayerView make_context_view(DBuf& dk, DBuf& dv) {
    return {
        .k               = Tensor(dk.p, DType::BF16, {kD, kWindow, kKVHeads}),
        .v               = Tensor(dv.p, DType::BF16, {kD, kWindow, kKVHeads}),
        .k_scale         = Tensor(),
        .v_scale         = Tensor(),
        .capacity        = kWindow,
        .padded_capacity = kWindow,
        .num_kv_heads    = kKVHeads,
        .head_dim        = kD,
        .dtype           = DType::BF16,
        .quant_group     = 0,
    };
}

struct RouteOverride {
    bool enabled                = false;
    ops::detail::SwaRoute route = ops::detail::SwaRoute::SplitKv;
    int key_block               = 32;
    int split_capacity          = 0;
};

enum class InputProfile {
    Random,
    DominantKeyAndCancellation,
};

int run_case(int tokens, int context_length, bool capture_graph, bool check_immutability,
             RouteOverride route = {}, InputProfile profile = InputProfile::Random) {
    const std::size_t qn      = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t qkvn    = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t cache_n = static_cast<std::size_t>(kD) * kWindow * kKVHeads;

    std::vector<float> q(qn);
    std::vector<float> query_k(qkvn);
    std::vector<float> query_v(qkvn);
    std::vector<float> context_k(cache_n);
    std::vector<float> context_v(cache_n);
    fill_uniform(q, 101u + static_cast<unsigned>(tokens * 31 + context_length), -0.35f, 0.35f);
    fill_uniform(query_k, 211u + static_cast<unsigned>(tokens * 37 + context_length), -0.4f, 0.4f);
    fill_uniform(query_v, 307u + static_cast<unsigned>(tokens * 41 + context_length), -0.8f, 0.8f);
    fill_uniform(context_k, 401u + static_cast<unsigned>(tokens * 43 + context_length), -0.4f,
                 0.4f);
    fill_uniform(context_v, 503u + static_cast<unsigned>(tokens * 47 + context_length), -0.8f,
                 0.8f);
    if (profile == InputProfile::DominantKeyAndCancellation) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(query_k.begin(), query_k.end(), 0.0f);
        std::fill(context_k.begin(), context_k.end(), 0.0f);
        for (int token = 0; token < tokens; ++token) {
            for (int q_head = 0; q_head < kQHeads; ++q_head) {
                q[q_index(0, q_head, token)] = 1.0f;
            }
            for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
                for (int d = 0; d < kD; ++d) {
                    query_v[query_kv_index(d, kv_head, token)] =
                        ((token + d) & 1) == 0 ? 64.0f : -64.0f;
                }
            }
        }
        for (int slot = 0; slot < kWindow; ++slot) {
            for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
                for (int d = 0; d < kD; ++d) {
                    context_v[context_index(d, kv_head, slot)] =
                        ((slot + d) & 1) == 0 ? 64.0f : -64.0f;
                }
            }
        }
        const int dominant_slot = (context_length - 3) & (kWindow - 1);
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            context_k[context_index(0, kv_head, dominant_slot)] = 80.0f;
            for (int d = 0; d < kD; ++d) {
                context_v[context_index(d, kv_head, dominant_slot)] = 0.5f;
            }
        }
    }
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    std::vector<double> reference;
    fp64_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length, reference);
    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) positions[static_cast<std::size_t>(i)] = context_length + i;

    DBuf dq         = to_device_bf16(q);
    DBuf dqk        = to_device_bf16(query_k);
    DBuf dqv        = to_device_bf16(query_v);
    DBuf dck        = to_device_bf16(context_k);
    DBuf dcv        = to_device_bf16(context_v);
    DBuf dpositions = to_device_i32(positions);
    GuardedDBuf dout(qn * sizeof(std::uint16_t));
    dout.fill(0x7f);

    Tensor tq(dq.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tqk(dqk.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tqv(dqv.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tp(dpositions.p, DType::I32, {tokens});
    Tensor tout(dout.data(), DType::BF16, {kD, kQHeads, tokens});
    CyclicKVCacheLayerView context = make_context_view(dck, dcv);
    DeviceArena workspace(ops::swa_workspace_bytes(tokens));
    const ops::SwaContextExecutionEnvelope envelope{
        static_cast<std::uint32_t>(context_length),
        static_cast<std::uint32_t>(context_length),
    };
    auto workspace_scope = workspace.scope();
    ops::detail::SwaPlan forced_plan;
    Tensor partial_acc;
    Tensor partial_m;
    Tensor partial_l;
    if (route.enabled) {
        forced_plan           = ops::detail::swa_resolve_plan(tokens, envelope);
        forced_plan.route     = route.route;
        forced_plan.key_block = route.key_block;
        forced_plan.split_capacity =
            route.route == ops::detail::SwaRoute::Direct
                ? 1
                : (route.split_capacity == 0 ? forced_plan.split_capacity : route.split_capacity);
        partial_acc =
            workspace.alloc(DType::BF16, {kD, kQHeads, tokens, forced_plan.split_capacity});
        partial_m = workspace.alloc(DType::FP32, {kQHeads, tokens, forced_plan.split_capacity});
        partial_l = workspace.alloc(DType::FP32, {kQHeads, tokens, forced_plan.split_capacity});
    }
    const auto launch = [&](cudaStream_t stream) {
        if (!route.enabled) {
            ops::swa(tq, tqk, tqv, tp, kScale, context, envelope, workspace, tout, stream);
        } else {
            ops::detail::swa_launch(tq, tqk, tqv, tp, kScale, context, forced_plan, partial_acc,
                                    partial_m, partial_l, tout, stream);
        }
    };

    if (capture_graph) {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate SWA");
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                   "cudaStreamBeginCapture SWA");
        launch(stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture SWA");
        cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                   "cudaGraphInstantiate SWA");
        cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch SWA first");
        cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch SWA second");
        cuda_synchronize(stream);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    } else {
        launch(nullptr);
        cuda_synchronize();
    }

    std::string label = "swa T=" + std::to_string(tokens) + " L=" + std::to_string(context_length) +
                        (capture_graph ? " graph" : "");
    if (route.enabled) {
        label += " " + std::string(ops::detail::swa_route_name(route.route)) +
                 " kb=" + std::to_string(route.key_block);
    }
    if (profile == InputProfile::DominantKeyAndCancellation) {
        label += " dominant-key cancellation";
    }
    const auto got = from_device_bf16(dout.data(), qn);
    int failures   = verify(label.c_str(), got, reference, Tolerance::attention_bf16());
    failures += dout.verify_guards((label + " output guards").c_str());
    if (check_immutability) {
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
        failures += verify_exact((label + " q unchanged").c_str(),
                                 from_device<std::uint16_t>(dq, qn), q_expected);
        failures += verify_exact((label + " query k unchanged").c_str(),
                                 from_device<std::uint16_t>(dqk, qkvn), qk_expected);
        failures += verify_exact((label + " query v unchanged").c_str(),
                                 from_device<std::uint16_t>(dqv, qkvn), qv_expected);
        failures += verify_exact((label + " positions unchanged").c_str(),
                                 from_device<int>(dpositions, positions.size()), positions);
        failures += verify_exact((label + " context k unchanged").c_str(),
                                 from_device<std::uint16_t>(dck, cache_n), ck_expected);
        failures += verify_exact((label + " context v unchanged").c_str(),
                                 from_device<std::uint16_t>(dcv, cache_n), cv_expected);
    }
    return failures;
}

int invalid_scale_case() {
    constexpr int tokens      = 1;
    const std::size_t qn      = static_cast<std::size_t>(kD) * kQHeads;
    const std::size_t qkvn    = static_cast<std::size_t>(kD) * kKVHeads;
    const std::size_t cache_n = static_cast<std::size_t>(kD) * kWindow * kKVHeads;
    DBuf dq(qn * sizeof(std::uint16_t)), dqk(qkvn * sizeof(std::uint16_t)),
        dqv(qkvn * sizeof(std::uint16_t)), dck(cache_n * sizeof(std::uint16_t)),
        dcv(cache_n * sizeof(std::uint16_t)), dpositions(sizeof(std::int32_t)),
        dout(qn * sizeof(std::uint16_t));
    Tensor tq(dq.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tqk(dqk.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tqv(dqv.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tp(dpositions.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kD, kQHeads, tokens});
    auto context = make_context_view(dck, dcv);
    DeviceArena workspace(ops::swa_workspace_bytes(tokens));

    int failures = 0;
    for (const float scale :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        try {
            ops::swa(tq, tqk, tqv, tp, scale, context, {0, 0}, workspace, tout, nullptr);
            std::cerr << "swa accepted a non-finite scale\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    return failures;
}

int equal_logit_boundary_case(int tokens, int context_length, int marked_position,
                              double expected_token0, double expected_token1) {
    const std::size_t qn      = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t qkvn    = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t cache_n = static_cast<std::size_t>(kD) * kWindow * kKVHeads;
    std::vector<float> q(qn, 0.0f), query_k(qkvn, 0.0f), query_v(qkvn, 0.0f);
    std::vector<float> context_k(cache_n, 0.0f), context_v(cache_n, 0.0f);
    for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (int d = 0; d < kD; ++d) {
            context_v[context_index(d, kv_head, marked_position & (kWindow - 1))] = 512.0f;
        }
    }
    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) positions[static_cast<std::size_t>(i)] = context_length + i;
    DBuf dq         = to_device_bf16(q);
    DBuf dqk        = to_device_bf16(query_k);
    DBuf dqv        = to_device_bf16(query_v);
    DBuf dck        = to_device_bf16(context_k);
    DBuf dcv        = to_device_bf16(context_v);
    DBuf dpositions = to_device_i32(positions);
    GuardedDBuf dout(qn * sizeof(std::uint16_t));
    dout.fill(0x55);
    Tensor tq(dq.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tqk(dqk.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tqv(dqv.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tp(dpositions.p, DType::I32, {tokens});
    Tensor tout(dout.data(), DType::BF16, {kD, kQHeads, tokens});
    CyclicKVCacheLayerView context = make_context_view(dck, dcv);
    DeviceArena workspace(ops::swa_workspace_bytes(tokens));
    ops::swa(
        tq, tqk, tqv, tp, kScale, context,
        {static_cast<std::uint32_t>(context_length), static_cast<std::uint32_t>(context_length)},
        workspace, tout, nullptr);
    cuda_synchronize();

    std::vector<double> expected(qn, 0.0);
    for (int h = 0; h < kQHeads; ++h) {
        for (int d = 0; d < kD; ++d) {
            expected[q_index(d, h, 0)] = expected_token0;
            if (tokens > 1) expected[q_index(d, h, 1)] = expected_token1;
        }
    }
    const std::string label = "swa boundary L=" + std::to_string(context_length) +
                              " marked=" + std::to_string(marked_position);
    int failures = verify(label.c_str(), from_device_bf16(dout.data(), qn), expected,
                          Tolerance::attention_bf16());
    failures += dout.verify_guards((label + " guards").c_str());
    return failures;
}

int all_query_rows_visible_case() {
    constexpr int tokens      = 16;
    const std::size_t qn      = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t qkvn    = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t cache_n = static_cast<std::size_t>(kD) * kWindow * kKVHeads;
    std::vector<float> q(qn, 0.0f), query_k(qkvn, 0.0f), query_v(qkvn, 0.0f);
    std::vector<float> context(cache_n, 0.0f);
    for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (int d = 0; d < kD; ++d) { query_v[query_kv_index(d, kv_head, tokens - 1)] = 1.0f; }
    }
    DBuf dq = to_device_bf16(q), dqk = to_device_bf16(query_k), dqv = to_device_bf16(query_v),
         dck = to_device_bf16(context), dcv = to_device_bf16(context),
         dpositions =
             to_device_i32(std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    DBuf dout(qn * sizeof(std::uint16_t));
    Tensor tq(dq.p, DType::BF16, {kD, kQHeads, tokens});
    Tensor tqk(dqk.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tqv(dqv.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tp(dpositions.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kD, kQHeads, tokens});
    auto context_view = make_context_view(dck, dcv);
    DeviceArena workspace(ops::swa_workspace_bytes(tokens));
    ops::swa(tq, tqk, tqv, tp, kScale, context_view, {0, 0}, workspace, tout, nullptr);
    cuda_synchronize();
    std::vector<double> expected(qn, 1.0 / tokens);
    return verify("swa all query rows visible", from_device_bf16(dout, qn), expected,
                  Tolerance::attention_bf16());
}

int predicate_contract_case() {
    const auto allowed = [](int a, int b) {
        return std::abs(static_cast<std::int64_t>(a) - static_cast<std::int64_t>(b)) < kWindow;
    };
    int failures = 0;
    for (const int delta : {-4096, -4095, 4095, 4096}) {
        const bool expected = std::abs(delta) == 4095;
        if (allowed(100000, 100000 + delta) != expected) {
            std::cerr << "swa predicate mismatch at delta=" << delta << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 0;
    }

    int failures = predicate_contract_case();
    for (int tokens = 1; tokens <= 16; ++tokens) {
        failures += run_case(tokens, 19 + tokens * 5, false, tokens == 1 || tokens == 16);
    }
    failures += run_case(1, 0, true, false);
    failures += run_case(16, 0, false, false);
    failures += run_case(2, 1, false, false);
    for (const int tokens : {1, 8, 16}) {
        failures += run_case(tokens, 4095, false, false);
        failures += run_case(tokens, 4096, tokens == 16, tokens == 16);
        failures += run_case(tokens, 8194, tokens == 16, false);
    }
    failures += run_case(12, 4097, false, false,
                         {.enabled        = true,
                          .route          = ops::detail::SwaRoute::SplitKv,
                          .key_block      = 64,
                          .split_capacity = 16});
    failures +=
        run_case(4, 257, false, false,
                 {.enabled = true, .route = ops::detail::SwaRoute::Direct, .key_block = 32});
    failures += run_case(16, 262144, true, false);
    failures += run_case(16, 4096, false, false, {}, InputProfile::DominantKeyAndCancellation);
    failures += equal_logit_boundary_case(16, 4095, 0, 512.0 / (4095.0 + 16.0), 0.0);
    failures += equal_logit_boundary_case(16, 4096, 0, 0.0, 0.0);
    failures += equal_logit_boundary_case(16, 4096, 1, 512.0 / (4095.0 + 16.0), 0.0);
    failures += equal_logit_boundary_case(16, 8192, 4097, 512.0 / (4095.0 + 16.0), 0.0);
    failures += all_query_rows_visible_case();
    failures += invalid_scale_case();

    if (failures != 0) {
        std::cerr << "swa failures=" << failures << '\n';
        return 1;
    }
    std::cout << "swa: PASS\n";
    return 0;
}
