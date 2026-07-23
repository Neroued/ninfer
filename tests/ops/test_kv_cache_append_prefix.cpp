#include "ninfer/ops/kv_cache_append_prefix.h"

#include "ops/launcher/kv_cache_append_prefix.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD            = 128;
constexpr int kKVHeads      = 8;
constexpr int kLinearCap    = 64;
constexpr int kLinearPadded = 72;
constexpr int kWindow       = 4096;

std::size_t input_index(int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(int d, int head, int slot, int padded) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(padded) * static_cast<std::size_t>(head));
}

std::vector<std::uint16_t> patterned_bits(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(count);
    std::uint32_t state = seed;
    for (auto& bit : bits) {
        state = state * 1664525u + 1013904223u;
        bit   = static_cast<std::uint16_t>(state >> 16);
    }
    return bits;
}

void apply_oracle(std::vector<std::uint16_t>& cache_k, std::vector<std::uint16_t>& cache_v,
                  const std::vector<std::uint16_t>& k, const std::vector<std::uint16_t>& v,
                  int count, int position0, int padded, bool cyclic) {
    for (int token = 0; token < count; ++token) {
        const int position = position0 + token;
        const int slot     = cyclic ? position & (kWindow - 1) : position;
        for (int head = 0; head < kKVHeads; ++head) {
            for (int d = 0; d < kD; ++d) {
                const auto src = input_index(d, head, token);
                const auto dst = cache_index(d, head, slot, padded);
                cache_k[dst]   = k[src];
                cache_v[dst]   = v[src];
            }
        }
    }
}

KVCacheLayerView linear_view(GuardedDBuf& k, GuardedDBuf& v) {
    return {
        .k              = Tensor(k.data(), DType::BF16, {kD, kLinearPadded, kKVHeads}),
        .v              = Tensor(v.data(), DType::BF16, {kD, kLinearPadded, kKVHeads}),
        .k_scale        = Tensor(),
        .v_scale        = Tensor(),
        .max_context    = kLinearCap,
        .padded_context = kLinearPadded,
        .num_kv_heads   = kKVHeads,
        .head_dim       = kD,
        .dtype          = DType::BF16,
        .quant_group    = 0,
    };
}

CyclicKVCacheLayerView cyclic_view(GuardedDBuf& k, GuardedDBuf& v) {
    return {
        .k               = Tensor(k.data(), DType::BF16, {kD, kWindow, kKVHeads}),
        .v               = Tensor(v.data(), DType::BF16, {kD, kWindow, kKVHeads}),
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

int run_exact_case(
    int tokens, int count, bool cyclic,
    ops::detail::KVCacheAppendPrefixRoute route = ops::detail::KVCacheAppendPrefixRoute::Flat32,
    bool force_route = false, int envelope_min = 0) {
    const int padded              = cyclic ? kWindow : kLinearPadded;
    const int position            = cyclic ? 2 * kWindow - 2 : 17;
    const std::size_t input_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t cache_count = static_cast<std::size_t>(kD) * padded * kKVHeads;
    const auto host_k             = patterned_bits(input_count, 0x10203040u + tokens);
    const auto host_v             = patterned_bits(input_count, 0x50607080u + count);
    const auto seeded_k           = patterned_bits(cache_count, 0x90a0b0c0u);
    const auto seeded_v           = patterned_bits(cache_count, 0xd0e0f001u);
    auto expected_k               = seeded_k;
    auto expected_v               = seeded_v;
    const bool valid_count        = count >= envelope_min && count <= tokens;
    if (valid_count) {
        apply_oracle(expected_k, expected_v, host_k, host_v, count, position, padded, cyclic);
    }

    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) host_positions[static_cast<std::size_t>(i)] = position + i;
    std::vector<std::int32_t> host_count{count};
    DBuf dk = to_device(host_k), dv = to_device(host_v), dpos = to_device(host_positions),
         dcount = to_device(host_count);
    GuardedDBuf cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDBuf cache_v(cache_count * sizeof(std::uint16_t));
    cache_k.copy_from_host(seeded_k.data(), cache_k.bytes());
    cache_v.copy_from_host(seeded_v.data(), cache_v.bytes());

    Tensor tk(dk.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tp(dpos.p, DType::I32, {tokens});
    Tensor tc(dcount.p, DType::I32, {1});
    const ops::KVCacheAppendPrefixExecutionEnvelope envelope{
        static_cast<std::uint32_t>(envelope_min),
        static_cast<std::uint32_t>(tokens),
    };
    if (cyclic) {
        auto view = cyclic_view(cache_k, cache_v);
        if (force_route) {
            auto plan  = ops::detail::kv_cache_append_prefix_resolve_plan(tokens, envelope);
            plan.route = route;
            ops::detail::kv_cache_append_prefix_launch(tk, tv, tp, tc, view, plan, nullptr);
        } else {
            ops::kv_cache_append_prefix(tk, tv, tp, tc, envelope, view, nullptr);
        }
    } else {
        auto view = linear_view(cache_k, cache_v);
        if (force_route) {
            auto plan  = ops::detail::kv_cache_append_prefix_resolve_plan(tokens, envelope);
            plan.route = route;
            ops::detail::kv_cache_append_prefix_launch(tk, tv, tp, tc, view, plan, nullptr);
        } else {
            ops::kv_cache_append_prefix(tk, tv, tp, tc, envelope, view, nullptr);
        }
    }
    cuda_synchronize();

    const std::string label =
        std::string("kv prefix ") + (cyclic ? "cyclic" : "linear") +
        " T=" + std::to_string(tokens) + " C=" + std::to_string(count) +
        (force_route ? " " + std::string(ops::detail::kv_cache_append_prefix_route_name(route))
                     : "");
    std::vector<std::uint16_t> got_k(cache_count), got_v(cache_count);
    cache_k.copy_to_host(got_k.data(), cache_k.bytes());
    cache_v.copy_to_host(got_v.data(), cache_v.bytes());
    int failures = verify_exact((label + " cache k").c_str(), got_k, expected_k);
    failures += verify_exact((label + " cache v").c_str(), got_v, expected_v);
    failures += verify_exact((label + " input k").c_str(),
                             from_device<std::uint16_t>(dk, input_count), host_k);
    failures += verify_exact((label + " input v").c_str(),
                             from_device<std::uint16_t>(dv, input_count), host_v);
    failures +=
        verify_exact((label + " positions").c_str(),
                     from_device<std::int32_t>(dpos, host_positions.size()), host_positions);
    failures +=
        verify_exact((label + " count").c_str(), from_device<std::int32_t>(dcount, 1), host_count);
    failures += cache_k.verify_guards((label + " cache k guards").c_str());
    failures += cache_v.verify_guards((label + " cache v guards").c_str());
    return failures;
}

int graph_and_resume_case() {
    constexpr int tokens          = 16;
    constexpr int resume_tokens   = 2;
    constexpr int position        = 3 * kWindow - 3;
    const std::size_t first_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t next_count  = static_cast<std::size_t>(kD) * kKVHeads * resume_tokens;
    const std::size_t cache_count = static_cast<std::size_t>(kD) * kWindow * kKVHeads;
    const auto first_k            = patterned_bits(first_count, 0x11112222u);
    const auto first_v            = patterned_bits(first_count, 0x33334444u);
    const auto next_k             = patterned_bits(next_count, 0x55556666u);
    const auto next_v             = patterned_bits(next_count, 0x77778888u);
    const auto seeded_k           = patterned_bits(cache_count, 0x9999aaaau);
    const auto seeded_v           = patterned_bits(cache_count, 0xbbbbccccu);
    std::vector<std::int32_t> first_positions(tokens), next_positions(resume_tokens);
    for (int i = 0; i < tokens; ++i) first_positions[static_cast<std::size_t>(i)] = position + i;
    std::vector<std::int32_t> count{0};

    DBuf dfirst_k = to_device(first_k), dfirst_v = to_device(first_v),
         dfirst_pos = to_device(first_positions), dnext_k = to_device(next_k),
         dnext_v = to_device(next_v), dnext_pos(resume_tokens * sizeof(std::int32_t)),
         dcount  = to_device(count);
    GuardedDBuf cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDBuf cache_v(cache_count * sizeof(std::uint16_t));
    Tensor tk(dfirst_k.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tv(dfirst_v.p, DType::BF16, {kD, kKVHeads, tokens});
    Tensor tp(dfirst_pos.p, DType::I32, {tokens});
    Tensor tc(dcount.p, DType::I32, {1});
    Tensor tnext_k(dnext_k.p, DType::BF16, {kD, kKVHeads, resume_tokens});
    Tensor tnext_v(dnext_v.p, DType::BF16, {kD, kKVHeads, resume_tokens});
    Tensor tnext_pos(dnext_pos.p, DType::I32, {resume_tokens});
    auto view = cyclic_view(cache_k, cache_v);

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate prefix graph");
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCapture prefix");
    ops::kv_cache_append_prefix(tk, tv, tp, tc, {0, tokens}, view, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture prefix");
    cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
               "cudaGraphInstantiate prefix");

    int failures = 0;
    for (int committed = 0; committed <= tokens; ++committed) {
        cache_k.copy_from_host(seeded_k.data(), cache_k.bytes());
        cache_v.copy_from_host(seeded_v.data(), cache_v.bytes());
        cuda_check(cudaMemcpyAsync(dcount.p, &committed, sizeof(committed), cudaMemcpyHostToDevice,
                                   stream),
                   "copy graph count");
        cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch prefix");
        cuda_synchronize(stream);

        auto expected_k = seeded_k;
        auto expected_v = seeded_v;
        apply_oracle(expected_k, expected_v, first_k, first_v, committed, position, kWindow, true);
        std::vector<std::uint16_t> got_k(cache_count), got_v(cache_count);
        cache_k.copy_to_host(got_k.data(), cache_k.bytes());
        cache_v.copy_to_host(got_v.data(), cache_v.bytes());
        const std::string label = "kv prefix graph C=" + std::to_string(committed);
        failures += verify_exact((label + " cache k").c_str(), got_k, expected_k);
        failures += verify_exact((label + " cache v").c_str(), got_v, expected_v);

        for (int i = 0; i < resume_tokens; ++i) {
            next_positions[static_cast<std::size_t>(i)] = position + committed + i;
        }
        dnext_pos.copy_from_host(next_positions.data(), dnext_pos.bytes);
        const int resume_count = resume_tokens;
        dcount.copy_from_host(&resume_count, sizeof(resume_count));
        ops::kv_cache_append_prefix(tnext_k, tnext_v, tnext_pos, tc, {resume_tokens, resume_tokens},
                                    view, nullptr);
        cuda_synchronize();
        apply_oracle(expected_k, expected_v, next_k, next_v, resume_tokens, position + committed,
                     kWindow, true);
        cache_k.copy_to_host(got_k.data(), cache_k.bytes());
        cache_v.copy_to_host(got_v.data(), cache_v.bytes());
        failures += verify_exact((label + " resumed cache k").c_str(), got_k, expected_k);
        failures += verify_exact((label + " resumed cache v").c_str(), got_v, expected_v);
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    failures += cache_k.verify_guards("kv prefix graph cache k guards");
    failures += cache_v.verify_guards("kv prefix graph cache v guards");
    failures += verify_exact("kv prefix graph input k",
                             from_device<std::uint16_t>(dfirst_k, first_count), first_k);
    failures += verify_exact("kv prefix graph input v",
                             from_device<std::uint16_t>(dfirst_v, first_count), first_v);
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
        for (int count = 0; count <= tokens; ++count) {
            failures += run_exact_case(tokens, count, false);
            failures += run_exact_case(tokens, count, true);
        }
    }
    for (const auto route : {
             ops::detail::KVCacheAppendPrefixRoute::Flat16,
             ops::detail::KVCacheAppendPrefixRoute::Flat32,
             ops::detail::KVCacheAppendPrefixRoute::Persistent32,
             ops::detail::KVCacheAppendPrefixRoute::Token,
         }) {
        failures += run_exact_case(16, 7, false, route, true);
        failures += run_exact_case(16, 7, true, route, true);
    }
    failures += run_exact_case(16, 17, false);
    failures +=
        run_exact_case(16, 4, true, ops::detail::KVCacheAppendPrefixRoute::Flat32, false, 5);
    failures += graph_and_resume_case();

    if (failures != 0) {
        std::cerr << "kv_cache_append_prefix failures=" << failures << '\n';
        return 1;
    }
    std::cout << "kv_cache_append_prefix: PASS\n";
    return 0;
}
