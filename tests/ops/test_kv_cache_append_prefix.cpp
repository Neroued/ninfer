#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim      = 128;
constexpr int kKVHeads      = 8;
constexpr int kLinearCap    = 64;
constexpr int kLinearPadded = 72;
constexpr int kWindow       = 4096;

std::size_t input_index(int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(int d, int head, int slot, int padded) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
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

void append_oracle(std::vector<std::uint16_t>& cache_k, std::vector<std::uint16_t>& cache_v,
                   const std::vector<std::uint16_t>& input_k,
                   const std::vector<std::uint16_t>& input_v,
                   const std::vector<std::int32_t>& positions, int commit_count, int padded,
                   bool cyclic) {
    for (int token = 0; token < commit_count; ++token) {
        const int position = positions[static_cast<std::size_t>(token)];
        const int slot     = cyclic ? position % kWindow : position;
        for (int head = 0; head < kKVHeads; ++head) {
            for (int d = 0; d < kHeadDim; ++d) {
                const auto src = input_index(d, head, token);
                const auto dst = cache_index(d, head, slot, padded);
                cache_k[dst]   = input_k[src];
                cache_v[dst]   = input_v[src];
            }
        }
    }
}

KVCacheLayerView linear_view(GuardedDeviceBuffer& k, GuardedDeviceBuffer& v) {
    return {
        .k              = Tensor(k.data(), DType::BF16, {kHeadDim, kLinearPadded, kKVHeads}),
        .v              = Tensor(v.data(), DType::BF16, {kHeadDim, kLinearPadded, kKVHeads}),
        .k_scale        = Tensor(),
        .v_scale        = Tensor(),
        .max_context    = kLinearCap,
        .padded_context = kLinearPadded,
        .num_kv_heads   = kKVHeads,
        .head_dim       = kHeadDim,
        .k_dtype        = DType::BF16,
        .v_dtype        = DType::BF16,
        .k_quant_group  = 0,
        .v_quant_group  = 0,
    };
}

// DFlash's local cyclic caches are fixed BF16 independently of the target KV format.
CyclicKVCacheLayerView cyclic_view(GuardedDeviceBuffer& k, GuardedDeviceBuffer& v) {
    return {
        .k               = Tensor(k.data(), DType::BF16, {kHeadDim, kWindow, kKVHeads}),
        .v               = Tensor(v.data(), DType::BF16, {kHeadDim, kWindow, kKVHeads}),
        .k_scale         = Tensor(),
        .v_scale         = Tensor(),
        .capacity        = kWindow,
        .padded_capacity = kWindow,
        .num_kv_heads    = kKVHeads,
        .head_dim        = kHeadDim,
        .dtype           = DType::BF16,
        .quant_group     = 0,
    };
}

int run_case(int tokens, int commit_count, int first_position, bool cyclic, int min_count = 0) {
    const int padded              = cyclic ? kWindow : kLinearPadded;
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count = static_cast<std::size_t>(kHeadDim) * padded * kKVHeads;
    const auto host_k = patterned_bits(input_count, 0x10203040u + static_cast<unsigned>(tokens));
    const auto host_v =
        patterned_bits(input_count, 0x50607080u + static_cast<unsigned>(commit_count));
    const auto initial_k = patterned_bits(cache_count, 0x90a0b0c0u);
    const auto initial_v = patterned_bits(cache_count, 0xd0e0f001u);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) {
        positions[static_cast<std::size_t>(i)] = first_position + i;
    }
    auto expected_k = initial_k;
    auto expected_v = initial_v;
    append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, padded, cyclic);

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({commit_count});
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
    cache_v.copy_from_host(initial_v.data(), cache_v.bytes());

    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens});
    Tensor count_tensor(d_count.p, DType::I32, {1});
    const ops::KVCacheAppendPrefixExecutionEnvelope envelope{
        .min_count = static_cast<std::uint32_t>(min_count),
        .max_count = static_cast<std::uint32_t>(tokens),
    };
    if (cyclic) {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, envelope,
                                    cyclic_view(cache_k, cache_v), nullptr);
    } else {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, envelope,
                                    linear_view(cache_k, cache_v), nullptr);
    }
    cuda_synchronize();

    const std::string label = std::string("kv_cache_append_prefix ") +
                              (cyclic ? "cyclic" : "linear") + " T=" + std::to_string(tokens) +
                              " C=" + std::to_string(commit_count) +
                              " min=" + std::to_string(min_count);
    int failures =
        verify_exact((label + " cache k").c_str(),
                     from_device<std::uint16_t>(cache_k.data(), cache_count), expected_k);
    failures += verify_exact((label + " cache v").c_str(),
                             from_device<std::uint16_t>(cache_v.data(), cache_count), expected_v);
    failures += verify_exact((label + " input k unchanged").c_str(),
                             from_device<std::uint16_t>(d_k, input_count), host_k);
    failures += verify_exact((label + " input v unchanged").c_str(),
                             from_device<std::uint16_t>(d_v, input_count), host_v);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<std::int32_t>(d_positions, positions.size()), positions);
    failures += verify_exact((label + " count unchanged").c_str(),
                             from_device<std::int32_t>(d_count, 1), {commit_count});
    failures += cache_k.verify_guards((label + " cache k guards").c_str());
    failures += cache_v.verify_guards((label + " cache v guards").c_str());
    return failures;
}

int graph_replay_case() {
    constexpr int tokens          = 16;
    constexpr int first_position  = 2 * kWindow - 4;
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count = static_cast<std::size_t>(kHeadDim) * kWindow * kKVHeads;
    const auto host_k             = patterned_bits(input_count, 0x11223344u);
    const auto host_v             = patterned_bits(input_count, 0x55667788u);
    const auto initial_k          = patterned_bits(cache_count, 0x99aabbccu);
    const auto initial_v          = patterned_bits(cache_count, 0xddeeff01u);
    std::vector<std::int32_t> positions(tokens);
    for (int i = 0; i < tokens; ++i) positions[static_cast<std::size_t>(i)] = first_position + i;

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({0});
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens});
    Tensor count_tensor(d_count.p, DType::I32, {1});
    auto cache = cyclic_view(cache_k, cache_v);

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create kv append stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin kv append capture");
    ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, {0, tokens}, cache, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end kv append capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate kv append graph");

    int failures = 0;
    for (const int commit_count : std::array{0, 7, tokens}) {
        cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
        cache_v.copy_from_host(initial_v.data(), cache_v.bytes());
        d_count.copy_from_host(&commit_count, sizeof(commit_count));
        cuda_check(cudaGraphLaunch(executable, stream), "launch kv append graph");
        cuda_synchronize(stream);

        auto expected_k = initial_k;
        auto expected_v = initial_v;
        append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, kWindow,
                      true);
        const std::string label = "kv_cache_append_prefix graph C=" + std::to_string(commit_count);
        failures +=
            verify_exact((label + " cache k").c_str(),
                         from_device<std::uint16_t>(cache_k.data(), cache_count), expected_k);
        failures +=
            verify_exact((label + " cache v").c_str(),
                         from_device<std::uint16_t>(cache_v.data(), cache_count), expected_v);
        failures += verify_exact((label + " count unchanged").c_str(),
                                 from_device<std::int32_t>(d_count, 1), {commit_count});
    }

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    failures += verify_exact("kv append graph input k unchanged",
                             from_device<std::uint16_t>(d_k, input_count), host_k);
    failures += verify_exact("kv append graph input v unchanged",
                             from_device<std::uint16_t>(d_v, input_count), host_v);
    failures += verify_exact("kv append graph positions unchanged",
                             from_device<std::int32_t>(d_positions, positions.size()), positions);
    failures += cache_k.verify_guards("kv append graph cache k guards");
    failures += cache_v.verify_guards("kv append graph cache v guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "kv_cache_append_prefix: SKIP (CUDA unavailable)\n";
        return 0;
    }

    int failures = 0;
    failures += run_case(1, 0, 0, false);
    failures += run_case(1, 1, 11, false);
    failures += run_case(16, 7, 17, false, 5);
    failures += run_case(16, 16, 40, false);
    failures += run_case(1, 0, kWindow - 1, true);
    failures += run_case(1, 1, 2 * kWindow - 1, true);
    failures += run_case(16, 7, 2 * kWindow - 2, true);
    failures += run_case(16, 16, 3 * kWindow - 8, true, 16);
    failures += graph_replay_case();

    if (failures != 0) {
        std::cerr << "kv_cache_append_prefix failures=" << failures << '\n';
        return 1;
    }
    std::cout << "kv_cache_append_prefix: PASS\n";
    return 0;
}
