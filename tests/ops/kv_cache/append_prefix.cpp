#include "ops/suite.h"
#include "ops/float_codec.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/kv_cache_append.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::float_codec;

namespace {

constexpr int kHeadDim       = 128;
constexpr int kKVHeads       = 8;
constexpr int kPage          = 64;
constexpr int kLogicalPages  = 3;
constexpr int kPhysicalPages = 6;
constexpr int kDFlash2Window = 2048;
constexpr int kDFlashWindow  = 4096;

std::uint16_t bf16_bits_to_f16_bits(std::uint16_t bits) {
    return f32_to_f16_bits(bf16_to_f32(bits));
}

std::size_t input_index(int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t cyclic_cache_index(int d, int head, int slot, int capacity) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(capacity) * static_cast<std::size_t>(head));
}

std::size_t paged_cache_index(int d, int head, int position,
                              const std::vector<std::int32_t>& mapping) {
    const int physical_page = mapping.at(static_cast<std::size_t>(position / kPage));
    const int page_offset   = position % kPage;
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(page_offset) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(physical_page) +
                     static_cast<std::size_t>(kPhysicalPages) * static_cast<std::size_t>(head)));
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

std::vector<std::uint16_t> finite_patterned_bf16_bits(std::size_t count, std::uint32_t seed) {
    auto bits = patterned_bits(count, seed);
    for (auto& value : bits) {
        if ((value & 0x7f80u) == 0x7f80u) { value ^= 0x0080u; }
    }
    return bits;
}

void append_oracle(std::vector<std::uint16_t>& cache_k, std::vector<std::uint16_t>& cache_v,
                   const std::vector<std::uint16_t>& input_k,
                   const std::vector<std::uint16_t>& input_v,
                   const std::vector<std::int32_t>& positions, int commit_count, bool cyclic,
                   const std::vector<std::int32_t>& mapping, int cyclic_capacity = 0) {
    for (int token = 0; token < commit_count; ++token) {
        const int position = positions[static_cast<std::size_t>(token)];
        const int slot     = cyclic ? position % cyclic_capacity : 0;
        for (int head = 0; head < kKVHeads; ++head) {
            for (int d = 0; d < kHeadDim; ++d) {
                const auto src = input_index(d, head, token);
                const auto dst = cyclic ? cyclic_cache_index(d, head, slot, cyclic_capacity)
                                        : paged_cache_index(d, head, position, mapping);
                cache_k[dst]   = input_k[src];
                cache_v[dst]   = bf16_bits_to_f16_bits(input_v[src]);
            }
        }
    }
}

PagedKVBatchLayerView paged_view(GuardedDeviceBuffer& k, GuardedDeviceBuffer& v,
                                 DeviceBuffer& block_table, int table_rows = 1) {
    return {
        .k_pages      = Tensor(k.span(), DType::BF16, {kHeadDim, kPage, kPhysicalPages, kKVHeads}),
        .v_pages      = Tensor(v.span(), DType::FP16, {kHeadDim, kPage, kPhysicalPages, kKVHeads}),
        .block_tables = Tensor(block_table.span(), DType::I32, {kLogicalPages, table_rows}),
        .head_dim     = kHeadDim,
        .num_kv_heads = kKVHeads,
        .storage      = KvCacheStorage::BFloat16,
    };
}

CyclicKVCacheLayerView cyclic_view(GuardedDeviceBuffer& k, GuardedDeviceBuffer& v, int capacity,
                                   int slot_count = 1) {
    return {
        .k        = Tensor(k.span(), DType::BF16, {kHeadDim, capacity, kKVHeads, slot_count}),
        .v        = Tensor(v.span(), DType::FP16, {kHeadDim, capacity, kKVHeads, slot_count}),
        .capacity = static_cast<std::uint32_t>(capacity),
        .padded_capacity = static_cast<std::uint32_t>(capacity),
        .num_kv_heads    = kKVHeads,
        .head_dim        = kHeadDim,
        .slot_count      = slot_count,
    };
}

int run_case(int tokens, int commit_count, int first_position, bool cyclic,
             std::vector<std::int32_t> mapping = {}, int min_count = 0,
             int cyclic_capacity = kDFlashWindow) {
    if (!cyclic && mapping.size() != kLogicalPages) {
        throw std::invalid_argument("paged prefix case requires a complete mapping");
    }
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count = static_cast<std::size_t>(kHeadDim) * kKVHeads *
                                    (cyclic ? cyclic_capacity : kPage * kPhysicalPages);
    const auto host_k = patterned_bits(input_count, 0x10203040u + static_cast<unsigned>(tokens));
    const auto host_v =
        finite_patterned_bf16_bits(input_count, 0x50607080u + static_cast<unsigned>(commit_count));
    const auto initial_k = patterned_bits(cache_count, 0x90a0b0c0u);
    const auto initial_v = patterned_bits(cache_count, 0xd0e0f001u);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) {
        positions[static_cast<std::size_t>(i)] = first_position + i;
    }
    auto expected_k = initial_k;
    auto expected_v = initial_v;
    append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, cyclic, mapping,
                  cyclic_capacity);

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({commit_count});
    DeviceBuffer d_selector  = to_device<std::int32_t>({0});
    DeviceBuffer d_table     = cyclic
                                   ? DeviceBuffer(ninfer::test::cuda_environment().context.device(), 1)
                                   : to_device(mapping);
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
    cache_v.copy_from_host(initial_v.data(), cache_v.bytes());

    Tensor k(d_k.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.span(), DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.span(), DType::I32, {1});
    Tensor selector_tensor(d_selector.span(), DType::I32, {1});
    const ops::KVCacheAppendPrefixExecutionEnvelope envelope{
        .min_count = static_cast<std::uint32_t>(min_count),
        .max_count = static_cast<std::uint32_t>(tokens),
    };
    if (cyclic) {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    cyclic_view(cache_k, cache_v, cyclic_capacity),
                                    ninfer::test::cuda_environment().default_execution);
    } else {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    paged_view(cache_k, cache_v, d_table),
                                    ninfer::test::cuda_environment().default_execution);
    }
    cuda_synchronize();

    const std::string label = std::string("kv_cache_append_prefix ") +
                              (cyclic ? "cyclic" : "paged") + " T=" + std::to_string(tokens) +
                              " C=" + std::to_string(commit_count) +
                              " min=" + std::to_string(min_count) +
                              (cyclic ? " capacity=" + std::to_string(cyclic_capacity) : "");
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

int cyclic_graph_replay_case(int capacity, int tokens) {
    const int first_position      = 2 * capacity - 4;
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count = static_cast<std::size_t>(kHeadDim) * capacity * kKVHeads;
    const auto host_k             = patterned_bits(input_count, 0x11223344u);
    const auto host_v             = finite_patterned_bf16_bits(input_count, 0x55667788u);
    const auto initial_k          = patterned_bits(cache_count, 0x99aabbccu);
    const auto initial_v          = patterned_bits(cache_count, 0xddeeff01u);
    std::vector<std::int32_t> positions(tokens);
    for (int i = 0; i < tokens; ++i) positions[static_cast<std::size_t>(i)] = first_position + i;

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({0});
    DeviceBuffer d_lane      = to_device<std::int32_t>({0});
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    Tensor k(d_k.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.span(), DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.span(), DType::I32, {1});
    Tensor lane_tensor(d_lane.span(), DType::I32, {1});
    auto cache = cyclic_view(cache_k, cache_v, capacity);

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create kv append stream");
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin kv append capture");
    const ops::KVCacheAppendPrefixExecutionEnvelope envelope{0, static_cast<std::uint32_t>(tokens)};
    ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, lane_tensor, envelope, cache,
                                stream_execution);
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
        append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, true, {},
                      capacity);
        const std::string label =
            "kv_cache_append_prefix cyclic graph capacity=" + std::to_string(capacity) +
            " C=" + std::to_string(commit_count);
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

int paged_graph_replay_case() {
    constexpr int tokens          = 16;
    constexpr int first_position  = 60;
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count =
        static_cast<std::size_t>(kHeadDim) * kPage * kKVHeads * kPhysicalPages;
    const auto host_k    = patterned_bits(input_count, 0x12345678u);
    const auto host_v    = finite_patterned_bf16_bits(input_count, 0x87654321u);
    const auto initial_k = patterned_bits(cache_count, 0xabcdef01u);
    const auto initial_v = patterned_bits(cache_count, 0x10fedcbau);
    std::vector<std::int32_t> positions(tokens);
    for (int i = 0; i < tokens; ++i) positions[static_cast<std::size_t>(i)] = first_position + i;

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({0});
    DeviceBuffer d_row       = to_device<std::int32_t>({0});
    DeviceBuffer d_table     = to_device<std::int32_t>({0, 1, 2});
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    Tensor k(d_k.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.span(), DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.span(), DType::I32, {1});
    Tensor row_tensor(d_row.span(), DType::I32, {1});
    auto cache = paged_view(cache_k, cache_v, d_table);

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create paged kv append stream");
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin paged kv append capture");
    ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, row_tensor, {0, tokens}, cache,
                                stream_execution);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end paged kv append capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate paged kv append graph");

    const std::array<int, 3> counts{0, 7, tokens};
    const std::array<std::array<std::int32_t, kLogicalPages>, 3> mappings{
        std::array<std::int32_t, kLogicalPages>{0, 1, 2},
        std::array<std::int32_t, kLogicalPages>{2, 3, 4},
        std::array<std::int32_t, kLogicalPages>{5, 1, 4},
    };
    int failures = 0;
    for (std::size_t replay = 0; replay < counts.size(); ++replay) {
        const int commit_count = counts[replay];
        const std::vector<std::int32_t> mapping(mappings[replay].begin(), mappings[replay].end());
        cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
        cache_v.copy_from_host(initial_v.data(), cache_v.bytes());
        d_count.copy_from_host(&commit_count, sizeof(commit_count));
        d_table.copy_from_host(mapping.data(), mapping.size() * sizeof(std::int32_t));
        cuda_check(cudaGraphLaunch(executable, stream), "launch paged kv append graph");
        cuda_synchronize(stream);

        auto expected_k = initial_k;
        auto expected_v = initial_v;
        append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, false,
                      mapping);
        const std::string label =
            "kv_cache_append_prefix paged graph C=" + std::to_string(commit_count);
        failures +=
            verify_exact((label + " cache k").c_str(),
                         from_device<std::uint16_t>(cache_k.data(), cache_count), expected_k);
        failures +=
            verify_exact((label + " cache v").c_str(),
                         from_device<std::uint16_t>(cache_v.data(), cache_count), expected_v);
        failures += verify_exact((label + " block table unchanged").c_str(),
                                 from_device<std::int32_t>(d_table, mapping.size()), mapping);
    }

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    failures += cache_k.verify_guards("paged kv append graph cache k guards");
    failures += cache_v.verify_guards("paged kv append graph cache v guards");
    return failures;
}

int batch_selector_case(bool cyclic, int cyclic_capacity = kDFlashWindow) {
    constexpr int tokens = 3;
    constexpr int batch  = 2;
    const std::vector<std::int32_t> counts{1, 3};
    const std::vector<std::int32_t> selectors{1, 0};
    const std::vector<std::int32_t> positions =
        cyclic ? std::vector<std::int32_t>{cyclic_capacity - 1,
                                           cyclic_capacity,
                                           cyclic_capacity + 1,
                                           5,
                                           6,
                                           7}
               : std::vector<std::int32_t>{63, 64, 65, 5, 6, 7};
    const std::size_t row_input_count  = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t lane_cache_count = static_cast<std::size_t>(kHeadDim) * kKVHeads *
                                         (cyclic ? cyclic_capacity : kPage * kPhysicalPages);
    const auto host_k    = patterned_bits(row_input_count * batch, 0x31415926u);
    const auto host_v    = finite_patterned_bf16_bits(row_input_count * batch, 0x27182818u);
    const auto initial_k = patterned_bits(lane_cache_count * (cyclic ? batch : 1), 0x16180339u);
    const auto initial_v = patterned_bits(lane_cache_count * (cyclic ? batch : 1), 0x57721566u);
    const std::vector<std::int32_t> tables{0, 1, 2, 3, 4, 5};
    auto expected_k = initial_k;
    auto expected_v = initial_v;

    for (int b = 0; b < batch; ++b) {
        const std::vector<std::int32_t> mapping(
            tables.begin() +
                static_cast<std::ptrdiff_t>(selectors[static_cast<std::size_t>(b)] * kLogicalPages),
            tables.begin() + static_cast<std::ptrdiff_t>(
                                 (selectors[static_cast<std::size_t>(b)] + 1) * kLogicalPages));
        for (int token = 0; token < counts[static_cast<std::size_t>(b)]; ++token) {
            const int position = positions[static_cast<std::size_t>(b * tokens + token)];
            for (int head = 0; head < kKVHeads; ++head) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const std::size_t src =
                        static_cast<std::size_t>(b) * row_input_count + input_index(d, head, token);
                    const std::size_t dst =
                        cyclic ? static_cast<std::size_t>(selectors[static_cast<std::size_t>(b)]) *
                                         lane_cache_count +
                                     cyclic_cache_index(d, head, position % cyclic_capacity,
                                                        cyclic_capacity)
                               : paged_cache_index(d, head, position, mapping);
                    expected_k[dst] = host_k[src];
                    expected_v[dst] = bf16_bits_to_f16_bits(host_v[src]);
                }
            }
        }
    }

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_counts    = to_device(counts);
    DeviceBuffer d_selectors = to_device(selectors);
    DeviceBuffer d_tables    = cyclic
                                   ? DeviceBuffer(ninfer::test::cuda_environment().context.device(), 1)
                                   : to_device(tables);
    GuardedDeviceBuffer cache_k(initial_k.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(initial_v.size() * sizeof(std::uint16_t));
    cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
    cache_v.copy_from_host(initial_v.data(), cache_v.bytes());

    Tensor k(d_k.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, batch});
    Tensor v(d_v.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, batch});
    Tensor position_tensor(d_positions.span(), DType::I32, {tokens, batch});
    Tensor count_tensor(d_counts.span(), DType::I32, {batch});
    Tensor selector_tensor(d_selectors.span(), DType::I32, {batch});
    constexpr ops::KVCacheAppendPrefixExecutionEnvelope envelope{0, tokens};
    if (cyclic) {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    cyclic_view(cache_k, cache_v, cyclic_capacity, batch),
                                    ninfer::test::cuda_environment().default_execution);
    } else {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    paged_view(cache_k, cache_v, d_tables, batch),
                                    ninfer::test::cuda_environment().default_execution);
    }
    cuda_synchronize();

    const std::string label =
        std::string("kv_cache_append_prefix B=2 ") +
        (cyclic ? "cyclic lanes capacity=" + std::to_string(cyclic_capacity) : "paged rows");
    int failures =
        verify_exact((label + " k").c_str(),
                     from_device<std::uint16_t>(cache_k.data(), expected_k.size()), expected_k);
    failures +=
        verify_exact((label + " v").c_str(),
                     from_device<std::uint16_t>(cache_v.data(), expected_v.size()), expected_v);
    failures += cache_k.verify_guards((label + " k guards").c_str());
    failures += cache_v.verify_guards((label + " v guards").c_str());
    return failures;
}

int cyclic_decode_batch_case(int capacity) {
    constexpr int tokens = 8;
    constexpr int batch  = 8;
    const std::array<std::int32_t, batch> counts{0, 1, 2, 3, 4, 5, 7, 8};
    const std::array<std::int32_t, batch> lanes{7, 0, 6, 1, 5, 2, 4, 3};
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens) * batch);
    for (int b = 0; b < batch; ++b) {
        const int first = 2 * capacity - 3 + 11 * b;
        for (int token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(b * tokens + token)] = first + token;
        }
    }

    const std::size_t row_input_count  = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t lane_cache_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * capacity;
    const auto host_k                  = patterned_bits(row_input_count * batch, 0x2468ace0u);
    const auto host_v    = finite_patterned_bf16_bits(row_input_count * batch, 0x13579bdfu);
    const auto initial_k = patterned_bits(lane_cache_count * batch, 0x10293847u);
    const auto initial_v = patterned_bits(lane_cache_count * batch, 0x56473829u);
    auto expected_k      = initial_k;
    auto expected_v      = initial_v;
    for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < counts[static_cast<std::size_t>(b)]; ++token) {
            const int position = positions[static_cast<std::size_t>(b * tokens + token)];
            for (int head = 0; head < kKVHeads; ++head) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const std::size_t src =
                        static_cast<std::size_t>(b) * row_input_count + input_index(d, head, token);
                    const std::size_t dst =
                        static_cast<std::size_t>(lanes[static_cast<std::size_t>(b)]) *
                            lane_cache_count +
                        cyclic_cache_index(d, head, position % capacity, capacity);
                    expected_k[dst] = host_k[src];
                    expected_v[dst] = bf16_bits_to_f16_bits(host_v[src]);
                }
            }
        }
    }

    const std::vector<std::int32_t> host_counts(counts.begin(), counts.end());
    const std::vector<std::int32_t> host_lanes(lanes.begin(), lanes.end());
    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_counts    = to_device(host_counts);
    DeviceBuffer d_lanes     = to_device(host_lanes);
    GuardedDeviceBuffer cache_k(initial_k.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(initial_v.size() * sizeof(std::uint16_t));
    cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
    cache_v.copy_from_host(initial_v.data(), cache_v.bytes());

    Tensor k(d_k.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, batch});
    Tensor v(d_v.span(), DType::BF16, {kHeadDim, kKVHeads, tokens, batch});
    Tensor position_tensor(d_positions.span(), DType::I32, {tokens, batch});
    Tensor count_tensor(d_counts.span(), DType::I32, {batch});
    Tensor lane_tensor(d_lanes.span(), DType::I32, {batch});
    ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, lane_tensor, {0, tokens},
                                cyclic_view(cache_k, cache_v, capacity, batch),
                                ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label =
        "kv_cache_append_prefix cyclic decode B=8 capacity=" + std::to_string(capacity);
    int failures =
        verify_exact((label + " k").c_str(),
                     from_device<std::uint16_t>(cache_k.data(), expected_k.size()), expected_k);
    failures +=
        verify_exact((label + " v").c_str(),
                     from_device<std::uint16_t>(cache_v.data(), expected_v.size()), expected_v);
    failures += cache_k.verify_guards((label + " k guards").c_str());
    failures += cache_v.verify_guards((label + " v guards").c_str());
    return failures;
}

// Cyclic prefix qualification: the independent exact oracle above owns the codec. The
// destination remains live across calls, so every comparison also checks untouched slots,
// padding and unselected lanes.
class CyclicPrefixFixture {
    static constexpr int Lanes = 8;
    static constexpr std::array<int, Lanes> LaneOrder{7, 0, 4, 2, 6, 1, 5, 3};
    int capacity_, padded_;
    std::vector<std::uint16_t> expected_k_, expected_v_, initial_k_, initial_v_;
    GuardedDeviceBuffer cache_k_, cache_v_;
    std::array<int, Lanes> frontier_{};

public:
    explicit CyclicPrefixFixture(int capacity)
        : capacity_(capacity), padded_(capacity + 8),
          expected_k_(
              patterned_bits(static_cast<std::size_t>(kHeadDim) * padded_ * kKVHeads * Lanes, 91)),
          expected_v_(patterned_bits(expected_k_.size(), 137)), cache_k_(expected_k_.size() * 2),
          cache_v_(expected_v_.size() * 2) {
        initial_k_ = expected_k_;
        initial_v_ = expected_v_;
        cache_k_.copy_from_host(expected_k_.data(), expected_k_.size() * 2);
        cache_v_.copy_from_host(expected_v_.data(), expected_v_.size() * 2);
        for (int lane = 0; lane < Lanes; ++lane) frontier_[lane] = (lane + 2) * capacity - 3;
    }

    // kind=0 empty, kind=1 mixed zero/one/short/full, kind=2 full.
    int run(int width, int batch, int kind, bool replay, int upper = -1) {
        expected_k_ = initial_k_;
        expected_v_ = initial_v_;
        cache_k_.copy_from_host(expected_k_.data(), expected_k_.size() * 2);
        cache_v_.copy_from_host(expected_v_.data(), expected_v_.size() * 2);
        for (int lane = 0; lane < Lanes; ++lane) frontier_[lane] = (lane + 2) * capacity_ - 3;
        if (upper < 0) upper = width;
        const auto elements = static_cast<std::size_t>(kHeadDim) * kKVHeads * width * batch;
        std::vector<std::uint16_t> input_k(elements), input_v(elements);
        std::vector<int> positions(width * batch), counts(batch), lanes(batch);
        GuardedDeviceBuffer dk(elements * 2), dv(elements * 2), dp(positions.size() * 4),
            dc(batch * 4), dl(batch * 4);
        Tensor k(dk.span(), DType::BF16, {kHeadDim, kKVHeads, width, batch}),
            v(dv.span(), DType::BF16, {kHeadDim, kKVHeads, width, batch}),
            p(dp.span(), DType::I32, {width, batch}), c(dc.span(), DType::I32, {batch}),
            l(dl.span(), DType::I32, {batch});
        CyclicKVCacheLayerView cache{
            .k        = Tensor(cache_k_.span(), DType::BF16, {kHeadDim, padded_, kKVHeads, Lanes}),
            .v        = Tensor(cache_v_.span(), DType::FP16, {kHeadDim, padded_, kKVHeads, Lanes}),
            .capacity = static_cast<std::uint32_t>(capacity_),
            .padded_capacity = static_cast<std::uint32_t>(padded_),
            .num_kv_heads    = kKVHeads,
            .head_dim        = kHeadDim,
            .slot_count      = Lanes};
        SlottedStateFixture key_state(cache.k, 3), value_state(cache.v, 3);
        cache.k = key_state.tensor;
        cache.v = value_state.tensor;
        const ops::KVCacheAppendPrefixExecutionEnvelope envelope{0,
                                                                 static_cast<std::uint32_t>(upper)};
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        const auto launch = [&] {
            ops::kv_cache_append_prefix(k, v, p, c, l, envelope, cache, device.execution_view());
        };
        int failures = 0;
        for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
            for (std::size_t i = 0; i < elements; ++i) {
                auto kb = static_cast<std::uint16_t>(i * 40503 + width * 17 + phase * 103);
                auto vb = static_cast<std::uint16_t>(i * 17351 + width * 31 + phase * 211);
                if ((kb & 0x7f80u) == 0x7f80u) kb ^= 0x0080u;
                if ((vb & 0x7f80u) == 0x7f80u) vb ^= 0x0080u;
                input_k[i] = kb;
                input_v[i] = vb;
            }
            for (int b = 0; b < batch; ++b) {
                lanes[b]       = LaneOrder[(b + 3 * phase) % Lanes];
                const int mode = (b + phase) % 4;
                counts[b]      = kind == 0   ? 0
                                 : kind == 2 ? upper
                                 : mode == 0 ? upper
                                 : mode == 1 ? std::max(0, upper - 1)
                                 : mode == 2 ? std::min(1, upper)
                                             : 0;
                for (int t = 0; t < width; ++t) {
                    const auto src =
                        (static_cast<std::size_t>(b) * width + t) * kKVHeads * kHeadDim;
                    positions[b * width + t] = t < counts[b] ? frontier_[lanes[b]] + t : -1234567;
                    if (t >= counts[b]) {
                        std::fill_n(input_k.begin() + src, kKVHeads * kHeadDim, 0x7fc1);
                        std::fill_n(input_v.begin() + src, kKVHeads * kHeadDim, 0x7fff);
                        continue;
                    }
                    const int slot = positions[b * width + t] % capacity_;
                    for (int h = 0; h < kKVHeads; ++h)
                        for (int d = 0; d < kHeadDim; ++d) {
                            const auto source = src + h * kHeadDim + d;
                            const auto dst =
                                (((static_cast<std::size_t>(lanes[b]) * kKVHeads + h) * padded_ +
                                  slot) *
                                 kHeadDim) +
                                d;
                            expected_k_[dst] = input_k[source];
                            expected_v_[dst] = bf16_bits_to_f16_bits(input_v[source]);
                        }
                }
                frontier_[lanes[b]] += counts[b];
            }
            dk.copy_from_host(input_k.data(), input_k.size() * 2);
            dv.copy_from_host(input_v.data(), input_v.size() * 2);
            dp.copy_from_host(positions.data(), positions.size() * 4);
            dc.copy_from_host(counts.data(), batch * 4);
            dl.copy_from_host(lanes.data(), batch * 4);
            cuda_synchronize();
            if (replay && phase == 0) {
                definition.capture(device.stream(), launch);
                graph.instantiate(definition);
            }
            if (replay)
                graph.launch(device.stream());
            else
                launch();
            cuda_synchronize(device.stream());
            const std::string label =
                "cyclic prefix capacity=" + std::to_string(capacity_) +
                " Wc=" + std::to_string(width) + " B=" + std::to_string(batch) +
                " upper=" + std::to_string(upper) + " kind=" + std::to_string(kind) +
                (replay ? " graph " : " eager ") + std::to_string(phase);
            key_state.copy_back();
            value_state.copy_back();
            failures += key_state.verify_guards("cyclic K") + value_state.verify_guards("cyclic V");
            failures += verify_exact(
                (label + " cache K").c_str(),
                from_device<std::uint16_t>(cache_k_.data(), expected_k_.size()), expected_k_);
            failures += verify_exact(
                (label + " cache V").c_str(),
                from_device<std::uint16_t>(cache_v_.data(), expected_v_.size()), expected_v_);
            failures += verify_exact((label + " input K unchanged").c_str(),
                                     from_device<std::uint16_t>(dk.data(), elements), input_k);
            failures += verify_exact((label + " input V unchanged").c_str(),
                                     from_device<std::uint16_t>(dv.data(), elements), input_v);
            failures += verify_exact((label + " positions unchanged").c_str(),
                                     from_device<int>(dp.data(), positions.size()), positions);
            failures += verify_exact((label + " counts unchanged").c_str(),
                                     from_device<int>(dc.data(), batch), counts);
            failures += verify_exact((label + " lanes unchanged").c_str(),
                                     from_device<int>(dl.data(), batch), lanes);
            failures += cache_k_.verify_guards(label) + cache_v_.verify_guards(label) +
                        dk.verify_guards(label) + dv.verify_guards(label) +
                        dp.verify_guards(label) + dc.verify_guards(label) + dl.verify_guards(label);
        }
        return failures;
    }
};

} // namespace

void add_prefix_append_cases(ninfer::test::OpSuite& suite) {
    const auto add = [&](int tokens, int count, int position, bool cyclic,
                         std::vector<int> mapping = {}, int min_count = 0,
                         int capacity = kDFlashWindow) {
        const auto id = std::string("append/") + (cyclic ? "cyclic" : "paged") + "/t" +
                        std::to_string(tokens) + "/count" + std::to_string(count) + "/position" +
                        std::to_string(position) + "/bound" + std::to_string(min_count);
        suite.add(
            id,
            [=] { return run_case(tokens, count, position, cyclic, mapping, min_count, capacity); },
            {{"format", "bf16"}, {"execution", "eager"}});
    };
    add(1, 0, 0, false, {0, 1, 2});
    add(1, 1, 63, false, {2, 3, 4});
    add(16, 7, 60, false, {5, 1, 4}, 5);
    add(16, 16, 120, false, {2, 5, 0});
    add(1, 0, kDFlashWindow - 1, true);
    add(1, 1, 2 * kDFlashWindow - 1, true);
    add(16, 7, 2 * kDFlashWindow - 2, true);
    add(16, 16, 3 * kDFlashWindow - 8, true, {}, 16);
    add(1, 0, kDFlash2Window - 1, true, {}, 0, kDFlash2Window);
    add(1, 1, 2 * kDFlash2Window - 1, true, {}, 0, kDFlash2Window);
    add(8, 7, 2 * kDFlash2Window - 2, true, {}, 0, kDFlash2Window);
    add(8, 8, 3 * kDFlash2Window - 4, true, {}, 8, kDFlash2Window);
    add(kDFlash2Window, kDFlash2Window, 3 * kDFlash2Window - kDFlash2Window / 2, true, {},
        kDFlash2Window, kDFlash2Window);
    suite.add("append/cyclic/graph/capacity4096/t16",
              [] { return cyclic_graph_replay_case(kDFlashWindow, 16); },
              {{"format", "bf16"}, {"execution", "graph"}});
    suite.add("append/cyclic/graph/capacity2048/t8",
              [] { return cyclic_graph_replay_case(kDFlash2Window, 8); },
              {{"format", "bf16"}, {"execution", "graph"}});
    suite.add("append/paged/graph/remap", [] { return paged_graph_replay_case(); },
              {{"format", "bf16"}, {"execution", "graph"}});
    suite.add("append/cyclic/selected_slots/capacity4096",
              [] { return batch_selector_case(true, kDFlashWindow); }, {{"format", "bf16"}});
    suite.add("append/paged/table_rows", [] { return batch_selector_case(false); },
              {{"format", "bf16"}});
    suite.add("append/cyclic/decode/capacity4096",
              [] { return cyclic_decode_batch_case(kDFlashWindow); }, {{"format", "bf16"}});
    suite.add("append/cyclic/decode/capacity2048",
              [] { return cyclic_decode_batch_case(kDFlash2Window); }, {{"format", "bf16"}});

    struct PrefixCase {
        int width, batch, kind;
        bool graph;
        int upper = -1;
    };

    for (int capacity : {2048, 4096}) {
        std::vector<PrefixCase> cases;
        const auto append = [&](int w, int b, int kind, bool graph, int upper = -1) {
            cases.push_back({w, b, kind, graph, upper});
        };
        if (capacity == 2048) {
            for (int b = 1; b <= 8; ++b)
                for (int w : {1, 2, 3, 7, 8, 9, 15, 16}) append(w, b, 1, b == 1 || b == 8);
            for (int w : {1, 3, 8, 9, 16})
                for (int b : {1, 8}) {
                    append(w, b, 0, true);
                    append(w, b, 0, true, 0);
                    append(w, b, 2, true);
                }
            append(16, 8, 1, true, 7);
            for (int w : {17, 127, 128, 129, 2048}) append(w, 1, 2, true);
            append(65, 1, 1, true);
            append(17, 8, 1, true);
            append(4097, 1, 1, true, 16);
        } else {
            for (int w : {1, 3, 8, 9, 16, 17})
                for (int b : {1, 8}) append(w, b, 1, true);
            append(4096, 1, 2, true);
        }
        std::vector<OpCase> catalogue;
        for (auto c : cases)
            catalogue.push_back({"append/prefix/capacity" + std::to_string(capacity) + "/w" +
                                     std::to_string(c.width) + "/b" + std::to_string(c.batch) +
                                     "/kind" + std::to_string(c.kind) + "/bound" +
                                     std::to_string(c.upper),
                                 {{"format", "bf16"}, {"execution", c.graph ? "graph" : "eager"}}});
        suite.add(std::move(catalogue), [=](auto selected) {
            CyclicPrefixFixture fixture(capacity);
            int failures = 0;
            for (auto i : selected) {
                const auto c = cases[i];
                failures += fixture.run(c.width, c.batch, c.kind, c.graph, c.upper);
            }
            return failures;
        });
    }
}
