#include "ops/suite.h"
#include "ninfer/ops/softmax_attention.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "../../../tools/support/paged_kv_fixture.h"

#include "core/arena.h"
#include "ops/graph_patch_check.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using ops::ContextBucket;
using ops::ContextBucketRange;

namespace {

constexpr int kD       = 128;
constexpr int kQHeads  = 32;
constexpr int kKVHeads = 8;
constexpr int kPage    = 64;
constexpr float kScale = 0.08838834764831844055f;
constexpr ops::AttentionHeadGeometry kGeometry{kD, kQHeads, kKVHeads};
constexpr ContextBucketRange kBuckets{ContextBucket::K2, ContextBucket::K256};

constexpr ReductionCriterion kContextQueryBf16Criterion{
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

enum class MappingPattern {
    Identity,
    Offset,
    Fragmented,
};

std::vector<std::int32_t> page_mapping(int logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
    for (int page = 0; page < logical_pages; ++page) {
        switch (pattern) {
        case MappingPattern::Identity:
            mapping[static_cast<std::size_t>(page)] = page;
            break;
        case MappingPattern::Offset:
            mapping[static_cast<std::size_t>(page)] = logical_pages + page;
            break;
        case MappingPattern::Fragmented:
            mapping[static_cast<std::size_t>(page)] = 2 * page + 1;
            break;
        }
    }
    return mapping;
}

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::size_t physical_context_index(int d, int kv_head, int position, int physical_page,
                                   int physical_pages) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(position % kPage) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(physical_page) +
                     static_cast<std::size_t>(physical_pages) * static_cast<std::size_t>(kv_head)));
}

void scatter_context_mapping(std::vector<float>& physical, const std::vector<float>& logical,
                             int logical_stride, int storage_capacity,
                             const std::vector<std::int32_t>& mapping, int physical_pages) {
    for (int position = 0; position < storage_capacity; ++position) {
        const int physical_page = mapping[static_cast<std::size_t>(position / kPage)];
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) {
                physical[physical_context_index(d, kv_head, position, physical_page,
                                                physical_pages)] =
                    logical[context_index(d, kv_head, position, logical_stride)];
            }
        }
    }
}

std::vector<float> scatter_context(const std::vector<float>& logical, int logical_stride,
                                   int storage_capacity, int physical_pages,
                                   const std::vector<std::int32_t>& mapping, std::uint32_t seed) {
    std::vector<float> physical(static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages);
    fill_uniform(physical, seed, -0.9f, 0.9f);
    round_to_bf16(physical);
    scatter_context_mapping(physical, logical, logical_stride, storage_capacity, mapping,
                            physical_pages);
    return physical;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<std::uint16_t> fp16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const __half value = __float2half_rn(values[i]);
        std::memcpy(&bits[i], &value, sizeof(value));
    }
    return bits;
}

void context_attention_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                              const std::vector<float>& query_v,
                              const std::vector<float>& context_k,
                              const std::vector<float>& context_v, int tokens, int context_length,
                              int valid_columns, int padded_context, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);
    const int key_count = context_length + valid_columns;
    naive_dense_softmax_attention(
        kGeometry, valid_columns, key_count, static_cast<double>(kScale),
        [&](int d, int head, int token) { return static_cast<double>(q[q_index(d, head, token)]); },
        [&](int d, int head, int key) {
            return key < context_length
                       ? static_cast<double>(context_k[context_index(d, head, key, padded_context)])
                       : static_cast<double>(
                             query_k[query_kv_index(d, head, key - context_length)]);
        },
        [&](int d, int head, int key) {
            return key < context_length
                       ? static_cast<double>(context_v[context_index(d, head, key, padded_context)])
                       : static_cast<double>(
                             query_v[query_kv_index(d, head, key - context_length)]);
        },
        [](int, int) { return true; },
        [&](int d, int head, int token, double value) { out[q_index(d, head, token)] = value; });
}

KVPageGeometry context_geometry() {
    return {kPage, {{DType::BF16, kD, kKVHeads}, {DType::FP16, kD, kKVHeads}}};
}

void upload_context(tool::PackedKVFixture& packed, std::size_t plane,
                    const std::vector<std::uint16_t>& values, int physical_pages) {
    Tensor input({const_cast<std::uint16_t*>(values.data()), values.size() * 2, Device::cpu()},
                 plane == 0 ? DType::BF16 : DType::FP16, {kD, kPage, physical_pages, kKVHeads});
    packed.upload_plane(plane, input, true);
}

std::vector<std::uint16_t> download_context(const tool::PackedKVFixture& packed, std::size_t plane,
                                            int physical_pages) {
    std::vector<std::uint16_t> values(static_cast<std::size_t>(kD) * kPage * physical_pages *
                                      kKVHeads);
    Tensor output({values.data(), values.size() * 2, Device::cpu()},
                  plane == 0 ? DType::BF16 : DType::FP16, {kD, kPage, physical_pages, kKVHeads});
    packed.download_plane(plane, output, true);
    return values;
}

PagedKVBatchLayerView make_context_view(tool::PackedKVFixture& packed, DeviceBuffer& offsets,
                                        int logical_pages, int table_rows = 1) {
    Tensor table(offsets.span(), DType::I64, {logical_pages, table_rows});
    return {.k_pages      = packed.plane(0, table),
            .v_pages      = packed.plane(1, table),
            .head_dim     = kD,
            .num_kv_heads = kKVHeads,
            .storage      = KvCacheStorage::BFloat16};
}

enum class InputProfile {
    Random,
    QueryVisibility,
};

int run_case(int tokens, int context_length, ContextBucket bucket, InputProfile profile,
             MappingPattern mapping_pattern, int valid_columns, int table_pages) {
    if (valid_columns < 0) valid_columns = tokens;
    const int populated_pages        = (context_length + kPage - 1) / kPage;
    const int logical_pages          = std::max({populated_pages, table_pages, 1});
    const int logical_stride         = std::max(populated_pages, 1) * kPage;
    const int physical_pages         = 2 * populated_pages + 1;
    const std::size_t q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_kv_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t logical_context_count =
        static_cast<std::size_t>(kD) * logical_stride * kKVHeads;
    std::vector<float> q(q_count);
    std::vector<float> query_k(query_kv_count);
    std::vector<float> query_v(query_kv_count);
    std::vector<float> context_k(logical_context_count);
    std::vector<float> context_v(logical_context_count);

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
    context_attention_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length,
                             valid_columns, logical_stride, reference);

    auto mapping = page_mapping(populated_pages, mapping_pattern);
    mapping.resize(logical_pages, -1);
    const int scattered_rows            = populated_pages * kPage;
    const std::vector<float> physical_k = scatter_context(context_k, logical_stride, scattered_rows,
                                                          physical_pages, mapping, 6007u + seed);
    const std::vector<float> physical_v = scatter_context(context_v, logical_stride, scattered_rows,
                                                          physical_pages, mapping, 7001u + seed);

    const auto q_expected         = bf16_bits(q);
    const auto query_k_expected   = bf16_bits(query_k);
    const auto query_v_expected   = bf16_bits(query_v);
    const auto context_k_expected = bf16_bits(physical_k);
    const auto context_v_expected = fp16_bits(physical_v);
    const std::vector<int> length_expected{context_length};
    const std::vector<int> valid_expected{valid_columns};
    const std::vector<int> table_row_expected{0};

    DeviceBuffer d_q       = to_device(q_expected);
    DeviceBuffer d_query_k = to_device(query_k_expected);
    DeviceBuffer d_query_v = to_device(query_v_expected);
    tool::PackedKVFixture packed(cuda_environment().context.device(), context_geometry(),
                                 physical_pages);
    upload_context(packed, 0, context_k_expected, physical_pages);
    upload_context(packed, 1, context_v_expected, physical_pages);
    const auto offsets       = packed.table_offsets(mapping);
    DeviceBuffer d_table     = to_device(offsets);
    DeviceBuffer d_length    = to_device_i32(length_expected);
    DeviceBuffer d_valid     = to_device_i32(valid_expected);
    DeviceBuffer d_table_row = to_device_i32(table_row_expected);
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.span(), DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.span(), DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.span(), DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor length_tensor(d_length.span(), DType::I32, {1});
    Tensor valid_tensor(d_valid.span(), DType::I32, {1});
    Tensor table_row_tensor(d_table_row.span(), DType::I32, {1});
    Tensor out_tensor(d_out.span(), DType::BF16, {kD, kQHeads, tokens, 1});
    PagedKVBatchLayerView context     = make_context_view(packed, d_table, logical_pages);
    const std::size_t workspace_bytes = ops::context_softmax_attention_workspace_capacity_bytes(
        kGeometry, bucket, 1, tokens, tokens);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(workspace_bytes, 1));
    MemoryArena workspace(workspace_storage.span());

    ops::context_softmax_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                   valid_tensor, table_row_tensor, kGeometry, kScale, context,
                                   bucket, workspace, out_tensor,
                                   ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    std::string label = "context_softmax_attention T=" + std::to_string(tokens) +
                        " V=" + std::to_string(valid_columns) +
                        " L=" + std::to_string(context_length) +
                        " mapping=" + mapping_name(mapping_pattern) +
                        " bucket=" + std::string(ops::context_bucket_name(bucket)) +
                        " table_pages=" + std::to_string(logical_pages);
    if (profile == InputProfile::QueryVisibility) label += " query-visibility";

    int failures =
        valid_columns == 0
            ? verify_exact(label.c_str(), from_device<std::uint16_t>(d_out.data(), q_count),
                           std::vector<std::uint16_t>(q_count, 0))
            : verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count), reference,
                               kContextQueryBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(d_q, q_count), q_expected);
    failures +=
        verify_exact((label + " query k unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_k, query_kv_count), query_k_expected);
    failures +=
        verify_exact((label + " query v unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_v, query_kv_count), query_v_expected);
    failures += verify_exact((label + " context k unchanged").c_str(),
                             download_context(packed, 0, physical_pages), context_k_expected);
    failures += verify_exact((label + " context v unchanged").c_str(),
                             download_context(packed, 1, physical_pages), context_v_expected);
    failures += verify_exact((label + " block table unchanged").c_str(),
                             from_device<std::int64_t>(d_table, mapping.size()),
                             packed.table_offsets(mapping));
    failures += verify_exact((label + " context length unchanged").c_str(),
                             from_device<int>(d_length, 1), length_expected);
    failures += verify_exact((label + " valid columns unchanged").c_str(),
                             from_device<int>(d_valid, 1), valid_expected);
    failures += verify_exact((label + " table row unchanged").c_str(),
                             from_device<int>(d_table_row, 1), table_row_expected);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int graph_mapping_replay_case() {
    constexpr int tokens            = 1;
    constexpr int context_length    = 65;
    constexpr int logical_pages     = 2;
    constexpr int logical_capacity  = logical_pages * kPage;
    constexpr int physical_pages    = 6;
    constexpr int logical_stride    = 128;
    const std::size_t q_count       = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_count   = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t context_count = static_cast<std::size_t>(kD) * logical_stride * kKVHeads;
    const std::size_t physical_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(q_count);
    std::vector<float> query_k(query_count);
    std::vector<float> query_v(query_count);
    std::vector<float> context_k(context_count);
    std::vector<float> context_v(context_count);
    fill_uniform(q, 11003u, -0.35f, 0.35f);
    fill_uniform(query_k, 12007u, -0.4f, 0.4f);
    fill_uniform(query_v, 13001u, -0.8f, 0.8f);
    fill_uniform(context_k, 14009u, -0.4f, 0.4f);
    fill_uniform(context_v, 15013u, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    const std::array<std::array<std::int32_t, logical_pages>, 3> replay_mappings{
        std::array<std::int32_t, logical_pages>{0, 1},
        std::array<std::int32_t, logical_pages>{2, 3},
        std::array<std::int32_t, logical_pages>{5, 4},
    };
    std::vector<float> physical_k(physical_count);
    std::vector<float> physical_v(physical_count);
    for (const auto& replay_mapping : replay_mappings) {
        const std::vector<std::int32_t> mapping(replay_mapping.begin(), replay_mapping.end());
        scatter_context_mapping(physical_k, context_k, logical_stride, logical_capacity, mapping,
                                physical_pages);
        scatter_context_mapping(physical_v, context_v, logical_stride, logical_capacity, mapping,
                                physical_pages);
    }

    std::vector<double> reference;
    context_attention_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length,
                             tokens, logical_stride, reference);

    DeviceBuffer d_q       = to_device(bf16_bits(q));
    DeviceBuffer d_query_k = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v = to_device(bf16_bits(query_v));
    tool::PackedKVFixture packed(cuda_environment().context.device(), context_geometry(),
                                 physical_pages);
    upload_context(packed, 0, bf16_bits(physical_k), physical_pages);
    upload_context(packed, 1, fp16_bits(physical_v), physical_pages);
    DeviceBuffer d_table     = to_device(packed.table_offsets(replay_mappings[0]));
    DeviceBuffer d_length    = to_device_i32({context_length});
    DeviceBuffer d_valid     = to_device_i32({tokens});
    DeviceBuffer d_table_row = to_device_i32({0});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    Tensor q_tensor(d_q.span(), DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.span(), DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.span(), DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor length_tensor(d_length.span(), DType::I32, {1});
    Tensor valid_tensor(d_valid.span(), DType::I32, {1});
    Tensor table_row_tensor(d_table_row.span(), DType::I32, {1});
    Tensor out_tensor(d_out.span(), DType::BF16, {kD, kQHeads, tokens, 1});
    PagedKVBatchLayerView context = make_context_view(packed, d_table, logical_pages);
    constexpr auto bucket         = ContextBucket::K2;
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           ops::context_softmax_attention_workspace_capacity_bytes(
                                               kGeometry, bucket, 1, tokens, tokens));
    MemoryArena workspace(workspace_storage.span());

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create context-attention stream");
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin context-attention capture");
    ops::context_softmax_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                   valid_tensor, table_row_tensor, kGeometry, kScale, context,
                                   bucket, workspace, out_tensor, stream_execution);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end context-attention capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate context-attention graph");

    int failures = 0;
    for (const auto& replay_mapping : replay_mappings) {
        const auto offsets = packed.table_offsets(replay_mapping);
        d_table.copy_from_host(offsets.data(), offsets.size() * sizeof(std::int64_t));
        cuda_check(cudaGraphLaunch(executable, stream), "launch context-attention graph");
        cuda_synchronize(stream);
        const std::vector<std::int32_t> mapping(replay_mapping.begin(), replay_mapping.end());
        const std::string label =
            "context_softmax_attention graph mapping=" + std::to_string(mapping[0]) + "," +
            std::to_string(mapping[1]);
        failures += verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count),
                                     reference, kContextQueryBf16Criterion);
        failures += verify_exact((label + " block table unchanged").c_str(),
                                 from_device<std::int64_t>(d_table, mapping.size()),
                                 packed.table_offsets(mapping));
    }

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    failures += d_out.verify_guards("context-attention graph output guards");
    return failures;
}

int batch_table_case(int batch, bool empty = false, bool direct_graph = false) {
    constexpr int tokens            = 2;
    constexpr int cache_rows        = 2;
    constexpr int logical_pages     = 2;
    constexpr int logical_capacity  = logical_pages * kPage;
    constexpr int physical_pages    = 6;
    const std::size_t row_q_count   = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t row_kv_count  = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t logical_count = static_cast<std::size_t>(kD) * logical_capacity * kKVHeads;
    const std::size_t physical_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(row_q_count * batch);
    std::vector<float> query_k(row_kv_count * batch);
    std::vector<float> query_v(row_kv_count * batch);
    std::vector<float> logical_k(logical_count * cache_rows);
    std::vector<float> logical_v(logical_count * cache_rows);
    std::vector<float> physical_k(physical_count);
    std::vector<float> physical_v(physical_count);
    fill_uniform(q, 16001u, -0.35f, 0.35f);
    fill_uniform(query_k, 17011u, -0.4f, 0.4f);
    fill_uniform(query_v, 18013u, -0.8f, 0.8f);
    fill_uniform(logical_k, 19001u, -0.4f, 0.4f);
    fill_uniform(logical_v, 20011u, -0.8f, 0.8f);
    fill_uniform(physical_k, 21001u, -0.9f, 0.9f);
    fill_uniform(physical_v, 22003u, -0.9f, 0.9f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(logical_k);
    round_to_bf16(logical_v);
    round_to_bf16(physical_k);
    round_to_bf16(physical_v);

    const std::vector<std::int32_t> block_tables{0, 2, 5, 3};
    for (int row = 0; row < cache_rows; ++row) {
        const std::vector<float> row_k(
            logical_k.begin() + static_cast<std::ptrdiff_t>(row * logical_count),
            logical_k.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_count));
        const std::vector<float> row_v(
            logical_v.begin() + static_cast<std::ptrdiff_t>(row * logical_count),
            logical_v.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_count));
        const std::vector<std::int32_t> mapping(
            block_tables.begin() + static_cast<std::ptrdiff_t>(row * logical_pages),
            block_tables.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_pages));
        scatter_context_mapping(physical_k, row_k, logical_capacity, logical_capacity, mapping,
                                physical_pages);
        scatter_context_mapping(physical_v, row_v, logical_capacity, logical_capacity, mapping,
                                physical_pages);
    }

    std::vector<std::int32_t> lengths(batch), valid(batch), table_rows(batch);
    for (int b = 0; b < batch; ++b) {
        valid[b]      = empty ? 0 : (2 - b % 3);
        lengths[b]    = valid[b] == 0 ? -1 : b % 2 == 0 ? 65 : 1;
        table_rows[b] = valid[b] == 0 ? -1 : 1 - b % 2;
    }
    DeviceBuffer d_q       = to_device(bf16_bits(q));
    DeviceBuffer d_query_k = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v = to_device(bf16_bits(query_v));
    tool::PackedKVFixture packed(cuda_environment().context.device(), context_geometry(),
                                 physical_pages);
    upload_context(packed, 0, bf16_bits(physical_k), physical_pages);
    upload_context(packed, 1, fp16_bits(physical_v), physical_pages);
    DeviceBuffer d_tables     = to_device(packed.table_offsets(block_tables));
    DeviceBuffer d_lengths    = to_device(lengths);
    DeviceBuffer d_valid      = to_device(valid);
    DeviceBuffer d_table_rows = to_device(table_rows);
    GuardedDeviceBuffer d_out(row_q_count * batch * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.span(), DType::BF16, {kD, kQHeads, tokens, batch});
    Tensor query_k_tensor(d_query_k.span(), DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor query_v_tensor(d_query_v.span(), DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor length_tensor(d_lengths.span(), DType::I32, {batch});
    Tensor valid_tensor(d_valid.span(), DType::I32, {batch});
    Tensor table_row_tensor(d_table_rows.span(), DType::I32, {batch});
    Tensor out_tensor(d_out.span(), DType::BF16, {kD, kQHeads, tokens, batch});
    auto context          = make_context_view(packed, d_tables, logical_pages, cache_rows);
    constexpr auto bucket = ContextBucket::K2;

    std::vector<double> expected(row_q_count * batch);
    for (int b = 0; b < batch; ++b) {
        if (valid[b] == 0) continue;
        const auto q_begin     = q.begin() + static_cast<std::ptrdiff_t>(b * row_q_count);
        const auto kv_begin    = query_k.begin() + static_cast<std::ptrdiff_t>(b * row_kv_count);
        const int selected_row = table_rows[static_cast<std::size_t>(b)];
        const auto context_begin =
            logical_k.begin() + static_cast<std::ptrdiff_t>(selected_row * logical_count);
        const std::vector<float> row_q(q_begin, q_begin + row_q_count);
        const std::vector<float> row_query_k(kv_begin, kv_begin + row_kv_count);
        const auto query_v_begin = query_v.begin() + static_cast<std::ptrdiff_t>(b * row_kv_count);
        const std::vector<float> row_query_v(query_v_begin, query_v_begin + row_kv_count);
        const std::vector<float> row_context_k(context_begin, context_begin + logical_count);
        const auto context_v_begin =
            logical_v.begin() + static_cast<std::ptrdiff_t>(selected_row * logical_count);
        const std::vector<float> row_context_v(context_v_begin, context_v_begin + logical_count);
        std::vector<double> row_expected;
        context_attention_oracle(row_q, row_query_k, row_query_v, row_context_k, row_context_v,
                                 tokens, lengths[static_cast<std::size_t>(b)],
                                 valid[static_cast<std::size_t>(b)], logical_capacity,
                                 row_expected);
        std::copy(row_expected.begin(), row_expected.end(),
                  expected.begin() + static_cast<std::ptrdiff_t>(b * row_q_count));
    }

    auto& device     = cuda_environment().context;
    const auto bytes = ops::context_softmax_attention_workspace_capacity_bytes(
        kGeometry, bucket, batch, tokens, tokens);
    DeviceBuffer workspace_storage(device.device(), std::max<std::size_t>(bytes, 1));
    MemoryArena workspace(workspace_storage.span());
    const auto launch = [&](CudaExecutionView cuda) {
        ops::context_softmax_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                       valid_tensor, table_row_tensor, kGeometry, kScale, context,
                                       bucket, workspace, out_tensor, cuda);
    };
    if (direct_graph) {
        const auto cuda = device.execution_view();
        CapturedKernelSequence seq(cuda, launch);
        if (bytes != 0 || seq.params().size() != 2) {
            std::cerr << "context attention direct graph requires two nodes and zero workspace\n";
            return 1;
        }
        FlatGraphPatchHarness graph(seq);
        graph.patch_and_launch(seq, cuda);
        cuda_synchronize(cuda.stream());
    } else {
        launch(cuda_environment().default_execution);
        cuda_synchronize();
    }

    int failures = verify_reduction("context_softmax_attention mixed lengths and table rows",
                                    from_device_bf16(d_out.data(), row_q_count * batch), expected,
                                    kContextQueryBf16Criterion);
    failures += d_out.verify_guards("context-attention batch output guards");
    return failures;
}

int graph_bucket_case(int tokens) {
    constexpr int batch = 4, longest = 140000, physical_pages = 2;
    constexpr int logical_pages = (longest + kPage - 1) / kPage;
    const std::size_t q_count   = std::size_t(kD) * kQHeads * tokens;
    const std::size_t kv_count  = std::size_t(kD) * kKVHeads * tokens;
    const auto component        = [](int head, int d) {
        return (head % 2 == 0 ? 1.F : -1.F) * (d % 2 == 0 ? 1.F : 0.5F);
    };
    constexpr std::array page_values{0.25F, 0.5F};
    constexpr std::array query_values{0.5F, 0.25F, -0.25F, -0.5F};
    std::vector<float> query_v(kv_count * batch);
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < tokens; ++t)
            for (int h = 0; h < kKVHeads; ++h)
                for (int d = 0; d < kD; ++d)
                    query_v[b * kv_count + query_kv_index(d, h, t)] =
                        query_values[b] * component(h, d);

    const std::size_t plane_size = std::size_t(kD) * kPage * kKVHeads * physical_pages;
    std::vector<float> page_v(plane_size);
    for (int page = 0; page < physical_pages; ++page)
        for (int t = 0; t < kPage; ++t)
            for (int h = 0; h < kKVHeads; ++h)
                for (int d = 0; d < kD; ++d)
                    page_v[physical_context_index(d, h, t, page, physical_pages)] =
                        page_values[page] * component(h, d);
    std::vector<std::int32_t> mapping(logical_pages * batch);
    for (int row = 0; row < batch; ++row)
        for (int page = 0; page < logical_pages; ++page)
            mapping[row * logical_pages + page] = (row + page) % physical_pages;

    auto& device    = cuda_environment().context;
    const auto cuda = device.execution_view();
    tool::PackedKVFixture packed(device.device(), context_geometry(), physical_pages);
    upload_context(packed, 0, std::vector<std::uint16_t>(plane_size), physical_pages);
    upload_context(packed, 1, fp16_bits(page_v), physical_pages);
    auto d_q       = to_device(std::vector<std::uint16_t>(q_count * batch));
    auto d_query_k = to_device(std::vector<std::uint16_t>(kv_count * batch));
    auto d_query_v = to_device(bf16_bits(query_v));
    auto d_table   = to_device(packed.table_offsets(mapping));
    auto d_lengths = to_device_i32({0, 0, 0, 0});
    auto d_valid   = to_device_i32({0, 0, 0, 0});
    auto d_rows    = to_device_i32({-1, -1, -1, -1});
    GuardedDeviceBuffer out(q_count * batch * sizeof(std::uint16_t));
    Tensor tq(d_q.span(), DType::BF16, {kD, kQHeads, tokens, batch});
    Tensor tk(d_query_k.span(), DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor tv(d_query_v.span(), DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor lengths(d_lengths.span(), DType::I32, {batch});
    Tensor valid(d_valid.span(), DType::I32, {batch});
    Tensor rows(d_rows.span(), DType::I32, {batch});
    Tensor to(out.span(), DType::BF16, {kD, kQHeads, tokens, batch});
    const auto context = make_context_view(packed, d_table, logical_pages, batch);
    const auto bytes   = ops::context_softmax_attention_workspace_capacity_bytes(
        kGeometry, kBuckets, batch, tokens, tokens);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena workspace(scratch.span());
    const auto launch = [&](ContextBucket bucket, CudaExecutionView execution) {
        ops::context_softmax_attention(tq, tk, tv, lengths, valid, rows, kGeometry, kScale, context,
                                       bucket, workspace, to, execution);
    };

    for (auto unsupported :
         {ContextBucket::K512, ContextBucket::M1, static_cast<ContextBucket>(255)}) {
        try {
            launch(unsupported, cuda);
            cuda_synchronize(cuda.stream());
            std::cerr << "context attention accepted an unsupported execution bucket\n";
            return 1;
        } catch (const std::invalid_argument&) {}
    }
    const auto profile = [&](ContextBucket bucket) {
        return ops::context_softmax_attention_graph_profile(kGeometry, bucket, batch, tokens);
    };
    const auto base_profile = profile(ContextBucket::K2);
    std::vector<CapturedKernelSequence> captures;
    int failures = 0;
    kBuckets.for_each([&](ContextBucket bucket) {
        captures.emplace_back(cuda, [&](auto execution) { launch(bucket, execution); });
        const bool base =
            bucket <= ContextBucket::K64 || (bucket == ContextBucket::K128 && tokens <= 8);
        if (captures.back().params().size() != 2 || (profile(bucket) == base_profile) != base ||
            (base && !same_kernel_launches(captures.front(), captures.back()))) {
            std::cerr << "context attention T=" << tokens
                      << " bucket=" << ops::context_bucket_name(bucket)
                      << ": unexpected graph profile or captured launch\n";
            ++failures;
        }
    });
    if (failures || profile(ContextBucket::K128) == profile(ContextBucket::K256))
        return failures + 1;

    struct Step {
        ContextBucket bucket;
        std::array<int, batch> lengths;
    };

    constexpr std::array steps{
        Step{ContextBucket::K2, {0, 128, 512, -1}},
        Step{ContextBucket::K8, {128, 4096, 0, -1}},
        Step{ContextBucket::K64, {0, 128, 4096, -1}},
        Step{ContextBucket::K128, {4096, 128, 0, -1}},
        Step{ContextBucket::K256, {0, 128, 4096, longest}},
        Step{ContextBucket::K2, {128, 0, 96, -1}},
    };
    FlatGraphPatchHarness patched(captures.front()), updated(captures.front());
    for (const auto& [bucket, history] : steps) {
        const std::array live{tokens - 1, 1, 2, history[3] < 0 ? 0 : 3};
        const std::array selectors{2, 0, 3, live[3] ? 1 : -1};
        d_lengths.copy_from_host(history.data(), sizeof(history));
        d_valid.copy_from_host(live.data(), sizeof(live));
        d_rows.copy_from_host(selectors.data(), sizeof(selectors));

        // With zero logits the independent softmax is the mean over all logical keys. Repeated
        // page mappings keep the 140k history compact, and both updates must round to these bits.
        std::vector<std::uint16_t> expected(q_count * batch);
        for (int b = 0; b < batch; ++b) {
            if (live[b] == 0) continue;
            double sum = live[b] * static_cast<double>(query_values[b]);
            for (int start = 0; start < history[b]; start += kPage) {
                const int page = mapping[selectors[b] * logical_pages + start / kPage];
                sum += std::min(kPage, history[b] - start) * static_cast<double>(page_values[page]);
            }
            const double mean = sum / (history[b] + live[b]);
            for (int t = 0; t < live[b]; ++t)
                for (int h = 0; h < kQHeads; ++h)
                    for (int d = 0; d < kD; ++d)
                        expected[b * q_count + q_index(d, h, t)] =
                            f32_to_bf16(static_cast<float>(mean * component(h / 4, d)));
        }
        const auto label = "context attention T=" + std::to_string(tokens) +
                           " bucket=" + std::string(ops::context_bucket_name(bucket));
        const auto& seq   = captures[static_cast<std::size_t>(bucket)];
        const auto verify = [&](const std::string& mode, auto update) {
            out.fill(0xff);
            scratch.fill(0xa7);
            update(seq, cuda);
            cuda_synchronize(cuda.stream());
            const auto check = label + " " + mode;
            return verify_exact(check.c_str(),
                                from_device<std::uint16_t>(out.data(), expected.size()), expected) +
                   out.verify_guards(check) + scratch.verify_guards(check);
        };
        failures += verify("node patch", [&](const auto& target, auto execution) {
            patched.patch_and_launch(target, execution);
        });
        failures += verify("whole-graph update", [&](const auto& target, auto execution) {
            updated.update_whole_graph_and_launch(target, execution);
        });
    }
    if (workspace.used() != 0 || workspace.peak_used() != bytes) {
        std::cerr << "context attention workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

} // namespace

void add_context_attention_cases(ninfer::test::OpSuite& suite) {
    suite.add(
        "context_attention/capacity",
        [] {
            int failures               = 0;
            const std::size_t interval = ops::context_softmax_attention_workspace_capacity_bytes(
                kGeometry, kBuckets, 1, 1, 16);
            const std::size_t witness =
                std::max(ops::context_softmax_attention_workspace_capacity_bytes(
                             kGeometry, ContextBucket::K256, 1, 8, 8),
                         ops::context_softmax_attention_workspace_capacity_bytes(
                             kGeometry, ContextBucket::K256, 1, 16, 16));
            if (interval != witness) {
                std::cerr << "context-attention interval capacity missed a token-band endpoint\n";
                ++failures;
            }
            const auto reject = [&](auto&& fn) {
                try {
                    fn();
                    std::cerr << "context-attention accepted an invalid profile or range\n";
                    ++failures;
                } catch (const std::invalid_argument&) {}
            };
            reject([] {
                (void)ops::context_softmax_attention_workspace_capacity_bytes(kGeometry, kBuckets,
                                                                              1, 9, 8);
            });
            reject([] {
                (void)ops::context_softmax_attention_workspace_capacity_bytes(
                    kGeometry, {ContextBucket::K8, ContextBucket::K2}, 1, 1, 16);
            });
            for (auto bucket :
                 {ContextBucket::K512, ContextBucket::M1, static_cast<ContextBucket>(255)}) {
                reject([&] {
                    (void)ops::context_softmax_attention_workspace_capacity_bytes(
                        kGeometry, {ContextBucket::K2, bucket}, 1, 1, 16);
                });
                reject([&] {
                    (void)ops::context_softmax_attention_graph_profile(kGeometry, bucket, 1, 8);
                });
            }
            return failures;
        },
        {{"format", "bf16"}});
    const auto add = [&](int tokens, int length, ContextBucket bucket = ContextBucket::K2,
                         MappingPattern mapping = MappingPattern::Identity, int valid = -1,
                         int table_pages = 0, InputProfile input = InputProfile::Random) {
        const auto id = "context_attention/t" + std::to_string(tokens) + "/l" +
                        std::to_string(length) + "/bucket" +
                        std::string(ops::context_bucket_name(bucket)) + "/valid" +
                        std::to_string(valid) + "/mapping" + std::to_string(int(mapping)) +
                        "/input" + std::to_string(int(input));
        suite.add(
            id,
            [=] { return run_case(tokens, length, bucket, input, mapping, valid, table_pages); },
            {{"format", "bf16"}, {"execution", "eager"}});
    };
    add(1, 0);
    add(2, 1);
    add(1, 63);
    add(2, 64, ContextBucket::K2, MappingPattern::Offset);
    add(4, 65, ContextBucket::K2, MappingPattern::Fragmented);
    add(8, 95, ContextBucket::K8, MappingPattern::Fragmented);
    add(16, 65, ContextBucket::K128, MappingPattern::Fragmented, 16, 1025);
    add(16, 257);
    add(1, 4096, ContextBucket::K8, MappingPattern::Fragmented);
    add(4, 96, ContextBucket::K2, MappingPattern::Fragmented, 4, 2);
    add(4, 0, ContextBucket::K2, MappingPattern::Identity, 4, 2, InputProfile::QueryVisibility);
    add(8, 0, ContextBucket::K2, MappingPattern::Identity, 0);
    add(8, 65, ContextBucket::K2, MappingPattern::Fragmented, 0);
    suite.add("context_attention/graph_mapping", graph_mapping_replay_case,
              {{"format", "bf16"}, {"execution", "graph"}});
    for (int tokens : {4, 16})
        suite.add("context_attention/cross_bucket/t" + std::to_string(tokens),
                  [=] { return graph_bucket_case(tokens); },
                  {{"format", "bf16"}, {"execution", "graph"}});
    for (const int batch : {2, 17})
        suite.add("context_attention/batch_table/b" + std::to_string(batch),
                  [=] { return batch_table_case(batch); },
                  {{"suite", batch > 8 ? "concurrency" : "context_attention"}, {"format", "bf16"}});
    suite.add("context_attention/batch_table/b129_empty",
              [] { return batch_table_case(129, true); },
              {{"suite", "concurrency"}, {"format", "bf16"}});
    suite.add("context_attention/batch_table/b256_direct_graph",
              [] { return batch_table_case(256, false, true); },
              {{"suite", "concurrency"}, {"format", "bf16"}, {"execution", "graph"}});
}
