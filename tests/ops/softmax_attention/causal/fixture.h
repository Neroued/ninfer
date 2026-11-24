#pragma once

#include "ops/float_codec.h"
#include "core/arena.h"
#include "cuda_fixture.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/suite.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::test::causal_attention {
using namespace float_codec;

constexpr std::int32_t kHeadDim          = 256;
constexpr std::int32_t kQuantGroup       = 64;
constexpr std::int32_t kQuantGroups      = kHeadDim / kQuantGroup;
constexpr std::int32_t kFp8QuantGroup    = kHeadDim;
constexpr std::int32_t kFp8QuantGroups   = 2;
constexpr std::int32_t kNvfp4QuantGroup  = 16;
constexpr std::int32_t kNvfp4QuantGroups = kHeadDim / kNvfp4QuantGroup;
constexpr std::int32_t kNvfp4CodeBytes   = kHeadDim / 2;
constexpr float kAttentionScale          = 0.0625f;
constexpr std::uint16_t kOutputCanary    = 0x7fc1u;

// Append and cached entries use one fixed criterion for each registered storage profile; token
// count, geometry, execution envelope, and private launch route do not select or relax it. Native
// INT8-G64 / FP8 Q quantization is an accepted production approximation. The unchanged FP64 oracle
// includes its error. Small, unit-RMS and RMS1.8 Q/K fixtures qualify that budget. These are
// conformance-domain budgets, not bounds for arbitrary BF16 inputs or KV quality scores.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 2.0e-2,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ 4.0e-2,
};

constexpr ReductionCriterion kAttentionFp8Criterion{
    /*relative_l2*/ 8.0e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-1,
};

constexpr ReductionCriterion kAttentionNvfp4Criterion{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-2,
};

constexpr ReductionCriterion kAttentionK8V4Criterion{
    /*relative_l2*/ 8.0e-2,
    /*gross_absolute*/ 5.0e-3,
    /*gross_relative_to_max_reference*/ 1.1e-1,
};

struct TestVectorLayout {
    DType code_dtype;
    std::int32_t code_extent;
    DType scale_dtype;
    std::int32_t scale_extent;
};

struct TestCacheLayout {
    TestVectorLayout key;
    TestVectorLayout value;
};

TestCacheLayout test_cache_layout(KvCacheStorage storage);

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

constexpr Geometry kGeometries[] = {
    {"d256-h24-kv4", 24, 4},
    {"d256-h16-kv2", 16, 2},
};

ops::AttentionHeadGeometry op_geometry(const Geometry& geometry);

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
    bool zero_q        = false;
    bool graph_replay  = false;
    float qk_amplitude = 0.25f;
};

enum class MappingPattern { Identity, Offset, Fragmented };

const char* mapping_name(MappingPattern pattern);

std::int32_t align_up_page(std::int32_t value);

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern);

std::vector<std::int32_t> make_block_table(std::int32_t populated_pages, std::int32_t table_pages,
                                           MappingPattern pattern);

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token);

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token);

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d);

std::size_t logical_plane_index(std::int32_t leading_extent, const Geometry& geometry,
                                std::int32_t padded_context, std::int32_t head,
                                std::int32_t position, std::int32_t leading);

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group);

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context);

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context);

std::size_t fp8_scale_index(const Geometry& geometry, std::int32_t padded_context,
                            std::int32_t head, std::int32_t position);

std::size_t fp8_scale_elements(const Geometry& geometry, std::int32_t padded_context);

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading);

std::size_t physical_plane_elements(std::int32_t leading_extent, const Geometry& geometry,
                                    std::int32_t physical_pages);

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity,
                             std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages) {
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(physical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

template <typename T>
void scatter_paged_into(const std::vector<T>& logical, std::int32_t leading_extent,
                        const Geometry& geometry, std::int32_t logical_capacity,
                        std::span<const std::int32_t> block_table, std::vector<T>& physical) {
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
}

template <typename T>
std::vector<T> gather_paged(std::span<const T> physical, std::int32_t leading_extent,
                            const Geometry& geometry, std::int32_t logical_capacity,
                            std::span<const std::int32_t> block_table) {
    std::vector<T> logical(static_cast<std::size_t>(leading_extent) * logical_capacity *
                           static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t target = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                logical[target] =
                    physical[paged_index(leading_extent, geometry, page, head, position, leading)];
            }
        }
    }
    return logical;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi);

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values);

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits);



float round_to_f16(float value);

std::vector<std::uint16_t> to_f16_bits(const std::vector<float>& values);

struct HostCache {
    Geometry geometry;
    KvCacheStorage storage;
    std::int32_t max_context;
    std::int32_t logical_capacity; // Page-rounded populated prefix, independent of table capacity.
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_fp16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint8_t> k_fp8;
    std::vector<std::uint8_t> v_fp8;
    std::vector<std::uint8_t> k_fp8_scale, v_fp8_scale;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
    std::vector<std::uint8_t> k_nvfp4;
    std::vector<std::uint8_t> v_nvfp4;
    std::vector<std::uint8_t> k_nvfp4_scale;
    std::vector<std::uint8_t> v_nvfp4_scale;
};

void encode_group(std::span<const float> source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset);

void normalized_hadamard_d256(std::array<float, kHeadDim>& values);

void normalized_hadamard_d256(std::array<double, kHeadDim>& values);

void encode_nvfp4_rotated_row(std::span<const float> source, std::size_t source_base,
                              std::vector<std::uint8_t>& codes, std::size_t code_base,
                              std::vector<std::uint8_t>& scales, std::size_t scale_base);

void encode_fp8_row(const std::array<float, kHeadDim>& values, std::vector<std::uint8_t>& codes,
                    std::size_t code_base, std::vector<std::uint8_t>& scales,
                    std::size_t scale_offset);

void encode_fp8_rotated_row(std::span<const float> source, std::size_t source_base,
                            std::vector<std::uint8_t>& codes, std::size_t code_base,
                            std::vector<std::uint8_t>& scales, std::size_t scale_offset);

void encode_rotated_key_row(std::span<const float> source, std::size_t source_base,
                            std::vector<std::int8_t>& codes, std::size_t code_base,
                            std::vector<std::uint16_t>& scales, std::size_t scale_base);

HostCache make_cache(const Geometry& geometry, KvCacheStorage storage, std::int32_t max_context,
                     std::uint32_t seed, float qk_amplitude = 0.25f);

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions);

double cache_value(const HostCache& cache, bool key, int head, int position, int d);

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions);

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

class DeviceCache {
public:
    // The table reserves the execution envelope; only the populated prefix owns physical pages.
    // Future entries remain invalid, just as in a growing paged cache.
    DeviceCache(const HostCache& cache, MappingPattern mapping, std::int32_t table_capacity);

    PagedKVLayerView view();

    PagedKVBatchLayerView batch_view();

    HostCache snapshot() const;

    int verify_guards(const std::string& label) const;

private:
    Geometry geometry_;
    KvCacheStorage storage_;
    TestCacheLayout layout_;
    std::int32_t max_context_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::size_t k_code_elements_;
    std::size_t v_code_elements_;
    std::size_t k_scale_elements_;
    std::size_t v_scale_elements_;
    tool::PackedKVFixture packed_;
    GuardedDeviceBuffer block_table_;
};

class BatchDeviceCache {
public:
    BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping,
                     std::int32_t table_capacity);

    PagedKVBatchLayerView view();

    using Bytes = std::array<std::vector<std::uint8_t>, 4>;

    Bytes snapshot_bytes() const;

    void copy_from(const BatchDeviceCache& source);

    PagedKVLayerView single_view(int row);

    int verify_untouched(const std::string& label, const Bytes& before,
                         std::span<const int> positions, std::span<const int> lanes,
                         std::span<const int> valid, int width) const;

    int verify(const std::string& label, std::span<const HostCache> expected) const;

private:
    [[nodiscard]] std::span<const std::int32_t> row_table(std::size_t row) const;

    void scatter_bf16_rows(std::span<const HostCache> rows, std::vector<std::uint16_t>& k,
                           std::vector<std::uint16_t>& v) const;

    void upload_rows(std::span<const HostCache> rows);

    Geometry geometry_;
    KvCacheStorage storage_;
    TestCacheLayout layout_;
    std::size_t rows_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_tables_host_;
    std::size_t k_code_elements_;
    std::size_t v_code_elements_;
    std::size_t k_scale_elements_;
    std::size_t v_scale_elements_;
    tool::PackedKVFixture packed_;
    GuardedDeviceBuffer block_tables_;
};

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected);

int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected);

int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected);

const char* cache_name(KvCacheStorage storage);

ReductionCriterion attention_criterion(KvCacheStorage storage);

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion);

std::string case_label(const char* entry, const Geometry& geometry, KvCacheStorage storage,
                       const AttentionCase& test_case, MappingPattern mapping);

void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v);

struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
    bool graph_replay = false;
    // Optional per-replay contexts exercise large live-length changes with stable device views.
    std::vector<std::vector<std::int32_t>> replay_contexts;
};

int run_append_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                    const AttentionCase& test_case, MappingPattern mapping,
                    std::span<const int> oracle_queries         = {},
                    std::span<const std::uint32_t> graph_limits = {});
int run_cached_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                    const AttentionCase& test_case, MappingPattern mapping,
                    std::span<const int> oracle_queries = {});
int run_batch_case(CudaExecutionView execution, const Geometry& geometry, KvCacheStorage storage,
                   const BatchAttentionCase& test_case, int envelope_max = 0,
                   std::span<const std::uint32_t> graph_limits = {});
int verify_workspace_capacity_contract(CudaExecutionView, KvCacheStorage);
void add_cases(OpSuite&);

} // namespace ninfer::test::causal_attention
