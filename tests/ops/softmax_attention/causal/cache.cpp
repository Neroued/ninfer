#include "fixture.h"

namespace ninfer::test::causal_attention {

TestCacheLayout test_cache_layout(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {{DType::BF16, kHeadDim, DType::U8, 0}, {DType::FP16, kHeadDim, DType::U8, 0}};
    case KvCacheStorage::Int8Group64:
        return {{DType::I8, kHeadDim, DType::FP16, kQuantGroups},
                {DType::I8, kHeadDim, DType::FP16, kQuantGroups}};
    case KvCacheStorage::Fp8E4M3Row256:
        return {{DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups},
                {DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups}};
    case KvCacheStorage::Nvfp4Group16:
        return {{DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups},
                {DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups}};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return {{DType::FP8_E4M3FN, kHeadDim, DType::FP16, kFp8QuantGroups},
                {DType::U8, kNvfp4CodeBytes, DType::U8, kNvfp4QuantGroups}};
    }
    throw std::invalid_argument("unsupported test KV storage");
}

ops::AttentionHeadGeometry op_geometry(const Geometry& geometry) {
    return {kHeadDim, geometry.q_heads, geometry.kv_heads};
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

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    }
    return 0;
}

std::vector<std::int32_t> make_block_table(std::int32_t populated_pages, std::int32_t table_pages,
                                           MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(table_pages), -1);
    switch (pattern) {
    case MappingPattern::Identity:
        for (std::int32_t page = 0; page < populated_pages; ++page) { table[page] = page; }
        break;
    case MappingPattern::Offset:
        for (std::int32_t page = 0; page < populated_pages; ++page) { table[page] = page + 1; }
        break;
    case MappingPattern::Fragmented:
        for (std::int32_t page = 0; page < populated_pages; ++page) { table[page] = 2 * page + 1; }
        break;
    }
    return table;
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t logical_plane_index(std::int32_t leading_extent, const Geometry& geometry,
                                std::int32_t padded_context, std::int32_t head,
                                std::int32_t position, std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t fp8_scale_index(const Geometry& geometry, std::int32_t padded_context,
                            std::int32_t head, std::int32_t position) {
    (void)geometry;
    return static_cast<std::size_t>(position) +
           static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head);
}

std::size_t fp8_scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::size_t>(physical_page)));
}

std::size_t physical_plane_elements(std::int32_t leading_extent, const Geometry& geometry,
                                    std::int32_t physical_pages) {
    return static_cast<std::size_t>(leading_extent) * kPagedKVPageSize * geometry.kv_heads *
           physical_pages;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

float round_to_f16(float value) { return f16_bits_to_f32(f32_to_f16_bits(value)); }

std::vector<std::uint16_t> to_f16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_f16_bits(values[i]); }
    return bits;
}

void encode_group(std::span<const float> source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const float unrounded_scale    = absmax / 127.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -127, 127);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

void normalized_hadamard_d256(std::array<float, kHeadDim>& values) {
    for (std::int32_t stride = 1; stride < kHeadDim; stride *= 2) {
        for (std::int32_t base = 0; base < kHeadDim; base += 2 * stride) {
            for (std::int32_t offset = 0; offset < stride; ++offset) {
                const float low  = values[static_cast<std::size_t>(base + offset)];
                const float high = values[static_cast<std::size_t>(base + offset + stride)];
                values[static_cast<std::size_t>(base + offset)]          = low + high;
                values[static_cast<std::size_t>(base + offset + stride)] = low - high;
            }
        }
    }
    for (float& value : values) { value *= 0x1p-4f; }
}

void normalized_hadamard_d256(std::array<double, kHeadDim>& values) {
    static const auto matrix = [] {
        std::array<std::array<double, kHeadDim>, kHeadDim> result{};
        for (unsigned row = 0; row < kHeadDim; ++row)
            for (unsigned column = 0; column < kHeadDim; ++column)
                result[row][column] = (std::popcount(row & column) & 1) ? -0.0625 : 0.0625;
        return result;
    }();
    const auto input = values;
    for (int row = 0; row < kHeadDim; ++row) {
        double sum = 0.0;
        for (int column = 0; column < kHeadDim; ++column)
            sum += matrix[row][column] * input[column];
        values[row] = sum;
    }
}

void encode_nvfp4_rotated_row(std::span<const float> source, std::size_t source_base,
                              std::vector<std::uint8_t>& codes, std::size_t code_base,
                              std::vector<std::uint8_t>& scales, std::size_t scale_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    for (std::int32_t group = 0; group < kNvfp4QuantGroups; ++group) {
        const std::int32_t d0 = group * kNvfp4QuantGroup;
        float absmax          = 0.0f;
        for (std::int32_t i = 0; i < kNvfp4QuantGroup; ++i) {
            absmax = std::max(absmax, std::abs(rotated[static_cast<std::size_t>(d0 + i)]));
        }
        std::uint8_t scale_code = 0;
        float represented_scale = 0.0f;
        if (absmax != 0.0f) {
            const float bounded = std::clamp(absmax / 6.0f, std::ldexp(1.0f, -9), 448.0f);
            scale_code          = encode_e4m3fn_rne_satfinite(bounded);
            represented_scale   = decode_e4m3fn(scale_code);
        }
        scales[scale_base + static_cast<std::size_t>(group)] = scale_code;
        for (std::int32_t pair = 0; pair < kNvfp4QuantGroup / 2; ++pair) {
            const std::int32_t low_d  = d0 + 2 * pair;
            const std::int32_t high_d = low_d + 1;
            const std::uint8_t low =
                represented_scale == 0.0f
                    ? 0
                    : encode_e2m1_rne_satfinite(rotated[static_cast<std::size_t>(low_d)] /
                                                represented_scale);
            const std::uint8_t high =
                represented_scale == 0.0f
                    ? 0
                    : encode_e2m1_rne_satfinite(rotated[static_cast<std::size_t>(high_d)] /
                                                represented_scale);
            codes[code_base + static_cast<std::size_t>(group * 8 + pair)] =
                static_cast<std::uint8_t>(low | (high << 4));
        }
    }
}

void encode_fp8_row(const std::array<float, kHeadDim>& values, std::vector<std::uint8_t>& codes,
                    std::size_t code_base, std::vector<std::uint16_t>& scales,
                    std::size_t scale_offset) {
    float absmax = 0.0f;
    for (const float value : values) absmax = std::max(absmax, std::abs(value));
    std::uint16_t scale_bits = 0;
    float represented_scale  = 0.0f;
    if (absmax != 0.0f) {
        const float raw_scale = absmax / 448.0f;
        scale_bits        = f32_to_f16_bits(std::clamp(raw_scale, std::ldexp(1.0f, -24), 65504.0f));
        represented_scale = f16_bits_to_f32(scale_bits);
    }
    scales[scale_offset]   = scale_bits;
    const float reciprocal = represented_scale == 0.0f ? 0.0f : 1.0f / represented_scale;
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        codes[code_base + static_cast<std::size_t>(d)] =
            represented_scale == 0.0f
                ? 0
                : encode_e4m3fn_rne_satfinite(values[static_cast<std::size_t>(d)] * reciprocal);
    }
}

void encode_fp8_rotated_row(std::span<const float> source, std::size_t source_base,
                            std::vector<std::uint8_t>& codes, std::size_t code_base,
                            std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    encode_fp8_row(rotated, codes, code_base, scales, scale_offset);
}

void encode_rotated_key_row(std::span<const float> source, std::size_t source_base,
                            std::vector<std::int8_t>& codes, std::size_t code_base,
                            std::vector<std::uint16_t>& scales, std::size_t scale_base) {
    std::array<float, kHeadDim> rotated{};
    for (std::int32_t d = 0; d < kHeadDim; ++d) {
        rotated[static_cast<std::size_t>(d)] = source[source_base + static_cast<std::size_t>(d)];
    }
    normalized_hadamard_d256(rotated);
    for (std::int32_t group = 0; group < kQuantGroups; ++group) {
        const std::size_t d = static_cast<std::size_t>(group * kQuantGroup);
        encode_group(rotated, d, codes, code_base + d, scales,
                     scale_base + static_cast<std::size_t>(group));
    }
}

HostCache make_cache(const Geometry& geometry, KvCacheStorage storage, std::int32_t max_context,
                     std::uint32_t seed, float qk_amplitude) {
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k = make_bf16_values(elements, seed, -qk_amplitude, qk_amplitude);
    std::vector<float> logical_v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    HostCache cache{geometry, storage, max_context, logical_capacity};
    if (storage == KvCacheStorage::BFloat16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_fp16 = to_f16_bits(logical_v);
        return cache;
    }

    if (storage == KvCacheStorage::Nvfp4Group16) {
        const std::size_t code_elements =
            static_cast<std::size_t>(kNvfp4CodeBytes) * logical_capacity * geometry.kv_heads;
        const std::size_t scale_elements =
            static_cast<std::size_t>(kNvfp4QuantGroups) * logical_capacity * geometry.kv_heads;
        cache.k_nvfp4.assign(code_elements, 0);
        cache.v_nvfp4.assign(code_elements, 0);
        cache.k_nvfp4_scale.assign(scale_elements, 0);
        cache.v_nvfp4_scale.assign(scale_elements, 0);
        parallel_rows(geometry.kv_heads * logical_capacity,
                      std::uint64_t(geometry.kv_heads) * logical_capacity * kHeadDim,
                      [&](int first, int last) {
                          for (int row = first; row < last; ++row) {
                              const int head     = row / logical_capacity,
                                        position = row % logical_capacity;

                              const std::size_t source =
                                  cache_index(geometry, logical_capacity, head, position, 0);
                              const std::size_t code = logical_plane_index(
                                  kNvfp4CodeBytes, geometry, logical_capacity, head, position, 0);
                              const std::size_t scale = logical_plane_index(
                                  kNvfp4QuantGroups, geometry, logical_capacity, head, position, 0);
                              encode_nvfp4_rotated_row(logical_k, source, cache.k_nvfp4, code,
                                                       cache.k_nvfp4_scale, scale);
                              encode_nvfp4_rotated_row(logical_v, source, cache.v_nvfp4, code,
                                                       cache.v_nvfp4_scale, scale);
                          }
                      });
        return cache;
    }

    if (storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        const std::size_t fp8_scales = fp8_scale_elements(geometry, logical_capacity);
        const std::size_t v_codes =
            static_cast<std::size_t>(kNvfp4CodeBytes) * logical_capacity * geometry.kv_heads;
        const std::size_t v_scales =
            static_cast<std::size_t>(kNvfp4QuantGroups) * logical_capacity * geometry.kv_heads;
        cache.k_fp8.assign(elements, 0);
        cache.k_scale.assign(fp8_scales, 0);
        cache.v_nvfp4.assign(v_codes, 0);
        cache.v_nvfp4_scale.assign(v_scales, 0);
        parallel_rows(geometry.kv_heads * logical_capacity,
                      std::uint64_t(geometry.kv_heads) * logical_capacity * kHeadDim,
                      [&](int first, int last) {
                          for (int row = first; row < last; ++row) {
                              const int head     = row / logical_capacity,
                                        position = row % logical_capacity;

                              const std::size_t source =
                                  cache_index(geometry, logical_capacity, head, position, 0);
                              const std::size_t k_scale =
                                  fp8_scale_index(geometry, logical_capacity, head, position);
                              const std::size_t v_code = logical_plane_index(
                                  kNvfp4CodeBytes, geometry, logical_capacity, head, position, 0);
                              const std::size_t v_scale = logical_plane_index(
                                  kNvfp4QuantGroups, geometry, logical_capacity, head, position, 0);
                              encode_fp8_rotated_row(logical_k, source, cache.k_fp8, source,
                                                     cache.k_scale, k_scale);
                              encode_nvfp4_rotated_row(logical_v, source, cache.v_nvfp4, v_code,
                                                       cache.v_nvfp4_scale, v_scale);
                          }
                      });
        return cache;
    }

    const std::size_t scales = storage == KvCacheStorage::Int8Group64
                                   ? scale_elements(geometry, logical_capacity)
                                   : fp8_scale_elements(geometry, logical_capacity);
    cache.k_scale.assign(scales, 0);
    cache.v_scale.assign(scales, 0);
    if (storage == KvCacheStorage::Fp8E4M3Row256) {
        cache.k_fp8.assign(elements, 0);
        cache.v_fp8.assign(elements, 0);
        parallel_rows(geometry.kv_heads * logical_capacity,
                      std::uint64_t(geometry.kv_heads) * logical_capacity * kHeadDim,
                      [&](int first, int last) {
                          for (int row = first; row < last; ++row) {
                              const int head     = row / logical_capacity,
                                        position = row % logical_capacity;

                              const std::size_t code =
                                  cache_index(geometry, logical_capacity, head, position, 0);
                              const std::size_t scale =
                                  fp8_scale_index(geometry, logical_capacity, head, position);
                              encode_fp8_rotated_row(logical_k, code, cache.k_fp8, code,
                                                     cache.k_scale, scale);
                              std::array<float, kHeadDim> v_row{};
                              for (std::int32_t d = 0; d < kHeadDim; ++d) {
                                  v_row[static_cast<std::size_t>(d)] =
                                      logical_v[code + static_cast<std::size_t>(d)];
                              }
                              encode_fp8_row(v_row, cache.v_fp8, code, cache.v_scale, scale);
                          }
                      });
        return cache;
    }

    cache.k_i8.assign(elements, 0);
    cache.v_i8.assign(elements, 0);
    parallel_rows(
        geometry.kv_heads * logical_capacity,
        std::uint64_t(geometry.kv_heads) * logical_capacity * kHeadDim, [&](int first, int last) {
            for (int row = first; row < last; ++row) {
                const int head = row / logical_capacity, position = row % logical_capacity;

                const std::size_t code = cache_index(geometry, logical_capacity, head, position, 0);
                const std::size_t scale =
                    scale_index(geometry, logical_capacity, head, position, 0);
                encode_rotated_key_row(logical_k, code, cache.k_i8, code, cache.k_scale, scale);
                for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                    const std::int32_t d = group * kQuantGroup;
                    const std::size_t group_code =
                        cache_index(geometry, logical_capacity, head, position, d);
                    const std::size_t group_scale =
                        scale_index(geometry, logical_capacity, head, position, group);
                    encode_group(logical_v, group_code, cache.v_i8, group_code, cache.v_scale,
                                 group_scale);
                }
            }
        });
    return cache;
}

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.storage == KvCacheStorage::BFloat16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                    cache.v_fp16[target] = f32_to_f16_bits(v[source]);
                }
                continue;
            }

            const std::size_t source = kv_input_index(geometry, head, 0, token);
            const std::size_t target =
                cache_index(geometry, cache.logical_capacity, head, position, 0);
            if (cache.storage == KvCacheStorage::Nvfp4Group16) {
                const std::size_t code = logical_plane_index(
                    kNvfp4CodeBytes, geometry, cache.logical_capacity, head, position, 0);
                const std::size_t scale = logical_plane_index(
                    kNvfp4QuantGroups, geometry, cache.logical_capacity, head, position, 0);
                encode_nvfp4_rotated_row(k, source, cache.k_nvfp4, code, cache.k_nvfp4_scale,
                                         scale);
                encode_nvfp4_rotated_row(v, source, cache.v_nvfp4, code, cache.v_nvfp4_scale,
                                         scale);
                continue;
            }
            if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
                const std::size_t k_scale =
                    fp8_scale_index(geometry, cache.logical_capacity, head, position);
                const std::size_t v_code = logical_plane_index(
                    kNvfp4CodeBytes, geometry, cache.logical_capacity, head, position, 0);
                const std::size_t v_scale = logical_plane_index(
                    kNvfp4QuantGroups, geometry, cache.logical_capacity, head, position, 0);
                encode_fp8_rotated_row(k, source, cache.k_fp8, target, cache.k_scale, k_scale);
                encode_nvfp4_rotated_row(v, source, cache.v_nvfp4, v_code, cache.v_nvfp4_scale,
                                         v_scale);
                continue;
            }
            if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
                const std::size_t scale =
                    fp8_scale_index(geometry, cache.logical_capacity, head, position);
                encode_fp8_rotated_row(k, source, cache.k_fp8, target, cache.k_scale, scale);
                std::array<float, kHeadDim> v_row{};
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    v_row[static_cast<std::size_t>(d)] =
                        v[kv_input_index(geometry, head, d, token)];
                }
                encode_fp8_row(v_row, cache.v_fp8, target, cache.v_scale, scale);
                continue;
            }
            const std::size_t scale =
                scale_index(geometry, cache.logical_capacity, head, position, 0);
            encode_rotated_key_row(k, source, cache.k_i8, target, cache.k_scale, scale);
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d           = group * kQuantGroup;
                const std::size_t group_source = kv_input_index(geometry, head, d, token);
                const std::size_t group_target =
                    cache_index(geometry, cache.logical_capacity, head, position, d);
                const std::size_t group_scale =
                    scale_index(geometry, cache.logical_capacity, head, position, group);
                encode_group(v, group_source, cache.v_i8, group_target, cache.v_scale, group_scale);
            }
        }
    }
}

double cache_value(const HostCache& cache, bool key, int head, int position, int d) {
    const std::size_t row   = std::size_t(head) * cache.logical_capacity + position;
    const std::size_t index = row * kHeadDim + d;
    if (cache.storage == KvCacheStorage::BFloat16)
        return key ? double(bf16_to_f32(cache.k_bf16[index]))
                   : double(f16_bits_to_f32(cache.v_fp16[index]));
    if (cache.storage == KvCacheStorage::Int8Group64) {
        const auto& codes  = key ? cache.k_i8 : cache.v_i8;
        const auto& scales = key ? cache.k_scale : cache.v_scale;
        return double(codes[index]) *
               double(f16_bits_to_f32(scales[row * kQuantGroups + d / kQuantGroup]));
    }
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256 ||
        (key && cache.storage == KvCacheStorage::Fp8KeyNvfp4Value)) {
        const auto& codes  = key ? cache.k_fp8 : cache.v_fp8;
        const auto& scales = key ? cache.k_scale : cache.v_scale;
        return double(decode_e4m3fn(codes[index])) * double(f16_bits_to_f32(scales[row]));
    }
    const auto& codes  = key ? cache.k_nvfp4 : cache.v_nvfp4;
    const auto& scales = key ? cache.k_nvfp4_scale : cache.v_nvfp4_scale;
    const auto byte    = codes[row * kNvfp4CodeBytes + d / 2];
    const auto code    = (d & 1) ? byte >> 4 : byte & 15;
    return double(decode_e2m1(code)) *
           double(decode_e4m3fn(scales[row * kNvfp4QuantGroups + d / kNvfp4QuantGroup]));
}

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected) {
    int failures = 0;
    if (expected.storage == KvCacheStorage::BFloat16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
        failures += verify_exact((label + " cache-v").c_str(), got.v_fp16, expected.v_fp16);
    } else if (expected.storage == KvCacheStorage::Int8Group64) {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);


        failures += verify_exact((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    } else if (expected.storage == KvCacheStorage::Fp8E4M3Row256) {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_fp8, expected.k_fp8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);


        failures += verify_exact((label + " cache-v-code").c_str(), got.v_fp8, expected.v_fp8);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    } else if (expected.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_fp8, expected.k_fp8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);


        failures += verify_exact((label + " cache-v-code").c_str(), got.v_nvfp4, expected.v_nvfp4);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_nvfp4_scale,
                                 expected.v_nvfp4_scale);
    } else {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_nvfp4, expected.k_nvfp4);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_nvfp4_scale,
                                 expected.k_nvfp4_scale);


        failures += verify_exact((label + " cache-v-code").c_str(), got.v_nvfp4, expected.v_nvfp4);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_nvfp4_scale,
                                 expected.v_nvfp4_scale);
    }
    return failures;
}

int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected) {
    int failures = verify_exact(
        label.c_str(), copy_from_guarded<std::uint16_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected) {
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::int32_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

const char* cache_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-g64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3fn-row256";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4-g16";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

std::string case_label(const char* entry, const Geometry& geometry, KvCacheStorage storage,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(storage) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max) +
           " qk_amplitude=" + std::to_string(test_case.qk_amplitude) +
           (test_case.graph_replay ? " graph-replay" : "");
}

void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v) {
    if (tokens == 0) return;
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        k[kv_input_index(geometry, 0, d, 0)]               = 0.0f;
        v[kv_input_index(geometry, 0, kQuantGroup + d, 0)] = 0.0f;
    }
    k[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = -1.0f;
    v[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = 1.0f;
}

} // namespace ninfer::test::causal_attention
