#include "ninfer/ops/kv_cache_append.h"
#include "ops/op_tester.h"

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
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim       = 128;
constexpr int kKVHeads       = 8;
constexpr int kPage          = 64;
constexpr int kLogicalPages  = 3;
constexpr int kPhysicalPages = 6;
constexpr int kWindow        = 4096;
constexpr int kFullHeadDim   = 256;
constexpr int kFullGroup     = 64;
constexpr int kFullGroups    = kFullHeadDim / kFullGroup;
constexpr int kFullFp8Groups = 1;

std::vector<std::uint16_t> patterned_bits(std::size_t count, std::uint32_t seed);

std::size_t full_input_index(int d, int head, int token, int kv_heads) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kFullHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(token));
}

std::size_t full_cache_index(int leading_extent, int leading, int head, int position,
                             int physical_page, int kv_heads) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPage) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(head) +
                     static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(physical_page)));
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) ++half_mantissa;
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    const auto lower     = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

float decode_e4m3fn_positive(std::uint8_t code) {
    const int exponent = (code >> 3) & 0x0f;
    const int mantissa = code & 0x07;
    if (exponent == 0) return std::ldexp(static_cast<float>(mantissa), -9);
    return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
}

std::uint8_t encode_e4m3fn_rne_satfinite(float value) {
    const bool negative   = std::signbit(value);
    const float magnitude = std::abs(value);
    std::uint8_t selected = 0x7e;
    if (magnitude < 448.0f) {
        for (std::uint8_t upper = 1; upper <= 0x7e; ++upper) {
            const float upper_value = decode_e4m3fn_positive(upper);
            if (upper_value < magnitude) continue;
            const std::uint8_t lower = static_cast<std::uint8_t>(upper - 1);
            const float lower_value  = decode_e4m3fn_positive(lower);
            const float lower_error  = magnitude - lower_value;
            const float upper_error  = upper_value - magnitude;
            selected                 = lower_error < upper_error   ? lower
                                       : upper_error < lower_error ? upper
                                                                   : ((lower & 1U) == 0U ? lower : upper);
            break;
        }
    }
    return static_cast<std::uint8_t>(selected | (negative ? 0x80U : 0U));
}

void normalized_hadamard_d256_host(std::array<float, kFullHeadDim>& values) {
    for (int block = 0; block < 8; ++block) {
        const int block_begin = block * 32;
        for (int stride = 1; stride <= 16; stride <<= 1) {
            for (int base = 0; base < 32; base += 2 * stride) {
                for (int offset = 0; offset < stride; ++offset) {
                    const int low_index  = block_begin + base + offset;
                    const int high_index = low_index + stride;
                    const float low      = values[static_cast<std::size_t>(low_index)];
                    const float high     = values[static_cast<std::size_t>(high_index)];
                    values[static_cast<std::size_t>(low_index)]  = low + high;
                    values[static_cast<std::size_t>(high_index)] = low - high;
                }
            }
        }
    }
    for (int span = 1; span < 8; span <<= 1) {
        for (int base = 0; base < 8; base += 2 * span) {
            for (int offset = 0; offset < span; ++offset) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int low_index  = lane + 32 * (base + offset);
                    const int high_index = lane + 32 * (base + offset + span);
                    const float low      = values[static_cast<std::size_t>(low_index)];
                    const float high     = values[static_cast<std::size_t>(high_index)];
                    values[static_cast<std::size_t>(low_index)]  = low + high;
                    values[static_cast<std::size_t>(high_index)] = low - high;
                }
            }
        }
    }
    for (float& value : values) value *= 0x1p-4f;
}

void encode_full_fp8_row(const std::array<float, kFullHeadDim>& values,
                         std::vector<std::uint8_t>& codes, int head, int position,
                         int physical_page, int kv_heads, std::vector<std::uint16_t>& scales) {
    float absmax = 0.0f;
    for (const float value : values) absmax = std::max(absmax, std::abs(value));
    std::uint16_t scale_bits = 0;
    float scale              = 0.0f;
    if (absmax != 0.0f) {
        const float raw_scale = absmax / 448.0f;
        const float bounded   = std::clamp(raw_scale, std::ldexp(1.0f, -24), 65504.0f);
        scale_bits            = f32_to_f16_bits(bounded);
        scale                 = f16_bits_to_f32(scale_bits);
    }
    const float inverse = scale == 0.0f ? 0.0f : 1.0f / scale;
    for (int d = 0; d < kFullHeadDim; ++d) {
        const auto target =
            full_cache_index(kFullHeadDim, d, head, position, physical_page, kv_heads);
        codes[target] =
            scale == 0.0f
                ? 0
                : encode_e4m3fn_rne_satfinite(values[static_cast<std::size_t>(d)] * inverse);
    }
    scales[full_cache_index(kFullFp8Groups, 0, head, position, physical_page, kv_heads)] =
        scale_bits;
}

void encode_full_group(const std::vector<float>& source, std::size_t source_base,
                       std::vector<std::int8_t>& codes, int head, int position, int physical_page,
                       int group, int kv_heads, std::vector<std::uint16_t>& scales) {
    float absmax = 0.0f;
    for (int i = 0; i < kFullGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }
    const std::uint16_t scale_bits = f32_to_f16_bits(absmax / 127.0f);
    const float scale              = f16_bits_to_f32(scale_bits);
    const float inverse            = scale == 0.0f ? 0.0f : 1.0f / scale;
    for (int i = 0; i < kFullGroup; ++i) {
        const int code =
            scale == 0.0f
                ? 0
                : std::clamp(round_even_to_i32(source[source_base + static_cast<std::size_t>(i)] *
                                               inverse),
                             -127, 127);
        const auto target = full_cache_index(kFullHeadDim, group * kFullGroup + i, head, position,
                                             physical_page, kv_heads);
        codes[target]     = static_cast<std::int8_t>(code);
    }
    scales[full_cache_index(kFullGroups, group, head, position, physical_page, kv_heads)] =
        scale_bits;
}

int full_append_case(int kv_heads, DType dtype, int tokens = 3) {
    const int first_position = tokens >= 128 ? 61 : 63;
    const int logical_pages  = (first_position + tokens + kPage - 1) / kPage;
    const int physical_pages = 2 * logical_pages + 1;
    std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
    for (int page = 0; page < logical_pages; ++page) {
        mapping[static_cast<std::size_t>(page)] = 2 * page + 1;
    }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = first_position + token;
    }
    const std::size_t input_count = static_cast<std::size_t>(kFullHeadDim) * kv_heads * tokens;
    const std::size_t code_count =
        static_cast<std::size_t>(kFullHeadDim) * kPage * kv_heads * physical_pages;
    const int scale_groups = dtype == DType::I8           ? kFullGroups
                             : dtype == DType::FP8_E4M3FN ? kFullFp8Groups
                                                          : 0;
    const std::size_t scale_count =
        static_cast<std::size_t>(scale_groups) * kPage * kv_heads * physical_pages;

    std::vector<float> host_k(input_count);
    std::vector<float> host_v(input_count);
    fill_uniform(host_k, 0x13579u + static_cast<std::uint32_t>(kv_heads), -0.75f, 0.75f);
    fill_uniform(host_v, 0x24680u + static_cast<std::uint32_t>(kv_heads), -1.25f, 1.25f);
    round_to_bf16(host_k);
    round_to_bf16(host_v);
    for (int i = 0; i < kFullGroup; ++i) {
        host_k[full_input_index(i, 0, 0, kv_heads)]              = 0.0f;
        host_v[full_input_index(kFullGroup + i, 0, 0, kv_heads)] = 0.0f;
    }
    if (dtype == DType::FP8_E4M3FN) {
        for (int d = 0; d < kFullHeadDim; ++d) {
            host_k[full_input_index(d, 0, 0, kv_heads)] = 0.0f;
            host_v[full_input_index(d, 0, 0, kv_heads)] = 0.0f;
        }
        host_v[full_input_index(0, 1, 0, kv_heads)] = 448.0f;
        host_v[full_input_index(1, 1, 0, kv_heads)] = 1.0625f;
        host_v[full_input_index(0, 0, 1, kv_heads)] = std::ldexp(1.0f, -20);
        host_v[full_input_index(0, 1, 1, kv_heads)] = bf16_to_f32(0x7f7fU);
        // With absmax=49, d1/s is exactly an E4M3 midpoint while d1*FP32(1/s) is one
        // FP32 ulp above it. This row locks the codec's ordered reciprocal-multiply semantics.
        for (int d = 0; d < kFullHeadDim; ++d) {
            host_v[full_input_index(d, 0, 2, kv_heads)] = 0.0f;
        }
        host_v[full_input_index(0, 0, 2, kv_heads)] = 49.0f;
        host_v[full_input_index(1, 0, 2, kv_heads)] = 0.0013885498046875f;
    }
    std::vector<std::uint16_t> input_k(input_count);
    std::vector<std::uint16_t> input_v(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        input_k[i] = f32_to_bf16(host_k[i]);
        input_v[i] = f32_to_bf16(host_v[i]);
    }

    DeviceBuffer d_k         = to_device(input_k);
    DeviceBuffer d_v         = to_device(input_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_mapping   = to_device(mapping);
    Tensor k(d_k.p, DType::BF16, {kFullHeadDim, kv_heads, tokens});
    Tensor v(d_v.p, DType::BF16, {kFullHeadDim, kv_heads, tokens});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens});

    GuardedDeviceBuffer cache_k(code_count * dtype_size(dtype));
    GuardedDeviceBuffer cache_v(code_count * dtype_size(dtype));
    const bool quantized = dtype == DType::I8 || dtype == DType::FP8_E4M3FN;
    GuardedDeviceBuffer scale_k(quantized ? scale_count * sizeof(std::uint16_t) : 1);
    GuardedDeviceBuffer scale_v(quantized ? scale_count * sizeof(std::uint16_t) : 1);
    PagedKVLayerView cache{
        .k_pages = Tensor(cache_k.data(), dtype, {kFullHeadDim, kPage, kv_heads, physical_pages}),
        .v_pages = Tensor(cache_v.data(), dtype, {kFullHeadDim, kPage, kv_heads, physical_pages}),
        .block_table  = Tensor(d_mapping.p, DType::I32, {logical_pages}),
        .head_dim     = kFullHeadDim,
        .num_kv_heads = kv_heads,
        .dtype        = dtype,
        .quant_group  = dtype == DType::I8           ? kFullGroup
                        : dtype == DType::FP8_E4M3FN ? kFullHeadDim
                                                     : 0,
    };

    int failures = 0;
    if (dtype == DType::BF16) {
        auto expected_k = patterned_bits(code_count, 0xabcdef01u);
        auto expected_v = patterned_bits(code_count, 0x10fedcbau);
        cache_k.copy_from_host(expected_k.data(), expected_k.size() * sizeof(std::uint16_t));
        cache_v.copy_from_host(expected_v.data(), expected_v.size() * sizeof(std::uint16_t));
        for (int token = 0; token < tokens; ++token) {
            const int position = positions[static_cast<std::size_t>(token)];
            const int page     = mapping[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kv_heads; ++head) {
                for (int d = 0; d < kFullHeadDim; ++d) {
                    const auto source = full_input_index(d, head, token, kv_heads);
                    const auto target =
                        full_cache_index(kFullHeadDim, d, head, position, page, kv_heads);
                    expected_k[target] = input_k[source];
                    expected_v[target] = input_v[source];
                }
            }
        }
        ops::kv_cache_append(k, v, position_tensor, cache, nullptr);
        cuda_synchronize();
        const std::string label = "kv_cache_append full bf16 Hkv=" + std::to_string(kv_heads) +
                                  " T=" + std::to_string(tokens) +
                                  " P=" + std::to_string(first_position);
        failures +=
            verify_exact((label + " k").c_str(),
                         from_device<std::uint16_t>(cache_k.data(), code_count), expected_k);
        failures +=
            verify_exact((label + " v").c_str(),
                         from_device<std::uint16_t>(cache_v.data(), code_count), expected_v);
    } else if (dtype == DType::I8) {
        std::vector<std::int8_t> expected_k(code_count, static_cast<std::int8_t>(0x55));
        std::vector<std::int8_t> expected_v(code_count, static_cast<std::int8_t>(0xaa));
        auto expected_scale_k = patterned_bits(scale_count, 0x01234567u);
        auto expected_scale_v = patterned_bits(scale_count, 0x89abcdefu);
        cache_k.copy_from_host(expected_k.data(), expected_k.size());
        cache_v.copy_from_host(expected_v.data(), expected_v.size());
        scale_k.copy_from_host(expected_scale_k.data(),
                               expected_scale_k.size() * sizeof(std::uint16_t));
        scale_v.copy_from_host(expected_scale_v.data(),
                               expected_scale_v.size() * sizeof(std::uint16_t));
        cache.k_scale_pages =
            Tensor(scale_k.data(), DType::FP16, {kFullGroups, kPage, kv_heads, physical_pages});
        cache.v_scale_pages =
            Tensor(scale_v.data(), DType::FP16, {kFullGroups, kPage, kv_heads, physical_pages});
        for (int token = 0; token < tokens; ++token) {
            const int position = positions[static_cast<std::size_t>(token)];
            const int page     = mapping[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kv_heads; ++head) {
                for (int group = 0; group < kFullGroups; ++group) {
                    const auto source = full_input_index(group * kFullGroup, head, token, kv_heads);
                    encode_full_group(host_v, source, expected_v, head, position, page, group,
                                      kv_heads, expected_scale_v);
                }
            }
        }
        ops::kv_cache_append(k, v, position_tensor, cache, nullptr);
        cuda_synchronize();
        const std::string label = "kv_cache_append full int8-g64 Hkv=" + std::to_string(kv_heads) +
                                  " T=" + std::to_string(tokens) +
                                  " P=" + std::to_string(first_position);
        failures += verify_exact((label + " v codes").c_str(),
                                 from_device<std::int8_t>(cache_v.data(), code_count), expected_v);
        failures +=
            verify_exact((label + " v scales").c_str(),
                         from_device<std::uint16_t>(scale_v.data(), scale_count), expected_scale_v);
        failures += scale_k.verify_guards((label + " k scale guards").c_str());
        failures += scale_v.verify_guards((label + " v scale guards").c_str());
    } else {
        std::vector<std::uint8_t> expected_k(code_count, 0x55U);
        std::vector<std::uint8_t> expected_v(code_count, 0xaaU);
        auto expected_scale_k = patterned_bits(scale_count, 0x01234567u);
        auto expected_scale_v = patterned_bits(scale_count, 0x89abcdefu);
        cache_k.copy_from_host(expected_k.data(), expected_k.size());
        cache_v.copy_from_host(expected_v.data(), expected_v.size());
        scale_k.copy_from_host(expected_scale_k.data(),
                               expected_scale_k.size() * sizeof(std::uint16_t));
        scale_v.copy_from_host(expected_scale_v.data(),
                               expected_scale_v.size() * sizeof(std::uint16_t));
        cache.k_scale_pages =
            Tensor(scale_k.data(), DType::FP16, {kFullFp8Groups, kPage, kv_heads, physical_pages});
        cache.v_scale_pages =
            Tensor(scale_v.data(), DType::FP16, {kFullFp8Groups, kPage, kv_heads, physical_pages});
        for (int token = 0; token < tokens; ++token) {
            const int position = positions[static_cast<std::size_t>(token)];
            const int page     = mapping[static_cast<std::size_t>(position / kPage)];
            for (int head = 0; head < kv_heads; ++head) {
                std::array<float, kFullHeadDim> k_row{};
                std::array<float, kFullHeadDim> v_row{};
                for (int d = 0; d < kFullHeadDim; ++d) {
                    const auto source                  = full_input_index(d, head, token, kv_heads);
                    k_row[static_cast<std::size_t>(d)] = host_k[source];
                    v_row[static_cast<std::size_t>(d)] = host_v[source];
                }
                normalized_hadamard_d256_host(k_row);
                encode_full_fp8_row(k_row, expected_k, head, position, page, kv_heads,
                                    expected_scale_k);
                encode_full_fp8_row(v_row, expected_v, head, position, page, kv_heads,
                                    expected_scale_v);
            }
        }
        ops::kv_cache_append(k, v, position_tensor, cache, nullptr);
        cuda_synchronize();
        const std::string label =
            "kv_cache_append full fp8-row256 Hkv=" + std::to_string(kv_heads) +
            " T=" + std::to_string(tokens) + " P=" + std::to_string(first_position);
        failures += verify_exact((label + " k codes").c_str(),
                                 from_device<std::uint8_t>(cache_k.data(), code_count), expected_k);
        failures += verify_exact((label + " v codes").c_str(),
                                 from_device<std::uint8_t>(cache_v.data(), code_count), expected_v);
        failures +=
            verify_exact((label + " k scales").c_str(),
                         from_device<std::uint16_t>(scale_k.data(), scale_count), expected_scale_k);
        failures +=
            verify_exact((label + " v scales").c_str(),
                         from_device<std::uint16_t>(scale_v.data(), scale_count), expected_scale_v);
        failures += scale_k.verify_guards((label + " k scale guards").c_str());
        failures += scale_v.verify_guards((label + " v scale guards").c_str());
    }
    const std::string label = "kv_cache_append full Hkv=" + std::to_string(kv_heads) +
                              " T=" + std::to_string(tokens) +
                              " P=" + std::to_string(first_position);
    failures += verify_exact((label + " input k unchanged").c_str(),
                             from_device<std::uint16_t>(d_k, input_count), input_k);
    failures += verify_exact((label + " input v unchanged").c_str(),
                             from_device<std::uint16_t>(d_v, input_count), input_v);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<std::int32_t>(d_positions, positions.size()), positions);
    failures += verify_exact((label + " block table unchanged").c_str(),
                             from_device<std::int32_t>(d_mapping, mapping.size()), mapping);
    failures += cache_k.verify_guards((label + " k guards").c_str());
    failures += cache_v.verify_guards((label + " v guards").c_str());
    return failures;
}

std::size_t input_index(int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t cyclic_cache_index(int d, int head, int slot) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(kWindow) * static_cast<std::size_t>(head));
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

void append_oracle(std::vector<std::uint16_t>& cache_k, std::vector<std::uint16_t>& cache_v,
                   const std::vector<std::uint16_t>& input_k,
                   const std::vector<std::uint16_t>& input_v,
                   const std::vector<std::int32_t>& positions, int commit_count, bool cyclic,
                   const std::vector<std::int32_t>& mapping) {
    for (int token = 0; token < commit_count; ++token) {
        const int position = positions[static_cast<std::size_t>(token)];
        const int slot     = cyclic ? position % kWindow : 0;
        for (int head = 0; head < kKVHeads; ++head) {
            for (int d = 0; d < kHeadDim; ++d) {
                const auto src = input_index(d, head, token);
                const auto dst = cyclic ? cyclic_cache_index(d, head, slot)
                                        : paged_cache_index(d, head, position, mapping);
                cache_k[dst]   = input_k[src];
                cache_v[dst]   = input_v[src];
            }
        }
    }
}

PagedKVBatchLayerView paged_view(GuardedDeviceBuffer& k, GuardedDeviceBuffer& v,
                                 DeviceBuffer& block_table, int table_rows = 1) {
    return {
        .k_pages      = Tensor(k.data(), DType::BF16, {kHeadDim, kPage, kPhysicalPages, kKVHeads}),
        .v_pages      = Tensor(v.data(), DType::BF16, {kHeadDim, kPage, kPhysicalPages, kKVHeads}),
        .block_tables = Tensor(block_table.p, DType::I32, {kLogicalPages, table_rows}),
        .head_dim     = kHeadDim,
        .num_kv_heads = kKVHeads,
        .dtype        = DType::BF16,
        .quant_group  = 0,
    };
}

CyclicKVCacheLayerView cyclic_view(GuardedDeviceBuffer& k, GuardedDeviceBuffer& v,
                                   int lane_capacity = 1) {
    return {
        .k        = Tensor(k.data(), DType::BF16, {kHeadDim, kWindow, kKVHeads, lane_capacity}),
        .v        = Tensor(v.data(), DType::BF16, {kHeadDim, kWindow, kKVHeads, lane_capacity}),
        .capacity = kWindow,
        .padded_capacity = kWindow,
        .num_kv_heads    = kKVHeads,
        .head_dim        = kHeadDim,
        .lane_capacity   = lane_capacity,
    };
}

int run_case(int tokens, int commit_count, int first_position, bool cyclic,
             std::vector<std::int32_t> mapping = {}, int min_count = 0) {
    if (!cyclic && mapping.size() != kLogicalPages) {
        throw std::invalid_argument("paged prefix case requires a complete mapping");
    }
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * (cyclic ? kWindow : kPage * kPhysicalPages);
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
    append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, cyclic, mapping);

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({commit_count});
    DeviceBuffer d_selector  = to_device<std::int32_t>({0});
    DeviceBuffer d_table     = cyclic ? DeviceBuffer(1) : to_device(mapping);
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
    cache_v.copy_from_host(initial_v.data(), cache_v.bytes());

    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.p, DType::I32, {1});
    Tensor selector_tensor(d_selector.p, DType::I32, {1});
    const ops::KVCacheAppendPrefixExecutionEnvelope envelope{
        .min_count = static_cast<std::uint32_t>(min_count),
        .max_count = static_cast<std::uint32_t>(tokens),
    };
    if (cyclic) {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    cyclic_view(cache_k, cache_v), nullptr);
    } else {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    paged_view(cache_k, cache_v, d_table), nullptr);
    }
    cuda_synchronize();

    const std::string label = std::string("kv_cache_append_prefix ") +
                              (cyclic ? "cyclic" : "paged") + " T=" + std::to_string(tokens) +
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

int cyclic_graph_replay_case() {
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
    DeviceBuffer d_lane      = to_device<std::int32_t>({0});
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.p, DType::I32, {1});
    Tensor lane_tensor(d_lane.p, DType::I32, {1});
    auto cache = cyclic_view(cache_k, cache_v);

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create kv append stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin kv append capture");
    ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, lane_tensor, {0, tokens},
                                cache, stream);
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
        append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, true, {});
        const std::string label =
            "kv_cache_append_prefix cyclic graph C=" + std::to_string(commit_count);
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
    const auto host_v    = patterned_bits(input_count, 0x87654321u);
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
    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.p, DType::I32, {1});
    Tensor row_tensor(d_row.p, DType::I32, {1});
    auto cache = paged_view(cache_k, cache_v, d_table);

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create paged kv append stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin paged kv append capture");
    ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, row_tensor, {0, tokens}, cache,
                                stream);
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

int batch_selector_case(bool cyclic) {
    constexpr int tokens = 3;
    constexpr int batch  = 2;
    const std::vector<std::int32_t> counts{1, 3};
    const std::vector<std::int32_t> selectors{1, 0};
    const std::vector<std::int32_t> positions =
        cyclic ? std::vector<std::int32_t>{kWindow - 1, kWindow, kWindow + 1, 5, 6, 7}
               : std::vector<std::int32_t>{63, 64, 65, 5, 6, 7};
    const std::size_t row_input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t lane_cache_count =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * (cyclic ? kWindow : kPage * kPhysicalPages);
    const auto host_k    = patterned_bits(row_input_count * batch, 0x31415926u);
    const auto host_v    = patterned_bits(row_input_count * batch, 0x27182818u);
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
                                     cyclic_cache_index(d, head, position % kWindow)
                               : paged_cache_index(d, head, position, mapping);
                    expected_k[dst] = host_k[src];
                    expected_v[dst] = host_v[src];
                }
            }
        }
    }

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_counts    = to_device(counts);
    DeviceBuffer d_selectors = to_device(selectors);
    DeviceBuffer d_tables    = cyclic ? DeviceBuffer(1) : to_device(tables);
    GuardedDeviceBuffer cache_k(initial_k.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(initial_v.size() * sizeof(std::uint16_t));
    cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
    cache_v.copy_from_host(initial_v.data(), cache_v.bytes());

    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens, batch});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens, batch});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens, batch});
    Tensor count_tensor(d_counts.p, DType::I32, {batch});
    Tensor selector_tensor(d_selectors.p, DType::I32, {batch});
    constexpr ops::KVCacheAppendPrefixExecutionEnvelope envelope{0, tokens};
    if (cyclic) {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    cyclic_view(cache_k, cache_v, batch), nullptr);
    } else {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    paged_view(cache_k, cache_v, d_tables, batch), nullptr);
    }
    cuda_synchronize();

    const std::string label =
        std::string("kv_cache_append_prefix B=2 ") + (cyclic ? "cyclic lanes" : "paged rows");
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

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "kv_cache_append: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    for (const int kv_heads : {4, 2}) {
        failures += full_append_case(kv_heads, DType::BF16);
        failures += full_append_case(kv_heads, DType::I8);
        failures += full_append_case(kv_heads, DType::FP8_E4M3FN);
    }
    failures += full_append_case(2, DType::I8, 129);
    failures += full_append_case(2, DType::FP8_E4M3FN, 129);
    failures += run_case(1, 0, 0, false, {0, 1, 2});
    failures += run_case(1, 1, 63, false, {2, 3, 4});
    failures += run_case(16, 7, 60, false, {5, 1, 4}, 5);
    failures += run_case(16, 16, 120, false, {2, 5, 0});
    failures += run_case(1, 0, kWindow - 1, true);
    failures += run_case(1, 1, 2 * kWindow - 1, true);
    failures += run_case(16, 7, 2 * kWindow - 2, true);
    failures += run_case(16, 16, 3 * kWindow - 8, true, {}, 16);
    failures += cyclic_graph_replay_case();
    failures += paged_graph_replay_case();
    failures += batch_selector_case(true);
    failures += batch_selector_case(false);

    if (failures != 0) {
        std::cerr << "kv_cache_append failures=" << failures << '\n';
        return 1;
    }
    std::cout << "kv_cache_append: PASS\n";
    return 0;
}
