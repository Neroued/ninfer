#include "core/arena.h"
#include "core/kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim       = 256;
constexpr std::int32_t kQuantGroup    = 64;
constexpr std::int32_t kQuantGroups   = kHeadDim / kQuantGroup;
constexpr float kAttentionScale       = 0.0625f;
constexpr std::uint16_t kOutputCanary = 0x7fc1u;

// The Op has three registered compute profiles. A1 and A3 use the same criterion for a given
// profile; token count, geometry, execution envelope, and private launch route do not select it.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

// Mixed-format caches (one BF16 side, one INT8-G64 side) keep BF16 tensor-core math and
// dequantize only the INT8 side. Their prompt route has its own qualified gross-error profile.
constexpr ReductionCriterion kAttentionMixedCriterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.8e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

constexpr Geometry kGeometries[] = {
    {"qwen3_6_27b", 24, 4},
    {"qwen3_6_35b_a3b", 16, 2},
};

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
};

std::int32_t align_up_128(std::int32_t value) { return ((value + 127) / 128) * 128; }

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
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
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
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
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
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

struct HostCache {
    Geometry geometry;
    DType k_dtype;
    DType v_dtype;
    std::int32_t max_context;
    std::int32_t padded_context;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_bf16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
};

void encode_group(const std::vector<float>& source, std::size_t source_base,
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

HostCache make_cache(const Geometry& geometry, DType k_dtype, DType v_dtype,
                     std::int32_t max_context, std::uint32_t seed) {
    const std::int32_t padded_context = align_up_128(max_context);
    const std::size_t elements        = cache_elements(geometry, padded_context);
    std::vector<float> logical_k      = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v      = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    HostCache cache{geometry, k_dtype, v_dtype, max_context, padded_context};
    if (k_dtype == DType::BF16) { cache.k_bf16 = to_bf16_bits(logical_k); }
    if (v_dtype == DType::BF16) { cache.v_bf16 = to_bf16_bits(logical_v); }
    const std::size_t scales = scale_elements(geometry, padded_context);
    if (k_dtype == DType::I8) {
        cache.k_i8.assign(elements, 0);
        cache.k_scale.assign(scales, 0);
    }
    if (v_dtype == DType::I8) {
        cache.v_i8.assign(elements, 0);
        cache.v_scale.assign(scales, 0);
    }
    if (k_dtype == DType::BF16 && v_dtype == DType::BF16) { return cache; }

    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < padded_context; ++position) {
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d   = group * kQuantGroup;
                const std::size_t code = cache_index(geometry, padded_context, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, padded_context, head, position, group);
                if (k_dtype == DType::I8) {
                    encode_group(logical_k, code, cache.k_i8, code, cache.k_scale, scale);
                }
                if (v_dtype == DType::I8) {
                    encode_group(logical_v, code, cache.v_i8, code, cache.v_scale, scale);
                }
            }
        }
    }
    return cache;
}

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.k_dtype == DType::BF16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.padded_context, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                }
            } else {
                for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                    const std::int32_t d     = group * kQuantGroup;
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.padded_context, head, position, d);
                    const std::size_t scale =
                        scale_index(geometry, cache.padded_context, head, position, group);
                    encode_group(k, source, cache.k_i8, target, cache.k_scale, scale);
                }
            }
            if (cache.v_dtype == DType::BF16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.padded_context, head, position, d);
                    cache.v_bf16[target] = f32_to_bf16(v[source]);
                }
            } else {
                for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                    const std::int32_t d     = group * kQuantGroup;
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.padded_context, head, position, d);
                    const std::size_t scale =
                        scale_index(geometry, cache.padded_context, head, position, group);
                    encode_group(v, source, cache.v_i8, target, cache.v_scale, scale);
                }
            }
        }
    }
}

double cache_value(const HostCache& cache, bool key, std::int32_t head, std::int32_t position,
                   std::int32_t d) {
    const std::size_t code  = cache_index(cache.geometry, cache.padded_context, head, position, d);
    const DType side_dtype  = key ? cache.k_dtype : cache.v_dtype;
    if (side_dtype == DType::BF16) {
        return static_cast<double>(bf16_to_f32(key ? cache.k_bf16[code] : cache.v_bf16[code]));
    }

    const std::size_t scale =
        scale_index(cache.geometry, cache.padded_context, head, position, d / kQuantGroup);
    const auto& codes   = key ? cache.k_i8 : cache.v_i8;
    const auto& scales  = key ? cache.k_scale : cache.v_scale;
    const float decoded = static_cast<float>(codes[code]) * f16_bits_to_f32(scales[scale]);
    return static_cast<double>(decoded);
}

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));

    std::vector<double> scores(static_cast<std::size_t>(positions.back()) + 1);
    std::vector<double> probabilities(scores.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < geometry.q_heads; ++q_head) {
            const std::int32_t kv_head = q_head / geometry.query_group();
            double max_score           = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[q_index(geometry, q_head, d, token)]) *
                           cache_value(cache, true, kv_head, position, d);
                }
                const double score = dot * static_cast<double>(kAttentionScale);
                scores[static_cast<std::size_t>(position)] = score;
                max_score                                  = std::max(max_score, score);
            }

            double sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                const double probability =
                    std::exp(scores[static_cast<std::size_t>(position)] - max_score);
                probabilities[static_cast<std::size_t>(position)] = probability;
                sum += probability;
            }
            for (std::int32_t position = 0; position < visible; ++position) {
                probabilities[static_cast<std::size_t>(position)] /= sum;
            }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (std::int32_t position = 0; position < visible; ++position) {
                    value += probabilities[static_cast<std::size_t>(position)] *
                             cache_value(cache, false, kv_head, position, d);
                }
                output[q_index(geometry, q_head, d, token)] = value;
            }
        }
    }
    return output;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

class DeviceCache {
public:
    explicit DeviceCache(const HostCache& cache)
        : geometry_(cache.geometry), k_dtype_(cache.k_dtype), v_dtype_(cache.v_dtype),
          max_context_(cache.max_context),
          padded_context_(cache.padded_context),
          code_elements_(cache_elements(geometry_, padded_context_)),
          scale_elements_(scale_elements(geometry_, padded_context_)),
          k_(code_elements_ *
             (k_dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (v_dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(k_dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(v_dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1) {
        if (k_dtype_ == DType::BF16) {
            k_.copy_from_host(cache.k_bf16.data(), cache.k_bf16.size() * sizeof(std::uint16_t));
        } else {
            k_.copy_from_host(cache.k_i8.data(), cache.k_i8.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(cache.k_scale.data(),
                                    cache.k_scale.size() * sizeof(std::uint16_t));
        }
        if (v_dtype_ == DType::BF16) {
            v_.copy_from_host(cache.v_bf16.data(), cache.v_bf16.size() * sizeof(std::uint16_t));
        } else {
            v_.copy_from_host(cache.v_i8.data(), cache.v_i8.size() * sizeof(std::int8_t));
            v_scale_.copy_from_host(cache.v_scale.data(),
                                    cache.v_scale.size() * sizeof(std::uint16_t));
        }
    }

    KVCacheLayerView view() {
        KVCacheLayerView result;
        result.k = Tensor(k_.data(), k_dtype_, {kHeadDim, padded_context_, geometry_.kv_heads});
        result.v = Tensor(v_.data(), v_dtype_, {kHeadDim, padded_context_, geometry_.kv_heads});
        result.max_context    = static_cast<std::uint32_t>(max_context_);
        result.padded_context = static_cast<std::uint32_t>(padded_context_);
        result.num_kv_heads   = geometry_.kv_heads;
        result.head_dim       = kHeadDim;
        result.k_dtype        = k_dtype_;
        result.v_dtype        = v_dtype_;
        if (k_dtype_ == DType::I8) {
            result.k_scale     = Tensor(k_scale_.data(), DType::FP16,
                                        {kQuantGroups, padded_context_, geometry_.kv_heads});
            result.k_quant_group = kQuantGroup;
        }
        if (v_dtype_ == DType::I8) {
            result.v_scale     = Tensor(v_scale_.data(), DType::FP16,
                                        {kQuantGroups, padded_context_, geometry_.kv_heads});
            result.v_quant_group = kQuantGroup;
        }
        return result;
    }

    HostCache snapshot() const {
        HostCache cache{geometry_, k_dtype_, v_dtype_, max_context_, padded_context_};
        if (k_dtype_ == DType::BF16) {
            cache.k_bf16 = copy_from_guarded<std::uint16_t>(k_, code_elements_);
        } else {
            cache.k_i8    = copy_from_guarded<std::int8_t>(k_, code_elements_);
            cache.k_scale = copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_);
        }
        if (v_dtype_ == DType::BF16) {
            cache.v_bf16 = copy_from_guarded<std::uint16_t>(v_, code_elements_);
        } else {
            cache.v_i8    = copy_from_guarded<std::int8_t>(v_, code_elements_);
            cache.v_scale = copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_);
        }
        return cache;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (k_dtype_ == DType::I8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
        }
        if (v_dtype_ == DType::I8) {
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        return failures;
    }

private:
    Geometry geometry_;
    DType k_dtype_;
    DType v_dtype_;
    std::int32_t max_context_;
    std::int32_t padded_context_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
};

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected) {
    int failures = 0;
    if (expected.k_dtype == DType::BF16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
    } else {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
    }
    if (expected.v_dtype == DType::BF16) {
        failures += verify_exact((label + " cache-v").c_str(), got.v_bf16, expected.v_bf16);
    } else {
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
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

const char* side_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "int8-g64"; }

std::string cache_name(DType k_dtype, DType v_dtype) {
    if (k_dtype == v_dtype) { return side_name(k_dtype); }
    return std::string("k-") + side_name(k_dtype) + "/v-" + side_name(v_dtype);
}

ReductionCriterion attention_criterion(DType k_dtype, DType v_dtype) {
    if (k_dtype != v_dtype) { return kAttentionMixedCriterion; }
    return k_dtype == DType::BF16 ? kAttentionBf16Criterion : kAttentionInt8Criterion;
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

std::string case_label(const char* entry, const Geometry& geometry, DType k_dtype, DType v_dtype,
                       const AttentionCase& test_case) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(k_dtype, v_dtype) +
           " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max);
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

void zero_cached_group(HostCache& cache, bool key, std::int32_t group) {
    const DType dtype = key ? cache.k_dtype : cache.v_dtype;
    if (dtype != DType::I8) return;
    constexpr std::int32_t head     = 0;
    constexpr std::int32_t position = 0;
    const std::int32_t d             = group * kQuantGroup;
    const std::size_t code =
        cache_index(cache.geometry, cache.padded_context, head, position, d);
    auto& codes = key ? cache.k_i8 : cache.v_i8;
    std::fill_n(codes.begin() + static_cast<std::ptrdiff_t>(code), kQuantGroup, 0);
    auto& scales = key ? cache.k_scale : cache.v_scale;
    scales[scale_index(cache.geometry, cache.padded_context, head, position, group)] = 0;
}

int run_append_case(const Geometry& geometry, DType k_dtype, DType v_dtype, std::uint32_t seed) {
    constexpr std::int32_t tokens      = 3;
    constexpr std::int32_t base        = 5;
    constexpr std::int32_t max_context = 16;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.kv_heads) * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    const std::vector<std::uint16_t> k_bits   = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits   = to_bf16_bits(v);
    const std::vector<std::int32_t> positions = {base, base + 1, base + 2};

    const HostCache initial = make_cache(geometry, k_dtype, v_dtype, max_context, seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    DeviceCache cache(initial);

    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpositions(positions.size() * sizeof(std::int32_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dpositions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dpositions.data(), DType::I32, {tokens});

    ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
    cuda_synchronize();

    const std::string label =
        std::string("gqa_kv_append ") + geometry.name + " " + cache_name(k_dtype, v_dtype);
    int failures = verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dpositions, positions);
    failures += cache.verify_guards(label);
    return failures;
}

int run_a1_case(const Geometry& geometry, DType k_dtype, DType v_dtype,
                 const AttentionCase& test_case, bool graph_replay = false) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache initial =
        make_cache(geometry, k_dtype, v_dtype, max_context, test_case.seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);
    DeviceCache cache(initial);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_bytes(geometry.q_heads, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};

    if (graph_replay) {
        cudaStream_t stream        = nullptr;
        cudaGraph_t graph          = nullptr;
        cudaGraphExec_t executable = nullptr;
        cuda_check(cudaStreamCreate(&stream), "create mixed GQA graph stream");
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                   "begin mixed GQA graph capture");
        ops::gqa_attention(tq, tk, tv, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                           stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "end mixed GQA graph capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                   "instantiate mixed GQA graph");
        cuda_check(cudaGraphLaunch(executable, stream), "launch mixed GQA graph first replay");
        cuda_check(cudaGraphLaunch(executable, stream), "launch mixed GQA graph second replay");
        cuda_synchronize(stream);
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    } else {
        ops::gqa_attention(tq, tk, tv, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                           nullptr);
        cuda_synchronize();
    }

    std::string label = case_label("gqa_attention", geometry, k_dtype, v_dtype, test_case);
    if (graph_replay) label += " graph-replay";
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(k_dtype, v_dtype));
    failures += verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, DType k_dtype, DType v_dtype,
                const AttentionCase& test_case) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    HostCache cache_host = make_cache(geometry, k_dtype, v_dtype, max_context, test_case.seed + 10u);
    zero_cached_group(cache_host, true, 0);
    zero_cached_group(cache_host, false, 1);
    const std::vector<double> reference = ideal_attention(q, cache_host, positions);
    DeviceCache cache(cache_host);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_bytes(geometry.q_heads, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    const std::string label =
        case_label("gqa_attention_cached", geometry, k_dtype, v_dtype, test_case);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(k_dtype, v_dtype));
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    failures += cache.verify_guards(label);
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    constexpr struct {
        DType k;
        DType v;
    } formats[] = {
        {DType::BF16, DType::BF16},
        {DType::I8, DType::I8},
        {DType::BF16, DType::I8},
        {DType::I8, DType::BF16},
    };
    for (const auto format : formats) {
        failures += run_append_case(geometry, format.k, format.v, 100u + geometry.q_heads);

        const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u}, {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {65, 0, 65, 205u},
        };
        for (const AttentionCase& test_case : a1_cases) {
            failures += run_a1_case(geometry, format.k, format.v, test_case);
        }

        const AttentionCase a3_cases[] = {
            {1, 31, 32, 301u},
            {7, 17, 512, 302u},
            {17, 31, 48, 303u},
        };
        for (const AttentionCase& test_case : a3_cases) {
            failures += run_a3_case(geometry, format.k, format.v, test_case);
        }

        if (format.k != format.v) {
            // Exercise real 1024-key prompt work and replay the captured mixed append/attention.
            failures +=
                run_a1_case(geometry, format.k, format.v, {16, 1008, 1024, 350u}, true);
        }

        if (geometry.q_heads == 16) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures += run_a1_case(geometry, format.k, format.v, {7, 17, 513, 401u});
            failures += run_a3_case(geometry, format.k, format.v, {7, 17, 513, 402u});
            failures += run_a3_case(geometry, format.k, format.v, {16, 17, 1024, 403u});
            failures += run_a3_case(geometry, format.k, format.v, {16, 17, 1025, 404u});
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " gqa_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}
