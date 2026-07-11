// Correctness + coverage for gqa_attention prefill/decode, against the frozen
// op-test standard (docs/kernel-development.md): fp64 golden from
// bf16-rounded q/k/v/cache inputs, device scalar pos for decode, composite
// tolerance attention_bf16.
#include "qus/core/arena.h"
#include "qus/core/kv_cache.h"
#include "qus/kernels/gqa_attention.h"
#include "model/gqa_prompt_ops.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

constexpr std::int32_t kHeadDim          = 256;
constexpr std::int32_t kQHeads           = 24;
constexpr std::int32_t kKVHeads          = 4;
constexpr std::int32_t kKvQuantGroup     = 64;
constexpr std::int32_t kKvQuantGroups    = kHeadDim / kKvQuantGroup;
constexpr float kScale                   = 0.0625f;
constexpr std::size_t kGqaWorkspaceBytes = 96ULL * 1024ULL * 1024ULL;

enum class DecodeInputMode { Random, Stress };

std::size_t q_index(std::int32_t q_head, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(q_head);
}

std::size_t q_index(std::int32_t q_head, std::int32_t d, std::int32_t token) {
    return q_index(q_head, d) + static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(kQHeads) *
                                    static_cast<std::size_t>(token);
}

std::size_t kv_tensor_index(std::int32_t kv_head, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kv_head);
}

std::size_t kv_tensor_index(std::int32_t kv_head, std::int32_t d, std::int32_t token) {
    return kv_tensor_index(kv_head, d) + static_cast<std::size_t>(kHeadDim) *
                                             static_cast<std::size_t>(kKVHeads) *
                                             static_cast<std::size_t>(token);
}

std::int32_t align_up_128(std::int32_t value) { return ((value + 127) / 128) * 128; }

std::size_t cache_index(std::int32_t kv_head, std::int32_t d, std::int32_t position,
                        std::int32_t padded_context) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(kv_head));
}

std::size_t cache_elements(std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(kKVHeads);
}

std::size_t scale_index(std::int32_t kv_head, std::int32_t group, std::int32_t position,
                        std::int32_t padded_context) {
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kKvQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(kv_head));
}

std::size_t scale_elements(std::int32_t padded_context) {
    return static_cast<std::size_t>(kKvQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(kKVHeads);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& h) {
    std::vector<std::uint16_t> b(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) b[i] = f32_to_bf16(h[i]);
    return b;
}

std::uint16_t f32_to_f16_bits(float f) {
    std::uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::uint32_t mant       = x & 0x007fffffu;
    const int exp            = static_cast<int>((x >> 23) & 0xffu) - 127 + 15;

    if (((x >> 23) & 0xffu) == 0xffu) {
        if (mant == 0) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    }
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (exp <= 0) {
        if (exp < -10) { return static_cast<std::uint16_t>(sign); }
        mant |= 0x00800000u;
        const int shift              = 1 - exp;
        const int total_shift        = shift + 13;
        const std::uint32_t half_msb = 1u << (total_shift - 1);
        const std::uint32_t mask     = (1u << total_shift) - 1u;
        std::uint32_t half_mant      = mant >> total_shift;
        const std::uint32_t rem      = mant & mask;
        if (rem > half_msb || (rem == half_msb && (half_mant & 1u) != 0u)) { ++half_mant; }
        return static_cast<std::uint16_t>(sign | half_mant);
    }

    mant += 0x00000fffu + ((mant >> 13) & 1u);
    std::uint32_t half_exp = static_cast<std::uint32_t>(exp);
    if ((mant & 0x00800000u) != 0u) {
        mant = 0;
        ++half_exp;
    }
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    return static_cast<std::uint16_t>(sign | (half_exp << 10) | (mant >> 13));
}

float f16_bits_to_f32(std::uint16_t h) {
    const int sign = (h & 0x8000u) ? -1 : 1;
    const int exp  = (h >> 10) & 0x1f;
    const int frac = h & 0x03ff;
    if (exp == 0) {
        if (frac == 0) { return sign < 0 ? -0.0f : 0.0f; }
        return static_cast<float>(sign) * std::ldexp(static_cast<float>(frac), -24);
    }
    if (exp == 31) {
        return frac == 0 ? static_cast<float>(sign) * std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(sign) *
           std::ldexp(1.0f + static_cast<float>(frac) / 1024.0f, exp - 15);
}

std::vector<std::uint16_t> from_device_bf16_bits(const void* p, std::size_t n) {
    std::vector<std::uint16_t> b(n);
    cudaMemcpy(b.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return b;
}

std::vector<std::int8_t> from_device_i8(const void* p, std::size_t n) {
    std::vector<std::int8_t> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(std::int8_t), cudaMemcpyDeviceToHost);
    return h;
}

std::vector<std::uint16_t> from_device_u16(const void* p, std::size_t n) {
    std::vector<std::uint16_t> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return h;
}

int check_bits_equal(const char* tag, const std::vector<std::uint16_t>& got,
                     const std::vector<std::uint16_t>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << tag << ": size mismatch got=" << got.size() << " expected=" << expected.size()
                  << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            std::cerr << tag << ": bf16 bit mismatch at " << i << " got=0x" << std::hex << got[i]
                      << " expected=0x" << expected[i] << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

int check_i8_equal(const char* tag, const std::vector<std::int8_t>& got,
                   const std::vector<std::int8_t>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << tag << ": size mismatch got=" << got.size() << " expected=" << expected.size()
                  << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            std::cerr << tag << ": int8 mismatch at " << i << " got=" << static_cast<int>(got[i])
                      << " expected=" << static_cast<int>(expected[i]) << '\n';
            return 1;
        }
    }
    return 0;
}

void cpu_gqa_decode(const std::vector<float>& q, const std::vector<float>& cache_k,
                    const std::vector<float>& cache_v, std::int32_t pos,
                    std::int32_t padded_context, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads), 0.0);
    const std::size_t window = static_cast<std::size_t>(pos) + 1;
    std::vector<double> scores(window);
    std::vector<double> probs(window);

    for (std::int32_t qh = 0; qh < kQHeads; ++qh) {
        const std::int32_t kvh = qh / 6;
        double max_score       = -std::numeric_limits<double>::infinity();
        for (std::int32_t j = 0; j <= pos; ++j) {
            double dot = 0.0;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                dot += static_cast<double>(q[q_index(qh, d)]) *
                       static_cast<double>(cache_k[cache_index(kvh, d, j, padded_context)]);
            }
            const double score                  = dot * static_cast<double>(kScale);
            scores[static_cast<std::size_t>(j)] = score;
            max_score                           = std::max(max_score, score);
        }

        double denom = 0.0;
        for (std::int32_t j = 0; j <= pos; ++j) {
            const double p = std::exp(scores[static_cast<std::size_t>(j)] - max_score);
            probs[static_cast<std::size_t>(j)] = p;
            denom += p;
        }
        for (double& p : probs) { p /= denom; }

        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            double acc = 0.0;
            for (std::int32_t j = 0; j <= pos; ++j) {
                acc += probs[static_cast<std::size_t>(j)] *
                       static_cast<double>(cache_v[cache_index(kvh, d, j, padded_context)]);
            }
            out[q_index(qh, d)] = acc;
        }
    }
}

void cpu_gqa_prefill(const std::vector<float>& q, const std::vector<float>& k,
                     const std::vector<float>& v, const std::vector<float>& initial_cache_k,
                     const std::vector<float>& initial_cache_v, std::int32_t tokens,
                     std::int32_t cache_offset, std::int32_t padded_context,
                     std::vector<float>& expected_cache_k, std::vector<float>& expected_cache_v,
                     std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                   static_cast<std::size_t>(tokens),
               0.0);
    expected_cache_k = initial_cache_k;
    expected_cache_v = initial_cache_v;

    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kKVHeads; ++h) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                expected_cache_k[cache_index(h, d, cache_offset + t, padded_context)] =
                    k[kv_tensor_index(h, d, t)];
                expected_cache_v[cache_index(h, d, cache_offset + t, padded_context)] =
                    v[kv_tensor_index(h, d, t)];
            }
        }
    }

    const std::int32_t cache_end = cache_offset + tokens;
    std::vector<double> scores(static_cast<std::size_t>(cache_end));
    std::vector<double> probs(static_cast<std::size_t>(cache_end));
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t query_abs = cache_offset + t;
        for (std::int32_t qh = 0; qh < kQHeads; ++qh) {
            const std::int32_t kvh = qh / 6;
            double max_score       = -std::numeric_limits<double>::infinity();
            for (std::int32_t j = 0; j <= query_abs; ++j) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[q_index(qh, d, t)]) *
                           static_cast<double>(
                               expected_cache_k[cache_index(kvh, d, j, padded_context)]);
                }
                const double score                  = dot * static_cast<double>(kScale);
                scores[static_cast<std::size_t>(j)] = score;
                max_score                           = std::max(max_score, score);
            }

            double denom = 0.0;
            for (std::int32_t j = 0; j <= query_abs; ++j) {
                const double p = std::exp(scores[static_cast<std::size_t>(j)] - max_score);
                probs[static_cast<std::size_t>(j)] = p;
                denom += p;
            }
            for (std::int32_t j = 0; j <= query_abs; ++j) {
                probs[static_cast<std::size_t>(j)] /= denom;
            }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (std::int32_t j = 0; j <= query_abs; ++j) {
                    acc += probs[static_cast<std::size_t>(j)] *
                           static_cast<double>(
                               expected_cache_v[cache_index(kvh, d, j, padded_context)]);
                }
                out[q_index(qh, d, t)] = acc;
            }
        }
    }
}

void cpu_quantize_group(const std::vector<float>& src, std::size_t src_base,
                        std::vector<std::int8_t>& code, std::size_t code_base,
                        std::vector<std::uint16_t>& scale, std::size_t scale_pos) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kKvQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(src[src_base + static_cast<std::size_t>(i)]));
    }

    const std::uint16_t scale_bits = f32_to_f16_bits(absmax > 0.0f ? absmax / 127.0f : 0.0f);
    scale[scale_pos]               = scale_bits;
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    if (stored_scale == 0.0f) {
        for (std::int32_t i = 0; i < kKvQuantGroup; ++i) {
            code[code_base + static_cast<std::size_t>(i)] = 0;
        }
        return;
    }

    for (std::int32_t i = 0; i < kKvQuantGroup; ++i) {
        int q = static_cast<int>(
            std::nearbyint(src[src_base + static_cast<std::size_t>(i)] / stored_scale));
        q                                             = std::max(-127, std::min(127, q));
        code[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(q);
    }
}

void cpu_quantize_append(const std::vector<float>& k, const std::vector<float>& v,
                         const std::vector<int>& positions, bool positions_are_base,
                         std::int32_t tokens, std::int32_t padded_context,
                         std::vector<std::int8_t>& expected_k, std::vector<std::int8_t>& expected_v,
                         std::vector<std::uint16_t>& expected_k_scale,
                         std::vector<std::uint16_t>& expected_v_scale) {
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t position = positions_are_base ? positions[0] + token : positions[token];
        for (std::int32_t kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (std::int32_t group = 0; group < kKvQuantGroups; ++group) {
                const std::int32_t d        = group * kKvQuantGroup;
                const std::size_t src_off   = kv_tensor_index(kv_head, d, token);
                const std::size_t code_off  = cache_index(kv_head, d, position, padded_context);
                const std::size_t scale_off = scale_index(kv_head, group, position, padded_context);
                cpu_quantize_group(k, src_off, expected_k, code_off, expected_k_scale, scale_off);
                cpu_quantize_group(v, src_off, expected_v, code_off, expected_v_scale, scale_off);
            }
        }
    }
}

std::vector<float> dequantize_cache_bf16(const std::vector<std::int8_t>& code,
                                         const std::vector<std::uint16_t>& scale,
                                         std::int32_t padded_context) {
    std::vector<float> out(code.size(), 0.0f);
    for (std::int32_t kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (std::int32_t position = 0; position < padded_context; ++position) {
            for (std::int32_t group = 0; group < kKvQuantGroups; ++group) {
                const float s =
                    f16_bits_to_f32(scale[scale_index(kv_head, group, position, padded_context)]);
                for (std::int32_t i = 0; i < kKvQuantGroup; ++i) {
                    const std::int32_t d  = group * kKvQuantGroup + i;
                    const std::size_t idx = cache_index(kv_head, d, position, padded_context);
                    out[idx] = bf16_to_f32(f32_to_bf16(static_cast<float>(code[idx]) * s));
                }
            }
        }
    }
    return out;
}

std::vector<float> dequantize_cache_f16(const std::vector<std::int8_t>& code,
                                        const std::vector<std::uint16_t>& scale,
                                        std::int32_t padded_context) {
    std::vector<float> out(code.size(), 0.0f);
    for (std::int32_t kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (std::int32_t position = 0; position < padded_context; ++position) {
            for (std::int32_t group = 0; group < kKvQuantGroups; ++group) {
                const float s =
                    f16_bits_to_f32(scale[scale_index(kv_head, group, position, padded_context)]);
                for (std::int32_t i = 0; i < kKvQuantGroup; ++i) {
                    const std::int32_t d  = group * kKvQuantGroup + i;
                    const std::size_t idx = cache_index(kv_head, d, position, padded_context);
                    out[idx] = f16_bits_to_f32(f32_to_f16_bits(static_cast<float>(code[idx]) * s));
                }
            }
        }
    }
    return out;
}

std::vector<float> dequantize_cache_f32(const std::vector<std::int8_t>& code,
                                        const std::vector<std::uint16_t>& scale,
                                        std::int32_t padded_context) {
    std::vector<float> out(code.size(), 0.0f);
    for (std::int32_t kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (std::int32_t position = 0; position < padded_context; ++position) {
            for (std::int32_t group = 0; group < kKvQuantGroups; ++group) {
                const float s =
                    f16_bits_to_f32(scale[scale_index(kv_head, group, position, padded_context)]);
                for (std::int32_t i = 0; i < kKvQuantGroup; ++i) {
                    const std::int32_t d  = group * kKvQuantGroup + i;
                    const std::size_t idx = cache_index(kv_head, d, position, padded_context);
                    out[idx]              = static_cast<float>(code[idx]) * s;
                }
            }
        }
    }
    return out;
}

std::vector<float> quantize_query_i8_roundtrip(const std::vector<float>& q, std::int32_t tokens) {
    std::vector<float> out(q.size(), 0.0f);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t q_head = 0; q_head < kQHeads; ++q_head) {
            for (std::int32_t group = 0; group < kKvQuantGroups; ++group) {
                const std::int32_t d0 = group * kKvQuantGroup;
                float absmax          = 0.0f;
                for (std::int32_t i = 0; i < kKvQuantGroup; ++i) {
                    absmax = std::max(absmax, std::abs(q[q_index(q_head, d0 + i, token)]));
                }
                const float scale = absmax > 0.0f ? absmax / 127.0f : 0.0f;
                const float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
                for (std::int32_t i = 0; i < kKvQuantGroup; ++i) {
                    const std::size_t idx = q_index(q_head, d0 + i, token);
                    int code = inv > 0.0f ? static_cast<int>(std::nearbyint(q[idx] * inv)) : 0;
                    code     = std::max(-127, std::min(127, code));
                    out[idx] = static_cast<float>(code) * scale;
                }
            }
        }
    }
    return out;
}

int one_int8_prefill_case(std::int32_t tokens, std::uint32_t seed, std::int32_t base = 0) {
    const std::size_t qn =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens);
    const std::size_t kvn =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);
    const std::int32_t total          = base + tokens;
    const std::int32_t padded_context = align_up_128(total);
    const std::size_t code_n          = cache_elements(padded_context);
    const std::size_t scale_n         = scale_elements(padded_context);

    std::vector<float> history_k(static_cast<std::size_t>(kHeadDim) * kKVHeads *
                                 static_cast<std::size_t>(base));
    std::vector<float> history_v(history_k.size());
    std::vector<float> q(qn), k(kvn), v(kvn);
    fill_uniform(history_k, seed + 3000u, -0.25f, 0.25f);
    fill_uniform(history_v, seed + 4000u, -1.0f, 1.0f);
    fill_uniform(q, seed, -0.25f, 0.25f);
    fill_uniform(k, seed + 1000u, -0.25f, 0.25f);
    fill_uniform(v, seed + 2000u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    round_to_bf16(history_k);
    round_to_bf16(history_v);

    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) { positions[static_cast<std::size_t>(t)] = base + t; }
    std::vector<std::int8_t> initial_k(code_n, 0), initial_v(code_n, 0);
    std::vector<std::uint16_t> initial_k_scale(scale_n, 0), initial_v_scale(scale_n, 0);
    if (base > 0) {
        std::vector<int> history_positions(static_cast<std::size_t>(base));
        for (std::int32_t t = 0; t < base; ++t) {
            history_positions[static_cast<std::size_t>(t)] = t;
        }
        cpu_quantize_append(history_k, history_v, history_positions, false, base, padded_context,
                            initial_k, initial_v, initial_k_scale, initial_v_scale);
    }
    std::vector<std::int8_t> expected_k = initial_k, expected_v = initial_v;
    std::vector<std::uint16_t> expected_k_scale = initial_k_scale;
    std::vector<std::uint16_t> expected_v_scale = initial_v_scale;
    cpu_quantize_append(k, v, positions, true, tokens, padded_context, expected_k, expected_v,
                        expected_k_scale, expected_v_scale);
    // Native-S8 QK uses an FP32-scale Q round-trip and exact code*FP16-scale K.
    // FP16 PV consumes code*FP16-scale V after the packed shared-memory FP16 rounding.
    const std::vector<float> cache_k =
        dequantize_cache_f32(expected_k, expected_k_scale, padded_context);
    const std::vector<float> cache_v =
        dequantize_cache_f16(expected_v, expected_v_scale, padded_context);
    std::vector<float> k_roundtrip(kvn), v_roundtrip(kvn);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kKVHeads; ++h) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                k_roundtrip[kv_tensor_index(h, d, t)] =
                    cache_k[cache_index(h, d, base + t, padded_context)];
                v_roundtrip[kv_tensor_index(h, d, t)] =
                    cache_v[cache_index(h, d, base + t, padded_context)];
            }
        }
    }

    const std::vector<float> ref_cache_k =
        dequantize_cache_f32(initial_k, initial_k_scale, padded_context);
    const std::vector<float> ref_cache_v =
        dequantize_cache_f16(initial_v, initial_v_scale, padded_context);
    const std::vector<float> q_roundtrip = quantize_query_i8_roundtrip(q, tokens);
    std::vector<float> ignored_k, ignored_v;
    std::vector<double> ref;
    cpu_gqa_prefill(q_roundtrip, k_roundtrip, v_roundtrip, ref_cache_k, ref_cache_v, tokens, base,
                    padded_context, ignored_k, ignored_v, ref);

    DBuf dq   = to_device_bf16(q);
    DBuf dk   = to_device_bf16(k);
    DBuf dv   = to_device_bf16(v);
    DBuf dpos = to_device_i32(positions);
    DBuf dout(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(kGqaWorkspaceBytes);

    const std::size_t arena_bytes =
        2 * (code_n + 256) + 2 * (scale_n * sizeof(std::uint16_t) + 256) + 4096;
    DeviceArena cache_arena(arena_bytes);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(total), kKVHeads, kHeadDim, DType::I8);
    cudaMemcpy(kv.k[0].data, initial_k.data(), code_n * sizeof(std::int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(kv.v[0].data, initial_v.data(), code_n * sizeof(std::int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(kv.k_scale[0].data, initial_k_scale.data(), scale_n * sizeof(std::uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(kv.v_scale[0].data, initial_v_scale.data(), scale_n * sizeof(std::uint16_t),
               cudaMemcpyHostToDevice);

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(dpos.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    kv.pos = static_cast<std::uint32_t>(base);
    kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    const std::string label =
        "gqa int8 prefill base=" + std::to_string(base) + " T=" + std::to_string(tokens);
    f += verify(label.c_str(), from_device_bf16(dout, qn), ref, Tolerance::attention_bf16());
    f += check_i8_equal((label + " k code").c_str(), from_device_i8(kv.k[0].data, code_n),
                        expected_k);
    f += check_i8_equal((label + " v code").c_str(), from_device_i8(kv.v[0].data, code_n),
                        expected_v);
    f += check_bits_equal((label + " k scale").c_str(),
                          from_device_u16(kv.k_scale[0].data, scale_n), expected_k_scale);
    f += check_bits_equal((label + " v scale").c_str(),
                          from_device_u16(kv.v_scale[0].data, scale_n), expected_v_scale);
    return f;
}

// Fused int8 decode/verify: `base` quantized history tokens already in the cache,
// then `tokens` (1..6) new tokens at [base, base+tokens). The kernel quantizes the
// new tokens into the cache (fused append, no separate kernel) AND reads every key
// (history and diagonal) back from the quantized cache for the int8-native QK. The
// reference therefore applies the int8 round-trip to the new tokens as well.
int one_int8_decode_case(std::int32_t base, std::int32_t tokens, std::uint32_t seed) {
    const std::size_t qn =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens);
    const std::size_t kvn =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);
    const std::int32_t total          = base + tokens;
    const std::int32_t padded_context = align_up_128(total);
    const std::size_t code_n          = cache_elements(padded_context);
    const std::size_t scale_n         = scale_elements(padded_context);

    std::vector<float> history_k(static_cast<std::size_t>(kHeadDim) * kKVHeads *
                                 static_cast<std::size_t>(base));
    std::vector<float> history_v(static_cast<std::size_t>(kHeadDim) * kKVHeads *
                                 static_cast<std::size_t>(base));
    std::vector<float> q(qn), k_new(kvn), v_new(kvn);
    fill_uniform(history_k, seed, -0.25f, 0.25f);
    fill_uniform(history_v, seed + 1000u, -1.0f, 1.0f);
    fill_uniform(q, seed + 2000u, -0.25f, 0.25f);
    fill_uniform(k_new, seed + 3000u, -0.25f, 0.25f);
    fill_uniform(v_new, seed + 4000u, -1.0f, 1.0f);
    round_to_bf16(history_k);
    round_to_bf16(history_v);
    round_to_bf16(q);
    round_to_bf16(k_new);
    round_to_bf16(v_new);

    std::vector<std::int8_t> initial_k(code_n, 0), initial_v(code_n, 0);
    std::vector<std::uint16_t> initial_k_scale(scale_n, 0), initial_v_scale(scale_n, 0);
    if (base > 0) {
        std::vector<int> history_positions(static_cast<std::size_t>(base));
        for (std::int32_t t = 0; t < base; ++t) {
            history_positions[static_cast<std::size_t>(t)] = t;
        }
        cpu_quantize_append(history_k, history_v, history_positions, false, base, padded_context,
                            initial_k, initial_v, initial_k_scale, initial_v_scale);
    }

    std::vector<std::int8_t> expected_k         = initial_k;
    std::vector<std::int8_t> expected_v         = initial_v;
    std::vector<std::uint16_t> expected_k_scale = initial_k_scale;
    std::vector<std::uint16_t> expected_v_scale = initial_v_scale;
    std::vector<int> new_positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        new_positions[static_cast<std::size_t>(t)] = base + t;
    }
    cpu_quantize_append(k_new, v_new, new_positions, false, tokens, padded_context, expected_k,
                        expected_v, expected_k_scale, expected_v_scale);

    // Native s8 QK uses an fp32-scale int8 Q round-trip and exact int8*scale K
    // values. PV consumes V after its int8*fp16-scale value is rounded to bf16.
    // Every key, including the newly appended diagonal, comes back from the cache.
    std::vector<float> full_cache_k =
        dequantize_cache_f32(expected_k, expected_k_scale, padded_context);
    std::vector<float> full_cache_v =
        dequantize_cache_bf16(expected_v, expected_v_scale, padded_context);
    std::vector<float> k_deq_new(kvn), v_deq_new(kvn);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kKVHeads; ++h) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                k_deq_new[kv_tensor_index(h, d, t)] =
                    full_cache_k[cache_index(h, d, base + t, padded_context)];
                v_deq_new[kv_tensor_index(h, d, t)] =
                    full_cache_v[cache_index(h, d, base + t, padded_context)];
            }
        }
    }
    std::vector<float> ref_cache_k =
        dequantize_cache_f32(initial_k, initial_k_scale, padded_context);
    std::vector<float> ref_cache_v =
        dequantize_cache_bf16(initial_v, initial_v_scale, padded_context);
    const std::vector<float> q_roundtrip = quantize_query_i8_roundtrip(q, tokens);
    std::vector<float> ignored_k, ignored_v;
    std::vector<double> ref;
    cpu_gqa_prefill(q_roundtrip, k_deq_new, v_deq_new, ref_cache_k, ref_cache_v, tokens, base,
                    padded_context, ignored_k, ignored_v, ref);

    DBuf dq   = to_device_bf16(q);
    DBuf dk   = to_device_bf16(k_new);
    DBuf dv   = to_device_bf16(v_new);
    DBuf dpos = to_device_i32(new_positions);
    DBuf dout(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(kGqaWorkspaceBytes);

    const std::size_t arena_bytes =
        2 * (code_n + 256) + 2 * (scale_n * sizeof(std::uint16_t) + 256) + 4096;
    DeviceArena cache_arena(arena_bytes);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(total), kKVHeads, kHeadDim, DType::I8);
    cudaMemcpy(kv.k[0].data, initial_k.data(), code_n * sizeof(std::int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(kv.v[0].data, initial_v.data(), code_n * sizeof(std::int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(kv.k_scale[0].data, initial_k_scale.data(), scale_n * sizeof(std::uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(kv.v_scale[0].data, initial_v_scale.data(), scale_n * sizeof(std::uint16_t),
               cudaMemcpyHostToDevice);

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(dpos.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    kv.pos                             = static_cast<std::uint32_t>(base);
    const std::uint32_t initial_kv_pos = kv.pos;
    kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    const std::string label =
        "gqa int8 decode base=" + std::to_string(base) + " T=" + std::to_string(tokens);
    f += verify(label.c_str(), from_device_bf16(dout, qn), ref, Tolerance::attention_bf16());
    f += check_i8_equal((label + " k code").c_str(), from_device_i8(kv.k[0].data, code_n),
                        expected_k);
    f += check_i8_equal((label + " v code").c_str(), from_device_i8(kv.v[0].data, code_n),
                        expected_v);
    f += check_bits_equal((label + " k scale").c_str(),
                          from_device_u16(kv.k_scale[0].data, scale_n), expected_k_scale);
    f += check_bits_equal((label + " v scale").c_str(),
                          from_device_u16(kv.v_scale[0].data, scale_n), expected_v_scale);
    if (kv.pos != initial_kv_pos) {
        std::cerr << label << ": decode op must not advance host KVCache.pos; got " << kv.pos
                  << '\n';
        ++f;
    }
    return f;
}

int one_decode_case(std::int32_t pos, std::uint32_t seed,
                    DecodeInputMode mode = DecodeInputMode::Random) {
    const std::size_t qn              = static_cast<std::size_t>(kHeadDim) * kQHeads;
    const std::size_t kvn             = static_cast<std::size_t>(kHeadDim) * kKVHeads;
    const std::int32_t padded_context = align_up_128(pos + 1);
    const std::size_t cache_n         = cache_elements(padded_context);

    std::vector<float> q(qn), k_new(kvn), v_new(kvn), cache_k(cache_n), cache_v(cache_n);
    if (mode == DecodeInputMode::Stress) {
        std::fill(q.begin(), q.end(), 8.0f);
        std::fill(k_new.begin(), k_new.end(), 8.0f);
        std::fill(cache_k.begin(), cache_k.end(), -8.0f);
        fill_uniform(v_new, seed + 2000u, -8.0f, 8.0f);
        fill_uniform(cache_v, seed + 4000u, -8.0f, 8.0f);
    } else {
        fill_uniform(q, seed, -0.25f, 0.25f);
        fill_uniform(k_new, seed + 1000u, -0.25f, 0.25f);
        fill_uniform(v_new, seed + 2000u, -1.0f, 1.0f);
        fill_uniform(cache_k, seed + 3000u, -0.25f, 0.25f);
        fill_uniform(cache_v, seed + 4000u, -1.0f, 1.0f);
    }
    round_to_bf16(q);
    round_to_bf16(k_new);
    round_to_bf16(v_new);
    round_to_bf16(cache_k);
    round_to_bf16(cache_v);

    std::vector<float> expected_cache_k = cache_k;
    std::vector<float> expected_cache_v = cache_v;
    for (std::int32_t h = 0; h < kKVHeads; ++h) {
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            expected_cache_k[cache_index(h, d, pos, padded_context)] = k_new[kv_tensor_index(h, d)];
            expected_cache_v[cache_index(h, d, pos, padded_context)] = v_new[kv_tensor_index(h, d)];
        }
    }

    std::vector<double> ref;
    cpu_gqa_decode(q, expected_cache_k, expected_cache_v, pos, padded_context, ref);

    DBuf dq   = to_device_bf16(q);
    DBuf dk   = to_device_bf16(k_new);
    DBuf dv   = to_device_bf16(v_new);
    DBuf dpos = to_device_i32(std::vector<int>{pos});
    DBuf dout(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(kGqaWorkspaceBytes);

    const std::size_t layer_bytes = cache_n * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(pos + 1), kKVHeads, kHeadDim,
               DType::BF16);
    const std::vector<std::uint16_t> cache_k_bits = bf16_bits(cache_k);
    const std::vector<std::uint16_t> cache_v_bits = bf16_bits(cache_v);
    cudaMemcpy(kv.k[0].data, cache_k_bits.data(), layer_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(kv.v[0].data, cache_v_bits.data(), layer_bytes, cudaMemcpyHostToDevice);

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, 1});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor tpos(dpos.p, DType::I32, {1});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, 1});

    kv.pos                             = static_cast<std::uint32_t>(pos);
    const std::uint32_t initial_kv_pos = kv.pos;
    kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, nullptr);
    cudaDeviceSynchronize();

    int f             = 0;
    std::string label = "gqa decode pos=" + std::to_string(pos);
    if (mode == DecodeInputMode::Stress) { label += " stress"; }
    f += verify(label.c_str(), from_device_bf16(dout, qn), ref, Tolerance::attention_bf16());
    f +=
        check_bits_equal((label + " k append").c_str(),
                         from_device_bf16_bits(kv.k[0].data, cache_n), bf16_bits(expected_cache_k));
    f +=
        check_bits_equal((label + " v append").c_str(),
                         from_device_bf16_bits(kv.v[0].data, cache_n), bf16_bits(expected_cache_v));
    if (kv.pos != initial_kv_pos) {
        std::cerr << label << ": decode op must not advance host KVCache.pos; got " << kv.pos
                  << '\n';
        ++f;
    }
    return f;
}

int one_prefill_case(std::int32_t tokens, std::uint32_t seed, std::int32_t cache_offset) {
    const std::size_t qn =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens);
    const std::size_t kvn =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);
    const std::int32_t padded_context = align_up_128(cache_offset + tokens);
    const std::size_t cache_n         = cache_elements(padded_context);

    std::vector<float> q(qn), k(kvn), v(kvn);
    fill_uniform(q, seed, -0.25f, 0.25f);
    fill_uniform(k, seed + 1000u, -0.25f, 0.25f);
    fill_uniform(v, seed + 2000u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    std::vector<float> cache_k(cache_n), cache_v(cache_n);
    fill_uniform(cache_k, seed + 3000u, -0.25f, 0.25f);
    fill_uniform(cache_v, seed + 4000u, -1.0f, 1.0f);
    round_to_bf16(cache_k);
    round_to_bf16(cache_v);

    std::vector<float> expected_cache_k, expected_cache_v;
    std::vector<double> ref;
    cpu_gqa_prefill(q, k, v, cache_k, cache_v, tokens, cache_offset, padded_context,
                    expected_cache_k, expected_cache_v, ref);

    DBuf dq = to_device_bf16(q);
    DBuf dk = to_device_bf16(k);
    DBuf dv = to_device_bf16(v);
    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        positions[static_cast<std::size_t>(t)] = cache_offset + t;
    }
    DBuf dpos = to_device_i32(positions);
    DBuf dout = DBuf(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(kGqaWorkspaceBytes);

    const std::size_t layer_bytes = cache_n * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(cache_offset + tokens), kKVHeads,
               kHeadDim, DType::BF16);
    const std::vector<std::uint16_t> cache_k_bits = bf16_bits(cache_k);
    const std::vector<std::uint16_t> cache_v_bits = bf16_bits(cache_v);
    cudaMemcpy(kv.k[0].data, cache_k_bits.data(), layer_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(kv.v[0].data, cache_v_bits.data(), layer_bytes, cudaMemcpyHostToDevice);

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(dpos.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    const std::uint32_t initial_kv_pos = 777u;
    kv.pos                             = initial_kv_pos;
    kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    std::string label =
        "gqa prefill offset=" + std::to_string(cache_offset) + " T=" + std::to_string(tokens);
    f += verify(label.c_str(), from_device_bf16(dout, qn), ref, Tolerance::attention_bf16());
    f += check_bits_equal((label + " k fill").c_str(), from_device_bf16_bits(kv.k[0].data, cache_n),
                          bf16_bits(expected_cache_k));
    f += check_bits_equal((label + " v fill").c_str(), from_device_bf16_bits(kv.v[0].data, cache_n),
                          bf16_bits(expected_cache_v));
    if (kv.pos != initial_kv_pos) {
        std::cerr << label << ": attention op must not advance host KVCache.pos; got " << kv.pos
                  << '\n';
        ++f;
    }
    return f;
}

int split_api_parity_case(DType cache_dtype, std::int32_t tokens, std::int32_t base,
                          std::uint32_t seed, bool final_row_only = false) {
    const std::size_t qn =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens);
    const std::size_t kvn =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);
    const std::int32_t total          = base + tokens;
    const std::int32_t padded_context = align_up_128(total);
    const std::size_t code_n          = cache_elements(padded_context);
    const std::size_t scale_n         = scale_elements(padded_context);

    std::vector<float> q(qn), k(kvn), v(kvn);
    fill_uniform(q, seed, -0.25f, 0.25f);
    fill_uniform(k, seed + 1000u, -0.25f, 0.25f);
    fill_uniform(v, seed + 2000u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) { positions[static_cast<std::size_t>(t)] = base + t; }

    DBuf dq         = to_device_bf16(q);
    DBuf dk         = to_device_bf16(k);
    DBuf dv         = to_device_bf16(v);
    DBuf dpos       = to_device_i32(positions);
    DBuf dout_full  = DBuf(qn * sizeof(std::uint16_t));
    DBuf dout_split = DBuf(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(kGqaWorkspaceBytes);

    const std::size_t arena_bytes =
        cache_dtype == DType::BF16
            ? 2 * (code_n * sizeof(std::uint16_t) + 256) + 4096
            : 2 * (code_n + 256) + 2 * (scale_n * sizeof(std::uint16_t) + 256) + 4096;
    DeviceArena full_arena(arena_bytes);
    DeviceArena split_arena(arena_bytes);
    KVCache full_cache(full_arena, 1, static_cast<std::uint32_t>(total), kKVHeads, kHeadDim,
                       cache_dtype);
    KVCache split_cache(split_arena, 1, static_cast<std::uint32_t>(total), kKVHeads, kHeadDim,
                        cache_dtype);
    const std::size_t code_bytes = code_n * dtype_size(cache_dtype);
    for (KVCache* cache : {&full_cache, &split_cache}) {
        cudaMemset(cache->k[0].data, 0, code_bytes);
        cudaMemset(cache->v[0].data, 0, code_bytes);
        if (cache_dtype == DType::I8) {
            cudaMemset(cache->k_scale[0].data, 0, scale_n * sizeof(std::uint16_t));
            cudaMemset(cache->v_scale[0].data, 0, scale_n * sizeof(std::uint16_t));
        }
        cache->pos = static_cast<std::uint32_t>(base);
    }

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(dpos.p, DType::I32, {tokens});
    Tensor tout_full(dout_full.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    const std::size_t q_col_elems   = static_cast<std::size_t>(kHeadDim) * kQHeads;
    auto* q_split_ptr               = static_cast<std::uint16_t*>(dq.p);
    auto* pos_split_ptr             = static_cast<std::int32_t*>(dpos.p);
    const std::int32_t split_tokens = final_row_only ? 1 : tokens;
    if (final_row_only) {
        q_split_ptr += static_cast<std::size_t>(tokens - 1) * q_col_elems;
        pos_split_ptr += tokens - 1;
    }
    Tensor tq_split(q_split_ptr, DType::BF16, {kHeadDim, kQHeads, split_tokens});
    Tensor tpos_split(pos_split_ptr, DType::I32, {split_tokens});
    Tensor tout_split(dout_split.p, DType::BF16, {kHeadDim, kQHeads, split_tokens});

    kernels::gqa_attention(tq, tk, tv, tpos, kScale, full_cache, 0, ws, tout_full, nullptr);
    kernels::gqa_kv_append(tk, tv, tpos, split_cache, 0, nullptr);
    kernels::gqa_attention_cached(tq_split, tpos_split, kScale, split_cache, 0, tout_split,
                                  nullptr);
    cudaDeviceSynchronize();

    const std::string dtype_label = cache_dtype == DType::BF16 ? "bf16" : "int8";
    const std::string label = "gqa split API " + dtype_label + " base=" + std::to_string(base) +
                              " T=" + std::to_string(tokens) + (final_row_only ? " final-row" : "");
    std::vector<double> expected = from_device_bf16(dout_full, qn);
    if (final_row_only) {
        expected.erase(expected.begin(), expected.end() - static_cast<std::ptrdiff_t>(q_col_elems));
    }
    int f = 0;
    f += verify(label.c_str(), from_device_bf16(dout_split, q_col_elems * split_tokens), expected,
                Tolerance::attention_bf16());
    if (cache_dtype == DType::BF16) {
        f += check_bits_equal((label + " k cache").c_str(),
                              from_device_bf16_bits(split_cache.k[0].data, code_n),
                              from_device_bf16_bits(full_cache.k[0].data, code_n));
        f += check_bits_equal((label + " v cache").c_str(),
                              from_device_bf16_bits(split_cache.v[0].data, code_n),
                              from_device_bf16_bits(full_cache.v[0].data, code_n));
    } else {
        f += check_i8_equal((label + " k code").c_str(),
                            from_device_i8(split_cache.k[0].data, code_n),
                            from_device_i8(full_cache.k[0].data, code_n));
        f += check_i8_equal((label + " v code").c_str(),
                            from_device_i8(split_cache.v[0].data, code_n),
                            from_device_i8(full_cache.v[0].data, code_n));
        f += check_bits_equal((label + " k scale").c_str(),
                              from_device_u16(split_cache.k_scale[0].data, scale_n),
                              from_device_u16(full_cache.k_scale[0].data, scale_n));
        f += check_bits_equal((label + " v scale").c_str(),
                              from_device_u16(split_cache.v_scale[0].data, scale_n),
                              from_device_u16(full_cache.v_scale[0].data, scale_n));
    }
    return f;
}

int one_prefill_decode_consistency_case(std::int32_t tokens, std::uint32_t seed) {
    const std::size_t q_prefill_n =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens);
    const std::size_t kv_prefill_n =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);
    constexpr std::size_t q_decode_n =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads);
    constexpr std::size_t kv_decode_n =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kKVHeads);
    const std::int32_t padded_context = align_up_128(tokens + 1);
    const std::size_t cache_n         = cache_elements(padded_context);

    std::vector<float> q_prefill(q_prefill_n), k_prefill(kv_prefill_n), v_prefill(kv_prefill_n);
    std::vector<float> q_decode(q_decode_n), k_decode(kv_decode_n), v_decode(kv_decode_n);
    fill_uniform(q_prefill, seed, -0.25f, 0.25f);
    fill_uniform(k_prefill, seed + 1000u, -0.25f, 0.25f);
    fill_uniform(v_prefill, seed + 2000u, -1.0f, 1.0f);
    fill_uniform(q_decode, seed + 3000u, -0.25f, 0.25f);
    fill_uniform(k_decode, seed + 4000u, -0.25f, 0.25f);
    fill_uniform(v_decode, seed + 5000u, -1.0f, 1.0f);
    round_to_bf16(q_prefill);
    round_to_bf16(k_prefill);
    round_to_bf16(v_prefill);
    round_to_bf16(q_decode);
    round_to_bf16(k_decode);
    round_to_bf16(v_decode);

    std::vector<float> expected_cache_k(cache_n), expected_cache_v(cache_n);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kKVHeads; ++h) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                expected_cache_k[cache_index(h, d, t, padded_context)] =
                    k_prefill[kv_tensor_index(h, d, t)];
                expected_cache_v[cache_index(h, d, t, padded_context)] =
                    v_prefill[kv_tensor_index(h, d, t)];
            }
        }
    }
    for (std::int32_t h = 0; h < kKVHeads; ++h) {
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            expected_cache_k[cache_index(h, d, tokens, padded_context)] =
                k_decode[kv_tensor_index(h, d)];
            expected_cache_v[cache_index(h, d, tokens, padded_context)] =
                v_decode[kv_tensor_index(h, d)];
        }
    }

    std::vector<double> ref;
    cpu_gqa_decode(q_decode, expected_cache_k, expected_cache_v, tokens, padded_context, ref);

    DBuf dq_prefill   = to_device_bf16(q_prefill);
    DBuf dk_prefill   = to_device_bf16(k_prefill);
    DBuf dv_prefill   = to_device_bf16(v_prefill);
    DBuf dout_prefill = DBuf(q_prefill_n * sizeof(std::uint16_t));
    DBuf dq_decode    = to_device_bf16(q_decode);
    DBuf dk_decode    = to_device_bf16(k_decode);
    DBuf dv_decode    = to_device_bf16(v_decode);
    std::vector<int> prefill_positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        prefill_positions[static_cast<std::size_t>(t)] = t;
    }
    DBuf dpos_prefill = to_device_i32(prefill_positions);
    DBuf dpos         = to_device_i32(std::vector<int>{tokens});
    DBuf dout_decode  = DBuf(q_decode_n * sizeof(std::uint16_t));
    WorkspaceArena ws(kGqaWorkspaceBytes);

    const std::size_t layer_bytes = cache_n * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(tokens + 1), kKVHeads, kHeadDim,
               DType::BF16);
    cudaMemset(kv.k[0].data, 0, layer_bytes);
    cudaMemset(kv.v[0].data, 0, layer_bytes);

    Tensor tq_prefill(dq_prefill.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk_prefill(dk_prefill.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv_prefill(dv_prefill.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos_prefill(dpos_prefill.p, DType::I32, {tokens});
    Tensor tout_prefill(dout_prefill.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tq_decode(dq_decode.p, DType::BF16, {kHeadDim, kQHeads, 1});
    Tensor tk_decode(dk_decode.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor tv_decode(dv_decode.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor tpos(dpos.p, DType::I32, {1});
    Tensor tout_decode(dout_decode.p, DType::BF16, {kHeadDim, kQHeads, 1});

    kernels::gqa_attention(tq_prefill, tk_prefill, tv_prefill, tpos_prefill, kScale, kv, 0, ws,
                           tout_prefill, nullptr);
    kv.pos                             = static_cast<std::uint32_t>(tokens);
    const std::uint32_t initial_kv_pos = kv.pos;
    kernels::gqa_attention(tq_decode, tk_decode, tv_decode, tpos, kScale, kv, 0, ws, tout_decode,
                           nullptr);
    cudaDeviceSynchronize();

    int f             = 0;
    std::string label = "gqa prefill->decode T=" + std::to_string(tokens);
    f += verify(label.c_str(), from_device_bf16(dout_decode, q_decode_n), ref,
                Tolerance::attention_bf16());
    f +=
        check_bits_equal((label + " k cache").c_str(), from_device_bf16_bits(kv.k[0].data, cache_n),
                         bf16_bits(expected_cache_k));
    f +=
        check_bits_equal((label + " v cache").c_str(), from_device_bf16_bits(kv.v[0].data, cache_n),
                         bf16_bits(expected_cache_v));
    if (kv.pos != initial_kv_pos) {
        std::cerr << label << ": attention ops must not advance host KVCache.pos; got " << kv.pos
                  << '\n';
        ++f;
    }
    return f;
}

int check_output_prefix_equal(const char* tag, const std::vector<std::uint16_t>& a,
                              const std::vector<std::uint16_t>& b, std::int32_t tokens) {
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t qh = 0; qh < kQHeads; ++qh) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t idx = q_index(qh, d, t);
                if (a[idx] != b[idx]) {
                    std::cerr << tag << ": prefix output changed at token=" << t << " qh=" << qh
                              << " d=" << d << " got=0x" << std::hex << b[idx] << " expected=0x"
                              << a[idx] << std::dec << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}

int one_future_token_isolation_case() {
    constexpr std::int32_t tokens = 6;
    constexpr std::int32_t base   = 17;
    const std::size_t qn =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens);
    const std::size_t kvn =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);
    const std::int32_t padded_context = align_up_128(base + tokens);
    const std::size_t cache_n         = cache_elements(padded_context);

    std::vector<float> q(qn), k_a(kvn), v_a(kvn), k_b, v_b, cache_k(cache_n), cache_v(cache_n);
    fill_uniform(q, 601u, -0.25f, 0.25f);
    fill_uniform(k_a, 1601u, -0.25f, 0.25f);
    fill_uniform(v_a, 2601u, -1.0f, 1.0f);
    fill_uniform(cache_k, 3601u, -0.25f, 0.25f);
    fill_uniform(cache_v, 4601u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k_a);
    round_to_bf16(v_a);
    round_to_bf16(cache_k);
    round_to_bf16(cache_v);
    k_b = k_a;
    v_b = v_a;
    for (std::int32_t h = 0; h < kKVHeads; ++h) {
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            k_b[kv_tensor_index(h, d, tokens - 1)] = 6.0f;
            v_b[kv_tensor_index(h, d, tokens - 1)] = -6.0f;
        }
    }
    round_to_bf16(k_b);
    round_to_bf16(v_b);

    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) { positions[static_cast<std::size_t>(t)] = base + t; }

    auto run = [&](const std::vector<float>& k, const std::vector<float>& v) {
        DBuf dq   = to_device_bf16(q);
        DBuf dk   = to_device_bf16(k);
        DBuf dv   = to_device_bf16(v);
        DBuf dpos = to_device_i32(positions);
        DBuf dout(qn * sizeof(std::uint16_t));
        WorkspaceArena ws(kGqaWorkspaceBytes);

        const std::size_t layer_bytes = cache_n * sizeof(std::uint16_t);
        DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
        KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(base + tokens), kKVHeads, kHeadDim,
                   DType::BF16);
        const std::vector<std::uint16_t> cache_k_bits = bf16_bits(cache_k);
        const std::vector<std::uint16_t> cache_v_bits = bf16_bits(cache_v);
        cudaMemcpy(kv.k[0].data, cache_k_bits.data(), layer_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(kv.v[0].data, cache_v_bits.data(), layer_bytes, cudaMemcpyHostToDevice);

        Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
        Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
        Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
        Tensor tpos(dpos.p, DType::I32, {tokens});
        Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});
        kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, nullptr);
        cudaDeviceSynchronize();
        return from_device_bf16_bits(dout.p, qn);
    };

    const std::vector<std::uint16_t> baseline = run(k_a, v_a);
    const std::vector<std::uint16_t> changed  = run(k_b, v_b);
    return check_output_prefix_equal("gqa small-T future isolation", baseline, changed, tokens - 1);
}

int one_graph_relaunch_positions_case() {
    constexpr std::int32_t tokens      = 2;
    constexpr std::int32_t base0       = 1;
    constexpr std::int32_t base1       = 3;
    constexpr std::int32_t max_context = base1 + tokens;
    const std::size_t qn =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens);
    const std::size_t kvn =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);
    const std::int32_t padded_context = align_up_128(max_context);
    const std::size_t cache_n         = cache_elements(padded_context);

    std::vector<float> q(qn), k(kvn), v(kvn), cache_k(cache_n), cache_v(cache_n);
    fill_uniform(q, 701u, -0.25f, 0.25f);
    fill_uniform(k, 1701u, -0.25f, 0.25f);
    fill_uniform(v, 2701u, -1.0f, 1.0f);
    fill_uniform(cache_k, 3701u, -0.25f, 0.25f);
    fill_uniform(cache_v, 4701u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    round_to_bf16(cache_k);
    round_to_bf16(cache_v);

    DBuf dq   = to_device_bf16(q);
    DBuf dk   = to_device_bf16(k);
    DBuf dv   = to_device_bf16(v);
    DBuf dpos = to_device_i32(std::vector<int>{base0, base0 + 1});
    DBuf dout(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(kGqaWorkspaceBytes);

    const std::size_t layer_bytes = cache_n * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(max_context), kKVHeads, kHeadDim,
               DType::BF16);
    const std::vector<std::uint16_t> cache_k_bits = bf16_bits(cache_k);
    const std::vector<std::uint16_t> cache_v_bits = bf16_bits(cache_v);
    cudaMemcpy(kv.k[0].data, cache_k_bits.data(), layer_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(kv.v[0].data, cache_v_bits.data(), layer_bytes, cudaMemcpyHostToDevice);

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(dpos.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    kernels::gqa_attention(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    int f = 0;
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    const int next_positions[2] = {base1, base1 + 1};
    cudaMemcpy(dpos.p, next_positions, sizeof(next_positions), cudaMemcpyHostToDevice);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> expected_cache_k = cache_k;
    std::vector<float> expected_cache_v = cache_v;
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kKVHeads; ++h) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                expected_cache_k[cache_index(h, d, base0 + t, padded_context)] =
                    k[kv_tensor_index(h, d, t)];
                expected_cache_v[cache_index(h, d, base0 + t, padded_context)] =
                    v[kv_tensor_index(h, d, t)];
                expected_cache_k[cache_index(h, d, base1 + t, padded_context)] =
                    k[kv_tensor_index(h, d, t)];
                expected_cache_v[cache_index(h, d, base1 + t, padded_context)] =
                    v[kv_tensor_index(h, d, t)];
            }
        }
    }
    f +=
        check_bits_equal("gqa graph relaunch k cache", from_device_bf16_bits(kv.k[0].data, cache_n),
                         bf16_bits(expected_cache_k));
    f +=
        check_bits_equal("gqa graph relaunch v cache", from_device_bf16_bits(kv.v[0].data, cache_n),
                         bf16_bits(expected_cache_v));

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return f;
}

template <typename Fn>
int expect_invalid(const char* label, Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int validation_checks() {
    constexpr std::int32_t pos = 1;
    const std::size_t qn       = static_cast<std::size_t>(kHeadDim) * kQHeads;
    const std::size_t kvn      = static_cast<std::size_t>(kHeadDim) * kKVHeads;
    DBuf dq(qn * sizeof(std::uint16_t));
    DBuf dk(kvn * sizeof(std::uint16_t));
    DBuf dv(kvn * sizeof(std::uint16_t));
    DBuf dpos = to_device_i32(std::vector<int>{pos});
    DBuf dout(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(kGqaWorkspaceBytes);
    const std::int32_t padded_context = align_up_128(pos + 1);
    const std::size_t layer_bytes     = cache_elements(padded_context) * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, pos + 1, kKVHeads, kHeadDim, DType::BF16);
    kv.pos = static_cast<std::uint32_t>(pos);

    Tensor q(dq.p, DType::BF16, {kHeadDim, kQHeads, 1});
    Tensor k(dk.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor v(dv.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor scalar_pos(dpos.p, DType::I32, {1});
    Tensor out(dout.p, DType::BF16, {kHeadDim, kQHeads, 1});

    int f = 0;
    f += expect_invalid("gqa q shape", [&] {
        Tensor bad(dq.p, DType::BF16, {kHeadDim, kQHeads, 2});
        kernels::gqa_attention(bad, k, v, scalar_pos, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa k shape", [&] {
        Tensor bad(dk.p, DType::BF16, {kHeadDim, kKVHeads, 2});
        kernels::gqa_attention(q, bad, v, scalar_pos, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa zero T", [&] {
        Tensor bad = q;
        bad.ne[2]  = 0;
        kernels::gqa_attention(bad, k, v, scalar_pos, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa positions shape", [&] {
        Tensor bad(dpos.p, DType::I32, {2});
        kernels::gqa_attention(q, k, v, bad, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa positions dtype", [&] {
        Tensor bad = scalar_pos;
        bad.dtype  = DType::BF16;
        kernels::gqa_attention(q, k, v, bad, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa null positions", [&] {
        Tensor bad(nullptr, DType::I32, {1});
        kernels::gqa_attention(q, k, v, bad, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa T exceeds max_context", [&] {
        Tensor q2(dq.p, DType::BF16, {kHeadDim, kQHeads, 2});
        Tensor k2(dk.p, DType::BF16, {kHeadDim, kKVHeads, 2});
        Tensor v2(dv.p, DType::BF16, {kHeadDim, kKVHeads, 2});
        Tensor pos2(dpos.p, DType::I32, {2});
        Tensor out2(dout.p, DType::BF16, {kHeadDim, kQHeads, 2});
        DeviceArena tiny_cache_arena(
            2 * (cache_elements(align_up_128(1)) * sizeof(std::uint16_t) + 256) + 4096);
        KVCache tiny_kv(tiny_cache_arena, 1, 1, kKVHeads, kHeadDim, DType::BF16);
        kernels::gqa_attention(q2, k2, v2, pos2, kScale, tiny_kv, 0, ws, out2, nullptr);
    });
    return f;
}

} // namespace

int main(int argc, char** argv) {
    bool long_decode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--long-decode") {
            long_decode = true;
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            return 2;
        }
    }

    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int f = 0;
    // T<=6 and T>6 use separate native-S8 QK kernels. Cover the prompt path at
    // tile tails and with nonzero history spanning multiple key tiles.
    f += one_int8_decode_case(0, 6, 903u);
    f += one_int8_prefill_case(65, 904u);
    f += one_int8_prefill_case(192, 911u);
    f += one_int8_prefill_case(65, 917u, 384);
    f += one_int8_decode_case(17, 1, 905u);
    f += one_int8_decode_case(2882, 1, 906u);
    f += one_int8_decode_case(100, 2, 907u);
    f += one_int8_decode_case(100, 3, 908u);
    f += one_int8_decode_case(100, 4, 909u);
    f += one_int8_decode_case(2048, 4, 910u);
    f += one_int8_decode_case(100, 5, 912u);
    // Protect the short-context launch specializations: full-CTA T=5, the
    // narrow T=6 boundary, and the dynamic-smem Bc=64 path with its split cap.
    f += one_int8_decode_case(256, 5, 914u);
    f += one_int8_decode_case(128, 6, 915u);
    f += one_int8_decode_case(8192, 6, 916u);
    for (std::int32_t tokens = 1; tokens <= 6; ++tokens) {
        f += one_prefill_case(tokens, 100u + static_cast<std::uint32_t>(tokens), 0);
        f += one_prefill_case(tokens, 200u + static_cast<std::uint32_t>(tokens), 1);
        f += one_prefill_case(tokens, 300u + static_cast<std::uint32_t>(tokens), 17);
        f += one_prefill_case(tokens, 400u + static_cast<std::uint32_t>(tokens), 128);
    }
    f += one_prefill_case(7, 107u, 0);
    f += one_prefill_case(128, 113u, 0);
    f += one_prefill_case(512, 127u, 0);
    f += one_prefill_case(128, 137u, 128);
    f += one_prefill_case(512, 139u, 128);
    f += one_prefill_case(65, 149u, 384);
    f += one_prefill_decode_consistency_case(128, 131u);
    f += split_api_parity_case(DType::BF16, 1, 17, 1401u);
    f += split_api_parity_case(DType::BF16, 128, 17, 1402u);
    f += split_api_parity_case(DType::I8, 1, 17, 1403u);
    f += split_api_parity_case(DType::I8, 128, 17, 1404u);
    f += split_api_parity_case(DType::BF16, 1024, 17, 1405u, true);
    f += split_api_parity_case(DType::I8, 1024, 17, 1406u, true);
    f += one_decode_case(1, 11u);
    f += one_decode_case(17, 17u);
    f += one_decode_case(2048, 23u);
    f += one_decode_case(2882, 29u);
    f += one_decode_case(8191, 37u);
    if (long_decode) {
        f += one_decode_case(32768, 41u);
        f += one_int8_decode_case(40800, 1, 913u);
    }
    f += one_decode_case(17, 31u, DecodeInputMode::Stress);
    f += one_future_token_isolation_case();
    f += one_graph_relaunch_positions_case();
    f += validation_checks();

    std::cout << (f ? "FAIL" : "OK") << " gqa_attention correctness\n";
    return f ? 1 : 0;
}
