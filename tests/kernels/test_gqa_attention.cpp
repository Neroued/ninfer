// Correctness + coverage for gqa_attention prefill/decode, against the frozen
// op-test standard (docs/l1-op-test-standard.md): fp64 golden from
// bf16-rounded q/k/v/cache inputs, device scalar pos for decode, composite
// tolerance attention_bf16.
#include "qus/core/arena.h"
#include "qus/core/kv_cache.h"
#include "qus/kernels/gqa_attention.h"
#include "kernels/launcher/gqa_attention.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

int one_int8_quantize_append_case(std::int32_t tokens, std::int32_t max_context,
                                  std::vector<int> positions, bool positions_are_base,
                                  std::uint32_t seed, const char* tag) {
    const std::size_t kvn =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);
    const std::int32_t padded_context = align_up_128(max_context);
    const std::size_t code_n          = cache_elements(padded_context);
    const std::size_t scale_n         = scale_elements(padded_context);

    std::vector<float> k(kvn), v(kvn);
    fill_uniform(k, seed, -0.5f, 0.5f);
    fill_uniform(v, seed + 1000u, -2.0f, 2.0f);

    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < kKVHeads; ++head) {
            for (std::int32_t d = 0; d < kKvQuantGroup; ++d) {
                k[kv_tensor_index(head, d, token)] = 0.0f;
            }
            k[kv_tensor_index(head, kKvQuantGroup, token)]         = 12.0f;
            k[kv_tensor_index(head, kKvQuantGroup + 1, token)]     = -0.125f;
            v[kv_tensor_index(head, 2 * kKvQuantGroup, token)]     = -9.0f;
            v[kv_tensor_index(head, 2 * kKvQuantGroup + 7, token)] = 0.25f;
        }
    }
    round_to_bf16(k);
    round_to_bf16(v);

    std::vector<std::int8_t> expected_k(code_n, static_cast<std::int8_t>(0x5a));
    std::vector<std::int8_t> expected_v(code_n, static_cast<std::int8_t>(0x5b));
    std::vector<std::uint16_t> expected_k_scale(scale_n, 0x6a6au);
    std::vector<std::uint16_t> expected_v_scale(scale_n, 0x6b6bu);
    cpu_quantize_append(k, v, positions, positions_are_base, tokens, padded_context, expected_k,
                        expected_v, expected_k_scale, expected_v_scale);

    DBuf dk   = to_device_bf16(k);
    DBuf dv   = to_device_bf16(v);
    DBuf dpos = to_device_i32(positions);
    const std::size_t arena_bytes =
        2 * (code_n + 256) + 2 * (scale_n * sizeof(std::uint16_t) + 256) + 4096;
    DeviceArena cache_arena(arena_bytes);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(max_context), kKVHeads, kHeadDim,
               DType::I8);
    cudaMemset(kv.k[0].data, 0x5a, kv.k[0].bytes());
    cudaMemset(kv.v[0].data, 0x5b, kv.v[0].bytes());
    cudaMemset(kv.k_scale[0].data, 0x6a, kv.k_scale[0].bytes());
    cudaMemset(kv.v_scale[0].data, 0x6b, kv.v_scale[0].bytes());

    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tpos(dpos.p, DType::I32, {tokens});
    kernels::detail::gqa_attention_kv_quantize_append_launch(tk, tv, tpos, kv, 0,
                                                             positions_are_base, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f += check_i8_equal((std::string(tag) + " k code").c_str(),
                        from_device_i8(kv.k[0].data, code_n), expected_k);
    f += check_i8_equal((std::string(tag) + " v code").c_str(),
                        from_device_i8(kv.v[0].data, code_n), expected_v);
    f += check_bits_equal((std::string(tag) + " k scale").c_str(),
                          from_device_u16(kv.k_scale[0].data, scale_n), expected_k_scale);
    f += check_bits_equal((std::string(tag) + " v scale").c_str(),
                          from_device_u16(kv.v_scale[0].data, scale_n), expected_v_scale);
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
    f += one_int8_quantize_append_case(3, 16, std::vector<int>{5, 6, 7}, true, 901u,
                                       "gqa int8 quant append prefill");
    f += one_int8_quantize_append_case(4, 32, std::vector<int>{2, 5, 9, 10}, false, 902u,
                                       "gqa int8 quant append decode");
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
    f += one_decode_case(1, 11u);
    f += one_decode_case(17, 17u);
    f += one_decode_case(2048, 23u);
    f += one_decode_case(2882, 29u);
    f += one_decode_case(8191, 37u);
    if (long_decode) { f += one_decode_case(32768, 41u); }
    f += one_decode_case(17, 31u, DecodeInputMode::Stress);
    f += one_future_token_isolation_case();
    f += one_graph_relaunch_positions_case();
    f += validation_checks();

    std::cout << (f ? "FAIL" : "OK") << " gqa_attention correctness\n";
    return f ? 1 : 0;
}
