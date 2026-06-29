// Correctness + coverage for gqa_attention prefill/decode, against the frozen
// op-test standard (docs/l1-op-test-standard.md): fp64 golden from
// bf16-rounded q/k/v/cache inputs, device scalar pos for decode, composite
// tolerance attention_bf16.
#include "qus/core/arena.h"
#include "qus/core/kv_cache.h"
#include "qus/kernels/gqa_attention.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads  = 24;
constexpr std::int32_t kKVHeads = 4;
constexpr float kScale          = 0.0625f;
constexpr std::size_t kDecodeWorkspaceBytes = 16ULL * 1024ULL * 1024ULL;

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

std::int32_t align_up_128(std::int32_t value) {
    return ((value + 127) / 128) * 128;
}

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

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& h) {
    std::vector<std::uint16_t> b(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) b[i] = f32_to_bf16(h[i]);
    return b;
}

std::vector<std::uint16_t> from_device_bf16_bits(const void* p, std::size_t n) {
    std::vector<std::uint16_t> b(n);
    cudaMemcpy(b.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return b;
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
                     const std::vector<float>& v, std::int32_t tokens,
                     std::int32_t padded_context,
                     std::vector<float>& expected_cache_k, std::vector<float>& expected_cache_v,
                     std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kQHeads) *
                   static_cast<std::size_t>(tokens),
               0.0);
    expected_cache_k.assign(cache_elements(padded_context), 0.0f);
    expected_cache_v.assign(expected_cache_k.size(), 0.0f);

    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kKVHeads; ++h) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                expected_cache_k[cache_index(h, d, t, padded_context)] =
                    k[kv_tensor_index(h, d, t)];
                expected_cache_v[cache_index(h, d, t, padded_context)] =
                    v[kv_tensor_index(h, d, t)];
            }
        }
    }

    std::vector<double> scores(static_cast<std::size_t>(tokens));
    std::vector<double> probs(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t qh = 0; qh < kQHeads; ++qh) {
            const std::int32_t kvh = qh / 6;
            double max_score       = -std::numeric_limits<double>::infinity();
            for (std::int32_t j = 0; j <= t; ++j) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[q_index(qh, d, t)]) *
                           static_cast<double>(k[kv_tensor_index(kvh, d, j)]);
                }
                const double score                  = dot * static_cast<double>(kScale);
                scores[static_cast<std::size_t>(j)] = score;
                max_score                           = std::max(max_score, score);
            }

            double denom = 0.0;
            for (std::int32_t j = 0; j <= t; ++j) {
                const double p = std::exp(scores[static_cast<std::size_t>(j)] - max_score);
                probs[static_cast<std::size_t>(j)] = p;
                denom += p;
            }
            for (std::int32_t j = 0; j <= t; ++j) { probs[static_cast<std::size_t>(j)] /= denom; }

            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (std::int32_t j = 0; j <= t; ++j) {
                    acc += probs[static_cast<std::size_t>(j)] *
                           static_cast<double>(v[kv_tensor_index(kvh, d, j)]);
                }
                out[q_index(qh, d, t)] = acc;
            }
        }
    }
}

int one_decode_case(std::int32_t pos, std::uint32_t seed,
                    DecodeInputMode mode = DecodeInputMode::Random) {
    const std::size_t qn      = static_cast<std::size_t>(kHeadDim) * kQHeads;
    const std::size_t kvn     = static_cast<std::size_t>(kHeadDim) * kKVHeads;
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
            expected_cache_k[cache_index(h, d, pos, padded_context)] =
                k_new[kv_tensor_index(h, d)];
            expected_cache_v[cache_index(h, d, pos, padded_context)] =
                v_new[kv_tensor_index(h, d)];
        }
    }

    std::vector<double> ref;
    cpu_gqa_decode(q, expected_cache_k, expected_cache_v, pos, padded_context, ref);

    DBuf dq   = to_device_bf16(q);
    DBuf dk   = to_device_bf16(k_new);
    DBuf dv   = to_device_bf16(v_new);
    DBuf dpos = to_device_i32(std::vector<int>{pos});
    DBuf dout(qn * sizeof(std::uint16_t));
    WorkspaceArena ws(kDecodeWorkspaceBytes);

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

    kv.pos = static_cast<std::uint32_t>(pos);
    const std::uint32_t initial_kv_pos = kv.pos;
    kernels::gqa_attention_decode(tq, tk, tv, tpos, kScale, kv, 0, ws, tout, nullptr);
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

int one_prefill_case(std::int32_t tokens, std::uint32_t seed) {
    const std::size_t qn =
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens);
    const std::size_t kvn =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * static_cast<std::size_t>(tokens);

    std::vector<float> q(qn), k(kvn), v(kvn);
    fill_uniform(q, seed, -0.25f, 0.25f);
    fill_uniform(k, seed + 1000u, -0.25f, 0.25f);
    fill_uniform(v, seed + 2000u, -1.0f, 1.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    std::vector<float> expected_cache_k, expected_cache_v;
    std::vector<double> ref;
    const std::int32_t padded_context = align_up_128(tokens);
    cpu_gqa_prefill(q, k, v, tokens, padded_context, expected_cache_k, expected_cache_v, ref);

    DBuf dq   = to_device_bf16(q);
    DBuf dk   = to_device_bf16(k);
    DBuf dv   = to_device_bf16(v);
    DBuf dout = DBuf(qn * sizeof(std::uint16_t));

    const std::size_t cache_n     = cache_elements(padded_context);
    const std::size_t layer_bytes = cache_n * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(tokens), kKVHeads, kHeadDim, DType::BF16);
    cudaMemset(kv.k[0].data, 0, layer_bytes);
    cudaMemset(kv.v[0].data, 0, layer_bytes);

    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv(dv.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQHeads, tokens});

    kernels::gqa_attention_prefill(tq, tk, tv, kScale, kv, 0, tout, nullptr);
    cudaDeviceSynchronize();

    int f             = 0;
    std::string label = "gqa prefill T=" + std::to_string(tokens);
    f += verify(label.c_str(), from_device_bf16(dout, qn), ref, Tolerance::attention_bf16());
    f += check_bits_equal((label + " k fill").c_str(), from_device_bf16_bits(kv.k[0].data, cache_n),
                          bf16_bits(expected_cache_k));
    f += check_bits_equal((label + " v fill").c_str(), from_device_bf16_bits(kv.v[0].data, cache_n),
                          bf16_bits(expected_cache_v));
    if (kv.pos != 0) {
        std::cerr << label << ": prefill op must not advance host KVCache.pos; got " << kv.pos
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
    DBuf dpos         = to_device_i32(std::vector<int>{tokens});
    DBuf dout_decode  = DBuf(q_decode_n * sizeof(std::uint16_t));
    WorkspaceArena ws(kDecodeWorkspaceBytes);

    const std::size_t layer_bytes = cache_n * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, static_cast<std::uint32_t>(tokens + 1), kKVHeads, kHeadDim,
               DType::BF16);
    cudaMemset(kv.k[0].data, 0, layer_bytes);
    cudaMemset(kv.v[0].data, 0, layer_bytes);

    Tensor tq_prefill(dq_prefill.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk_prefill(dk_prefill.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tv_prefill(dv_prefill.p, DType::BF16, {kHeadDim, kKVHeads, tokens});
    Tensor tout_prefill(dout_prefill.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tq_decode(dq_decode.p, DType::BF16, {kHeadDim, kQHeads, 1});
    Tensor tk_decode(dk_decode.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor tv_decode(dv_decode.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor tpos(dpos.p, DType::I32, {1});
    Tensor tout_decode(dout_decode.p, DType::BF16, {kHeadDim, kQHeads, 1});

    kernels::gqa_attention_prefill(tq_prefill, tk_prefill, tv_prefill, kScale, kv, 0, tout_prefill,
                                   nullptr);
    kv.pos = static_cast<std::uint32_t>(tokens);
    const std::uint32_t initial_kv_pos = kv.pos;
    kernels::gqa_attention_decode(tq_decode, tk_decode, tv_decode, tpos, kScale, kv, 0, ws,
                                  tout_decode, nullptr);
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
    WorkspaceArena ws(kDecodeWorkspaceBytes);
    const std::int32_t padded_context = align_up_128(pos + 1);
    const std::size_t layer_bytes = cache_elements(padded_context) * sizeof(std::uint16_t);
    DeviceArena cache_arena(2 * (layer_bytes + 256) + 4096);
    KVCache kv(cache_arena, 1, pos + 1, kKVHeads, kHeadDim, DType::BF16);
    kv.pos = static_cast<std::uint32_t>(pos);

    Tensor q(dq.p, DType::BF16, {kHeadDim, kQHeads, 1});
    Tensor k(dk.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor v(dv.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor scalar_pos(dpos.p, DType::I32, {1});
    Tensor out(dout.p, DType::BF16, {kHeadDim, kQHeads, 1});

    int f = 0;
    f += expect_invalid("gqa prefill q shape", [&] {
        Tensor bad(dq.p, DType::BF16, {kHeadDim, kQHeads, 2});
        kernels::gqa_attention_prefill(bad, k, v, kScale, kv, 0, out, nullptr);
    });
    f += expect_invalid("gqa prefill k shape", [&] {
        Tensor bad(dk.p, DType::BF16, {kHeadDim, kKVHeads, 2});
        kernels::gqa_attention_prefill(q, bad, v, kScale, kv, 0, out, nullptr);
    });
    f += expect_invalid("gqa prefill zero T", [&] {
        Tensor bad = q;
        bad.ne[2]  = 0;
        kernels::gqa_attention_prefill(bad, k, v, kScale, kv, 0, out, nullptr);
    });
    f += expect_invalid("gqa decode pos dtype", [&] {
        Tensor bad = scalar_pos;
        bad.dtype  = DType::BF16;
        kernels::gqa_attention_decode(q, k, v, bad, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa decode pos shape", [&] {
        Tensor bad(dpos.p, DType::I32, {2});
        kernels::gqa_attention_decode(q, k, v, bad, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa decode null pos", [&] {
        Tensor bad(nullptr, DType::I32, {1});
        kernels::gqa_attention_decode(q, k, v, bad, kScale, kv, 0, ws, out, nullptr);
    });
    f += expect_invalid("gqa decode q shape", [&] {
        Tensor bad(dq.p, DType::BF16, {kHeadDim, kQHeads, 2});
        kernels::gqa_attention_decode(bad, k, v, scalar_pos, kScale, kv, 0, ws, out, nullptr);
    });
    return f;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int f = 0;
    f += one_prefill_case(1, 101u);
    f += one_prefill_case(7, 107u);
    f += one_prefill_case(128, 113u);
    f += one_prefill_case(512, 127u);
    f += one_prefill_decode_consistency_case(128, 131u);
    f += one_decode_case(1, 11u);
    f += one_decode_case(17, 17u);
    f += one_decode_case(2048, 23u);
    f += one_decode_case(17, 31u, DecodeInputMode::Stress);
    f += validation_checks();

    std::cout << (f ? "FAIL" : "OK") << " gqa_attention_decode correctness\n";
    return f ? 1 : 0;
}
