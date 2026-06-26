// Correctness + coverage for gqa_attention_decode, against the frozen op-test
// standard (docs/l1-op-test-standard.md): fp64 golden from bf16-rounded
// q/k/v/cache inputs, device scalar pos, composite tolerance attention_bf16.
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

enum class DecodeInputMode { Random, Stress };

std::size_t q_index(std::int32_t q_head, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(q_head);
}

std::size_t kv_tensor_index(std::int32_t kv_head, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kv_head);
}

std::size_t cache_index(std::int32_t kv_head, std::int32_t d, std::int32_t position) {
    return (static_cast<std::size_t>(position) * static_cast<std::size_t>(kHeadDim) +
            static_cast<std::size_t>(d)) *
               static_cast<std::size_t>(kKVHeads) +
           static_cast<std::size_t>(kv_head);
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
                    const std::vector<float>& cache_v, std::int32_t pos, std::vector<double>& out) {
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
                       static_cast<double>(cache_k[cache_index(kvh, d, j)]);
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
                       static_cast<double>(cache_v[cache_index(kvh, d, j)]);
            }
            out[q_index(qh, d)] = acc;
        }
    }
}

int one_decode_case(std::int32_t pos, std::uint32_t seed,
                    DecodeInputMode mode = DecodeInputMode::Random) {
    const std::size_t qn      = static_cast<std::size_t>(kHeadDim) * kQHeads;
    const std::size_t kvn     = static_cast<std::size_t>(kHeadDim) * kKVHeads;
    const std::size_t cache_n = kvn * (static_cast<std::size_t>(pos) + 1);

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
            expected_cache_k[cache_index(h, d, pos)] = k_new[kv_tensor_index(h, d)];
            expected_cache_v[cache_index(h, d, pos)] = v_new[kv_tensor_index(h, d)];
        }
    }

    std::vector<double> ref;
    cpu_gqa_decode(q, expected_cache_k, expected_cache_v, pos, ref);

    DBuf dq   = to_device_bf16(q);
    DBuf dk   = to_device_bf16(k_new);
    DBuf dv   = to_device_bf16(v_new);
    DBuf dpos = to_device_i32(std::vector<int>{pos});
    DBuf dout(qn * sizeof(std::uint16_t));

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

    kernels::gqa_attention_decode(tq, tk, tv, tpos, kScale, kv, 0, tout, nullptr);
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
    if (kv.pos != 0) {
        std::cerr << label << ": decode op must not advance host KVCache.pos; got " << kv.pos
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
    DeviceArena cache_arena(2 * kvn * (pos + 1) * sizeof(std::uint16_t) + 4096);
    KVCache kv(cache_arena, 1, pos + 1, kKVHeads, kHeadDim, DType::BF16);

    Tensor q(dq.p, DType::BF16, {kHeadDim, kQHeads, 1});
    Tensor k(dk.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor v(dv.p, DType::BF16, {kHeadDim, kKVHeads, 1});
    Tensor scalar_pos(dpos.p, DType::I32, {1});
    Tensor out(dout.p, DType::BF16, {kHeadDim, kQHeads, 1});

    int f = 0;
    f += expect_invalid("gqa prefill unsupported", [&] {
        kernels::gqa_attention_prefill(q, k, v, kScale, kv, 0, out, nullptr);
    });
    f += expect_invalid("gqa decode pos dtype", [&] {
        Tensor bad = scalar_pos;
        bad.dtype  = DType::BF16;
        kernels::gqa_attention_decode(q, k, v, bad, kScale, kv, 0, out, nullptr);
    });
    f += expect_invalid("gqa decode pos shape", [&] {
        Tensor bad(dpos.p, DType::I32, {2});
        kernels::gqa_attention_decode(q, k, v, bad, kScale, kv, 0, out, nullptr);
    });
    f += expect_invalid("gqa decode null pos", [&] {
        Tensor bad(nullptr, DType::I32, {1});
        kernels::gqa_attention_decode(q, k, v, bad, kScale, kv, 0, out, nullptr);
    });
    f += expect_invalid("gqa decode q shape", [&] {
        Tensor bad(dq.p, DType::BF16, {kHeadDim, kQHeads, 2});
        kernels::gqa_attention_decode(bad, k, v, scalar_pos, kScale, kv, 0, out, nullptr);
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
    f += one_decode_case(1, 11u);
    f += one_decode_case(17, 17u);
    f += one_decode_case(2048, 23u);
    f += one_decode_case(17, 31u, DecodeInputMode::Stress);
    f += validation_checks();

    std::cout << (f ? "FAIL" : "OK") << " gqa_attention_decode correctness\n";
    return f ? 1 : 0;
}
