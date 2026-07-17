// Correctness + coverage for rope, against the frozen op-test standard
// (docs/op-development.md): fp64 golden from bf16-rounded q/k,
// positions read from device, composite tolerance bf16_elementwise.
#include "ninfer/ops/rope.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads  = 24;
constexpr std::int32_t kKHeads  = 4;
constexpr int kRotaryDim        = 64;
constexpr float kTheta          = 1.0e7f;

std::size_t tensor_size(std::int32_t heads, std::int32_t T) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(heads) *
           static_cast<std::size_t>(T);
}

std::size_t index_of(std::int32_t heads, std::int32_t t, std::int32_t h, std::int32_t d) {
    return (static_cast<std::size_t>(t) * static_cast<std::size_t>(heads) +
            static_cast<std::size_t>(h)) *
               static_cast<std::size_t>(kHeadDim) +
           static_cast<std::size_t>(d);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& h) {
    std::vector<std::uint16_t> b(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) b[i] = f32_to_bf16(h[i]);
    return b;
}

std::vector<std::uint16_t> from_device_bf16_bits(const DBuf& d, std::size_t n) {
    std::vector<std::uint16_t> b(n);
    cudaMemcpy(b.data(), d.p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return b;
}

std::vector<std::uint16_t> from_device_bf16_bits_ptr(const void* p, std::size_t n) {
    std::vector<std::uint16_t> b(n);
    cudaMemcpy(b.data(), p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return b;
}

std::vector<double> from_device_bf16_ptr(const void* p, std::size_t n) {
    const std::vector<std::uint16_t> b = from_device_bf16_bits_ptr(p, n);
    std::vector<double> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = double(bf16_to_f32(b[i]));
    return o;
}

DBuf to_device_bf16_unaligned(const std::vector<float>& h) {
    std::vector<std::uint16_t> b(h.size() + 1);
    for (std::size_t i = 0; i < h.size(); ++i) b[i + 1] = f32_to_bf16(h[i]);
    DBuf d(b.size() * sizeof(std::uint16_t));
    cudaMemcpy(d.p, b.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

void cpu_rope_one(const std::vector<float>& in, const std::vector<int>& positions,
                  std::int32_t heads, std::int32_t T, int rotary_dim, float theta,
                  std::vector<double>& out) {
    out.resize(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) out[i] = static_cast<double>(in[i]);

    const int half = rotary_dim / 2;
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t h = 0; h < heads; ++h) {
            for (int i = 0; i < half; ++i) {
                const double freq =
                    std::pow(static_cast<double>(theta),
                             -2.0 * static_cast<double>(i) / static_cast<double>(rotary_dim));
                const double ang       = static_cast<double>(positions[t]) * freq;
                const double c         = std::cos(ang);
                const double s         = std::sin(ang);
                const std::size_t idx1 = index_of(heads, t, h, i);
                const std::size_t idx2 = index_of(heads, t, h, i + half);
                const double x1        = static_cast<double>(in[idx1]);
                const double x2        = static_cast<double>(in[idx2]);
                out[idx1]              = x1 * c - x2 * s;
                out[idx2]              = x2 * c + x1 * s;
            }
        }
    }
}

int check_passthrough_bits(const char* tag, const std::vector<std::uint16_t>& got,
                           const std::vector<std::uint16_t>& before, std::int32_t heads,
                           std::int32_t T, int rotary_dim) {
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t h = 0; h < heads; ++h) {
            for (std::int32_t d = rotary_dim; d < kHeadDim; ++d) {
                const std::size_t idx = index_of(heads, t, h, d);
                if (got[idx] != before[idx]) {
                    std::cerr << tag << ": pass-through dim changed at t=" << t << " h=" << h
                              << " d=" << d << " got_bits=0x" << std::hex << got[idx]
                              << " before_bits=0x" << before[idx] << std::dec << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}

int check_all_bits_same(const char* tag, const std::vector<std::uint16_t>& got,
                        const std::vector<std::uint16_t>& before) {
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != before[i]) {
            std::cerr << tag << ": identity changed index " << i << " got_bits=0x" << std::hex
                      << got[i] << " before_bits=0x" << before[i] << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

int one_shape(const char* tag, std::int32_t T, std::uint32_t seed) {
    const std::size_t qn = tensor_size(kQHeads, T);
    const std::size_t kn = tensor_size(kKHeads, T);
    std::vector<float> q(qn), k(kn);
    std::vector<int> positions(static_cast<std::size_t>(T));
    fill_uniform(q, seed, -8.f, 8.f);
    fill_uniform(k, seed + 1000u, -8.f, 8.f);
    fill_iota_i32(positions, 0);
    round_to_bf16(q);
    round_to_bf16(k);

    const std::vector<std::uint16_t> q_before = bf16_bits(q);
    const std::vector<std::uint16_t> k_before = bf16_bits(k);

    std::vector<double> q_ref, k_ref;
    cpu_rope_one(q, positions, kQHeads, T, kRotaryDim, kTheta, q_ref);
    cpu_rope_one(k, positions, kKHeads, T, kRotaryDim, kTheta, k_ref);

    DBuf dpos = to_device_i32(positions);
    DBuf dq = to_device_bf16(q), dk = to_device_bf16(k);
    Tensor tpos(dpos.p, DType::I32, {T});
    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, T});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKHeads, T});

    ops::rope(tpos, kRotaryDim, kTheta, tq, tk, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f += verify((std::string(tag) + " q").c_str(), from_device_bf16(dq, qn), q_ref,
                Tolerance::bf16_elementwise());
    f += verify((std::string(tag) + " k").c_str(), from_device_bf16(dk, kn), k_ref,
                Tolerance::bf16_elementwise());
    f += check_passthrough_bits((std::string(tag) + " q").c_str(), from_device_bf16_bits(dq, qn),
                                q_before, kQHeads, T, kRotaryDim);
    f += check_passthrough_bits((std::string(tag) + " k").c_str(), from_device_bf16_bits(dk, kn),
                                k_before, kKHeads, T, kRotaryDim);
    return f;
}

int unaligned_data_case() {
    constexpr std::int32_t T = 7;
    const std::size_t qn     = tensor_size(kQHeads, T);
    const std::size_t kn     = tensor_size(kKHeads, T);
    std::vector<float> q(qn), k(kn);
    std::vector<int> positions(static_cast<std::size_t>(T));
    fill_uniform(q, 1234u, -8.f, 8.f);
    fill_uniform(k, 4321u, -8.f, 8.f);
    fill_iota_i32(positions, 0);
    round_to_bf16(q);
    round_to_bf16(k);

    const std::vector<std::uint16_t> q_before = bf16_bits(q);
    const std::vector<std::uint16_t> k_before = bf16_bits(k);

    std::vector<double> q_ref, k_ref;
    cpu_rope_one(q, positions, kQHeads, T, kRotaryDim, kTheta, q_ref);
    cpu_rope_one(k, positions, kKHeads, T, kRotaryDim, kTheta, k_ref);

    DBuf dpos = to_device_i32(positions);
    DBuf dq = to_device_bf16_unaligned(q), dk = to_device_bf16_unaligned(k);
    auto* qptr = static_cast<std::uint16_t*>(dq.p) + 1;
    auto* kptr = static_cast<std::uint16_t*>(dk.p) + 1;
    Tensor tpos(dpos.p, DType::I32, {T});
    Tensor tq(qptr, DType::BF16, {kHeadDim, kQHeads, T});
    Tensor tk(kptr, DType::BF16, {kHeadDim, kKHeads, T});

    ops::rope(tpos, kRotaryDim, kTheta, tq, tk, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f += verify("rope unaligned q", from_device_bf16_ptr(qptr, qn), q_ref,
                Tolerance::bf16_elementwise());
    f += verify("rope unaligned k", from_device_bf16_ptr(kptr, kn), k_ref,
                Tolerance::bf16_elementwise());
    f += check_passthrough_bits("rope unaligned q", from_device_bf16_bits_ptr(qptr, qn), q_before,
                                kQHeads, T, kRotaryDim);
    f += check_passthrough_bits("rope unaligned k", from_device_bf16_bits_ptr(kptr, kn), k_before,
                                kKHeads, T, kRotaryDim);
    return f;
}

int identity_positions_zero_case() {
    constexpr std::int32_t T = 7;
    const std::size_t qn     = tensor_size(kQHeads, T);
    const std::size_t kn     = tensor_size(kKHeads, T);
    std::vector<float> q(qn), k(kn);
    std::vector<int> positions(static_cast<std::size_t>(T), 0);
    fill_uniform(q, 2026u, -8.f, 8.f);
    fill_uniform(k, 3026u, -8.f, 8.f);
    round_to_bf16(q);
    round_to_bf16(k);
    const std::vector<std::uint16_t> q_before = bf16_bits(q);
    const std::vector<std::uint16_t> k_before = bf16_bits(k);

    DBuf dpos = to_device_i32(positions);
    DBuf dq = to_device_bf16(q), dk = to_device_bf16(k);
    Tensor tpos(dpos.p, DType::I32, {T});
    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, T});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKHeads, T});
    ops::rope(tpos, kRotaryDim, kTheta, tq, tk, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f +=
        check_all_bits_same("rope positions=0 identity q", from_device_bf16_bits(dq, qn), q_before);
    f +=
        check_all_bits_same("rope positions=0 identity k", from_device_bf16_bits(dk, kn), k_before);
    return f;
}

int split_api_parity_case(std::int32_t T, std::uint32_t seed) {
    const std::size_t qn = tensor_size(kQHeads, T);
    const std::size_t kn = tensor_size(kKHeads, T);
    std::vector<float> q(qn), k(kn);
    std::vector<int> positions(static_cast<std::size_t>(T));
    fill_uniform(q, seed, -8.f, 8.f);
    fill_uniform(k, seed + 1000u, -8.f, 8.f);
    fill_iota_i32(positions, 17);
    round_to_bf16(q);
    round_to_bf16(k);

    std::vector<double> q_ref, k_ref;
    cpu_rope_one(q, positions, kQHeads, T, kRotaryDim, kTheta, q_ref);
    cpu_rope_one(k, positions, kKHeads, T, kRotaryDim, kTheta, k_ref);

    DBuf dpos     = to_device_i32(positions);
    DBuf dq_fused = to_device_bf16(q), dk_fused = to_device_bf16(k);
    DBuf dq_split = to_device_bf16(q), dk_split = to_device_bf16(k);
    Tensor tpos(dpos.p, DType::I32, {T});
    Tensor tq_fused(dq_fused.p, DType::BF16, {kHeadDim, kQHeads, T});
    Tensor tk_fused(dk_fused.p, DType::BF16, {kHeadDim, kKHeads, T});
    Tensor tq_split(dq_split.p, DType::BF16, {kHeadDim, kQHeads, T});
    Tensor tk_split(dk_split.p, DType::BF16, {kHeadDim, kKHeads, T});

    ops::rope(tpos, kRotaryDim, kTheta, tq_fused, tk_fused, nullptr);
    ops::rope(tpos, kRotaryDim, kTheta, tq_split, nullptr);
    ops::rope(tpos, kRotaryDim, kTheta, tk_split, nullptr);
    cudaDeviceSynchronize();

    const std::string label = "rope split API T=" + std::to_string(T);
    int f                   = 0;
    f += verify((label + " q vs FP64").c_str(), from_device_bf16(dq_split, qn), q_ref,
                Tolerance::bf16_elementwise());
    f += verify((label + " k vs FP64").c_str(), from_device_bf16(dk_split, kn), k_ref,
                Tolerance::bf16_elementwise());
    f += check_all_bits_same((label + " q").c_str(), from_device_bf16_bits(dq_split, qn),
                             from_device_bf16_bits(dq_fused, qn));
    f += check_all_bits_same((label + " k").c_str(), from_device_bf16_bits(dk_split, kn),
                             from_device_bf16_bits(dk_fused, kn));
    return f;
}

void cpu_rope_nd(const std::vector<float>& input, const std::vector<int>& positions, int axes,
                 int head_dim, int heads, int tokens, int rotary_dim, float theta,
                 std::vector<double>& output) {
    output.assign(input.begin(), input.end());
    const int half = rotary_dim / 2;
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int pair = 0; pair < half; ++pair) {
                int axis;
                double exponent;
                if (axes == 2) {
                    axis     = pair / 18;
                    exponent = -2.0 * static_cast<double>(pair % 18) / 36.0;
                } else {
                    axis     = axes == 3 ? pair % 3 : 0;
                    exponent = -2.0 * static_cast<double>(pair) / rotary_dim;
                }
                const double frequency = std::pow(static_cast<double>(theta), exponent);
                const double angle =
                    static_cast<double>(positions[axis * tokens + token]) * frequency;
                const std::size_t base =
                    (static_cast<std::size_t>(token) * heads + head) * head_dim;
                const double first         = input[base + pair];
                const double second        = input[base + pair + half];
                output[base + pair]        = first * std::cos(angle) - second * std::sin(angle);
                output[base + pair + half] = second * std::cos(angle) + first * std::sin(angle);
            }
        }
    }
}

int text_mrope_case() {
    constexpr int tokens = 7;
    std::vector<float> q(tensor_size(kQHeads, tokens));
    std::vector<float> k(tensor_size(kKHeads, tokens));
    std::vector<int> positions(3 * tokens);
    fill_uniform(q, 3001u, -8.0f, 8.0f);
    fill_uniform(k, 3002u, -8.0f, 8.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    for (int token = 0; token < tokens; ++token) {
        positions[token]              = 10 + token;
        positions[tokens + token]     = 20 + token * 2;
        positions[2 * tokens + token] = 30 + token * 3;
    }
    std::vector<double> q_ref;
    std::vector<double> k_ref;
    cpu_rope_nd(q, positions, 3, kHeadDim, kQHeads, tokens, kRotaryDim, kTheta, q_ref);
    cpu_rope_nd(k, positions, 3, kHeadDim, kKHeads, tokens, kRotaryDim, kTheta, k_ref);
    DBuf dpos = to_device_i32(positions);
    DBuf dq = to_device_bf16(q), dk = to_device_bf16(k);
    DBuf dq_single = to_device_bf16(q), dk_single = to_device_bf16(k);
    Tensor tpos(dpos.p, DType::I32, {tokens, 3});
    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk(dk.p, DType::BF16, {kHeadDim, kKHeads, tokens});
    Tensor tq_single(dq_single.p, DType::BF16, {kHeadDim, kQHeads, tokens});
    Tensor tk_single(dk_single.p, DType::BF16, {kHeadDim, kKHeads, tokens});
    ops::rope(tpos, kRotaryDim, kTheta, tq, tk, nullptr);
    ops::rope(tpos, kRotaryDim, kTheta, tq_single, nullptr);
    ops::rope(tpos, kRotaryDim, kTheta, tk_single, nullptr);
    cudaDeviceSynchronize();
    int failures = 0;
    failures += verify("text MRoPE q", from_device_bf16(dq, q.size()), q_ref,
                       Tolerance::bf16_elementwise());
    failures += verify("text MRoPE k", from_device_bf16(dk, k.size()), k_ref,
                       Tolerance::bf16_elementwise());
    failures += verify("text MRoPE q single", from_device_bf16(dq_single, q.size()), q_ref,
                       Tolerance::bf16_elementwise());
    failures += verify("text MRoPE k single", from_device_bf16(dk_single, k.size()), k_ref,
                       Tolerance::bf16_elementwise());
    return failures;
}

int text_35b_case(int tokens, int axes) {
    constexpr int q_heads = 16;
    constexpr int k_heads = 2;
    std::vector<float> q(tensor_size(q_heads, tokens));
    std::vector<float> k(tensor_size(k_heads, tokens));
    std::vector<int> positions(static_cast<std::size_t>(axes) * tokens);
    fill_uniform(q, 3501u + static_cast<std::uint32_t>(tokens), -8.0F, 8.0F);
    fill_uniform(k, 3502u + static_cast<std::uint32_t>(tokens), -8.0F, 8.0F);
    round_to_bf16(q);
    round_to_bf16(k);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(axis) * tokens + token] = 100 * axis + token;
        }
    }

    std::vector<double> q_ref;
    std::vector<double> k_ref;
    cpu_rope_nd(q, positions, axes, kHeadDim, q_heads, tokens, kRotaryDim, kTheta, q_ref);
    cpu_rope_nd(k, positions, axes, kHeadDim, k_heads, tokens, kRotaryDim, kTheta, k_ref);

    DBuf dpos    = to_device_i32(positions);
    DBuf dq_pair = to_device_bf16(q);
    DBuf dk_pair = to_device_bf16(k);
    DBuf dq_one  = to_device_bf16(q);
    DBuf dk_one  = to_device_bf16(k);
    Tensor tpos(dpos.p, DType::I32, {tokens, axes});
    Tensor tq_pair(dq_pair.p, DType::BF16, {kHeadDim, q_heads, tokens});
    Tensor tk_pair(dk_pair.p, DType::BF16, {kHeadDim, k_heads, tokens});
    Tensor tq_one(dq_one.p, DType::BF16, {kHeadDim, q_heads, tokens});
    Tensor tk_one(dk_one.p, DType::BF16, {kHeadDim, k_heads, tokens});
    ops::rope(tpos, kRotaryDim, kTheta, tq_pair, tk_pair, nullptr);
    ops::rope(tpos, kRotaryDim, kTheta, tq_one, nullptr);
    ops::rope(tpos, kRotaryDim, kTheta, tk_one, nullptr);
    cudaDeviceSynchronize();

    const std::string label =
        "text 35B axes=" + std::to_string(axes) + " T=" + std::to_string(tokens);
    const auto q_pair_bits = from_device_bf16_bits(dq_pair, q.size());
    const auto k_pair_bits = from_device_bf16_bits(dk_pair, k.size());
    int failures           = 0;
    failures += verify((label + " q").c_str(), from_device_bf16(dq_pair, q.size()), q_ref,
                       Tolerance::bf16_elementwise());
    failures += verify((label + " k").c_str(), from_device_bf16(dk_pair, k.size()), k_ref,
                       Tolerance::bf16_elementwise());
    failures += check_passthrough_bits((label + " q passthrough").c_str(), q_pair_bits,
                                       bf16_bits(q), q_heads, tokens, kRotaryDim);
    failures += check_passthrough_bits((label + " k passthrough").c_str(), k_pair_bits,
                                       bf16_bits(k), k_heads, tokens, kRotaryDim);
    failures += verify((label + " q single vs FP64").c_str(), from_device_bf16(dq_one, q.size()),
                       q_ref, Tolerance::bf16_elementwise());
    failures += verify((label + " k single vs FP64").c_str(), from_device_bf16(dk_one, k.size()),
                       k_ref, Tolerance::bf16_elementwise());
    failures += check_all_bits_same((label + " q single parity").c_str(),
                                    from_device_bf16_bits(dq_one, q.size()), q_pair_bits);
    failures += check_all_bits_same((label + " k single parity").c_str(),
                                    from_device_bf16_bits(dk_one, k.size()), k_pair_bits);
    return failures;
}

int vision_rope_packed_case() {
    constexpr int head_dim = 72;
    constexpr int heads    = 16;
    constexpr int tokens   = 11;
    constexpr int hidden   = head_dim * heads;
    constexpr int qkv      = hidden * 3;
    std::vector<float> q(static_cast<std::size_t>(tokens) * hidden);
    std::vector<float> k(static_cast<std::size_t>(tokens) * hidden);
    std::vector<float> packed(static_cast<std::size_t>(tokens) * qkv, 0.0f);
    std::vector<int> positions(2 * tokens);
    fill_uniform(q, 4001u, -8.0f, 8.0f);
    fill_uniform(k, 4002u, -8.0f, 8.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    for (int token = 0; token < tokens; ++token) {
        positions[token]          = token / 4;
        positions[tokens + token] = token % 4;
        std::copy_n(q.data() + static_cast<std::size_t>(token) * hidden, hidden,
                    packed.data() + static_cast<std::size_t>(token) * qkv);
        std::copy_n(k.data() + static_cast<std::size_t>(token) * hidden, hidden,
                    packed.data() + static_cast<std::size_t>(token) * qkv + hidden);
    }
    std::vector<double> q_ref;
    std::vector<double> k_ref;
    cpu_rope_nd(q, positions, 2, head_dim, heads, tokens, head_dim, 10000.0f, q_ref);
    cpu_rope_nd(k, positions, 2, head_dim, heads, tokens, head_dim, 10000.0f, k_ref);
    DBuf dpacked_pair   = to_device_bf16(packed);
    DBuf dpacked_single = to_device_bf16(packed);
    DBuf dpos           = to_device_i32(positions);
    Tensor tq(dpacked_pair.p, DType::BF16, {head_dim, heads, tokens});
    tq.nb[2]  = qkv * 2;
    Tensor tk = tq;
    tk.data   = static_cast<unsigned char*>(dpacked_pair.p) + hidden * 2;
    Tensor tq_one(dpacked_single.p, DType::BF16, {head_dim, heads, tokens});
    tq_one.nb[2]  = qkv * 2;
    Tensor tk_one = tq_one;
    tk_one.data   = static_cast<unsigned char*>(dpacked_single.p) + hidden * 2;
    Tensor tpos(dpos.p, DType::I32, {tokens, 2});
    ops::rope(tpos, head_dim, 10000.0f, tq, tk, nullptr);
    ops::rope(tpos, head_dim, 10000.0f, tq_one, nullptr);
    ops::rope(tpos, head_dim, 10000.0f, tk_one, nullptr);
    cudaDeviceSynchronize();
    const std::vector<std::uint16_t> pair_bits = from_device_bf16_bits(dpacked_pair, packed.size());
    const std::vector<std::uint16_t> single_bits =
        from_device_bf16_bits(dpacked_single, packed.size());
    std::vector<double> q_got(q.size()), k_got(k.size());
    std::vector<double> q_single_got(q.size()), k_single_got(k.size());
    for (int token = 0; token < tokens; ++token) {
        for (int i = 0; i < hidden; ++i) {
            q_got[static_cast<std::size_t>(token) * hidden + i] =
                bf16_to_f32(pair_bits[static_cast<std::size_t>(token) * qkv + i]);
            k_got[static_cast<std::size_t>(token) * hidden + i] =
                bf16_to_f32(pair_bits[static_cast<std::size_t>(token) * qkv + hidden + i]);
            q_single_got[static_cast<std::size_t>(token) * hidden + i] =
                bf16_to_f32(single_bits[static_cast<std::size_t>(token) * qkv + i]);
            k_single_got[static_cast<std::size_t>(token) * hidden + i] =
                bf16_to_f32(single_bits[static_cast<std::size_t>(token) * qkv + hidden + i]);
        }
    }
    int failures = 0;
    failures += verify("vision RoPE packed q", q_got, q_ref, Tolerance::bf16_elementwise());
    failures += verify("vision RoPE packed k", k_got, k_ref, Tolerance::bf16_elementwise());
    failures +=
        verify("vision RoPE packed q single", q_single_got, q_ref, Tolerance::bf16_elementwise());
    failures +=
        verify("vision RoPE packed k single", k_single_got, k_ref, Tolerance::bf16_elementwise());
    return failures;
}

template <typename Fn>
int expect_invalid(const char* label, Fn&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

template <typename Fn>
int expect_overflow(const char* label, Fn&& fn) {
    try {
        fn();
    } catch (const std::overflow_error&) { return 0; } catch (const std::invalid_argument& e) {
        std::cerr << label << ": expected overflow_error, got invalid_argument: " << e.what()
                  << '\n';
        return 1;
    }
    std::cerr << label << ": expected overflow_error\n";
    return 1;
}

int validation_checks() {
    int f = 0;
    Tensor pos(nullptr, DType::I32, {7});
    Tensor q(nullptr, DType::BF16, {kHeadDim, kQHeads, 7});
    Tensor k(nullptr, DType::BF16, {kHeadDim, kKHeads, 7});

    try {
        Tensor empty_pos(nullptr, DType::I32, {1});
        Tensor empty_q(nullptr, DType::BF16, {kHeadDim, kQHeads, 1});
        Tensor empty_k(nullptr, DType::BF16, {kHeadDim, kKHeads, 1});
        empty_pos.ne[0] = 0;
        empty_q.ne[2]   = 0;
        empty_k.ne[2]   = 0;
        ops::rope(empty_pos, kRotaryDim, kTheta, empty_q, empty_k, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "validation empty T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    f += expect_invalid("validation q dtype", [&] {
        Tensor bad = q;
        bad.dtype  = DType::FP32;
        ops::rope(pos, kRotaryDim, kTheta, bad, k, nullptr);
    });
    f += expect_invalid("validation k dtype", [&] {
        Tensor bad = k;
        bad.dtype  = DType::FP32;
        ops::rope(pos, kRotaryDim, kTheta, q, bad, nullptr);
    });
    f += expect_invalid("validation positions dtype", [&] {
        Tensor bad = pos;
        bad.dtype  = DType::BF16;
        ops::rope(bad, kRotaryDim, kTheta, q, k, nullptr);
    });
    f += expect_invalid("validation q head dim", [&] {
        Tensor bad = q;
        bad.ne[0]  = 255;
        ops::rope(pos, kRotaryDim, kTheta, bad, k, nullptr);
    });
    f += expect_invalid("validation k head dim", [&] {
        Tensor bad = k;
        bad.ne[0]  = 255;
        ops::rope(pos, kRotaryDim, kTheta, q, bad, nullptr);
    });
    f += expect_invalid("validation q T mismatch", [&] {
        Tensor bad = q;
        bad.ne[2]  = 8;
        ops::rope(pos, kRotaryDim, kTheta, bad, k, nullptr);
    });
    f += expect_invalid("validation k T mismatch", [&] {
        Tensor bad = k;
        bad.ne[2]  = 8;
        ops::rope(pos, kRotaryDim, kTheta, q, bad, nullptr);
    });
    f += expect_invalid("validation positions shape rank", [&] {
        Tensor bad = pos;
        bad.ne[1]  = 2;
        ops::rope(bad, kRotaryDim, kTheta, q, k, nullptr);
    });
    f += expect_invalid("validation positions length", [&] {
        Tensor bad = pos;
        bad.ne[0]  = 8;
        ops::rope(bad, kRotaryDim, kTheta, q, k, nullptr);
    });
    f += expect_invalid("validation q non-contiguous", [&] {
        Tensor bad = q;
        bad.nb[0]  = 4;
        ops::rope(pos, kRotaryDim, kTheta, bad, k, nullptr);
    });
    f += expect_invalid("validation k non-contiguous", [&] {
        Tensor bad = k;
        bad.nb[0]  = 4;
        ops::rope(pos, kRotaryDim, kTheta, q, bad, nullptr);
    });
    f += expect_invalid("validation positions non-contiguous", [&] {
        Tensor bad = pos;
        bad.nb[0]  = 8;
        ops::rope(bad, kRotaryDim, kTheta, q, k, nullptr);
    });
    f += expect_invalid("validation null data",
                        [&] { ops::rope(pos, kRotaryDim, kTheta, q, k, nullptr); });
    f += expect_invalid("validation negative dim", [&] {
        Tensor bad = q;
        bad.ne[2]  = -1;
        ops::rope(pos, kRotaryDim, kTheta, bad, k, nullptr);
    });
    f +=
        expect_invalid("validation rotary <= 0", [&] { ops::rope(pos, 0, kTheta, q, k, nullptr); });
    f += expect_invalid("validation rotary > head dim",
                        [&] { ops::rope(pos, 258, kTheta, q, k, nullptr); });
    f +=
        expect_invalid("validation rotary odd", [&] { ops::rope(pos, 63, kTheta, q, k, nullptr); });
    f += expect_invalid("validation theta finite positive",
                        [&] { ops::rope(pos, kRotaryDim, -1.0f, q, k, nullptr); });
    f += expect_overflow("validation overflow dims", [&] {
        Tensor huge_pos = pos;
        Tensor huge_q   = q;
        Tensor huge_k   = k;
        for (int d = 0; d < 4; ++d) {
            huge_pos.ne[d] = std::numeric_limits<std::int32_t>::max();
            huge_q.ne[d]   = std::numeric_limits<std::int32_t>::max();
            huge_k.ne[d]   = std::numeric_limits<std::int32_t>::max();
        }
        ops::rope(huge_pos, kRotaryDim, kTheta, huge_q, huge_k, nullptr);
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
    f += validation_checks();
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_shape("rope T=1", 1, seed);
        f += one_shape("rope T=7", 7, seed);
        f += one_shape("rope T=4096", 4096, seed);
    }
    f += identity_positions_zero_case();
    f += unaligned_data_case();
    f += split_api_parity_case(1, 2001u);
    f += split_api_parity_case(7, 2007u);
    f += split_api_parity_case(1024, 2024u);
    f += text_mrope_case();
    f += text_35b_case(1, 1);
    f += text_35b_case(1024, 3);
    f += vision_rope_packed_case();

    std::cout << (f ? "FAIL" : "OK") << " rope correctness\n";
    return f ? 1 : 0;
}
