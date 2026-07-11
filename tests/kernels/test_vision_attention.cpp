#include "qus/kernels/vision_attention.h"
#include "kernels/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

constexpr int kDim   = 72;
constexpr int kHeads = 16;

std::size_t index_of(int token, int head, int d) {
    return (static_cast<std::size_t>(token) * kHeads + head) * kDim + d;
}

void reference_attention(const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v, const std::vector<int>& cu,
                         std::vector<double>& out) {
    constexpr double scale = 1.0 / std::sqrt(72.0);
    for (std::size_t segment = 0; segment + 1 < cu.size(); ++segment) {
        const int begin = cu[segment];
        const int end   = cu[segment + 1];
        for (int query = begin; query < end; ++query) {
            for (int head = 0; head < kHeads; ++head) {
                std::vector<double> scores(static_cast<std::size_t>(end - begin));
                double maximum = -INFINITY;
                for (int key = begin; key < end; ++key) {
                    double dot = 0.0;
                    for (int d = 0; d < kDim; ++d) {
                        dot += static_cast<double>(q[index_of(query, head, d)]) *
                               k[index_of(key, head, d)];
                    }
                    scores[static_cast<std::size_t>(key - begin)] = dot * scale;
                    maximum                                       = std::max(maximum, dot * scale);
                }
                double sum = 0.0;
                for (double& score : scores) {
                    score = std::exp(score - maximum);
                    sum += score;
                }
                for (int d = 0; d < kDim; ++d) {
                    double value = 0.0;
                    for (int key = begin; key < end; ++key) {
                        value += scores[static_cast<std::size_t>(key - begin)] / sum *
                                 v[index_of(key, head, d)];
                    }
                    out[index_of(query, head, d)] = value;
                }
            }
        }
    }
}

int one_case(const std::vector<int>& cu, std::uint32_t seed, bool packed) {
    const int patches       = cu.back();
    const std::size_t plane = static_cast<std::size_t>(patches) * kHeads * kDim;
    std::vector<float> q(plane), k(plane), v(plane);
    fill_uniform(q, seed, -1.0f, 1.0f);
    fill_uniform(k, seed + 1, -1.0f, 1.0f);
    fill_uniform(v, seed + 2, -2.0f, 2.0f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    std::vector<double> reference(plane);
    reference_attention(q, k, v, cu, reference);

    DBuf dcu = to_device_i32(cu);
    DBuf dout(plane * 2);
    Tensor tq;
    Tensor tk;
    Tensor tv;
    DBuf storage(packed ? plane * 3 * 2 : 1);
    DBuf dq(packed ? 1 : plane * 2);
    DBuf dk(packed ? 1 : plane * 2);
    DBuf dv(packed ? 1 : plane * 2);
    if (packed) {
        std::vector<float> qkv(plane * 3);
        const std::size_t token_plane = static_cast<std::size_t>(kHeads) * kDim;
        for (int token = 0; token < patches; ++token) {
            const std::size_t src = static_cast<std::size_t>(token) * token_plane;
            const std::size_t dst = static_cast<std::size_t>(token) * token_plane * 3;
            std::copy_n(q.data() + src, token_plane, qkv.data() + dst);
            std::copy_n(k.data() + src, token_plane, qkv.data() + dst + token_plane);
            std::copy_n(v.data() + src, token_plane, qkv.data() + dst + token_plane * 2);
        }
        std::vector<std::uint16_t> bits(qkv.size());
        for (std::size_t i = 0; i < qkv.size(); ++i) bits[i] = f32_to_bf16(qkv[i]);
        cudaMemcpy(storage.p, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice);
        tq       = Tensor(storage.p, DType::BF16, {kDim, kHeads, patches});
        tq.nb[2] = static_cast<std::int64_t>(token_plane * 3 * 2);
        tk       = tq;
        tv       = tq;
        tk.data  = static_cast<unsigned char*>(storage.p) + token_plane * 2;
        tv.data  = static_cast<unsigned char*>(storage.p) + token_plane * 4;
    } else {
        std::vector<std::uint16_t> qb(q.size()), kb(k.size()), vb(v.size());
        for (std::size_t i = 0; i < q.size(); ++i) {
            qb[i] = f32_to_bf16(q[i]);
            kb[i] = f32_to_bf16(k[i]);
            vb[i] = f32_to_bf16(v[i]);
        }
        cudaMemcpy(dq.p, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dk.p, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dv.p, vb.data(), vb.size() * 2, cudaMemcpyHostToDevice);
        tq = Tensor(dq.p, DType::BF16, {kDim, kHeads, patches});
        tk = Tensor(dk.p, DType::BF16, {kDim, kHeads, patches});
        tv = Tensor(dv.p, DType::BF16, {kDim, kHeads, patches});
    }
    Tensor tcu(dcu.p, DType::I32, {static_cast<int>(cu.size())});
    Tensor tout(dout.p, DType::BF16, {kDim, kHeads, patches});
    WorkspaceArena workspace(256);
    kernels::vision_attention(tq, tk, tv, tcu, workspace, tout, nullptr);
    cudaDeviceSynchronize();
    return verify(packed ? "vision attention packed qkv" : "vision attention contiguous",
                  from_device_bf16(dout, plane), reference, Tolerance::attention_bf16());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int failures = 0;
    failures += one_case({0, 4, 11}, 1u, false);
    failures += one_case({0, 4, 11}, 7u, true);
    failures += one_case({0, 16}, 99u, true);
    failures += one_case({0, 256}, 2026u, true);
    std::cout << (failures ? "FAIL" : "OK") << " vision_attention correctness\n";
    return failures ? 1 : 0;
}
