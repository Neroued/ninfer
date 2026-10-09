#include "ninfer/ops/softmax_attention.h"

#include "core/arena.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim         = 256;
constexpr int kQueryHeads  = 24;
constexpr int kKVHeads     = 4;
constexpr float kScale     = 0.0625F; // 1/sqrt(256)
constexpr ops::AttentionHeadGeometry kGeometry{kDim, kQueryHeads, kKVHeads};

constexpr ReductionCriterion kNoncausalGQABf16Criterion{
    .relative_l2                     = 2.5e-3,
    .gross_absolute                  = 1e-3,
    .gross_relative_to_max_reference = 3.5e-3,
};

std::size_t index_q(int token, int head, int d) {
    return (static_cast<std::size_t>(token) * kQueryHeads + static_cast<std::size_t>(head)) *
               kDim +
           static_cast<std::size_t>(d);
}

std::size_t index_kv(int token, int head, int d) {
    return (static_cast<std::size_t>(token) * kKVHeads + static_cast<std::size_t>(head)) * kDim +
           static_cast<std::size_t>(d);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

void noncausal_gqa_oracle(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, const std::vector<int>& cu_seqlens,
                          std::vector<double>& out) {
    constexpr double scale = 0.0625; // 1/sqrt(256)
    const std::size_t tokens = static_cast<std::size_t>(cu_seqlens.back());
    out.assign(tokens * kQueryHeads * kDim, 0.0);

    for (std::size_t segment = 0; segment + 1 < cu_seqlens.size(); ++segment) {
        const int begin = cu_seqlens[segment];
        const int end   = cu_seqlens[segment + 1];
        naive_dense_softmax_attention(
            kGeometry, end - begin, end - begin, scale,
            [&](int d, int head, int token) {
                return static_cast<double>(q[index_q(begin + token, head, d)]);
            },
            [&](int d, int head, int token) {
                return static_cast<double>(k[index_kv(begin + token, head, d)]);
            },
            [&](int d, int head, int token) {
                return static_cast<double>(v[index_kv(begin + token, head, d)]);
            },
            [](int, int) { return true; },
            [&](int d, int head, int token, double value) {
                out[index_q(begin + token, head, d)] = value;
            });
    }
}

enum class InputProfile {
    Random,
    SegmentIsolation,
};

const char* input_name(InputProfile profile) {
    return profile == InputProfile::Random ? "random" : "segment-isolation";
}

int run_case(const std::vector<int>& cu_seqlens, std::uint32_t seed,
             InputProfile input_profile = InputProfile::Random) {
    const int tokens          = cu_seqlens.back();
    const std::size_t plane_q = static_cast<std::size_t>(kQueryHeads) * kDim;
    const std::size_t plane_kv = static_cast<std::size_t>(kKVHeads) * kDim;
    const std::size_t q_count  = static_cast<std::size_t>(tokens) * plane_q;
    const std::size_t kv_count = static_cast<std::size_t>(tokens) * plane_kv;

    std::vector<float> q(q_count);
    std::vector<float> k(kv_count);
    std::vector<float> v(kv_count);
    fill_uniform(q, seed, -1.0f, 1.0f);
    fill_uniform(k, seed + 1, -1.0f, 1.0f);
    fill_uniform(v, seed + 2, -2.0f, 2.0f);
    if (input_profile == InputProfile::SegmentIsolation) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        std::fill(v.begin(), v.end(), 0.0f);
        for (std::size_t segment = 0; segment + 1 < cu_seqlens.size(); ++segment) {
            const float segment_value = (segment & 1u) == 0 ? 4.0f : -3.0f;
            for (int token = cu_seqlens[segment]; token < cu_seqlens[segment + 1]; ++token) {
                std::fill_n(v.data() + static_cast<std::size_t>(token) * plane_kv, plane_kv,
                            segment_value);
            }
        }
    }
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    std::vector<double> reference;
    noncausal_gqa_oracle(q, k, v, cu_seqlens, reference);

    const auto q_expected = bf16_bits(q);
    const auto k_expected = bf16_bits(k);
    const auto v_expected = bf16_bits(v);

    DeviceBuffer q_storage = to_device(q_expected);
    DeviceBuffer k_storage = to_device(k_expected);
    DeviceBuffer v_storage = to_device(v_expected);
    Tensor q_tensor(q_storage.p, DType::BF16, {kDim, kQueryHeads, tokens});
    Tensor k_tensor(k_storage.p, DType::BF16, {kDim, kKVHeads, tokens});
    Tensor v_tensor(v_storage.p, DType::BF16, {kDim, kKVHeads, tokens});

    DeviceBuffer d_cu_seqlens = to_device_i32(cu_seqlens);
    Tensor cu_tensor(d_cu_seqlens.p, DType::I32, {static_cast<std::int32_t>(cu_seqlens.size())});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    Tensor out_tensor(d_out.data(), DType::BF16, {kDim, kQueryHeads, tokens});

    const std::int32_t segments      = static_cast<std::int32_t>(cu_seqlens.size()) - 1;
    const std::size_t workspace_bytes =
        ops::noncausal_gqa_attention_workspace_capacity_bytes(kGeometry, tokens, tokens, segments,
                                                              segments);
    DeviceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    ops::noncausal_gqa_attention(q_tensor, k_tensor, v_tensor, kGeometry, kScale, cu_tensor,
                                 workspace, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "noncausal_gqa_attention T=" + std::to_string(tokens) +
                              " S=" + std::to_string(segments) + " " + input_name(input_profile);
    int failures =
        verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count), reference,
                         kNoncausalGQABf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(q_storage, q_count), q_expected);
    failures += verify_exact((label + " k unchanged").c_str(),
                             from_device<std::uint16_t>(k_storage, kv_count), k_expected);
    failures += verify_exact((label + " v unchanged").c_str(),
                             from_device<std::uint16_t>(v_storage, kv_count), v_expected);
    failures += verify_exact((label + " cu_seqlens unchanged").c_str(),
                             from_device<int>(d_cu_seqlens, cu_seqlens.size()), cu_seqlens);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

} // namespace

int run_softmax_attention_noncausal_gqa_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    int failures = 0;
    if (ops::noncausal_gqa_attention_workspace_capacity_bytes(kGeometry, 4, 194, 1, 1) != 0 ||
        ops::noncausal_gqa_attention_workspace_capacity_bytes(kGeometry, 4, 194, 1, 3) !=
            ops::noncausal_gqa_attention_workspace_capacity_bytes(kGeometry, 194, 194, 3, 3)) {
        std::cerr
            << "noncausal_gqa_attention rectangular capacity missed its maximal legal pair\n";
        ++failures;
    }
    try {
        (void)ops::noncausal_gqa_attention_workspace_capacity_bytes(kGeometry, 1, 2, 3, 4);
        std::cerr << "noncausal_gqa_attention accepted an envelope without a legal segment pair\n";
        ++failures;
    } catch (const std::invalid_argument&) {}

    failures += run_case({0, 4}, 11u);
    failures += run_case({0, 4}, 1u);
    failures += run_case({0, 4}, 101u);
    failures += run_case({0, 8}, 77u);
    failures += run_case({0, 16}, 83u);
    failures += run_case({0, 3, 7, 12, 19}, 5u);
    failures += run_case({0, 64}, 23u);
    failures += run_case({0, 68}, 101u);
    failures += run_case({0, 64, 129, 194}, 31u);
    failures += run_case({0, 68, 136}, 57u);
    failures += run_case({0, 68, 136}, 91u, InputProfile::SegmentIsolation);
    failures += run_case({0, 256}, 2026u);
    failures += run_case({0, 1, 70, 71, 194}, 42u);

    if (failures != 0) {
        std::cerr << "noncausal_gqa_attention failures=" << failures << '\n';
        return 1;
    }
    std::cout << "noncausal_gqa_attention: PASS\n";
    return 0;
}
