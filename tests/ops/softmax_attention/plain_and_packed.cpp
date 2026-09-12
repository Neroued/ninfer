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

constexpr int kDim     = 72;
constexpr int kHeads   = 16;
constexpr float kScale = 0.11785113019775792073F;
constexpr ops::AttentionHeadGeometry kGeometry{kDim, kHeads, kHeads};

constexpr ReductionCriterion kPackedAttentionBf16Criterion{
    .relative_l2                     = 2.5e-3,
    .gross_absolute                  = 1e-3,
    .gross_relative_to_max_reference = 2.8e-3,
};

std::size_t index_of(int token, int head, int d) {
    return (static_cast<std::size_t>(token) * kHeads + static_cast<std::size_t>(head)) * kDim +
           static_cast<std::size_t>(d);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

void packed_attention_oracle(const std::vector<float>& q, const std::vector<float>& k,
                             const std::vector<float>& v, const std::vector<int>& cu_seqlens,
                             std::vector<double>& out) {
    constexpr double scale = 1.0 / std::sqrt(72.0);
    out.assign(q.size(), 0.0);

    for (std::size_t segment = 0; segment + 1 < cu_seqlens.size(); ++segment) {
        const int begin = cu_seqlens[segment];
        const int end   = cu_seqlens[segment + 1];
        naive_dense_softmax_attention(
            kGeometry, end - begin, end - begin, scale,
            [&](int d, int head, int token) {
                return static_cast<double>(q[index_of(begin + token, head, d)]);
            },
            [&](int d, int head, int token) {
                return static_cast<double>(k[index_of(begin + token, head, d)]);
            },
            [&](int d, int head, int token) {
                return static_cast<double>(v[index_of(begin + token, head, d)]);
            },
            [](int, int) { return true; },
            [&](int d, int head, int token, double value) {
                out[index_of(begin + token, head, d)] = value;
            });
    }
}

enum class StorageProfile {
    Contiguous,
    InterleavedQkv,
};

enum class PublicEntry {
    Plain,
    CuSeqlensArena,
    UniformSegments,
};

enum class InputProfile {
    Random,
    SegmentIsolation,
};

const char* storage_name(StorageProfile profile) {
    return profile == StorageProfile::Contiguous ? "contiguous" : "interleaved-qkv";
}

const char* entry_name(PublicEntry entry) {
    switch (entry) {
    case PublicEntry::Plain:
        return "plain";
    case PublicEntry::CuSeqlensArena:
        return "cu-arena";
    case PublicEntry::UniformSegments:
        return "uniform";
    }
    return "unknown";
}

int run_case(const std::vector<int>& cu_seqlens, std::uint32_t seed, StorageProfile storage_profile,
             PublicEntry entry, InputProfile input_profile = InputProfile::Random) {
    const int tokens              = cu_seqlens.back();
    const std::size_t token_plane = static_cast<std::size_t>(kHeads) * kDim;
    const std::size_t value_count = static_cast<std::size_t>(tokens) * token_plane;
    std::vector<float> q(value_count);
    std::vector<float> k(value_count);
    std::vector<float> v(value_count);
    fill_uniform(q, seed, -1.0f, 1.0f);
    fill_uniform(k, seed + 1, -1.0f, 1.0f);
    fill_uniform(v, seed + 2, -2.0f, 2.0f);
    if (input_profile == InputProfile::SegmentIsolation) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(k.begin(), k.end(), 0.0f);
        for (std::size_t segment = 0; segment + 1 < cu_seqlens.size(); ++segment) {
            const float segment_value = (segment & 1u) == 0 ? 4.0f : -3.0f;
            for (int token = cu_seqlens[segment]; token < cu_seqlens[segment + 1]; ++token) {
                std::fill_n(v.data() + static_cast<std::size_t>(token) * token_plane, token_plane,
                            segment_value);
            }
        }
    }
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    std::vector<double> reference;
    packed_attention_oracle(q, k, v, cu_seqlens, reference);

    const auto q_expected = bf16_bits(q);
    const auto k_expected = bf16_bits(k);
    const auto v_expected = bf16_bits(v);

    DeviceBuffer q_storage;
    DeviceBuffer k_storage;
    DeviceBuffer v_storage;
    DeviceBuffer interleaved_storage;
    Tensor q_tensor;
    Tensor k_tensor;
    Tensor v_tensor;
    std::vector<std::uint16_t> interleaved_expected;

    if (storage_profile == StorageProfile::Contiguous) {
        q_storage = to_device(q_expected);
        k_storage = to_device(k_expected);
        v_storage = to_device(v_expected);
        q_tensor  = Tensor(q_storage.p, DType::BF16, {kDim, kHeads, tokens});
        k_tensor  = Tensor(k_storage.p, DType::BF16, {kDim, kHeads, tokens});
        v_tensor  = Tensor(v_storage.p, DType::BF16, {kDim, kHeads, tokens});
    } else {
        interleaved_expected.resize(value_count * 3);
        for (int token = 0; token < tokens; ++token) {
            const std::size_t source = static_cast<std::size_t>(token) * token_plane;
            const std::size_t target = static_cast<std::size_t>(token) * token_plane * 3;
            std::copy_n(q_expected.data() + source, token_plane,
                        interleaved_expected.data() + target);
            std::copy_n(k_expected.data() + source, token_plane,
                        interleaved_expected.data() + target + token_plane);
            std::copy_n(v_expected.data() + source, token_plane,
                        interleaved_expected.data() + target + token_plane * 2);
        }
        interleaved_storage = to_device(interleaved_expected);
        q_tensor            = Tensor(interleaved_storage.p, DType::BF16, {kDim, kHeads, tokens});
        q_tensor.nb[2]      = static_cast<std::int64_t>(token_plane * 3 * sizeof(std::uint16_t));
        k_tensor            = q_tensor;
        v_tensor            = q_tensor;
        k_tensor.data =
            static_cast<std::uint8_t*>(interleaved_storage.p) + token_plane * sizeof(std::uint16_t);
        v_tensor.data = static_cast<std::uint8_t*>(interleaved_storage.p) +
                        token_plane * 2 * sizeof(std::uint16_t);
    }

    DeviceBuffer d_cu_seqlens = to_device_i32(cu_seqlens);
    Tensor cu_tensor(d_cu_seqlens.p, DType::I32, {static_cast<std::int32_t>(cu_seqlens.size())});
    GuardedDeviceBuffer d_out(value_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    Tensor out_tensor(d_out.data(), DType::BF16, {kDim, kHeads, tokens});

    const std::int32_t segments       = static_cast<std::int32_t>(cu_seqlens.size()) - 1;
    const std::size_t workspace_bytes = ops::packed_softmax_attention_workspace_capacity_bytes(
        kGeometry, tokens, tokens, segments, segments);
    DeviceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (entry == PublicEntry::Plain) {
        if (cu_seqlens.size() != 2) {
            throw std::logic_error("plain case requires exactly one segment");
        }
        ops::softmax_attention(q_tensor, k_tensor, v_tensor, kGeometry, kScale, workspace,
                               out_tensor, nullptr);
    } else if (entry == PublicEntry::CuSeqlensArena) {
        ops::packed_softmax_attention(q_tensor, k_tensor, v_tensor, kGeometry, kScale, cu_tensor,
                                      workspace, out_tensor, nullptr);
    } else {
        const int segment_length = cu_seqlens[1] - cu_seqlens[0];
        for (std::size_t segment = 1; segment + 1 < cu_seqlens.size(); ++segment) {
            if (cu_seqlens[segment + 1] - cu_seqlens[segment] != segment_length) {
                throw std::logic_error("uniform case requires equal segments");
            }
        }
        ops::packed_softmax_attention(q_tensor, k_tensor, v_tensor, kGeometry, kScale,
                                      segment_length, out_tensor, nullptr);
    }
    cuda_synchronize();

    const std::string label = "packed_softmax_attention T=" + std::to_string(tokens) +
                              " S=" + std::to_string(cu_seqlens.size() - 1) + " " +
                              storage_name(storage_profile) + " " + entry_name(entry);
    const std::string qualified_label =
        input_profile == InputProfile::SegmentIsolation ? label + " segment-isolation" : label;
    int failures =
        verify_reduction(qualified_label.c_str(), from_device_bf16(d_out.data(), value_count),
                         reference, kPackedAttentionBf16Criterion);
    failures += d_out.verify_guards((qualified_label + " output guards").c_str());
    if (storage_profile == StorageProfile::Contiguous) {
        failures += verify_exact((qualified_label + " q unchanged").c_str(),
                                 from_device<std::uint16_t>(q_storage, value_count), q_expected);
        failures += verify_exact((qualified_label + " k unchanged").c_str(),
                                 from_device<std::uint16_t>(k_storage, value_count), k_expected);
        failures += verify_exact((qualified_label + " v unchanged").c_str(),
                                 from_device<std::uint16_t>(v_storage, value_count), v_expected);
    } else {
        failures += verify_exact(
            (qualified_label + " qkv storage unchanged").c_str(),
            from_device<std::uint16_t>(interleaved_storage, interleaved_expected.size()),
            interleaved_expected);
    }
    if (entry != PublicEntry::UniformSegments) {
        failures += verify_exact((qualified_label + " cu_seqlens unchanged").c_str(),
                                 from_device<int>(d_cu_seqlens, cu_seqlens.size()), cu_seqlens);
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
            std::cerr << qualified_label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int run_softmax_attention_plain_and_packed_tests() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    int failures = 0;
    if (ops::packed_softmax_attention_workspace_capacity_bytes(kGeometry, 4, 194, 1, 1) != 0 ||
        ops::packed_softmax_attention_workspace_capacity_bytes(kGeometry, 4, 194, 1, 3) !=
            ops::packed_softmax_attention_workspace_capacity_bytes(kGeometry, 194, 194, 3, 3)) {
        std::cerr
            << "packed_softmax_attention rectangular capacity missed its maximal legal pair\n";
        ++failures;
    }
    try {
        (void)ops::packed_softmax_attention_workspace_capacity_bytes(kGeometry, 1, 2, 3, 4);
        std::cerr << "packed_softmax_attention accepted an envelope without a legal segment pair\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += run_case({0, 4}, 11u, StorageProfile::Contiguous, PublicEntry::Plain);
    failures += run_case({0, 4}, 1u, StorageProfile::Contiguous, PublicEntry::CuSeqlensArena);
    failures += run_case({0, 4, 11}, 7u, StorageProfile::InterleavedQkv,
                         PublicEntry::CuSeqlensArena, InputProfile::SegmentIsolation);
    failures += run_case({0, 64, 129, 194}, 31u, StorageProfile::InterleavedQkv,
                         PublicEntry::CuSeqlensArena);
    failures +=
        run_case({0, 68, 136}, 101u, StorageProfile::InterleavedQkv, PublicEntry::UniformSegments);
    failures +=
        run_case({0, 256}, 2026u, StorageProfile::InterleavedQkv, PublicEntry::CuSeqlensArena);

    if (failures != 0) {
        std::cerr << "packed_softmax_attention failures=" << failures << '\n';
        return 1;
    }
    std::cout << "packed_softmax_attention: PASS\n";
    return 0;
}
