#include "ops/suite.h"
#include "ninfer/ops/softmax_attention.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"

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

constexpr int kDim   = 72;
constexpr int kHeads = 16;
constexpr ops::AttentionHeadGeometry kGeometry{kDim, kHeads, kHeads};

constexpr ReductionCriterion kPackedAttentionBf16Criterion{
    .relative_l2    = 2.5e-3,
    .gross_absolute = 1e-3,
    // A BF16 result alone can round by 2^-8; allow the additional probability
    // representation while the normwise criterion controls overall accuracy.
    .gross_relative_to_max_reference = 8e-3,
};

std::size_t index_of(int dim, int token, int head, int d) {
    return (static_cast<std::size_t>(token) * kHeads + static_cast<std::size_t>(head)) * dim +
           static_cast<std::size_t>(d);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

void packed_attention_oracle(const std::vector<float>& q, const std::vector<float>& k,
                             const std::vector<float>& v, const std::vector<int>& cu_seqlens,
                             std::vector<double>& out, int dim) {
    const double scale = 1.0 / std::sqrt(double(dim));
    const ops::AttentionHeadGeometry geometry{dim, kHeads, kHeads};
    out.assign(q.size(), std::numeric_limits<double>::quiet_NaN());

    for (std::size_t segment = 0; segment + 1 < cu_seqlens.size(); ++segment) {
        const int begin = cu_seqlens[segment];
        const int end   = cu_seqlens[segment + 1];
        std::vector<int> queries;
        if (end - begin <= 768) {
            for (int i = begin; i < end; ++i) queries.push_back(i);
        } else {
            // Sample independent rows, evaluating each sampled row's complete key set.
            for (int offset :
                 {0, 1, 15, 31, 63, 127, (end - begin) / 2, end - begin - 2, end - begin - 1})
                queries.push_back(begin + offset);
        }
        naive_dense_softmax_attention(
            geometry, queries.size(), end - begin, scale,
            [&](int d, int head, int token) {
                return static_cast<double>(q[index_of(dim, queries[token], head, d)]);
            },
            [&](int d, int head, int token) {
                return static_cast<double>(k[index_of(dim, begin + token, head, d)]);
            },
            [&](int d, int head, int token) {
                return static_cast<double>(v[index_of(dim, begin + token, head, d)]);
            },
            [](int, int) { return true; },
            [&](int d, int head, int token, double value) {
                out[index_of(dim, queries[token], head, d)] = value;
            },
            true);
    }
}

enum class StorageProfile {
    Contiguous,
    InterleavedQkv,
    NaturalPadded,
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
    return profile == StorageProfile::Contiguous      ? "contiguous"
           : profile == StorageProfile::NaturalPadded ? "natural-padded"
                                                      : "interleaved-qkv";
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
             PublicEntry entry, InputProfile input_profile = InputProfile::Random,
             bool graph = false, int dim = 72) {
    const ops::AttentionHeadGeometry geometry{dim, kHeads, kHeads};
    const float scale              = 1.F / std::sqrt(float(dim));
    const int tokens               = cu_seqlens.back();
    const std::size_t token_plane  = static_cast<std::size_t>(kHeads) * dim;
    const std::size_t value_count  = static_cast<std::size_t>(tokens) * token_plane;
    const std::size_t prefix       = storage_profile == StorageProfile::NaturalPadded ? 1 : 0;
    const std::size_t token_stride = token_plane * 3 + prefix;
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
    packed_attention_oracle(q, k, v, cu_seqlens, reference, dim);

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
        q_tensor  = Tensor(q_storage.span(), DType::BF16, {dim, kHeads, tokens});
        k_tensor  = Tensor(k_storage.span(), DType::BF16, {dim, kHeads, tokens});
        v_tensor  = Tensor(v_storage.span(), DType::BF16, {dim, kHeads, tokens});
    } else {
        interleaved_expected.resize(prefix + token_stride * tokens, 0x5a5a);
        for (int token = 0; token < tokens; ++token) {
            const std::size_t source = static_cast<std::size_t>(token) * token_plane;
            const std::size_t target = prefix + static_cast<std::size_t>(token) * token_stride;
            std::copy_n(q_expected.data() + source, token_plane,
                        interleaved_expected.data() + target);
            std::copy_n(k_expected.data() + source, token_plane,
                        interleaved_expected.data() + target + token_plane);
            std::copy_n(v_expected.data() + source, token_plane,
                        interleaved_expected.data() + target + token_plane * 2);
        }
        interleaved_storage = to_device(interleaved_expected);
        q_tensor            = Tensor(interleaved_storage.span().subspan(prefix * 2), DType::BF16,
                                     {dim, kHeads, tokens});
        q_tensor.nb[2]      = static_cast<std::int64_t>(token_stride * sizeof(std::uint16_t));
        k_tensor            = q_tensor;
        v_tensor            = q_tensor;
        k_tensor.data =
            static_cast<std::uint8_t*>(q_tensor.data) + token_plane * sizeof(std::uint16_t);
        v_tensor.data =
            static_cast<std::uint8_t*>(q_tensor.data) + token_plane * 2 * sizeof(std::uint16_t);
    }

    DeviceBuffer d_cu_seqlens = to_device_i32(cu_seqlens);
    Tensor cu_tensor(d_cu_seqlens.span(), DType::I32,
                     {static_cast<std::int32_t>(cu_seqlens.size())});
    GuardedDeviceBuffer d_out((value_count + prefix) * sizeof(std::uint16_t));
    d_out.fill(0x7f);
    Tensor out_tensor(d_out.span().subspan(prefix * 2), DType::BF16, {dim, kHeads, tokens});

    const std::int32_t segments       = static_cast<std::int32_t>(cu_seqlens.size()) - 1;
    const std::size_t workspace_bytes = ops::packed_softmax_attention_workspace_capacity_bytes(
        geometry, tokens, tokens, segments, segments);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(256, workspace_bytes));
    MemoryArena workspace(workspace_storage.span());

    cuda_synchronize();
    const auto execution = cuda_environment().context.execution_view();
    const auto launch    = [&] {
        if (entry == PublicEntry::Plain) {
            if (cu_seqlens.size() != 2) {
                throw std::logic_error("plain case requires exactly one segment");
            }
            ops::softmax_attention(q_tensor, k_tensor, v_tensor, geometry, scale, workspace,
                                      out_tensor, execution);
        } else if (entry == PublicEntry::CuSeqlensArena) {
            ops::packed_softmax_attention(q_tensor, k_tensor, v_tensor, geometry, scale, cu_tensor,
                                             workspace, out_tensor, execution);
        } else {
            const int segment_length = cu_seqlens[1] - cu_seqlens[0];
            for (std::size_t segment = 1; segment + 1 < cu_seqlens.size(); ++segment) {
                if (cu_seqlens[segment + 1] - cu_seqlens[segment] != segment_length) {
                    throw std::logic_error("uniform case requires equal segments");
                }
            }
            ops::packed_softmax_attention(q_tensor, k_tensor, v_tensor, geometry, scale,
                                             segment_length, out_tensor, execution);
        }
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    if (graph) {
        definition.capture(execution.stream(), launch);
        executable.instantiate(definition);
        executable.launch(execution.stream());
    } else {
        launch();
    }
    cuda_synchronize();

    const std::string label = "packed_softmax_attention T=" + std::to_string(tokens) +
                              " S=" + std::to_string(cu_seqlens.size() - 1) + " " +
                              storage_name(storage_profile) + " " + entry_name(entry);
    const std::string qualified_label =
        input_profile == InputProfile::SegmentIsolation ? label + " segment-isolation" : label;
    const auto verify = [&] {
        const auto actual = from_device_bf16(out_tensor.data, value_count);
        std::vector<double> selected_actual, selected_reference;
        int failures = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) ++failures;
            if (std::isfinite(reference[i])) {
                selected_actual.push_back(actual[i]);
                selected_reference.push_back(reference[i]);
            }
        }
        return failures + verify_reduction(qualified_label, selected_actual, selected_reference,
                                           kPackedAttentionBf16Criterion);
    };
    int failures = verify();
    if (prefix)
        failures += verify_exact("output prefix", from_device<uint16_t>(d_out.data(), 1),
                                 std::vector<uint16_t>{0x7f7f});
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
    if (graph) {
        for (auto& value : v) value = -value;
        const auto changed_v = bf16_bits(v);
        if (storage_profile == StorageProfile::Contiguous) {
            CUDA_CHECK(cudaMemcpy(v_storage.data(), changed_v.data(), value_count * 2,
                                  cudaMemcpyHostToDevice));
        } else {
            for (int t = 0; t < tokens; ++t)
                std::copy_n(changed_v.data() + std::size_t(t) * token_plane, token_plane,
                            interleaved_expected.data() + prefix + std::size_t(t) * token_stride +
                                token_plane * 2);
            CUDA_CHECK(cudaMemcpy(interleaved_storage.data(), interleaved_expected.data(),
                                  interleaved_expected.size() * 2, cudaMemcpyHostToDevice));
        }
        auto changed_segments = cu_seqlens;
        if (entry == PublicEntry::CuSeqlensArena)
            for (std::size_t i = 1; i + 1 < changed_segments.size(); ++i)
                if (changed_segments[i] + 1 < changed_segments[i + 1]) ++changed_segments[i];
        CUDA_CHECK(cudaMemcpy(d_cu_seqlens.data(), changed_segments.data(),
                              changed_segments.size() * 4, cudaMemcpyHostToDevice));
        packed_attention_oracle(q, k, v, changed_segments, reference, dim);
        d_out.fill(0x7f);
        executable.launch(execution.stream());
        cuda_synchronize();
        failures += verify() + d_out.verify_guards("replayed output guards");
        failures +=
            verify_exact("replayed segment metadata",
                         from_device<int>(d_cu_seqlens, changed_segments.size()), changed_segments);
        if (workspace.used() != 0) ++failures;
    }
    return failures;
}

} // namespace

void add_plain_attention_cases(ninfer::test::OpSuite& suite) {
    suite.add(
        "packed_attention/capacity",
        [] {
            int failures = 0;
            if (ops::packed_softmax_attention_workspace_capacity_bytes(kGeometry, 4, 194, 1, 1) !=
                    0 ||
                ops::packed_softmax_attention_workspace_capacity_bytes(kGeometry, 4, 194, 1, 3) !=
                    ops::packed_softmax_attention_workspace_capacity_bytes(kGeometry, 194, 194, 3,
                                                                           3)) {
                std::cerr << "packed_softmax_attention rectangular capacity missed its maximal "
                             "legal pair\n";
                ++failures;
            }
            try {
                (void)ops::packed_softmax_attention_workspace_capacity_bytes(kGeometry, 1, 2, 3, 4);
                std::cerr << "packed_softmax_attention accepted an envelope without a legal "
                             "segment pair\n";
                ++failures;
            } catch (const std::invalid_argument&) {}
            return failures;
        },
        {{"format", "bf16"}});
    const auto add = [&](std::vector<int> offsets, unsigned seed, StorageProfile storage,
                         PublicEntry entry, InputProfile input = InputProfile::Random,
                         bool graph = false) {
        auto id =
            std::string("packed_attention/") + entry_name(entry) + "/" + storage_name(storage);
        for (int offset : offsets) id += "_" + std::to_string(offset);
        if (graph) id += "/graph";
        suite.add(id, [=] { return run_case(offsets, seed, storage, entry, input, graph); },
                  {{"format", "bf16"}, {"execution", graph ? "graph" : "eager"}});
    };
    add({0, 4}, 11u, StorageProfile::Contiguous, PublicEntry::Plain);
    add({0, 4}, 1u, StorageProfile::Contiguous, PublicEntry::CuSeqlensArena);
    add({0, 4, 11}, 7u, StorageProfile::InterleavedQkv, PublicEntry::CuSeqlensArena,
        InputProfile::SegmentIsolation);
    add({0, 64, 129, 194}, 31u, StorageProfile::InterleavedQkv, PublicEntry::CuSeqlensArena);
    add({0, 68, 136}, 101u, StorageProfile::InterleavedQkv, PublicEntry::UniformSegments);
    add({0, 256}, 2026u, StorageProfile::InterleavedQkv, PublicEntry::CuSeqlensArena);
    add({0, 33, 66, 99}, 501u, StorageProfile::InterleavedQkv, PublicEntry::UniformSegments);
    add({0, 512}, 503u, StorageProfile::InterleavedQkv, PublicEntry::UniformSegments,
        InputProfile::Random, true);
    add({0, 513}, 521u, StorageProfile::InterleavedQkv, PublicEntry::UniformSegments);
    add({0, 129, 258}, 509u, StorageProfile::InterleavedQkv, PublicEntry::UniformSegments,
        InputProfile::SegmentIsolation, true);
    const auto d64 = [&](std::vector<int> offsets, PublicEntry entry,
                         StorageProfile storage = StorageProfile::Contiguous, bool graph = false,
                         InputProfile input = InputProfile::Random) {
        auto id =
            std::string("packed_attention/D64/") + entry_name(entry) + "/" + storage_name(storage);
        for (int offset : offsets) id += "_" + std::to_string(offset);
        suite.add(id, [=] { return run_case(offsets, 7103, storage, entry, input, graph, 64); },
                  {{"format", "bf16"}, {"execution", graph ? "graph" : "eager"}});
    };
    for (int n : {1, 16, 17, 33, 64, 65, 255, 256, 257, 4097}) d64({0, n}, PublicEntry::Plain);
    d64({0, 1, 18, 81, 338}, PublicEntry::CuSeqlensArena, StorageProfile::InterleavedQkv, true);
    d64({0, 3, 6, 9}, PublicEntry::CuSeqlensArena, StorageProfile::InterleavedQkv, true);
    std::vector<int> short_segments(258, 0);
    for (int s = 1; s < int(short_segments.size()); ++s)
        short_segments[s] = short_segments[s - 1] + 1 + s % 17;
    d64(short_segments, PublicEntry::CuSeqlensArena, StorageProfile::InterleavedQkv, true);
    d64({0, 3, 20, 85}, PublicEntry::CuSeqlensArena, StorageProfile::Contiguous, false,
        InputProfile::SegmentIsolation);
    d64({0, 4096}, PublicEntry::CuSeqlensArena);
    d64({0, 65}, PublicEntry::Plain, StorageProfile::NaturalPadded);
    d64({0, 1}, PublicEntry::Plain, StorageProfile::NaturalPadded, true);
    d64({0, 3, 20, 85}, PublicEntry::CuSeqlensArena, StorageProfile::NaturalPadded, true);
    d64({0, 4096, 8192}, PublicEntry::CuSeqlensArena, StorageProfile::InterleavedQkv);
    d64({0, 2048, 4097}, PublicEntry::CuSeqlensArena, StorageProfile::InterleavedQkv);
    for (int n : {1, 16, 17, 33, 64, 128, 256, 513, 4096})
        d64({0, n, 2 * n}, PublicEntry::UniformSegments, StorageProfile::InterleavedQkv, n == 33);
    suite.add("packed_attention/D64/capacity", [] {
        const ops::AttentionHeadGeometry geometry{64, 16, 16};
        int failures =
            ops::packed_softmax_attention_workspace_capacity_bytes(geometry, 1, 4097, 1, 1) != 0;
        const auto capacity =
            ops::packed_softmax_attention_workspace_capacity_bytes(geometry, 1, 16385, 1, 4);
        for (int tokens : {4, 16, 17, 32, 65, 127, 4095, 8191, 8192, 16383, 16384, 16385})
            for (int segments : {1, 2, 3, 4})
                failures += ops::packed_softmax_attention_workspace_capacity_bytes(
                                geometry, tokens, tokens, segments, segments) > capacity;
        return failures;
    });
}
