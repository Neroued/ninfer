#include "ninfer/ops/dynamic_grouped_conv.h"
#include "core/decode_graph.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden          = 5120;
constexpr std::int32_t kMaximumWidth    = 16;
constexpr std::int32_t kGroups          = 320;
constexpr std::int32_t kTaps            = 2;
constexpr std::int32_t kSides           = 2;
constexpr std::int32_t kCoefficientRows = 1280;
constexpr std::int32_t kMaximumBatch    = 8;
constexpr std::int32_t kMaximumTokens   = kMaximumWidth * kMaximumBatch;
constexpr float kEps                    = 1.0e-6F;

constexpr ReductionCriterion kFinishDeltaCriterion{/*relative_l2=*/3.2e-3,
                                                   /*gross_absolute=*/4.0e-3,
                                                   /*gross_relative=*/4.5e-3};
constexpr ReductionCriterion kPreparedCriterion{/*relative_l2=*/3.6e-3,
                                                /*gross_absolute=*/5.0e-3,
                                                /*gross_relative=*/6.0e-3};

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float signed_pattern(std::uint32_t index, std::uint32_t seed, float scale) {
    const std::uint32_t mixed = mix32(index ^ (seed * 0x9e3779b9U));
    int centered              = static_cast<int>((mixed >> 8) & 0xffU) - 128;
    if (centered == 0) { centered = (index & 1U) == 0 ? 1 : -1; }
    return static_cast<float>(centered) * scale;
}

std::vector<std::uint16_t> make_residual() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kHidden) * kMaximumTokens);
    for (std::int32_t token = 0; token < kMaximumTokens; ++token) {
        const std::int32_t batch    = token / kMaximumWidth;
        const std::int32_t position = token % kMaximumWidth;
        for (std::int32_t hidden = 0; hidden < kHidden; ++hidden) {
            float value = signed_pattern(static_cast<std::uint32_t>(token * kHidden + hidden), 101U,
                                         1.0F / 192.0F);
            if (hidden == (batch * 659 + position * 83) % kHidden) {
                value += (position == 0 ? 1.5F : -1.25F) * static_cast<float>(batch + 1);
            }
            result[static_cast<std::size_t>(token) * kHidden + hidden] = f32_to_bf16(value);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_norm_weight() {
    std::vector<std::uint16_t> result(kHidden);
    for (std::int32_t hidden = 0; hidden < kHidden; ++hidden) {
        const float value = 0.875F + static_cast<float>((hidden * 17 + 11) % 65) * (0.25F / 64.0F);
        result[hidden]    = f32_to_bf16(value);
    }
    return result;
}

std::vector<std::uint16_t> make_base_kernel() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kHidden) * kTaps * kSides);
    for (std::int32_t side = 0; side < kSides; ++side) {
        for (std::int32_t tap = 0; tap < kTaps; ++tap) {
            for (std::int32_t hidden = 0; hidden < kHidden; ++hidden) {
                const float center =
                    side == 0 ? (tap == 0 ? 0.75F : -0.125F) : (tap == 0 ? 0.625F : 0.1875F);
                const float perturbation =
                    signed_pattern(static_cast<std::uint32_t>((side * 2 + tap) * kHidden + hidden),
                                   211U, 1.0F / 8192.0F);
                result[static_cast<std::size_t>(side * 2 + tap) * kHidden + hidden] =
                    f32_to_bf16(center + perturbation);
            }
        }
    }
    return result;
}

direct_bf16_weight::HostWeight make_projection_weight() {
    direct_bf16_weight::HostWeight result;
    result.n = kCoefficientRows;
    result.k = kHidden;
    result.bits.resize(static_cast<std::size_t>(result.n) * result.k);
    for (std::int32_t row = 0; row < result.n; ++row) {
        for (std::int32_t column = 0; column < result.k; ++column) {
            const std::uint32_t index = static_cast<std::uint32_t>(row * result.k + column);
            result.bits[static_cast<std::size_t>(row) * result.k + column] =
                f32_to_bf16(signed_pattern(index, 307U, 1.0F / 8192.0F));
        }
    }
    return result;
}

struct Reference {
    std::vector<double> normalized;
    std::vector<double> coefficients;
    std::vector<double> prepared;
    std::vector<double> finish_delta;
};

Reference compute_reference(const std::vector<std::uint16_t>& residual,
                            const std::vector<std::uint16_t>& norm_weight,
                            const std::vector<std::uint16_t>& base_kernel,
                            const direct_bf16_weight::HostWeight& projection_weight) {
    std::vector<double> normalized(static_cast<std::size_t>(kHidden) * kMaximumTokens);
    for (std::int32_t token = 0; token < kMaximumTokens; ++token) {
        const std::size_t token_begin = static_cast<std::size_t>(token) * kHidden;
        double sum_squares            = 0.0;
        for (std::int32_t hidden = 0; hidden < kHidden; ++hidden) {
            const double value = bf16_to_f32(residual[token_begin + hidden]);
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(kHidden) + kEps);
        for (std::int32_t hidden = 0; hidden < kHidden; ++hidden) {
            normalized[token_begin + hidden] =
                static_cast<double>(bf16_to_f32(residual[token_begin + hidden])) * inverse *
                static_cast<double>(bf16_to_f32(norm_weight[hidden]));
        }
    }

    std::vector<double> coefficients(static_cast<std::size_t>(kCoefficientRows) * kMaximumTokens,
                                     0.0);
    const unsigned available_threads = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t thread_count =
        std::min<std::int32_t>(kCoefficientRows, static_cast<std::int32_t>(available_threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (std::int32_t thread = 0; thread < thread_count; ++thread) {
        const std::int32_t row_begin = static_cast<std::int32_t>(
            static_cast<std::int64_t>(kCoefficientRows) * thread / thread_count);
        const std::int32_t row_end = static_cast<std::int32_t>(
            static_cast<std::int64_t>(kCoefficientRows) * (thread + 1) / thread_count);
        workers.emplace_back([&, row_begin, row_end] {
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                const std::uint16_t* weight_row =
                    projection_weight.bits.data() + static_cast<std::size_t>(row) * kHidden;
                for (std::int32_t token = 0; token < kMaximumTokens; ++token) {
                    const double* normalized_row =
                        normalized.data() + static_cast<std::size_t>(token) * kHidden;
                    double sum = 0.0;
                    for (std::int32_t hidden = 0; hidden < kHidden; ++hidden) {
                        sum += static_cast<double>(bf16_to_f32(weight_row[hidden])) *
                               normalized_row[hidden];
                    }
                    coefficients[static_cast<std::size_t>(token) * kCoefficientRows + row] = sum;
                }
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }

    Reference result;
    result.prepared.resize(static_cast<std::size_t>(kHidden) * kMaximumTokens);
    result.finish_delta.resize(static_cast<std::size_t>(kGroups) * kTaps * kMaximumTokens);
    for (std::int32_t token = 0; token < kMaximumTokens; ++token) {
        const std::int32_t position       = token % kMaximumWidth;
        const std::size_t normalized_base = static_cast<std::size_t>(token) * kHidden;
        for (std::int32_t group = 0; group < kGroups; ++group) {
            const double input_delta0 =
                coefficients[static_cast<std::size_t>(token) * kCoefficientRows + group];
            const double input_delta1 =
                coefficients[static_cast<std::size_t>(token) * kCoefficientRows + kGroups + group];
            for (std::int32_t tap = 0; tap < kTaps; ++tap) {
                const std::int32_t row = (2 + tap) * kGroups + group;
                result.finish_delta[(static_cast<std::size_t>(token) * kTaps + tap) * kGroups +
                                    group] =
                    coefficients[static_cast<std::size_t>(token) * kCoefficientRows + row];
            }
            for (std::int32_t channel = 0; channel < 16; ++channel) {
                const std::int32_t hidden = group * 16 + channel;
                const double base0        = bf16_to_f32(base_kernel[hidden]);
                const double base1        = bf16_to_f32(base_kernel[kHidden + hidden]);
                double value = (base0 + input_delta0) * normalized[normalized_base + hidden];
                if (position > 0) {
                    value +=
                        (base1 + input_delta1) * normalized[normalized_base - kHidden + hidden];
                }
                result.prepared[normalized_base + hidden] = value;
            }
        }
    }
    result.normalized   = std::move(normalized);
    result.coefficients = std::move(coefficients);
    return result;
}

int verify_preserved(std::string_view label, const DeviceBuffer& device,
                     const std::vector<std::uint16_t>& expected) {
    const std::string owned_label(label);
    return verify_exact(owned_label.c_str(), from_device<std::uint16_t>(device, expected.size()),
                        expected);
}

void set_reference_width(Reference& reference, const std::vector<std::uint16_t>& base, int width) {
    for (int token = 0; token < kMaximumTokens; ++token) {
        for (int h = 0; h < kHidden; ++h) {
            const int group = h / 16;
            const double d0 = reference.coefficients[token * kCoefficientRows + group];
            const double d1 = reference.coefficients[token * kCoefficientRows + kGroups + group];
            const std::size_t offset = static_cast<std::size_t>(token) * kHidden + h;
            double value             = (bf16_to_f32(base[h]) + d0) * reference.normalized[offset];
            if (token % width > 0)
                value +=
                    (bf16_to_f32(base[kHidden + h]) + d1) * reference.normalized[offset - kHidden];
            reference.prepared[offset] = value;
        }
    }
}

int run() {
    const auto residual_host = make_residual(), norm_host = make_norm_weight(),
               base_host = make_base_kernel();
    direct_bf16_weight::DeviceWeight projection_weight(make_projection_weight());
    Reference reference =
        compute_reference(residual_host, norm_host, base_host, projection_weight.host);
    DeviceBuffer residual_device = to_device(residual_host), norm_device = to_device(norm_host),
                 base_device = to_device(base_host);
    const auto capacity =
        ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(2, 16, 1, 8);
    WorkspaceArena workspace(capacity);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    Tensor norm(norm_device.p, DType::BF16, {kHidden});
    Tensor base(base_device.p, DType::BF16, {kHidden, kTaps, kSides});
    const Weight weight = projection_weight.view();
    int failures        = 0;
    for (int width = 2; width <= 16; ++width) {
        set_reference_width(reference, base_host, width);
        for (int batch = 1; batch <= 8; ++batch) {
            const std::size_t pe = static_cast<std::size_t>(kHidden) * width * batch;
            const std::size_t fe = static_cast<std::size_t>(kGroups) * 2 * width * batch;
            GuardedDeviceBuffer pd(pe * sizeof(std::uint16_t)), fd(fe * sizeof(std::uint16_t));
            Tensor residual(residual_device.p, DType::BF16, {kHidden, width, batch});
            Tensor prepared(pd.data(), DType::BF16, {kHidden, width, batch});
            Tensor finish(fd.data(), DType::BF16, {kGroups, kTaps, width, batch});
            const bool graph_case = (width == 3 && batch == 3) || (width == 7 && batch == 7) ||
                                    (width == 14 && batch == 7) || (width == 16 && batch == 8);
            for (int replay = 0; replay < (graph_case ? 2 : 1); ++replay) {
                pd.fill(0xff);
                fd.fill(0xff);
                // Fixture fills use the default stream; execution intentionally uses a nonblocking
                // stream.
                cuda_check(cudaStreamSynchronize(nullptr), "fixture fill synchronize");
                workspace.reset_peak();
                const auto launch = [&] {
                    ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm, kEps, base, weight,
                                                              prepared, finish, workspace, stream);
                };
                if (replay) {
                    DecodeGraphDefinition definition;
                    DecodeGraphExecutable executable;
                    definition.capture(stream, launch);
                    executable.instantiate(definition);
                    executable.launch(stream);
                    executable.launch(stream);
                    cuda_check(cudaStreamSynchronize(stream), "graph synchronize");
                } else
                    launch();
                cuda_synchronize();
                const std::string label = "dynamic conv prepare W=" + std::to_string(width) +
                                          " B=" + std::to_string(batch) +
                                          " graph=" + std::to_string(replay);
                failures += verify_reduction(label + " prepared", from_device_bf16(pd.data(), pe),
                                             std::span<const double>(reference.prepared.data(), pe),
                                             kPreparedCriterion);
                failures +=
                    verify_reduction(label + " finish", from_device_bf16(fd.data(), fe),
                                     std::span<const double>(reference.finish_delta.data(), fe),
                                     kFinishDeltaCriterion);
                failures += pd.verify_guards(label);
                failures += fd.verify_guards(label);
                const auto exact =
                    ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                        width, width, batch, batch);
                if (workspace.used() != 0 || workspace.peak_used() != exact) {
                    std::cerr << label << ": workspace mismatch\n";
                    ++failures;
                }
            }
        }
    }
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    failures += verify_preserved("residual", residual_device, residual_host);
    failures += verify_preserved("norm", norm_device, norm_host);
    failures += verify_preserved("base", base_device, base_host);
    failures += projection_weight.verify_preserved("projection");
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        const int failures = run();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " rmsnorm_dynamic_grouped_conv_prepare\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "rmsnorm_dynamic_grouped_conv_prepare test: " << error.what() << '\n';
        return 1;
    }
}
