#include "ninfer/ops/rmsnorm_rope.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim         = 128;
constexpr int kQueryHeads      = 32;
constexpr int kKeyHeads        = 8;
constexpr double kEpsilon      = 1.0e-6;
constexpr double kTheta        = 1.0e7;
constexpr double kRelativeL2   = 1.85e-3;
constexpr double kPairRelative = 6.9e-3;

struct OracleResult {
    std::vector<double> output;
    std::vector<double> pair_scale;
};

std::size_t dense_index(int heads, int token, int head, int dim) {
    return (static_cast<std::size_t>(token) * heads + head) * kHeadDim + dim;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return values;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](float value) { return f32_to_bf16(value); });
    return bits;
}

std::vector<std::int32_t> make_positions(int tokens, int first_position) {
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = first_position + token;
    }
    return positions;
}

// Independent FP64 oracle for the complete public formula. It does not call either standalone Op
// or reproduce the production warp reduction, coefficient table, or range reduction.
OracleResult fused_oracle(const std::vector<float>& input, const std::vector<float>& weight,
                          const std::vector<std::int32_t>& positions, int heads) {
    const int tokens = static_cast<int>(positions.size());
    OracleResult result{
        .output     = std::vector<double>(input.size()),
        .pair_scale = std::vector<double>(input.size()),
    };
    std::vector<double> normalized(kHeadDim);
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            double sum_squares = 0.0;
            for (int dim = 0; dim < kHeadDim; ++dim) {
                const double value = input[dense_index(heads, token, head, dim)];
                sum_squares += value * value;
            }
            const double inverse =
                1.0 / std::sqrt(sum_squares / static_cast<double>(kHeadDim) + kEpsilon);
            for (int dim = 0; dim < kHeadDim; ++dim) {
                normalized[static_cast<std::size_t>(dim)] =
                    static_cast<double>(input[dense_index(heads, token, head, dim)]) * inverse *
                    static_cast<double>(weight[static_cast<std::size_t>(dim)]);
            }
            for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                const double exponent = -2.0 * static_cast<double>(pair) / kHeadDim;
                const double phase =
                    static_cast<double>(positions[static_cast<std::size_t>(token)]) *
                    std::pow(kTheta, exponent);
                const double cosine = std::cos(phase);
                const double sine   = std::sin(phase);
                const double first  = normalized[static_cast<std::size_t>(pair)];
                const double second = normalized[static_cast<std::size_t>(pair + kHeadDim / 2)];
                const double scale  = std::hypot(first, second);
                const std::size_t first_index = dense_index(heads, token, head, pair);
                const std::size_t second_index =
                    dense_index(heads, token, head, pair + kHeadDim / 2);
                result.output[first_index]      = first * cosine - second * sine;
                result.output[second_index]     = second * cosine + first * sine;
                result.pair_scale[first_index]  = scale;
                result.pair_scale[second_index] = scale;
            }
        }
    }
    return result;
}

int verify_profile(const std::string& label, const std::vector<double>& got,
                   const OracleResult& expected) {
    if (got.size() != expected.output.size() || got.size() != expected.pair_scale.size()) {
        std::cerr << label << ": result size mismatch\n";
        return 1;
    }
    double error_square_sum     = 0.0;
    double reference_square_sum = 0.0;
    double maximum_ratio        = 0.0;
    int violations              = 0;
    for (std::size_t index = 0; index < got.size(); ++index) {
        if (!std::isfinite(got[index]) || !std::isfinite(expected.output[index])) {
            std::cerr << label << ": non-finite value at index " << index << '\n';
            return 1;
        }
        const double error = std::abs(got[index] - expected.output[index]);
        const double scale = expected.pair_scale[index];
        const double limit = kPairRelative * scale;
        const double ratio = limit == 0.0
                                 ? (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                                 : error / limit;
        maximum_ratio      = std::max(maximum_ratio, ratio);
        if (error > limit) {
            ++violations;
            if (violations == 1) {
                std::cerr << label << ": pair-scaled mismatch at index " << index
                          << " error=" << error << " scale=" << scale << '\n';
            }
        }
        error_square_sum += error * error;
        reference_square_sum += expected.output[index] * expected.output[index];
    }
    const double relative_l2 = std::sqrt(error_square_sum / reference_square_sum);
    if (relative_l2 > kRelativeL2) {
        std::cerr << label << ": relative L2=" << relative_l2 << " exceeds " << kRelativeL2 << '\n';
        ++violations;
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=rmsnorm_rope count=" << got.size()
                  << " relative_l2=" << relative_l2 << " max_pair_limit_ratio=" << maximum_ratio
                  << " case=" << label << '\n';
    }
    if (violations != 0) {
        std::cerr << label << ": " << violations << " numerical violations\n";
        return 1;
    }
    return 0;
}

template <class Launch, class Reset>
void execute(Launch launch, Reset reset, bool graph) {
    if (!graph) {
        launch(nullptr);
        cuda_synchronize();
        return;
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(stream, [&] { launch(stream); });
        executable.instantiate(definition);
        for (int i = 0; i < 2; ++i) {
            reset(stream);
            executable.launch(stream);
            cuda_check(cudaStreamSynchronize(stream), "graph replay synchronize");
        }
    }
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
}

int run_pair_case(int width, int batch, int first_position, std::uint32_t seed,
                  bool graph = false) {
    const int tokens              = width * batch;
    const std::size_t q_count     = static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens;
    const std::size_t k_count     = static_cast<std::size_t>(kHeadDim) * kKeyHeads * tokens;
    const auto q                  = make_bf16_values(q_count, seed, -4.0F, 4.0F);
    const auto k                  = make_bf16_values(k_count, seed + 1U, -4.0F, 4.0F);
    const auto q_weight           = make_bf16_values(kHeadDim, seed + 2U, 0.25F, 1.75F);
    const auto k_weight           = make_bf16_values(kHeadDim, seed + 3U, 0.25F, 1.75F);
    const auto positions          = make_positions(tokens, first_position);
    const OracleResult q_expected = fused_oracle(q, q_weight, positions, kQueryHeads);
    const OracleResult k_expected = fused_oracle(k, k_weight, positions, kKeyHeads);
    const auto q_bits             = bf16_bits(q);
    const auto k_bits             = bf16_bits(k);
    const auto q_weight_bits      = bf16_bits(q_weight);
    const auto k_weight_bits      = bf16_bits(k_weight);

    GuardedDeviceBuffer q_device(q_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer k_device(k_count * sizeof(std::uint16_t));
    q_device.copy_from_host(q_bits.data(), q_device.bytes());
    k_device.copy_from_host(k_bits.data(), k_device.bytes());
    DeviceBuffer q_weight_device = to_device(q_weight_bits);
    DeviceBuffer k_weight_device = to_device(k_weight_bits);
    DeviceBuffer position_device = to_device(positions);
    Tensor q_tensor(q_device.data(), DType::BF16, {kHeadDim, kQueryHeads, width, batch});
    Tensor k_tensor(k_device.data(), DType::BF16, {kHeadDim, kKeyHeads, width, batch});
    Tensor q_weight_tensor(q_weight_device.p, DType::BF16, {kHeadDim});
    Tensor k_weight_tensor(k_weight_device.p, DType::BF16, {kHeadDim});
    Tensor position_tensor(position_device.p, DType::I32, {width, batch});
    execute(
        [&](cudaStream_t stream) {
            ops::rmsnorm_rope(position_tensor, q_weight_tensor, k_weight_tensor, q_tensor, k_tensor,
                              stream);
        },
        [&](cudaStream_t stream) {
            cuda_check(cudaMemcpyAsync(q_device.data(), q_bits.data(), q_device.bytes(),
                                       cudaMemcpyHostToDevice, stream),
                       "reset q");
            cuda_check(cudaMemcpyAsync(k_device.data(), k_bits.data(), k_device.bytes(),
                                       cudaMemcpyHostToDevice, stream),
                       "reset k");
        },
        graph);

    const std::string label = "rmsnorm_rope pair W=" + std::to_string(width) +
                              " graph=" + std::to_string(graph) + " B=" + std::to_string(batch) +
                              " P=" + std::to_string(first_position);
    int failures =
        verify_profile(label + " q", from_device_bf16(q_device.data(), q_count), q_expected);
    failures +=
        verify_profile(label + " k", from_device_bf16(k_device.data(), k_count), k_expected);
    failures += q_device.verify_guards(label + " q guards");
    failures += k_device.verify_guards(label + " k guards");
    failures +=
        verify_exact((label + " positions").c_str(),
                     from_device<std::int32_t>(position_device, positions.size()), positions);
    failures += verify_exact((label + " q weight").c_str(),
                             from_device<std::uint16_t>(q_weight_device, q_weight_bits.size()),
                             q_weight_bits);
    failures += verify_exact((label + " k weight").c_str(),
                             from_device<std::uint16_t>(k_weight_device, k_weight_bits.size()),
                             k_weight_bits);
    return failures;
}

int run_single_case(int tokens, int first_position, std::uint32_t seed, bool graph = false) {
    const std::size_t count     = static_cast<std::size_t>(kHeadDim) * kKeyHeads * tokens;
    const auto input            = make_bf16_values(count, seed, -4.0F, 4.0F);
    const auto weight           = make_bf16_values(kHeadDim, seed + 1U, 0.25F, 1.75F);
    const auto positions        = make_positions(tokens, first_position);
    const OracleResult expected = fused_oracle(input, weight, positions, kKeyHeads);
    const auto input_bits       = bf16_bits(input);
    const auto weight_bits      = bf16_bits(weight);

    GuardedDeviceBuffer input_device(count * sizeof(std::uint16_t));
    input_device.copy_from_host(input_bits.data(), input_device.bytes());
    DeviceBuffer weight_device   = to_device(weight_bits);
    DeviceBuffer position_device = to_device(positions);
    Tensor input_tensor(input_device.data(), DType::BF16, {kHeadDim, kKeyHeads, tokens});
    Tensor weight_tensor(weight_device.p, DType::BF16, {kHeadDim});
    Tensor position_tensor(position_device.p, DType::I32, {tokens});
    execute(
        [&](cudaStream_t stream) {
            ops::rmsnorm_rope(position_tensor, weight_tensor, input_tensor, stream);
        },
        [&](cudaStream_t stream) {
            cuda_check(cudaMemcpyAsync(input_device.data(), input_bits.data(), input_device.bytes(),
                                       cudaMemcpyHostToDevice, stream),
                       "reset K");
        },
        graph);

    const std::string label = "rmsnorm_rope single graph=" + std::to_string(graph) +
                              " T=" + std::to_string(tokens) +
                              " P=" + std::to_string(first_position);
    int failures = verify_profile(label, from_device_bf16(input_device.data(), count), expected);
    failures += input_device.verify_guards(label + " guards");
    failures +=
        verify_exact((label + " positions").c_str(),
                     from_device<std::int32_t>(position_device, positions.size()), positions);
    failures +=
        verify_exact((label + " weight").c_str(),
                     from_device<std::uint16_t>(weight_device, weight_bits.size()), weight_bits);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "rmsnorm_rope: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    for (int width = 2; width <= 16; ++width)
        for (int batch = 1; batch <= 8; ++batch)
            failures += run_pair_case(width, batch, width == 2 && batch == 1 ? 0 : 262000,
                                      0x1000U + width * 8 + batch);
    failures += run_pair_case(3, 3, 0, 0x1001U, true);
    failures += run_pair_case(16, 8, 262000, 0x1002U, true);
    failures += run_single_case(1, 0, 0x2001U, true);
    failures += run_single_case(2048, 260000, 0x2002U, true);
    failures += run_single_case(1, 0, 0x2001U);
    failures += run_single_case(8, 131'069, 0x2002U);
    failures += run_single_case(64, 262'080, 0x2003U);
    failures += run_single_case(1024, 130'048, 0x2004U);
    failures += run_single_case(2048, 260'032, 0x2005U);

    if (failures != 0) {
        std::cerr << "rmsnorm_rope failures=" << failures << '\n';
        return 1;
    }
    std::cout << "rmsnorm_rope: PASS\n";
    return 0;
}
