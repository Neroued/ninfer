#pragma once

#include "ninfer/ops/linear.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::test::linear {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
};

enum class CallForm : std::uint8_t {
    Policy,
    A16Convenience,
};

enum class Comparison : std::uint8_t {
    Full,
    Sampled,
};

struct Invocation {
    std::int32_t t;
    CallForm call_form       = CallForm::Policy;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    Comparison comparison;
    bool verify_input_preservation;
    std::span<const Invocation> invocations;
};

using WeightGenerator = quantized_weight::PackedWeight (*)(std::int32_t, std::int32_t,
                                                           std::uint32_t);

quantized_weight::PackedWeight make_q4g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed);
quantized_weight::PackedWeight make_q5g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed);
quantized_weight::PackedWeight make_q6g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed);
quantized_weight::PackedWeight make_w8g32_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed);
quantized_weight::PackedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed);
quantized_weight::PackedWeight make_fp8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed);

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t);

bool cuda_available();

int run_shape(std::string_view label, ActivationCompute activation_compute,
              WeightGenerator generator, const ShapeCase& shape);

} // namespace ninfer::test::linear
