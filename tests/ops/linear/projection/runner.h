#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/support.h"
#include "ops/linear/weights.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::test::linear {

enum class ActivationRecipe : std::uint8_t { TokenBlocks, Hashed };

using linear_suite::Suite;
using linear_weights::WeightGenerator;

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
    bool graph_replay        = false;
    std::string id;
};

enum class ActivationPattern : std::uint8_t { Dense, KTail, Cancellation };

using ActivationGenerator = std::vector<std::uint16_t> (*)(std::int32_t, std::int32_t,
                                                           std::uint32_t);
using IndexSampler        = std::vector<std::int32_t> (*)(std::int32_t);

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    Comparison comparison;
    bool verify_input_preservation;
    std::span<const Invocation> invocations;
    ActivationPattern activation_pattern = ActivationPattern::Dense;
    // Explicit fixture recipes preserve different represented inputs and sample coordinates.
    // Null callbacks select this family's standard activation and evenly spaced 32-point sample.
    ActivationGenerator activation_generator = nullptr;
    IndexSampler row_sampler                 = nullptr;
    IndexSampler column_sampler              = nullptr;
    bool verify_each_invocation              = false;
};

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t);

void add_shape(Suite& suite, QType format, ActivationRecipe recipe, WeightGenerator generator,
               const ShapeCase& shape);
void add_workspace_cases(Suite& suite, QType format, int n, int k);

// A declared token interval must reserve enough space for every public point it contains.
int verify_workspace_envelopes(QType qtype, std::int32_t n, std::int32_t k);

void add_bf16_cases(Suite& suite);
void add_q4_cases(Suite& suite);
void add_q5_cases(Suite& suite);
void add_q6_cases(Suite& suite);
void add_q8_cases(Suite& suite);

void add_fp8_cases(Suite& suite);
void add_nvfp4_cases(Suite& suite);
void add_bank_cases(Suite& suite);

} // namespace ninfer::test::linear
