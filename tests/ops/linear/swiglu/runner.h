#pragma once

#include "core/weight.h"
#include "ops/linear/support.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::quantized_weight {
struct PackedWeight;
}

namespace ninfer::test::linear_swiglu {

using linear_suite::Suite;

enum class CallForm { Policy, Convenience, RejectA16Extent };

struct Invocation {
    int tokens;
    bool graph    = false;
    CallForm form = CallForm::Policy;
    std::string id;
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ops::LinearPolicy policy;
};

void add_profile(Suite& suite, const Profile& profile, std::span<const std::int32_t> token_cases,
                 std::span<const std::int32_t> graph_cases = {});

void add_negative_gate(Suite& suite, const Profile& profile, std::span<const std::int32_t> tokens,
                       std::span<const std::int32_t> active_columns,
                       std::function<quantized_weight::PackedWeight()> make_weight);

void add_q4_cases(Suite& suite);
void add_q8_cases(Suite& suite);
void add_nvfp4_cases(Suite& suite);
void add_fp8_cases(Suite& suite);

void add_bank_cases(Suite& suite);

} // namespace ninfer::test::linear_swiglu
