#pragma once

#include "ninfer/ops/linear_add.h"
#include "ops/linear/support.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace ninfer::test::linear_add {

using linear_suite::Suite;
enum class FixturePattern : std::uint8_t { TokenBlocks, Hashed };
enum class Samples : std::uint8_t { All, BoundarySeven, Uniform32 };
enum class GraphReplay : std::uint8_t {
    None,
    OriginalZeroOriginal,
    OriginalNegative,
    OriginalTwice,
};

struct Invocation {
    std::int32_t tokens;
    std::optional<ops::LinearPolicy> policy{};
    GraphReplay graph = GraphReplay::None;
    std::optional<ops::LinearPolicy> graph_policy{};
    std::optional<ops::LinearPolicy> followup_policy{};
    std::string id;
};

struct InvocationCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    std::span<const Invocation> invocations;
    FixturePattern fixture = FixturePattern::TokenBlocks;
    Samples rows           = Samples::BoundarySeven;
    Samples columns        = Samples::All;
    bool interval_witness  = false;
};

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    std::span<const std::int32_t> route_starts;
    std::span<const std::int32_t> route_interiors;
    std::span<const std::int32_t> graph_tokens{};
    bool full_output = false;
    // Revisit each boundary in the residual tail after this many whole-wave columns.
    std::int32_t composite_offset = 0;
};

void add_cases(Suite& suite, QType format, const InvocationCase& shape);
void add_shape(Suite& suite, QType format, const ShapeCase& shape);

void add_bf16_cases(Suite& suite);
void add_q4_cases(Suite& suite);
void add_q5_cases(Suite& suite);
void add_q8_cases(Suite& suite);
void add_nvfp4_cases(Suite& suite);
void add_fp8_cases(Suite& suite);

void add_bank_cases(Suite& suite);

} // namespace ninfer::test::linear_add
