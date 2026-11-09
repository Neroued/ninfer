#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include "ninfer/types.h"

namespace xgrammar {
class Grammar;
}

namespace ninfer::text {

enum class SchemaType : std::uint8_t {
    Null    = 1U << 0U,
    Boolean = 1U << 1U,
    Integer = 1U << 2U,
    Number  = 1U << 3U,
    String  = 1U << 4U,
    Object  = 1U << 5U,
    Array   = 1U << 6U,
};

struct SchemaTypes {
    std::uint8_t bits = 0;
};

// Longest run of JSON whitespace the constrained language admits between two tokens and around
// the value: a newline and 62 columns of indentation (15 levels at 4 spaces) fit.
inline constexpr int kMaxJsonWhitespace = 64;

// Validate the supported JSON Schema dialect, retaining property order and literal data.
// Returns normalized schema text for the vendor converter; errors carry a JSON Pointer.
// The limits (depth, members, size: text/constraint_limits.h) are checked on the parsed source and
// on the normalized schema before anything recursive runs.
std::string prepare_json_schema(std::string_view source, const OutputConstraintLimits& limits = {});
xgrammar::Grammar build_json_grammar(const OutputConstraint& constraint,
                                     const OutputConstraintLimits& limits = {});

} // namespace ninfer::text
