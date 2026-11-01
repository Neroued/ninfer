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

// Validate the supported JSON Schema dialect, retaining property order and literal data.
// Returns normalized schema text for the vendor converter; errors carry a JSON Pointer.
std::string prepare_json_schema(std::string_view source);
xgrammar::Grammar build_json_grammar(const OutputConstraint& constraint);

} // namespace ninfer::text
