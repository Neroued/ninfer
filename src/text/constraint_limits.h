#pragma once

// capy-flash: input limits of a structured-output constraint (OutputConstraintLimits). They run
// before anything recursive or expensive sees the input: the JSON Schema converter of the vendor
// is recursive (a deep schema overflowed the stack), and compilation cost grows with size. Every
// rejection is RequestError(ConstraintLimitExceeded) naming the limit; JSON Schema rejections
// carry a JSON Pointer to the offending location.

#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <string_view>

namespace ninfer::text {

// The parsed schema document: nesting depth of objects and arrays, members of one object or
// array, and the number of values (a lower bound of the serialized size). Iterative.
void check_json_schema_limits(const nlohmann::ordered_json& schema,
                              const OutputConstraintLimits& limits);

// Source-level limits of any constraint kind: bytes; JSON Schema nesting (scanned without
// parsing), members and values; GBNF group nesting and rules; regex group nesting; choice entries
// and total bytes.
void check_constraint_limits(const OutputConstraint& constraint,
                             const OutputConstraintLimits& limits);

} // namespace ninfer::text
