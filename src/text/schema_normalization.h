#pragma once

#include <string>
#include <string_view>

namespace ninfer::text {
// Distribute proven common type/required assertions across anyOf branches.
// Unsupported intersections throw instead of weakening a schema.
std::string normalize_json_schema(std::string_view schema);
} // namespace ninfer::text
