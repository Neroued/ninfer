#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text {
struct ParsedJsonNumbers {
    nlohmann::ordered_json value;
    std::vector<std::string> inexact_numbers;
};

// Build the ordinary ordered DOM in one pass, retaining locations whose numeric spelling
// would change value. Consumers apply the precision contract only to active schemas.
ParsedJsonNumbers parse_json_numbers(std::string_view source);
std::optional<std::string> inexact_schema_number(const ParsedJsonNumbers& parsed,
                                                 const std::string& schema_pointer = {});
} // namespace ninfer::text
