#pragma once

#include <nlohmann/json.hpp>
#include <cstddef>
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

// capy-flash: byte offset of the first `{` or `[` (outside strings) that nests deeper than
// max_depth, or nullopt. Linear and allocation-free, so it can run on untrusted text before any
// parser or recursive consumer sees it. Malformed text is not diagnosed here.
std::optional<std::size_t> json_nesting_exceeds(std::string_view text, std::size_t max_depth);
} // namespace ninfer::text
