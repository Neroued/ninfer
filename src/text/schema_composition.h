#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::text {
struct NormalizedSchema {
    nlohmann::ordered_json schema;
    std::vector<std::pair<std::string, std::string>> origins;
};

// Reduce a source validated by prepare_json_schema before vocabulary compilation.
// References remain a graph; dialect and keyword validation belong to the caller.
NormalizedSchema normalize_schema_composition(const nlohmann::ordered_json& source,
                                              bool draft7 = false);
} // namespace ninfer::text
