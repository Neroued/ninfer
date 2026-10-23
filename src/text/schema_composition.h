#pragma once

#include "text/json_schema.h"
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::text {
// Direct type hints are useful without requiring a strict schema. Domain inspection and local
// reference resolution consume a schema already validated by prepare_json_schema.
bool read_schema_types(const nlohmann::ordered_json& schema, SchemaTypes& types);
std::uint8_t schema_type_domain(const nlohmann::ordered_json& schema,
                                const nlohmann::ordered_json& document);
const nlohmann::ordered_json& resolve_schema(const nlohmann::ordered_json& schema,
                                             const nlohmann::ordered_json& document);

struct NormalizedSchema {
    nlohmann::ordered_json schema;
    std::vector<std::pair<std::string, std::string>> origins;
};

// Reduce a source validated by prepare_json_schema before vocabulary compilation.
// References remain a graph; dialect and keyword validation belong to the caller.
NormalizedSchema normalize_schema_composition(const nlohmann::ordered_json& source,
                                              bool draft7 = false);
} // namespace ninfer::text
