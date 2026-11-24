#pragma once

#include "text/json_schema.h"
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::text {
// Type hints follow local references and type-bearing assertions without validating a schema
// or enforcing its value constraints. An unresolved domain returns false.
bool read_schema_types(const nlohmann::ordered_json& schema, const nlohmann::ordered_json& document,
                       SchemaTypes& types);
// Strict domain inspection and root resolution consume prepare_json_schema output.
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
