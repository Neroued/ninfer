#include "text/json_schema.h"
#include "text/schema_composition.h"
#include "text/json_input.h"
#include "ninfer/types.h"
#include "json_string_grammar.h"
#include "json_number.h"
#include <xgrammar/xgrammar.h>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;
using Kind = RequestErrorKind;

[[noreturn]] void fail(Kind kind, const std::string& path, const std::string& message) {
    throw RequestError(kind, message + " at " + (path.empty() ? "/" : path), path);
}

std::string child(const std::string& path, std::string_view key) {
    std::string result = path + '/';
    for (char c : key) {
        if (c == '~')
            result += "~0";
        else if (c == '/')
            result += "~1";
        else
            result += c;
    }
    return result;
}

bool annotation(std::string_view key) {
    static const std::unordered_set<std::string_view> keys{
        "title",    "description", "default", "examples", "$comment", "deprecated",
        "readOnly", "writeOnly",   "$schema", "$id",      "$defs",    "definitions"};
    return keys.contains(key);
}

bool matches_type(const Json& value, std::string_view type) {
    if (type == "object") return value.is_object();
    if (type == "array") return value.is_array();
    if (type == "string") return value.is_string();
    if (type == "boolean") return value.is_boolean();
    if (type == "null") return value.is_null();
    if (type == "number") return value.is_number();
    if (type == "integer")
        return value.is_number_integer() ||
               (value.is_number_float() && std::floor(value.get<double>()) == value.get<double>());
    return false;
}

void check_literal(const Json& value, const std::string& path) {
    if ((value.is_number_unsigned() &&
         value.get<std::uint64_t>() > std::uint64_t(std::numeric_limits<std::int64_t>::max())) ||
        (value.is_number_float() && std::abs(value.get<double>()) >= 0x1p63))
        fail(Kind::UnsupportedJsonSchema, path,
             "const/enum integers must fit signed 64-bit integers");
    if (value.is_object() || value.is_array())
        for (const auto& [key, item] : value.items()) check_literal(item, child(path, key));
}

class Schema {
public:
    bool source_validation = false;
    bool draft7            = false;
    Json root;
    std::vector<std::pair<std::string, std::string>> references;
    std::unordered_set<std::string> locations;

    void check(Json& node, const std::string& path) {
        locations.insert(path);
        if (node.is_boolean()) return;
        if (!node.is_object())
            fail(Kind::InvalidJsonSchema, path, "schema must be an object or boolean");
        static const std::unordered_set<std::string_view> keywords{"type",
                                                                   "properties",
                                                                   "required",
                                                                   "additionalProperties",
                                                                   "items",
                                                                   "prefixItems",
                                                                   "additionalItems",
                                                                   "minItems",
                                                                   "maxItems",
                                                                   "minLength",
                                                                   "maxLength",
                                                                   "pattern",
                                                                   "minimum",
                                                                   "maximum",
                                                                   "exclusiveMinimum",
                                                                   "exclusiveMaximum",
                                                                   "const",
                                                                   "enum",
                                                                   "anyOf",
                                                                   "oneOf",
                                                                   "allOf",
                                                                   "$ref"};
        for (const auto& [key, value] : node.items()) {
            if (!annotation(key) && !keywords.contains(key))
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "unsupported schema keyword: " + key);
        }
        if (node.contains("prefixItems") && draft7)
            fail(Kind::UnsupportedJsonSchema, child(path, "prefixItems"),
                 "draft-07 tuples use an items array");
        if (node.contains("additionalItems") && !draft7)
            fail(Kind::UnsupportedJsonSchema, child(path, "additionalItems"),
                 "2020-12 tuples use prefixItems and items");
        if (node.contains("$schema")) {
            const auto& dialect = node["$schema"];
            if (!dialect.is_string())
                fail(Kind::InvalidJsonSchema, child(path, "$schema"), "dialect must be a string");
            const auto name = dialect.get<std::string>();
            if (name != "https://json-schema.org/draft/2020-12/schema" &&
                name != "https://json-schema.org/draft/2020-12/schema#" &&
                name != "http://json-schema.org/draft-07/schema#" &&
                name != "https://json-schema.org/draft-07/schema#")
                fail(Kind::UnsupportedJsonSchema, child(path, "$schema"),
                     "unsupported JSON Schema dialect");
        }
        if (node.contains("$id")) {
            if (!node["$id"].is_string())
                fail(Kind::InvalidJsonSchema, child(path, "$id"), "$id must be a string");
            if (!path.empty())
                fail(Kind::UnsupportedJsonSchema, child(path, "$id"),
                     "nested reference scopes are not supported");
        }
        for (const char* key : {"$defs", "definitions"}) {
            if (!node.contains(key)) continue;
            if (!node[key].is_object())
                fail(Kind::InvalidJsonSchema, child(path, key), "definitions must be an object");
            for (auto& [name, definition] : node[key].items())
                check(definition, child(child(path, key), name));
        }
        for (const char* key : {"$ref", "anyOf", "oneOf", "allOf"}) {
            if (!node.contains(key)) continue;
            for (const auto& [sibling, value] : node.items()) {
                if ((!source_validation || (draft7 && std::string_view(key) == "$ref")) &&
                    sibling != key && !annotation(sibling))
                    fail(Kind::UnsupportedJsonSchema, child(path, sibling),
                         "assertion siblings of " + std::string(key) + " are not supported");
            }
            if (std::string_view(key) == "$ref") {
                if (!node[key].is_string())
                    fail(Kind::InvalidJsonSchema, child(path, key), "$ref must be a string");
                const auto ref = node[key].get<std::string>();
                if (ref != "#" && !ref.starts_with("#/"))
                    fail(Kind::UnsupportedJsonSchema, child(path, key),
                         "only local JSON Pointer references are supported");
                references.emplace_back(ref.substr(1), child(path, key));
            } else {
                if (!node[key].is_array() || node[key].empty())
                    fail(Kind::InvalidJsonSchema, child(path, key),
                         "schema alternatives must be a nonempty array");
                if (!source_validation && std::string_view(key) == "allOf" &&
                    node[key].size() != 1 &&
                    !std::all_of(node[key].begin(), node[key].end(), [](const auto& branch) {
                        return branch.is_object() &&
                               branch.value("type", std::string{}) == "string";
                    }))
                    fail(Kind::UnsupportedJsonSchema, child(path, key),
                         "allOf cannot be reduced to supported constraints");
                for (std::size_t i = 0; i < node[key].size(); ++i)
                    check(node[key][i], child(child(path, key), std::to_string(i)));
            }
            if (!source_validation) return;
        }
        std::set<std::string> types;
        if (node.contains("type")) {
            const auto add = [&](const Json& value) {
                if (!value.is_string())
                    fail(Kind::InvalidJsonSchema, child(path, "type"),
                         "type must name a JSON type");
                const auto type = value.get<std::string>();
                if (type != "object" && type != "array" && type != "string" && type != "integer" &&
                    type != "number" && type != "boolean" && type != "null")
                    fail(Kind::InvalidJsonSchema, child(path, "type"),
                         "unknown JSON type: " + type);
                if (!types.insert(type).second)
                    fail(Kind::InvalidJsonSchema, child(path, "type"), "duplicate JSON type");
            };
            if (node["type"].is_array()) {
                if (node["type"].empty())
                    fail(Kind::InvalidJsonSchema, child(path, "type"),
                         "type array must not be empty");
                for (const auto& type : node["type"]) add(type);
            } else
                add(node["type"]);
        }
        if (node.contains("const") || node.contains("enum")) {
            const bool constant = node.contains("const");
            const char* key     = constant ? "const" : "enum";
            for (const auto& [sibling, value] : node.items())
                if (!source_validation && sibling != key && sibling != "type" &&
                    !annotation(sibling))
                    fail(Kind::UnsupportedJsonSchema, child(path, sibling),
                         "finite values support only a type assertion alongside const/enum");
            if (node.contains("enum")) {
                if (!node["enum"].is_array() || node["enum"].empty())
                    fail(Kind::InvalidJsonSchema, child(path, "enum"),
                         "enum must be a nonempty array");
                check_literal(node["enum"], child(path, "enum"));
            }
            if (constant) check_literal(node["const"], child(path, "const"));
            if (!source_validation) {
                const auto admitted = [&](const Json& value) {
                    return types.empty() ||
                           std::any_of(types.begin(), types.end(),
                                       [&](const auto& t) { return matches_type(value, t); });
                };
                const auto impossible = [&] {
                    // Definitions remain addressable even when their containing schema admits no
                    // value.
                    for (auto it = node.begin(); it != node.end();) {
                        if (annotation(it.key()))
                            ++it;
                        else
                            it = node.erase(it);
                    }
                    node["allOf"] = Json::array({false});
                };
                if (constant) {
                    if (!admitted(node[key])) impossible();
                } else {
                    Json filtered = Json::array();
                    for (const auto& value : node[key])
                        if (admitted(value)) filtered.push_back(value);
                    if (filtered.empty())
                        impossible();
                    else
                        node[key] = std::move(filtered);
                }
                return;
            }
        }
        // Without type, typed assertions apply only to their own instance types. Preserve that
        // JSON Schema meaning instead of silently treating the whole schema as unrestricted.
        if (!source_validation && types.empty()) {
            bool assertions = false;
            for (const auto& [key, value] : node.items()) assertions |= !annotation(key);
            if (assertions) {
                node["type"] =
                    Json::array({"object", "array", "string", "number", "boolean", "null"});
                types = {"object", "array", "string", "number", "boolean", "null"};
            }
        }
        for (const char* key : {"minItems", "maxItems", "minLength", "maxLength"}) {
            if (!node.contains(key)) continue;
            const auto& value = node[key];
            if (!value.is_number_integer() || value < 0)
                fail(Kind::InvalidJsonSchema, child(path, key),
                     "bound must be a nonnegative integer");
            if (value > std::numeric_limits<std::int32_t>::max())
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "length bound exceeds the supported range");
        }
        xgrammar::NumberRange number_range;
        for (const char* key : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
            if (!node.contains(key)) continue;
            const auto& value = node[key];
            if (!value.is_number())
                fail(Kind::InvalidJsonSchema, child(path, key), "numeric bound must be a number");
            if (value.is_number_unsigned() &&
                value.get<std::uint64_t>() >
                    std::uint64_t(std::numeric_limits<std::int64_t>::max()))
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "integer bounds must fit signed 64-bit integers");
            if (!source_validation && types.contains("integer") && !value.is_number_integer())
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "integer interval was not reduced to whole-number bounds");
            if (types.contains("integer") && ((std::string_view(key) == "exclusiveMinimum" &&
                                               value == std::numeric_limits<std::int64_t>::max()) ||
                                              (std::string_view(key) == "exclusiveMaximum" &&
                                               value == std::numeric_limits<std::int64_t>::min())))
                fail(Kind::UnsupportedJsonSchema, child(path, key),
                     "exclusive bound requires integers beyond the supported range");
            auto endpoint = xgrammar::DecimalNumber::Parse(value.dump());
            const std::string_view name(key);
            if (name == "minimum" || name == "exclusiveMinimum")
                number_range.Lower(std::move(endpoint), name.starts_with("exclusive"));
            else
                number_range.Upper(std::move(endpoint), name.starts_with("exclusive"));
        }
        if (!source_validation && types.contains("number") && !number_range.HasPublishableValue())
            fail(Kind::UnsupportedJsonSchema, path, "numeric interval has no publishable value");
        if (node.contains("pattern")) {
            if (!node["pattern"].is_string())
                fail(Kind::InvalidJsonSchema, child(path, "pattern"), "pattern must be a string");
        }
        if (node.contains("pattern") && (source_validation || types.contains("string"))) {
            try {
                (void)xgrammar::JSONStringPattern(node["pattern"].get<std::string>());
            } catch (const std::exception& error) {
                fail(Kind::UnsupportedJsonSchema, child(path, "pattern"), error.what());
            }
        }
        if (node.contains("properties")) {
            if (!node["properties"].is_object())
                fail(Kind::InvalidJsonSchema, child(path, "properties"),
                     "properties must be an object");
            for (auto& [key, value] : node["properties"].items())
                check(value, child(child(path, "properties"), key));
        }
        if (node.contains("required")) {
            if (!node["required"].is_array())
                fail(Kind::InvalidJsonSchema, child(path, "required"), "required must be an array");
            std::set<std::string> names;
            for (const auto& item : node["required"]) {
                if (!item.is_string() || !names.insert(item.get<std::string>()).second)
                    fail(Kind::InvalidJsonSchema, child(path, "required"),
                         "required must contain distinct property names");
                if (!source_validation && (!node.contains("properties") ||
                                           !node["properties"].contains(item.get<std::string>())))
                    fail(Kind::UnsupportedJsonSchema, child(path, "required"),
                         "required properties must be declared in properties");
            }
        }
        for (const char* key :
             {"items", "prefixItems", "additionalItems", "additionalProperties"}) {
            if (!node.contains(key)) continue;
            const bool tuple = std::string_view(key) == "prefixItems" ||
                               (draft7 && std::string_view(key) == "items" && node[key].is_array());
            if (tuple) {
                if (!node[key].is_array() || node[key].empty())
                    fail(Kind::InvalidJsonSchema, child(path, key),
                         "tuple positions must be a nonempty array of schemas");
                for (std::size_t i = 0; i < node[key].size(); ++i)
                    check(node[key][i], child(child(path, key), std::to_string(i)));
            } else {
                check(node[key], child(path, key));
            }
        }
    }
};
} // namespace

std::string prepare_json_schema(std::string_view source) {
    Schema schema;
    try {
        auto parsed = parse_json_numbers(source);
        if (const auto pointer = inexact_schema_number(parsed))
            fail(Kind::UnsupportedJsonSchema, *pointer,
                 "numeric schema value cannot be preserved by the JSON number representation");
        schema.root = std::move(parsed.value);
    } catch (const Json::exception& error) { fail(Kind::InvalidJsonSchema, {}, error.what()); }
    schema.source_validation = true;
    schema.draft7            = schema.root.is_object() && schema.root.contains("$schema") &&
                    schema.root["$schema"].is_string() &&
                    schema.root["$schema"].get<std::string>().find("draft-07") != std::string::npos;
    schema.check(schema.root, {});
    for (const auto& [pointer, path] : schema.references) {
        try {
            const auto& target = schema.root.at(Json::json_pointer(pointer));
            if (!schema.locations.contains(pointer) ||
                (!target.is_object() && !target.is_boolean()))
                fail(Kind::UnsupportedJsonSchema, path,
                     "reference target is not a schema location");
        } catch (const Json::exception&) {
            fail(Kind::InvalidJsonSchema, path, "unresolved local reference");
        }
    }
    auto normalized = normalize_schema_composition(schema.root, schema.draft7);
    Schema prepared;
    prepared.root = std::move(normalized.schema);
    try {
        prepared.check(prepared.root, {});
    } catch (const RequestError& error) {
        std::string pointer = error.pointer();
        std::size_t matched = 0;
        for (const auto& [generated, source] : normalized.origins)
            if (generated.size() > matched &&
                (error.pointer() == generated || error.pointer().starts_with(generated + "/"))) {
                pointer = source + error.pointer().substr(generated.size());
                matched = generated.size();
            }
        if (pointer == error.pointer()) throw;
        std::string message      = error.what();
        const std::string suffix = " at " + (error.pointer().empty() ? "/" : error.pointer());
        if (message.ends_with(suffix)) message.resize(message.size() - suffix.size());
        throw RequestError(error.kind(), message + " at " + (pointer.empty() ? "/" : pointer),
                           pointer);
    }
    schema.root = std::move(prepared.root);
    if (schema.root.is_boolean() && !schema.root.get<bool>())
        fail(Kind::UnsatisfiableJsonSchema, {}, "schema cannot accept any value");
    return schema.root.dump();
}

xgrammar::Grammar build_json_grammar(const OutputConstraint& constraint) {
    try {
        if (!constraint.choices.empty())
            throw RequestError(Kind::InvalidJsonSchema,
                               "JSON constraints do not accept literal alternatives");
        std::string source;
        if (constraint.kind == OutputConstraintKind::JsonObject && constraint.source.empty())
            source = R"({"type":"object"})";
        else if (constraint.kind == OutputConstraintKind::JsonSchema)
            source = prepare_json_schema(constraint.source);
        else
            throw RequestError(Kind::InvalidJsonSchema,
                               "expected JSON object or schema constraint");
        return xgrammar::Grammar::FromJSONSchema(source, false, std::nullopt,
                                                 std::pair<std::string, std::string>{",", ":"},
                                                 false, std::nullopt, false, false);
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        const auto kind = error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema
                              ? Kind::UnsupportedJsonSchema
                          : error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema
                              ? Kind::UnsatisfiableJsonSchema
                              : Kind::InvalidJsonSchema;
        throw RequestError(kind,
                           std::string(error.what()) + " at " +
                               (error.pointer.empty() ? "/" : error.pointer),
                           error.pointer);
    } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(Kind::InvalidJsonSchema, error.what());
    } catch (const RequestError&) { throw; } catch (const std::invalid_argument& error) {
        throw RequestError(Kind::InvalidJsonSchema, error.what());
    }
}
} // namespace ninfer::text
