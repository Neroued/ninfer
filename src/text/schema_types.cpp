#include "text/json_schema.h"
#include "text/schema_composition.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <set>

namespace ninfer::text {
using Json = nlohmann::ordered_json;

namespace {
constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

bool schema_type(std::string_view name, SchemaType& type) {
    if (name == "null") {
        type = SchemaType::Null;
    } else if (name == "boolean") {
        type = SchemaType::Boolean;
    } else if (name == "integer") {
        type = SchemaType::Integer;
    } else if (name == "number") {
        type = SchemaType::Number;
    } else if (name == "string") {
        type = SchemaType::String;
    } else if (name == "object") {
        type = SchemaType::Object;
    } else if (name == "array") {
        type = SchemaType::Array;
    } else {
        return false;
    }
    return true;
}

bool compile_direct_types(const Json& type_definition, SchemaTypes& types) {
    types = {};
    if (type_definition.is_string()) {
        SchemaType type;
        if (!schema_type(type_definition.get_ref<const std::string&>(), type)) { return false; }
        types.bits = type_bit(type);
        return true;
    }
    if (!type_definition.is_array() || type_definition.empty()) { return false; }
    for (const Json& member : type_definition) {
        if (!member.is_string()) { return false; }
        SchemaType type;
        if (!schema_type(member.get_ref<const std::string&>(), type)) { return false; }
        types.bits |= type_bit(type);
    }
    return types.bits != 0;
}

bool read_types(const Json& schema, SchemaTypes& types) {
    if (!schema.is_object()) { return false; }
    const auto direct = schema.find("type");
    if (direct != schema.end()) { return compile_direct_types(*direct, types); }

    const auto any_of     = schema.find("anyOf");
    const auto one_of     = schema.find("oneOf");
    const bool has_any_of = any_of != schema.end();
    const bool has_one_of = one_of != schema.end();
    if (has_any_of == has_one_of) { return false; }

    const Json& alternatives = has_any_of ? *any_of : *one_of;
    if (!alternatives.is_array() || alternatives.empty()) { return false; }

    SchemaTypes combined;
    for (const Json& alternative : alternatives) {
        SchemaTypes branch;
        if (!read_types(alternative, branch)) { return false; }
        combined.bits |= branch.bits;
    }
    if (combined.bits == 0) { return false; }
    types = combined;
    return true;
}

const Json& resolve(const Json& node, const Json& document, std::set<std::string>& seen) {
    if (!node.is_object()) return node;
    if (node.contains("$ref") && node["$ref"].is_string()) {
        const auto ref = node["$ref"].get<std::string>();
        if ((ref != "#" && !ref.starts_with("#/")) || !seen.insert(ref).second)
            throw RequestError(RequestErrorKind::UnsupportedJsonSchema,
                               "schema requires resolvable local references");
        return resolve(document.at(Json::json_pointer(ref.substr(1))), document, seen);
    }
    if (node.contains("allOf") && node["allOf"].is_array() && node["allOf"].size() == 1)
        return resolve(node["allOf"][0], document, seen);
    return node;
}

std::uint8_t value_type(const Json& value) {
    if (value.is_string()) return type_bit(SchemaType::String);
    if (value.is_boolean()) return type_bit(SchemaType::Boolean);
    if (value.is_null()) return type_bit(SchemaType::Null);
    if (value.is_object()) return type_bit(SchemaType::Object);
    if (value.is_array()) return type_bit(SchemaType::Array);
    // JSON Schema classifies numbers by value: 1.0 belongs to the integer domain too.
    const bool integer =
        value.is_number_integer() ||
        (value.is_number_float() && std::floor(value.get<double>()) == value.get<double>());
    return type_bit(integer ? SchemaType::Integer : SchemaType::Number);
}

std::uint8_t domain(const Json& source, const Json& document, std::set<std::string> seen = {}) {
    if (source.is_boolean()) return source.get<bool>() ? 0x7f : 0;
    const Json& node = resolve(source, document, seen);
    if (!node.is_object()) return node.is_boolean() && !node.get<bool>() ? 0 : 0x7f;
    std::uint8_t types = 0x7f;
    if (node.contains("type")) {
        SchemaTypes direct;
        if (!compile_direct_types(node["type"], direct))
            throw RequestError(RequestErrorKind::InvalidJsonSchema, "invalid parameter type");
        types = direct.bits;
        if (types & type_bit(SchemaType::Number)) types |= type_bit(SchemaType::Integer);
    }
    if (node.contains("const")) return types & value_type(node["const"]);
    if (node.contains("enum") && node["enum"].is_array()) {
        std::uint8_t values = 0;
        for (const auto& item : node["enum"]) values |= value_type(item);
        return types & values;
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        if (!node.contains(key) || !node[key].is_array()) continue;
        std::uint8_t combined = 0;
        for (const auto& branch : node[key]) combined |= domain(branch, document, seen);
        types &= combined;
    }
    if (node.contains("allOf"))
        for (const auto& branch : node["allOf"]) types &= domain(branch, document, seen);
    return types;
}

} // namespace

bool read_schema_types(const Json& schema, SchemaTypes& types) { return read_types(schema, types); }

std::uint8_t schema_type_domain(const Json& schema, const Json& document) {
    return domain(schema, document);
}

const Json& resolve_schema(const Json& schema, const Json& document) {
    std::set<std::string> seen;
    return resolve(schema, document, seen);
}
} // namespace ninfer::text
