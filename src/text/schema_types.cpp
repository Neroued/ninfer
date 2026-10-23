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

struct TypeDomain {
    std::uint8_t bits = 0x7f;
    bool known        = false;

    void intersect(TypeDomain other) {
        bits &= other.bits;
        known |= other.known;
    }
};

// Encoding hints inspect only type-bearing assertions. Unknown constraints and unresolved
// references remain unconstrained; language validation belongs to prepare_json_schema.
TypeDomain domain(const Json& node, const Json& document, std::set<const Json*> seen = {}) {
    if (node.is_boolean()) return node.get<bool>() ? TypeDomain{} : TypeDomain{0, true};
    if (!node.is_object() || !seen.insert(&node).second) return {};
    TypeDomain result;
    if (node.contains("$ref") && node["$ref"].is_string()) {
        const auto ref = node["$ref"].get<std::string>();
        if (ref == "#" || ref.starts_with("#/")) {
            try {
                result.intersect(
                    domain(document.at(Json::json_pointer(ref.substr(1))), document, seen));
            } catch (const Json::exception&) {}
        }
        // draft-07 ignores assertion siblings of $ref.
        if (document.is_object() && document.contains("$schema") &&
            (document["$schema"] == "http://json-schema.org/draft-07/schema#" ||
             document["$schema"] == "https://json-schema.org/draft-07/schema#"))
            return result;
    }
    if (node.contains("type")) {
        SchemaTypes direct;
        if (compile_direct_types(node["type"], direct)) {
            if (direct.bits & type_bit(SchemaType::Number))
                direct.bits |= type_bit(SchemaType::Integer);
            result.intersect({direct.bits, true});
        }
    }
    if (node.contains("const")) result.intersect({value_type(node["const"]), true});
    if (node.contains("enum") && node["enum"].is_array()) {
        std::uint8_t values = 0;
        for (const auto& item : node["enum"]) values |= value_type(item);
        result.intersect({values, true});
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        if (!node.contains(key) || !node[key].is_array() || node[key].empty()) continue;
        TypeDomain combined{0, true};
        for (const auto& branch : node[key]) {
            const auto candidate = domain(branch, document, seen);
            combined.bits |= candidate.bits;
            combined.known &= candidate.known;
        }
        result.intersect(combined);
    }
    if (node.contains("allOf") && node["allOf"].is_array())
        for (const auto& branch : node["allOf"]) result.intersect(domain(branch, document, seen));
    return result;
}

} // namespace

bool read_schema_types(const Json& schema, const Json& document, SchemaTypes& types) {
    const auto result = domain(schema, document);
    types.bits        = result.known ? result.bits : 0;
    return types.bits != 0;
}

std::uint8_t schema_type_domain(const Json& schema, const Json& document) {
    return domain(schema, document).bits;
}

const Json& resolve_schema(const Json& schema, const Json& document) {
    std::set<std::string> seen;
    return resolve(schema, document, seen);
}
} // namespace ninfer::text
