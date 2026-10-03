#include "text/schema_normalization.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace ninfer::text {
namespace {
using Json = nlohmann::ordered_json;
using Types = std::vector<std::string>;

const std::unordered_set<std::string> annotations = {
    "$schema", "title", "description", "default", "examples", "$comment", "$defs", "definitions"};
const std::unordered_set<std::string> schema_maps = {
    "properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"};
const std::unordered_set<std::string> schema_children = {
    "items", "additionalItems", "additionalProperties", "contains", "propertyNames", "not",
    "if", "then", "else", "unevaluatedProperties", "unevaluatedItems", "contentSchema"};
const std::unordered_set<std::string> schema_arrays = {"anyOf", "allOf", "oneOf", "prefixItems"};

bool has(const Types& types, const std::string& type) {
    return std::find(types.begin(), types.end(), type) != types.end();
}

Types read_types(const Json& value) {
    static const std::unordered_set<std::string> valid = {
        "null", "boolean", "object", "array", "number", "integer", "string"};
    Types result;
    const auto append = [&](const Json& item) {
        if (!item.is_string() || !valid.contains(item.get<std::string>()))
            throw std::invalid_argument("type must contain JSON Schema primitive type names");
        const auto type = item.get<std::string>();
        if (!has(result, type)) result.push_back(type);
    };
    if (value.is_array()) {
        if (value.empty()) throw std::invalid_argument("type array must not be empty");
        for (const auto& item : value) append(item);
    } else {
        append(value);
    }
    if (has(result, "number"))
        result.erase(std::remove(result.begin(), result.end(), "integer"), result.end());
    return result;
}

Types intersect_types(const Types& left, const Types& right) {
    Types result;
    for (const auto& a : left) for (const auto& b : right) {
        std::string intersection;
        if (a == b) intersection = a;
        else if ((a == "number" && b == "integer") || (a == "integer" && b == "number"))
            intersection = "integer";
        if (!intersection.empty() && !has(result, intersection)) result.push_back(intersection);
    }
    if (has(result, "number"))
        result.erase(std::remove(result.begin(), result.end(), "integer"), result.end());
    return result;
}

Types read_required(const Json& value) {
    if (!value.is_array()) throw std::invalid_argument("required must be an array");
    Types result;
    for (const auto& item : value) {
        if (!item.is_string()) throw std::invalid_argument("required property names must be strings");
        const auto name = item.get<std::string>();
        if (!has(result, name)) result.push_back(name);
    }
    return result;
}

Json type_value(const Types& types) {
    if (types.size() == 1) return types.front();
    return Json(types);
}

class Normalizer {
public:
    explicit Normalizer(Json root) : root_(std::move(root)) {
        collect_sensitive_refs(root_);
    }

    Json normalize() {
        Json result = node(root_, {});
        if (result.is_boolean() && !result.get<bool>())
            throw std::invalid_argument("JSON schema has no satisfiable anyOf branches");
        return result;
    }

private:
    Json root_;
    bool composition_refs_ = false;

    void collect_sensitive_refs(const Json& schema) {
        if (!schema.is_object()) return;
        if (schema.contains("$ref") && schema["$ref"].is_string() &&
            schema["$ref"].get<std::string>().find("/anyOf/") != std::string::npos)
            composition_refs_ = true;
        for (const auto& [key, value] : schema.items()) {
            if (schema_maps.contains(key) && value.is_object())
                for (const auto& child : value) collect_sensitive_refs(child);
            else if (schema_children.contains(key)) {
                if (key == "items" && value.is_array())
                    for (const auto& child : value) collect_sensitive_refs(child);
                else collect_sensitive_refs(value);
            } else if (schema_arrays.contains(key) && value.is_array())
                for (const auto& child : value) collect_sensitive_refs(child);
        }
    }

    Json resolve(const std::string& ref) const {
        if (ref == "#") return root_;
        if (!ref.starts_with("#/") || ref.find('%') != std::string::npos)
            throw std::invalid_argument("type/required distribution supports only local JSON Pointer refs");
        try {
            const auto& target = root_.at(Json::json_pointer(ref.substr(1)));
            if (!target.is_object() && !target.is_boolean())
                throw std::invalid_argument("local $ref does not point to a schema");
            return target;
        } catch (const Json::exception& error) {
            throw std::invalid_argument(std::string("cannot resolve local schema $ref: ") + error.what());
        }
    }

    Json constrain(Json branch, const std::optional<Types>& common_types,
                   const Types& common_required, Types refs) {
        if (branch.is_boolean()) {
            if (!branch.get<bool>()) return false;
            branch = Json::object();
        }
        if (!branch.is_object()) throw std::invalid_argument("anyOf branches must be schemas");
        if (branch.contains("$ref")) {
            if (!branch["$ref"].is_string()) throw std::invalid_argument("$ref must be a string");
            for (const auto& [key, value] : branch.items())
                if (key != "$ref" && key != "type" && key != "required" && !annotations.contains(key))
                    throw std::invalid_argument("cannot distribute type/required through a $ref with complex siblings");
            const auto ref = branch["$ref"].get<std::string>();
            if (has(refs, ref)) throw std::invalid_argument("cyclic $ref in type/required distribution");
            refs.push_back(ref);
            Json expanded = node(resolve(ref), refs);
            const std::optional<Types> local_types = branch.contains("type")
                ? std::optional<Types>(read_types(branch["type"])) : std::nullopt;
            const Types local_required = branch.contains("required") ? read_required(branch["required"]) : Types{};
            if (local_types || !local_required.empty())
                expanded = constrain(std::move(expanded), local_types, local_required, refs);
            expanded = constrain(std::move(expanded), common_types, common_required, refs);
            if (expanded.is_object()) for (const auto& [key, value] : branch.items())
                if (annotations.contains(key)) expanded[key] = value;
            return expanded;
        }
        if (branch.contains("anyOf")) {
            if (common_types) {
                const Types types = branch.contains("type")
                    ? intersect_types(read_types(branch["type"]), *common_types) : *common_types;
                if (types.empty()) return false;
                branch["type"] = type_value(types);
            }
            if (!common_required.empty()) {
                Types required = branch.contains("required") ? read_required(branch["required"]) : Types{};
                for (const auto& name : common_required) if (!has(required, name)) required.push_back(name);
                branch["required"] = required;
            }
            return distribute(std::move(branch), refs);
        }
        if ((branch.contains("const") || branch.contains("enum")) && !common_required.empty())
            throw std::invalid_argument("required distribution through const/enum needs a literal intersection");
        if (common_types) {
            const Types types = branch.contains("type")
                ? intersect_types(read_types(branch["type"]), *common_types) : *common_types;
            if (types.empty()) return false;
            branch["type"] = type_value(types);
        }
        Types required = branch.contains("required") ? read_required(branch["required"]) : Types{};
        for (const auto& name : common_required) if (!has(required, name)) required.push_back(name);
        if (!required.empty()) branch["required"] = required;
        if (branch.contains("type") && branch["type"] == "object" &&
            branch.contains("additionalProperties") && branch["additionalProperties"] == false &&
            !branch.contains("patternProperties")) {
            const auto properties = branch.find("properties");
            if (properties != branch.end() && !properties->is_object())
                throw std::invalid_argument("properties must be an object");
            for (const auto& name : required)
                if (properties == branch.end() || !properties->contains(name)) return false;
        }
        return branch;
    }

    Json distribute(Json schema, const Types& refs) {
        if (!schema["anyOf"].is_array() || schema["anyOf"].empty())
            throw std::invalid_argument("anyOf must be a nonempty array");
        const bool constrained = schema.contains("type") || schema.contains("required");
        if (!constrained) {
            Json branches = Json::array();
            for (const auto& branch : schema["anyOf"])
                if (!branch.is_boolean() || branch.get<bool>()) branches.push_back(branch);
            if (branches.size() != schema["anyOf"].size() && composition_refs_)
                throw std::invalid_argument("cannot prune anyOf branches referenced by positional JSON Pointers");
            if (branches.empty()) return false;
            schema["anyOf"] = std::move(branches);
            return schema;
        }
        for (const auto& [key, value] : schema.items())
            if (key != "anyOf" && key != "type" && key != "required" && !annotations.contains(key))
                throw std::invalid_argument("anyOf type/required distribution cannot include other assertions");
        const std::optional<Types> common_types = schema.contains("type")
            ? std::optional<Types>(read_types(schema["type"])) : std::nullopt;
        const Types common_required = schema.contains("required") ? read_required(schema["required"]) : Types{};
        Json branches = Json::array();
        for (const auto& branch : schema["anyOf"]) {
            Json result = constrain(branch, common_types, common_required, refs);
            if (!result.is_boolean() || result.get<bool>()) branches.push_back(std::move(result));
        }
        if (branches.size() != schema["anyOf"].size() && composition_refs_)
            throw std::invalid_argument("cannot prune anyOf branches referenced by positional JSON Pointers");
        if (branches.empty()) return false;
        schema.erase("type");
        schema.erase("required");
        schema["anyOf"] = std::move(branches);
        return schema;
    }

    Json node(Json schema, const Types& refs) {
        if (schema.is_boolean()) return schema;
        if (!schema.is_object()) throw std::invalid_argument("schema nodes must be objects or booleans");
        for (auto& [key, value] : schema.items()) {
            if (schema_maps.contains(key) && value.is_object()) {
                for (auto& child : value) {
                    if (key == "dependencies" && child.is_array()) continue;
                    child = node(std::move(child), refs);
                }
            } else if (schema_children.contains(key)) {
                if (key == "items" && value.is_array())
                    for (auto& child : value) child = node(std::move(child), refs);
                else value = node(std::move(value), refs);
            } else if (schema_arrays.contains(key) && value.is_array()) {
                for (auto& child : value) child = node(std::move(child), refs);
            }
        }
        if (schema.contains("anyOf")) return distribute(std::move(schema), refs);
        return schema;
    }
};
} // namespace

std::string normalize_json_schema(std::string_view schema) {
    try {
        return Normalizer(Json::parse(schema)).normalize().dump();
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("invalid JSON schema for normalization: ") + error.what());
    }
}
} // namespace ninfer::text
