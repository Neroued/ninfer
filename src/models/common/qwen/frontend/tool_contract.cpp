#include "models/common/qwen/frontend/tool_contract.h"
#include "models/common/qwen/frontend/tool_format.h"
#include "text/json_schema.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen::frontend {
namespace {
using Json                = nlohmann::ordered_json;
using Contract            = ToolCallOutputContract;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;
using NormalizationPolicy = Contract::NormalizationPolicy;

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

[[noreturn]] void fail(std::string message, std::string path = {}) {
    throw RequestError(RequestErrorKind::InvalidToolConstraint, std::move(message),
                       std::move(path));
}

std::string parameter_path(const Contract::Tool& tool) {
    return "/" + std::to_string(tool.declaration_index) + "/parameters";
}

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

bool compile_direct_types(const Json& type_definition, TypeSet& types) {
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

bool compile_schema_types(const Json& schema, TypeSet& types) {
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

    TypeSet combined;
    for (const Json& alternative : alternatives) {
        TypeSet branch;
        if (!compile_schema_types(alternative, branch)) { return false; }
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
            fail("tool schema requires resolvable local references");
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
        TypeSet direct;
        if (!compile_direct_types(node["type"], direct)) fail("invalid tool parameter type");
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

void prepare_strict_parameters(Contract::Tool& tool) {
    const Json document = Json::parse(text::prepare_json_schema(tool.schema_json));
    std::set<std::string> seen;
    const Json& root = resolve(document, document, seen);
    if (!root.is_object() || root.value("type", std::string("object")) != "object")
        fail("tool '" + tool.name + "' requires an object parameter schema", parameter_path(tool));
    if (!root.contains("type") || !root.contains("additionalProperties") ||
        root["additionalProperties"] != false)
        fail("strict tool '" + tool.name + "' requires type object and additionalProperties=false",
             parameter_path(tool));
    for (const char* assertion : {"const", "enum", "anyOf", "oneOf"})
        if (root.contains(assertion))
            fail("tool '" + tool.name + "' requires a single declared object parameter list",
                 parameter_path(tool) + "/" + assertion);
    tool.parameters.clear();
    if (!root.contains("properties")) return;
    if (!root["properties"].is_object())
        fail("tool properties must be an object", parameter_path(tool));
    for (const auto& [name, schema] : root["properties"].items()) {
        if (!valid_parameter_name(name))
            fail("parameter name cannot be represented in Qwen tool format: " + name,
                 parameter_path(tool));
        Contract::Parameter parameter;
        parameter.name = name;
        if (compile_schema_types(schema, parameter.types))
            parameter.policy = NormalizationPolicy::DeclaredTypes;
        const auto types  = domain(schema, document);
        const bool string = (types & type_bit(SchemaType::String)) != 0;
        if (string && types != type_bit(SchemaType::String))
            fail("strict tool '" + tool.name + "' parameter '" + name +
                     "' mixes raw string and JSON value types",
                 parameter_path(tool));
        parameter.encoding = string ? Contract::Encoding::RawString : Contract::Encoding::JsonValue;
        tool.parameters.push_back(std::move(parameter));
    }
}
} // namespace

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons) {
    if (tool_jsons.empty()) return {};
    auto result = std::make_shared<Contract>();
    for (auto& declaration : models::frontend::parse_tool_declarations(tool_jsons)) {
        Contract::Tool tool{std::move(declaration), {}};
        const auto schema = Json::parse(tool.schema_json);
        // Encoding is selected before generation, never guessed from a value prefix.
        if (schema.is_object() && schema.contains("properties") &&
            schema["properties"].is_object()) {
            for (const auto& [name, property] : schema["properties"].items()) {
                Contract::Parameter parameter;
                parameter.name = name;
                if (compile_schema_types(property, parameter.types)) {
                    parameter.policy = NormalizationPolicy::DeclaredTypes;
                    // The established non-strict normalization of a string union is raw text.
                    parameter.encoding = parameter.types.bits & type_bit(SchemaType::String)
                                             ? Contract::Encoding::RawString
                                             : Contract::Encoding::JsonValue;
                }
                tool.parameters.push_back(std::move(parameter));
            }
        }
        result->tools.push_back(std::move(tool));
    }
    return result;
}

std::shared_ptr<const ToolCallOutputContract>
select_tool_call_contract(const std::shared_ptr<const ToolCallOutputContract>& declarations,
                          const ToolChoice& choice) {
    const auto tools = declarations ? std::span<const Contract::Tool>(declarations->tools)
                                    : std::span<const Contract::Tool>{};
    auto selection   = models::frontend::select_tools(tools, choice);
    if (!selection.constrained && declarations && selection.indices.size() == tools.size())
        return declarations;
    if (tools.empty()) return {};
    auto selected         = std::make_shared<Contract>();
    selected->required    = selection.required;
    selected->parallel    = selection.parallel;
    selected->constrained = selection.constrained;
    selected->tools.reserve(selection.indices.size());
    for (auto i : selection.indices) selected->tools.push_back(tools[i]);
    if (selected->constrained) {
        for (auto& tool : selected->tools) {
            if (!tool.strict) continue;
            try {
                prepare_strict_parameters(tool);
            } catch (const RequestError& error) {
                if (error.kind() == RequestErrorKind::InvalidToolConstraint ||
                    error.source() == RequestErrorSource::Tools)
                    throw;
                throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                                   parameter_path(tool) + error.pointer(),
                                   RequestErrorSource::Tools);
            } catch (const Json::exception& error) {
                fail("invalid parameters for tool '" + tool.name + "': " + error.what(),
                     parameter_path(tool));
            }
        }
    }
    return selected;
}
} // namespace ninfer::models::qwen::frontend
