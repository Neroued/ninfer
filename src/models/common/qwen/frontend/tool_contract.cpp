#include "models/common/qwen/frontend/tool_contract.h"
#include "models/common/qwen/frontend/tool_format.h"
#include "text/json_schema.h"
#include "text/schema_composition.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen::frontend {
namespace {
using Json                = nlohmann::ordered_json;
using Contract            = ToolCallOutputContract;
using SchemaType          = text::SchemaType;
using NormalizationPolicy = Contract::NormalizationPolicy;

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

[[noreturn]] void fail(std::string message, std::string path = {}) {
    throw RequestError(RequestErrorKind::InvalidToolConstraint, std::move(message),
                       std::move(path));
}

std::string parameter_path(const Contract::Tool& tool) {
    return "/" + std::to_string(tool.declaration_index) + "/parameters";
}

void prepare_strict_parameters(Contract::Tool& tool) {
    const Json document = Json::parse(text::prepare_json_schema(tool.schema_json));
    const Json& root    = text::resolve_schema(document, document);
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
        if (text::read_schema_types(schema, document, parameter.types))
            parameter.policy = NormalizationPolicy::DeclaredTypes;
        const auto types  = text::schema_type_domain(schema, document);
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
                if (text::read_schema_types(property, schema, parameter.types)) {
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
