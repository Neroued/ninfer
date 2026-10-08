#include "serve/request_validation.h"
#include "text/json_input.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

namespace {
void set_constraint(GenerationRequest& request, OutputConstraint constraint, std::string param) {
    if (request.constraint) bad_request("only one output constraint may be specified", param);
    request.constraint       = std::move(constraint);
    request.constraint_param = std::move(param);
}
} // namespace

void validate_schema_number_input(const text::ParsedJsonNumbers& parsed) {
    if (parsed.inexact_numbers.empty() || !parsed.value.is_object()) return;
    const auto check = [&](const std::string& pointer, const std::string& param) {
        if (const auto error = text::inexact_schema_number(parsed, pointer))
            bad_request(
                "numeric schema value cannot be preserved by the JSON number representation",
                param + error->substr(pointer.size()), "unsupported_json_schema");
    };
    const auto& body = parsed.value;
    for (const auto& [path, param] :
         {std::pair{"/response_format/json_schema/schema", "response_format.json_schema.schema"},
          std::pair{"/text/format/schema", "text.format.schema"},
          std::pair{"/output_config/format/schema", "output_config.format.schema"}})
        check(path, param);
    const auto tools = [&](auto&& self, const RequestJson& list, const std::string& path) -> void {
        if (!list.is_array()) return;
        for (std::size_t i = 0; i < list.size(); ++i) {
            const auto& tool = list[i];
            if (!tool.is_object()) continue;
            const auto pointer = path + '/' + std::to_string(i);
            if (tool.contains("type") && tool["type"] == "namespace" && tool.contains("tools")) {
                self(self, tool["tools"], pointer + "/tools");
                continue;
            }
            const bool nested      = tool.contains("function") && tool["function"].is_object();
            const auto& definition = nested ? tool["function"] : tool;
            if (!definition.contains("strict") || definition["strict"] != true) continue;
            const std::string key =
                definition.contains("input_schema") ? "input_schema" : "parameters";
            const auto schema = pointer + (nested ? "/function/" : "/") + key;
            check(schema, schema.substr(1));
        }
    };
    if (body.contains("tools")) tools(tools, body["tools"], "/tools");
}

void parse_json_output_format(const RequestJson& format, GenerationRequest& request,
                              const std::string& param, JsonFormatProtocol protocol) {
    if (!format.is_object() || !format.contains("type") || !format["type"].is_string())
        bad_request("output format must contain a string type", param);
    const auto type = format["type"].get<std::string>();
    if (type == "text" && protocol != JsonFormatProtocol::Anthropic) {
        if (format.size() != 1) bad_request("text format has no options", param);
        return;
    }
    if (type == "json_object" && protocol != JsonFormatProtocol::Anthropic) {
        if (format.size() != 1) bad_request("json_object format has no options", param);
        set_constraint(request, OutputConstraint::json_object(), param);
        return;
    }
    if (type != "json_schema") bad_request("unsupported output format type", param + ".type");
    const auto* spec         = &format;
    std::string schema_param = param;
    if (protocol == JsonFormatProtocol::Chat) {
        if (format.size() != 2 || !format.contains("json_schema") ||
            !format["json_schema"].is_object())
            bad_request("json_schema format requires a json_schema object", param);
        spec = &format["json_schema"];
        schema_param += ".json_schema";
    }
    for (const auto& [key, value] : spec->items()) {
        const bool allowed = key == "schema" ||
                             (key == "type" && protocol != JsonFormatProtocol::Chat) ||
                             (protocol != JsonFormatProtocol::Anthropic &&
                              (key == "name" || key == "description" || key == "strict"));
        if (!allowed) bad_request("unknown output format option: " + key, schema_param + "." + key);
    }
    if (protocol != JsonFormatProtocol::Anthropic) {
        if (!spec->contains("name") || !(*spec)["name"].is_string())
            bad_request("json_schema requires a name", schema_param + ".name");
        const auto name = (*spec)["name"].get<std::string>();
        if (name.empty() || name.size() > 64 ||
            !std::all_of(name.begin(), name.end(), [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                       c == '_' || c == '-';
            }))
            bad_request("schema name must contain 1..64 letters, digits, underscores or hyphens",
                        schema_param + ".name");
        if (spec->contains("description") && !(*spec)["description"].is_string())
            bad_request("schema description must be a string", schema_param + ".description");
        if (spec->contains("strict") && !(*spec)["strict"].is_null() &&
            !(*spec)["strict"].is_boolean())
            bad_request("schema strict must be a boolean", schema_param + ".strict");
    }
    if (!spec->contains("schema") ||
        (!(*spec)["schema"].is_object() && !(*spec)["schema"].is_boolean()))
        bad_request("schema must be an object or boolean", schema_param + ".schema");
    set_constraint(request, OutputConstraint::json_schema((*spec)["schema"].dump()),
                   schema_param + ".schema");
}

void parse_structured_outputs(const RequestJson& body, GenerationRequest& request) {
    if (body.contains("tool_constraints") && !body["tool_constraints"].is_null()) {
        if (!body["tool_constraints"].is_string())
            bad_request("tool_constraints must be auto or basic", "tool_constraints");
        const auto mode = body["tool_constraints"].get<std::string>();
        if (mode == "basic")
            request.tool_choice.constraints = ToolConstraintMode::Basic;
        else if (mode == "auto")
            request.tool_choice.constraints = ToolConstraintMode::Automatic;
        else
            bad_request("tool_constraints must be auto or basic", "tool_constraints");
    }
    if (request.tool_choice.allowed_names)
        for (const auto& name : *request.tool_choice.allowed_names)
            if (std::none_of(request.tools.begin(), request.tools.end(),
                             [&](const auto& tool) { return tool.name == name; }))
                bad_request("tool choice refers to undeclared tool: " + name, "tool_choice");
    if (request.tool_choice.mode == ToolChoiceMode::Required && !request.uses_tools())
        bad_request("required tool choice has no callable tools", "tool_choice");
    for (const char* alias :
         {"grammar", "guided_json", "guided_regex", "guided_choice", "guided_grammar"}) {
        if (body.contains(alias) && !body[alias].is_null())
            bad_request("use standard JSON output formats or structured_outputs", alias);
    }
    if (body.contains("structured_outputs") && !body["structured_outputs"].is_null()) {
        const auto& value = body["structured_outputs"];
        if (!value.is_object() || value.size() != 1)
            bad_request("structured_outputs requires exactly one of grammar, regex or choice",
                        "structured_outputs");
        const auto& kind   = value.begin().key();
        const auto& source = value.begin().value();
        const auto param   = "structured_outputs." + kind;
        if (kind == "choice") {
            if (!source.is_array() || source.empty())
                bad_request("choice requires a nonempty array of strings", param, "invalid_choice");
            std::vector<std::string> choices;
            choices.reserve(source.size());
            for (std::size_t i = 0; i < source.size(); ++i) {
                if (!source[i].is_string())
                    bad_request("choice entries must be strings", param + "/" + std::to_string(i),
                                "invalid_choice");
                choices.push_back(source[i].get<std::string>());
            }
            set_constraint(request, OutputConstraint::choice(std::move(choices)), param);
        } else if (kind == "regex") {
            if (!source.is_string()) bad_request("regex must be a string", param, "invalid_regex");
            set_constraint(request, OutputConstraint::regex(source.get<std::string>()), param);
        } else if (kind == "grammar") {
            if (!source.is_string() || source.get_ref<const std::string&>().empty())
                bad_request("grammar must be a nonempty string", param, "invalid_grammar");
            set_constraint(request, OutputConstraint::grammar(source.get<std::string>()), param);
        } else {
            bad_request("unknown structured_outputs option: " + kind, param);
        }
    }
    if (request.constraint && !request.stop_strings.empty())
        bad_request("output constraints require model EOS and cannot use custom stops",
                    request.constraint_param);
    if (request.constraint && request.uses_tools() &&
        request.constraint->kind != OutputConstraintKind::JsonObject &&
        request.constraint->kind != OutputConstraintKind::JsonSchema)
        bad_request("active tools can be combined with JSON output constraints",
                    request.constraint_param);
    if (request.constrains_tools() && !request.stop_strings.empty())
        bad_request("constrained tools require model EOS and cannot use custom stops", "stop");
}

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const RequestJson& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

std::optional<double> optional_number(const RequestJson& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

bool optional_bool(const RequestJson& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).get<bool>();
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

} // namespace ninfer::serve
