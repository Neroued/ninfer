#include "serve/request_validation.h"
#include "text/constraint_limits.h"
#include "text/json_input.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ninfer::serve {

namespace {

constexpr const char* kUnsupportedCombination = "unsupported_output_combination";
constexpr const char* kLimitExceeded          = "constraint_limit_exceeded";

ninfer::OutputConstraintLimits& configured_limits() noexcept {
    static ninfer::OutputConstraintLimits limits;
    return limits;
}

// Source-level limits of a grammar, regex or choice list (text/constraint_limits.h). A source the
// vendor lexer cannot read is left to the compiler, which reports it with its own error kind.
void check_source_limits(const ninfer::OutputConstraint& constraint, const std::string& param) {
    try {
        text::check_constraint_limits(constraint, output_constraint_limits());
    } catch (const ninfer::RequestError& error) {
        if (error.kind() != ninfer::RequestErrorKind::ConstraintLimitExceeded) { throw; }
        bad_request(error.what(), param, kLimitExceeded);
    } catch (const ApiException&) { throw; } catch (const std::exception&) {
    }
}

void set_constraint(GenerationRequest& request, ninfer::OutputConstraint constraint,
                    std::string param) {
    if (request.constraint) {
        bad_request("only one output constraint may be specified (already set by " +
                        request.constraint_param + ")",
                    param, kUnsupportedCombination);
    }
    request.constraint       = std::move(constraint);
    request.constraint_param = std::move(param);
}

const RequestJson* member(const RequestJson* object, const char* key) {
    if (object == nullptr || !object->is_object() || !object->contains(key)) { return nullptr; }
    return &object->at(key);
}

// Combinations the serve layer rejects before the Engine sees the request. Upstream checks the
// stop strings again in the frontend (UnsupportedOutputCombination); the earlier 400 here keeps
// the error on the request field and spends no prompt preparation.
void validate_constraint_combinations(const GenerationRequest& request) {
    if (!request.constraint) { return; }
    if (!request.stop_strings.empty()) {
        bad_request("output constraints require the model EOS and cannot use custom stop strings",
                    request.constraint_param, kUnsupportedCombination);
    }
    // JSON output combined with callable tools needs the tool-call grammar (Tappa B). Until then
    // a constrained request must not leave tools active: tool_choice "none" keeps them declared
    // but inactive.
    if (request.uses_tools()) {
        bad_request("output constraints cannot be combined with active tools yet; remove the "
                    "tools or set tool_choice to none",
                    request.constraint_param, kUnsupportedCombination);
    }
    // D7: continuing a final assistant message under a grammar needs the template continuation
    // content in the matcher; not supported in this release.
    if (request.continuation == ninfer::PromptContinuationMode::ContinueFinalAssistant) {
        bad_request("a final assistant prefill cannot be combined with an output constraint",
                    "messages", "assistant_prefill_not_supported");
    }
}

} // namespace

namespace {
std::uint32_t& configured_json_depth() noexcept {
    static std::uint32_t depth = 1000; // kDefaultMaxRequestJsonDepth (serve_options.h)
    return depth;
}
} // namespace

void configure_max_request_json_depth(std::uint32_t depth) { configured_json_depth() = depth; }

std::uint32_t max_request_json_depth() noexcept { return configured_json_depth(); }

void check_request_json_nesting(std::string_view text, const std::string& param) {
    if (const auto offset = text::json_nesting_exceeds(text, max_request_json_depth())) {
        bad_request("JSON nests deeper than " + std::to_string(max_request_json_depth()) +
                        " levels at byte " + std::to_string(*offset) +
                        " (--max-request-json-depth)",
                    param, "request_json_too_deep");
    }
}

void configure_output_constraint_limits(const ninfer::OutputConstraintLimits& limits) {
    configured_limits() = limits;
}

const ninfer::OutputConstraintLimits& output_constraint_limits() noexcept {
    return configured_limits();
}

void validate_schema_number_input(const text::ParsedJsonNumbers& parsed) {
    if (parsed.inexact_numbers.empty() || !parsed.value.is_object()) { return; }
    for (const auto& [path, param] :
         {std::pair{"/response_format/json_schema/schema", "response_format.json_schema.schema"},
          std::pair{"/text/format/schema", "text.format.schema"},
          std::pair{"/output_config/format/schema", "output_config.format.schema"}}) {
        // The search below is recursive: an over-deep schema stops here (capy-flash).
        const RequestJson::json_pointer pointer(path);
        if (parsed.value.contains(pointer)) {
            try {
                text::check_json_schema_limits(parsed.value.at(pointer),
                                               output_constraint_limits());
            } catch (const ninfer::RequestError& error) {
                bad_request(error.what(), param + error.pointer(), kLimitExceeded);
            }
        }
        if (const auto error = text::inexact_schema_number(parsed, path)) {
            bad_request(
                "numeric schema value cannot be preserved by the JSON number representation",
                param + error->substr(std::string_view(path).size()), "unsupported_json_schema");
        }
    }
}

bool has_output_format_schema(const RequestJson& body) {
    return member(member(member(&body, "response_format"), "json_schema"), "schema") != nullptr ||
           member(member(member(&body, "text"), "format"), "schema") != nullptr ||
           member(member(member(&body, "output_config"), "format"), "schema") != nullptr;
}

bool is_output_constraint_param(std::string_view param) noexcept {
    for (const std::string_view prefix :
         {"response_format", "text.format", "output_config.format", "structured_outputs"}) {
        if (param.starts_with(prefix) &&
            (param.size() == prefix.size() || param[prefix.size()] == '.' ||
             param[prefix.size()] == '/')) {
            return true;
        }
    }
    return false;
}

void parse_json_output_format(const RequestJson& format, GenerationRequest& request,
                              const std::string& param, JsonFormatProtocol protocol) {
    if (!format.is_object() || !format.contains("type") || !format.at("type").is_string()) {
        bad_request("output format must contain a string type", param);
    }
    const auto type = format.at("type").get<std::string>();
    if (type == "text" && protocol != JsonFormatProtocol::Anthropic) {
        // Chat accepted extra members next to {"type":"text"} before structured output and keeps
        // doing so (unconstrained requests do not change); Responses always rejected them.
        if (protocol == JsonFormatProtocol::Responses && format.size() != 1) {
            bad_request("text format has no options", param);
        }
        return;
    }
    if (type == "json_object" && protocol != JsonFormatProtocol::Anthropic) {
        if (format.size() != 1) { bad_request("json_object format has no options", param); }
        set_constraint(request, ninfer::OutputConstraint::json_object(), param);
        return;
    }
    if (type != "json_schema") { bad_request("unsupported output format type", param + ".type"); }
    const RequestJson* spec  = &format;
    std::string schema_param = param;
    if (protocol == JsonFormatProtocol::Chat) {
        if (format.size() != 2 || !format.contains("json_schema") ||
            !format.at("json_schema").is_object()) {
            bad_request("json_schema format requires a json_schema object", param);
        }
        spec = &format.at("json_schema");
        schema_param += ".json_schema";
    }
    for (const auto& [key, value] : spec->items()) {
        const bool allowed = key == "schema" ||
                             (key == "type" && protocol != JsonFormatProtocol::Chat) ||
                             (protocol != JsonFormatProtocol::Anthropic &&
                              (key == "name" || key == "description" || key == "strict"));
        if (!allowed) {
            bad_request("unknown output format option: " + key, schema_param + "." + key);
        }
    }
    if (protocol != JsonFormatProtocol::Anthropic) {
        if (!spec->contains("name") || !spec->at("name").is_string()) {
            bad_request("json_schema requires a name", schema_param + ".name");
        }
        const auto name = spec->at("name").get<std::string>();
        if (name.empty() || name.size() > 64 ||
            !std::all_of(name.begin(), name.end(), [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                       c == '_' || c == '-';
            })) {
            bad_request("schema name must contain 1..64 letters, digits, underscores or hyphens",
                        schema_param + ".name");
        }
        if (spec->contains("description") && !spec->at("description").is_string()) {
            bad_request("schema description must be a string", schema_param + ".description");
        }
        // strict is accepted either way: the schema is always enforced by the grammar, so
        // strict:false gets the same guarantee as strict:true (upstream semantics).
        if (spec->contains("strict") && !spec->at("strict").is_null() &&
            !spec->at("strict").is_boolean()) {
            bad_request("schema strict must be a boolean", schema_param + ".strict");
        }
    }
    if (!spec->contains("schema") ||
        (!spec->at("schema").is_object() && !spec->at("schema").is_boolean())) {
        bad_request("schema must be an object or boolean", schema_param + ".schema");
    }
    // Depth, members and value count are checked iteratively before anything recursive (dump,
    // the schema validation, the vendor converter) walks the document.
    const RequestJson& schema = spec->at("schema");
    const auto& limits        = output_constraint_limits();
    try {
        text::check_json_schema_limits(schema, limits);
    } catch (const ninfer::RequestError& error) {
        bad_request(error.what(), schema_param + ".schema" + error.pointer(), kLimitExceeded);
    }
    std::string source = schema.dump();
    if (source.size() > limits.max_bytes) {
        bad_request("JSON schema has " + std::to_string(source.size()) + " bytes; the limit is " +
                        std::to_string(limits.max_bytes) + " (--constraint-max-kib)",
                    schema_param + ".schema", kLimitExceeded);
    }
    set_constraint(request, ninfer::OutputConstraint::json_schema(std::move(source)),
                   schema_param + ".schema");
}

void parse_structured_outputs(const RequestJson& body, GenerationRequest& request) {
    // llama.cpp exposes grammar; vLLM previously exposed the guided_* spellings. Each promises
    // constrained generation, so they are rejected rather than ignored. An empty llama.cpp
    // grammar is the neutral client default and stays accepted, as before.
    for (const char* alias :
         {"grammar", "guided_json", "guided_regex", "guided_choice", "guided_grammar"}) {
        if (!body.contains(alias) || body.at(alias).is_null()) { continue; }
        const RequestJson& value = body.at(alias);
        if (std::string_view(alias) == "grammar" && value.is_string() &&
            value.get_ref<const std::string&>().empty()) {
            continue;
        }
        bad_request(std::string(alias) +
                        " is not supported; use response_format or structured_outputs "
                        "(grammar, regex or choice)",
                    alias, "constrained_decoding_not_supported");
    }
    if (body.contains("structured_outputs") && !body.at("structured_outputs").is_null()) {
        const auto& value = body.at("structured_outputs");
        if (!value.is_object() || value.size() != 1) {
            bad_request("structured_outputs requires exactly one of grammar, regex or choice",
                        "structured_outputs");
        }
        const auto& kind   = value.begin().key();
        const auto& source = value.begin().value();
        const auto param   = "structured_outputs." + kind;
        if (kind == "choice") {
            if (!source.is_array() || source.empty()) {
                bad_request("choice requires a nonempty array of strings", param, "invalid_choice");
            }
            if (source.size() > output_constraint_limits().max_items) {
                bad_request("choice has " + std::to_string(source.size()) +
                                " entries; the limit is " +
                                std::to_string(output_constraint_limits().max_items) +
                                " (--constraint-max-items)",
                            param, kLimitExceeded);
            }
            std::vector<std::string> choices;
            choices.reserve(source.size());
            for (std::size_t i = 0; i < source.size(); ++i) {
                if (!source[i].is_string()) {
                    bad_request("choice entries must be strings", param + "/" + std::to_string(i),
                                "invalid_choice");
                }
                choices.push_back(source[i].get<std::string>());
            }
            auto constraint = ninfer::OutputConstraint::choice(std::move(choices));
            check_source_limits(constraint, param);
            set_constraint(request, std::move(constraint), param);
        } else if (kind == "regex") {
            if (!source.is_string()) {
                bad_request("regex must be a string", param, "invalid_regex");
            }
            auto constraint = ninfer::OutputConstraint::regex(source.get<std::string>());
            check_source_limits(constraint, param);
            set_constraint(request, std::move(constraint), param);
        } else if (kind == "grammar") {
            if (!source.is_string() || source.get_ref<const std::string&>().empty()) {
                bad_request("grammar must be a nonempty string", param, "invalid_grammar");
            }
            auto constraint = ninfer::OutputConstraint::grammar(source.get<std::string>());
            check_source_limits(constraint, param);
            set_constraint(request, std::move(constraint), param);
        } else {
            bad_request("unknown structured_outputs option: " + kind, param);
        }
    }
    validate_constraint_combinations(request);
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
