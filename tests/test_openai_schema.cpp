#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/request_validation.h"
#include "serve/translate.h"
#include "text/json_input.h"
#include "tool_call_salvage_probe.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

template <typename Function>
bool throws_logic(Function&& function) {
    try {
        function();
    } catch (const std::logic_error&) { return true; }
    return false;
}

RequestLimits limits() { return RequestLimits{.default_max_tokens = 512}; }

Json base_request() {
    return Json{{"model", "qwen"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
}

OpenAIChatRequest parse(Json body) { return parse_chat_completion_request(body, limits()); }

ResolvedPromptSemantics semantics(const GenerationRequest& request) {
    ServeOptions server;
    return resolve_prompt_semantics(request, server);
}

ninfer::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), {});
}

ninfer::RequestOptions options(const GenerationRequest& request) {
    ServeOptions server;
    return to_request_options(request, server, semantics(request), true);
}

Json parse_sse(const std::string& event) {
    constexpr std::string_view prefix = "data: ";
    if (!event.starts_with(prefix) || !event.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    return Json::parse(event.substr(prefix.size(), event.size() - prefix.size() - 2));
}

int test_request_envelope_and_sampling() {
    int failures                  = 0;
    Json body                     = base_request();
    body["stream"]                = true;
    body["stream_options"]        = Json{{"include_usage", true}, {"include_obfuscation", false}};
    body["max_completion_tokens"] = 48;
    body["max_tokens"]            = 9;
    body["temperature"]           = 0.7;
    body["top_p"]                 = 0.8;
    body["presence_penalty"]      = 0.3;
    body["frequency_penalty"]     = -0.2;
    body["seed"]                  = -1;
    body["top_k"]                 = 17;
    body["min_p"]                 = 0.05;
    body["timings_per_token"]     = true;
    body["return_progress"]       = true;

    const OpenAIChatRequest request = parse(body);
    failures += check(request.model == "qwen", "model remains in OpenAI envelope");
    failures += check(request.stream && request.include_usage, "stream metadata parsed");
    failures += check(request.timings_per_token && request.return_progress,
                      "llama.cpp response observations remain in the protocol envelope");
    failures += check(request.output_tokens_explicit && request.generation.max_tokens == 48,
                      "max_completion_tokens wins and explicitness stays in envelope");
    failures += check(request.generation.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
                      "signed seed maps modulo 2^64");
    failures +=
        check(request.generation.sampling.top_k == 17 && request.generation.sampling.min_p == 0.05,
              "compatible sampler extensions parsed");
    const ninfer::RequestOptions translated = options(request.generation);
    failures +=
        check(translated.execution.sampling.top_k == 17, "top_k reaches Engine request options");
    failures +=
        check(translated.execution.sampling.min_p && *translated.execution.sampling.min_p == 0.05F,
              "min_p reaches Engine request options");
    failures +=
        check(translated.execution.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
              "signed seed reaches Engine request options");

    const OpenAIChatRequest defaults = parse(base_request());
    failures +=
        check(!defaults.stream && !defaults.include_usage && !defaults.output_tokens_explicit &&
                  !defaults.timings_per_token && !defaults.return_progress &&
                  defaults.generation.max_tokens == limits().default_max_tokens,
              "protocol defaults remain outside GenerationRequest");

    Json malformed              = base_request();
    malformed["stream_options"] = true;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "stream_options",
                      "malformed stream_options rejected");
    malformed                      = base_request();
    malformed["timings_per_token"] = "yes";
    failures += check(api_error([&] { (void)parse(malformed); }).param == "timings_per_token",
                      "non-boolean timings_per_token rejected");
    malformed                    = base_request();
    malformed["return_progress"] = 1;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "return_progress",
                      "non-boolean return_progress rejected");
    return failures;
}

int test_standard_field_policy() {
    int failures  = 0;
    auto rejected = [&](const char* key, Json value, const char* code) {
        Json body            = base_request();
        body[key]            = std::move(value);
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.param == key && error.code == code,
                          std::string(key) + " non-neutral value rejected");
    };

    rejected("n", 2, "n_not_supported");
    rejected("logit_bias", Json{{"12", 1}}, "logit_bias_not_supported");
    rejected("logprobs", true, "logprobs_not_supported");
    rejected("top_logprobs", 2, "logprobs_not_supported");
    rejected("modalities", Json::array({"text", "audio"}), "modality_not_supported");
    rejected("web_search_options", Json::object(), "web_search_not_supported");
    rejected("moderation", Json::object(), "moderation_not_supported");
    rejected("verbosity", "high", "verbosity_not_supported");
    rejected("store", true, "store_not_supported");
    rejected("functions", Json::array({Json{{"name", "legacy"}}}), "legacy_tools_not_supported");

    Json neutral                      = base_request();
    neutral["n"]                      = 1;
    neutral["logit_bias"]             = Json{{"12", 0}, {"13", 0.0}};
    neutral["logprobs"]               = false;
    neutral["top_logprobs"]           = 0;
    neutral["response_format"]        = Json{{"type", "text"}};
    neutral["modalities"]             = Json::array({"text"});
    neutral["audio"]                  = Json{{"voice", "alloy"}};
    neutral["prediction"]             = Json{{"type", "content"}, {"content", "expected"}};
    neutral["verbosity"]              = "medium";
    neutral["store"]                  = false;
    neutral["functions"]              = Json::array();
    neutral["function_call"]          = "auto";
    neutral["metadata"]               = Json{{"trace", "client"}};
    neutral["user"]                   = "user-1";
    neutral["safety_identifier"]      = "safe-1";
    neutral["prompt_cache_key"]       = "cache-1";
    neutral["prompt_cache_options"]   = Json{{"retention", "24h"}};
    neutral["prompt_cache_retention"] = "24h";
    neutral["service_tier"]           = "priority";
    neutral["future_unknown_field"]   = Json{{"value", 1}};
    failures += check(parse(neutral).generation.messages.size() == 1,
                      "neutral controls and advisory hints are accepted");

    Json zero_limit                     = base_request();
    zero_limit["max_completion_tokens"] = 0;
    const OpenAIChatRequest zero        = parse(zero_limit);
    failures += check(zero.output_tokens_explicit && zero.generation.max_tokens == 0,
                      "an explicit zero output limit reaches Engine's no-generation path");
    return failures;
}

// Structured output (constrained decoding), ported from Neroued/ninfer 08902f73, eb6696ae,
// 8ffa3131 and 2734a56e (test_openai_schema.cpp) and adapted to Tappa A: JSON output with active
// tools stays rejected until the tool grammar lands.
int test_constrained_decoding_extensions() {
    int failures = 0;

    // Unconstrained requests translate as before: no constraint, default engine tool choice,
    // no presence-penalty override, special tokens preserved exactly when tools are in play.
    for (Json body : {base_request(),
                      [] {
                          Json value                  = base_request();
                          value["response_format"]    = Json{{"type", "text"}};
                          value["grammar"]            = "";
                          value["structured_outputs"] = nullptr;
                          value["guided_json"]        = nullptr;
                          return value;
                      }(),
                      [] {
                          Json value     = base_request();
                          value["tools"] = Json::array(
                              {Json{{"type", "function"},
                                    {"function", Json{{"name", "lookup"},
                                                      {"parameters", Json{{"type", "object"}}}}}}});
                          value["stop"] = "END";
                          return value;
                      }()}) {
        const auto parsed = parse(body).generation;
        const auto engine = options(parsed);
        failures +=
            check(!parsed.constraint && parsed.constraint_param.empty() && !engine.constraint &&
                      engine.tool_choice.mode == ninfer::ToolChoiceMode::Auto &&
                      engine.tool_choice.constraints == ninfer::ToolConstraintMode::Automatic &&
                      !engine.tool_choice.allowed_names && engine.tool_choice.parallel &&
                      !engine.execution.sampling.presence_penalty &&
                      engine.output.preserve_special_tokens == parsed.uses_tools(),
                  "an unconstrained request changed its Engine options");
    }

    // response_format json_object / json_schema (strict either way), streaming included.
    Json body               = base_request();
    body["stream"]          = true;
    body["response_format"] = Json{{"type", "json_object"}};
    const auto json_object  = parse(body);
    failures += check(json_object.stream &&
                          options(json_object.generation).constraint ==
                              ninfer::OutputConstraint::json_object() &&
                          json_object.generation.constraint_param == "response_format",
                      "JSON object mode lost in Engine translation");
    const Json schema = {
        {"type", "object"},
        {"properties", {{"description", {{"type", "string"}}}, {"a", {{"type", "integer"}}}}},
        {"required", {"a"}}};
    for (const Json strict : {Json(true), Json(false), Json(nullptr)}) {
        body                    = base_request();
        body["response_format"] = Json{
            {"type", "json_schema"},
            {"json_schema",
             {{"name", "answer"}, {"description", "d"}, {"strict", strict}, {"schema", schema}}}};
        const auto typed = parse(body).generation;
        failures += check(options(typed).constraint ==
                                  ninfer::OutputConstraint::json_schema(schema.dump()) &&
                              typed.constraint_param == "response_format.json_schema.schema",
                          "JSON schema source/order or diagnostic location lost");
    }

    // structured_outputs: grammar, regex, choice (literals kept byte for byte).
    body                       = base_request();
    body["structured_outputs"] = Json{{"grammar", "root ::= \"yes\" | \"no\""}};
    const auto grammar         = parse(body).generation;
    failures += check(options(grammar).constraint ==
                              ninfer::OutputConstraint::grammar("root ::= \"yes\" | \"no\"") &&
                          grammar.constraint_param == "structured_outputs.grammar",
                      "GBNF must survive protocol-to-Engine translation");
    body["structured_outputs"] = Json{{"choice", {"", "yes", "你好"}}};
    failures += check(options(parse(body).generation).constraint ==
                          ninfer::OutputConstraint::choice({"", "yes", "你好"}),
                      "choice literals were changed in Engine translation");
    body["structured_outputs"] = Json{{"regex", ""}};
    failures +=
        check(options(parse(body).generation).constraint == ninfer::OutputConstraint::regex(""),
              "empty regex was dropped in Engine translation");

    // Malformed formats: 400 with the field that is wrong.
    const auto param_of = [&](Json format) {
        Json request               = base_request();
        request["response_format"] = std::move(format);
        return api_error([&] { (void)parse(request); });
    };
    failures += check(param_of(Json{{"type", "json_schema"}}).param == "response_format",
                      "json_schema without its object was accepted");
    failures += check(
        param_of(Json{{"type", "json_schema"}, {"json_schema", {{"schema", schema}}}}).param ==
            "response_format.json_schema.name",
        "json_schema without a name was accepted");
    failures += check(param_of(Json{{"type", "json_schema"},
                                    {"json_schema", {{"name", "bad name"}, {"schema", schema}}}})
                              .param == "response_format.json_schema.name",
                      "invalid schema name was accepted");
    failures += check(param_of(Json{{"type", "json_schema"},
                                    {"json_schema", {{"name", "a"}, {"schema", "object"}}}})
                              .param == "response_format.json_schema.schema",
                      "non-object schema was accepted");
    failures += check(
        param_of(Json{{"type", "json_schema"},
                      {"json_schema", {{"name", "a"}, {"schema", schema}, {"strict", "yes"}}}})
                .param == "response_format.json_schema.strict",
        "non-boolean strict was accepted");
    failures +=
        check(param_of(Json{{"type", "json_schema"},
                            {"json_schema", {{"name", "a"}, {"schema", schema}, {"extra", 1}}}})
                      .param == "response_format.json_schema.extra",
              "unknown schema option was accepted");
    failures += check(param_of(Json{{"type", "xml"}}).param == "response_format.type",
                      "unknown output format type was accepted");
    failures += check(param_of(Json{{"type", "json_object"}, {"x", 1}}).status == 400,
                      "json_object options were accepted");
    failures += check(param_of(Json("json")).param == "response_format",
                      "untyped response_format was accepted");
    for (const auto& value : {Json{{"grammar", ""}}, Json{{"json", Json::object()}}, Json("bad"),
                              Json{{"choice", Json::array()}}, Json{{"choice", {"a", 1}}},
                              Json{{"regex", 7}}, Json{{"regex", "a"}, {"choice", {"a"}}}}) {
        body                       = base_request();
        body["structured_outputs"] = value;
        failures += check(api_error([&] { (void)parse(body); }).status == 400,
                          "malformed structured_outputs accepted: " + value.dump());
    }
    body["structured_outputs"] = Json{{"choice", {"a", 1}}};
    const ApiError bad_choice  = api_error([&] { (void)parse(body); });
    failures += check(bad_choice.param == "structured_outputs.choice/1" &&
                          bad_choice.code == "invalid_choice",
                      "bad choice entry lost its index");
    for (const char* alias :
         {"grammar", "guided_json", "guided_regex", "guided_choice", "guided_grammar"}) {
        body                 = base_request();
        body[alias]          = "root ::= \"yes\"";
        const ApiError error = api_error([&] { (void)parse(body); });
        failures +=
            check(error.param == alias && error.code == "constrained_decoding_not_supported" &&
                      error.message.find(alias) != std::string::npos,
                  std::string(alias) + " constrained-decoding alias accepted");
    }

    // Combinations rejected before the Engine: two constraints, custom stops, active tools.
    body                       = base_request();
    body["response_format"]    = Json{{"type", "json_object"}};
    body["structured_outputs"] = Json{{"grammar", "root ::= \"x\""}};
    failures += check(api_error([&] { (void)parse(body); }).param == "structured_outputs.grammar",
                      "conflicting formats accepted");
    body.erase("structured_outputs");
    body["stop"]             = "yes";
    const ApiError with_stop = api_error([&] { (void)parse(body); });
    failures += check(with_stop.param == "response_format" &&
                          with_stop.code == "unsupported_output_combination",
                      "constraint with custom stops accepted");
    body.erase("stop");
    body["tools"] = Json::array(
        {Json{{"type", "function"},
              {"function", {{"name", "lookup"}, {"parameters", {{"type", "object"}}}}}}});
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "unsupported_output_combination",
              "constraint with active tools accepted before the tool grammar exists");
    body["tool_choice"]         = "none";
    const auto inactive         = parse(body).generation;
    const auto inactive_options = options(inactive);
    failures += check(inactive.constraint && !inactive.uses_tools() &&
                          !inactive_options.output.preserve_special_tokens,
                      "inactive tools blocked a constraint");
    body["tool_choice"] = "required";
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_choice_not_supported",
                      "forced tool_choice is still rejected in Tappa A");

    // D6: constrained requests sample without presence penalty unless the client sets one.
    body                    = base_request();
    body["response_format"] = Json{{"type", "json_object"}};
    failures += check(options(parse(body).generation).execution.sampling.presence_penalty ==
                          std::optional<float>(0.0F),
                      "constrained request kept a default presence penalty");
    body["presence_penalty"] = 0.5;
    failures += check(options(parse(body).generation).execution.sampling.presence_penalty ==
                          std::optional<float>(0.5F),
                      "client presence penalty was overridden");

    // Tool history without active tools: constrained output is plain text.
    body             = base_request();
    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "q"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_1"},
                                 {"type", "function"},
                                 {"function", {{"name", "lookup"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "42"}},
         Json{{"role", "user"}, {"content", "answer as JSON"}}});
    failures += check(options(parse(body).generation).output.preserve_special_tokens,
                      "tool history no longer preserves special tokens");
    body["response_format"] = Json{{"type", "json_object"}};
    failures += check(!options(parse(body).generation).output.preserve_special_tokens,
                      "constrained output preserved tool-call special tokens");

    // Engine errors map to 400 with the request field and the schema JSON Pointer.
    const auto error = request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::UnsupportedJsonSchema, "unsupported keyword",
                             "/properties/x/format"),
        "response_format.json_schema.schema");
    failures += check(error.status == 400 &&
                          error.param == "response_format.json_schema.schema/properties/x/format" &&
                          error.code == "unsupported_json_schema",
                      "schema error lost its source path");
    const std::vector<std::tuple<ninfer::RequestErrorKind, int, const char*, const char*>> kinds = {
        {ninfer::RequestErrorKind::InvalidGrammar, 400, "invalid_grammar",
         "structured_outputs.grammar"},
        {ninfer::RequestErrorKind::InvalidChoice, 400, "invalid_choice",
         "structured_outputs.grammar"},
        {ninfer::RequestErrorKind::InvalidRegex, 400, "invalid_regex",
         "structured_outputs.grammar"},
        {ninfer::RequestErrorKind::ConstraintDeadEnd, 400, "constraint_dead_end",
         "structured_outputs.grammar"},
        {ninfer::RequestErrorKind::InvalidJsonSchema, 400, "invalid_json_schema",
         "structured_outputs.grammar/x"},
        {ninfer::RequestErrorKind::UnsatisfiableJsonSchema, 400, "unsatisfiable_json_schema",
         "structured_outputs.grammar/x"},
        {ninfer::RequestErrorKind::ConstraintLimitExceeded, 400, "constraint_limit_exceeded",
         "structured_outputs.grammar/x"},
        {ninfer::RequestErrorKind::ConstraintBudgetExceeded, 400, "constraint_budget_exceeded",
         "structured_outputs.grammar"},
        {ninfer::RequestErrorKind::UnsupportedOutputCombination, 400,
         "unsupported_output_combination", "/x"},
        {ninfer::RequestErrorKind::InvalidToolConstraint, 400, "invalid_tool_constraint",
         "tools/x"},
        {ninfer::RequestErrorKind::InvalidToolOutput, 500, "invalid_tool_output", ""},
    };
    for (const auto& [kind, status, code, param] : kinds) {
        const auto mapped = request_error_to_api_error(ninfer::RequestError(kind, "m", "/x"),
                                                       "structured_outputs.grammar");
        failures += check(mapped.status == status && mapped.code == code && mapped.param == param,
                          std::string("structured-output error misclassified: ") + code);
    }
    const auto unlocated = request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::UnsupportedOutputCombination, "m"),
        "text.format.schema");
    failures += check(unlocated.param == "text.format.schema",
                      "combination error without pointer lost the constraint field");
    // Vendor errors carry a timestamp and the server's source path: not for clients (N6).
    const auto vendor = request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::InvalidRegex,
                             "[21:04:30] /home/u/xgrammar/cpp/regex_converter.cc:85: Regex parsing "
                             "error at position 3: Lookahead is not supported yet."),
        "structured_outputs.regex");
    failures += check(vendor.message ==
                          "Regex parsing error at position 3: Lookahead is not supported yet.",
                      "a server source path reached the client: " + vendor.message);
    return failures;
}

// capy-flash structured output limits (OutputConstraintLimits): over-limit inputs are 400s in the
// HTTP layer that name the limit and the location, before anything recursive sees them. Deep
// documents are built from text and only moved: copying or dumping them would recurse.
std::string nested_items(std::size_t levels) {
    std::string text;
    for (std::size_t level = 0; level < levels; ++level) text += R"({"type":"array","items":)";
    return text + R"({"type":"integer"})" + std::string(levels, '}');
}

int test_constraint_limits() {
    int failures            = 0;
    const auto schema_error = [&](const std::string& schema) {
        Json body = Json::parse(
            R"({"model":"qwen","messages":[{"role":"user","content":"hello"}],)"
            R"("response_format":{"type":"json_schema","json_schema":{"name":"x","schema":)" +
            schema + "}}}");
        return api_error([&] { (void)parse(std::move(body)); });
    };
    const auto limited = [](const ApiError& error, const std::string& param, const char* flag) {
        return error.status == 400 && error.code == "constraint_limit_exceeded" &&
               error.param.starts_with(param) && error.message.find(flag) != std::string::npos;
    };
    // B2: the audit's 5 000-level schema killed the server (stack overflow in the converter).
    for (const std::size_t levels : {300U, 5000U, 100000U}) {
        failures += check(limited(schema_error(nested_items(levels)),
                                  "response_format.json_schema.schema/items/items",
                                  "--constraint-max-depth"),
                          "deep schema was not rejected: " + std::to_string(levels));
    }
    failures += check(schema_error(nested_items(255)).status == 0,
                      "a schema at the depth limit (256 containers) was rejected");
    Json properties = Json::object();
    for (int i = 0; i < 5000; ++i) properties["p" + std::to_string(i)] = {{"type", "string"}};
    failures +=
        check(limited(schema_error(
                          Json{{"type", "object"}, {"properties", std::move(properties)}}.dump()),
                      "response_format.json_schema.schema/properties", "--constraint-max-items"),
              "an object with 5000 properties was not rejected");
    const auto wide = schema_error(
        Json{{"type", "string"}, {"description", std::string(300U * 1024U, 'x')}}.dump());
    failures += check(limited(wide, "response_format.json_schema.schema", "--constraint-max-kib") &&
                          wide.param == "response_format.json_schema.schema",
                      "a 300 KiB schema was not rejected");

    const auto outputs_error = [&](Json value) {
        Json body                  = base_request();
        body["structured_outputs"] = std::move(value);
        return api_error([&] { (void)parse(body); });
    };
    Json choices = Json::array();
    for (int i = 0; i < 5000; ++i) choices.push_back("v" + std::to_string(i));
    failures += check(limited(outputs_error(Json{{"choice", std::move(choices)}}),
                              "structured_outputs.choice", "--constraint-max-items"),
                      "a 5000-entry choice was not rejected");
    const std::string open(300, '('), close(300, ')');
    failures += check(limited(outputs_error(Json{{"regex", open + "a" + close}}),
                              "structured_outputs.regex", "--constraint-max-depth"),
                      "a regex nested 300 groups deep was not rejected");
    failures +=
        check(limited(outputs_error(Json{{"grammar", "root ::= " + open + "\"a\"" + close}}),
                      "structured_outputs.grammar", "--constraint-max-depth"),
              "a GBNF nested 300 groups deep was not rejected");
    failures += check(outputs_error(Json{{"grammar", "root ::= \"" + open + "\" [(]"}}).status == 0,
                      "parentheses inside GBNF literals counted as groups");
    failures += check(outputs_error(Json{{"regex", "[(]{3}\\((a)"}}).status == 0,
                      "escaped or bracketed parentheses counted as regex groups");

    // Tool-call arguments are JSON in a string: checked where they are read (capy-flash).
    Json history        = base_request();
    history["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "hi"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls", Json::array({Json{{"id", "c1"},
                                               {"type", "function"},
                                               {"function",
                                                {{"name", "f"},
                                                 {"arguments", std::string(2000, '[') +
                                                                   std::string(2000, ']')}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", "c1"}, {"content", "ok"}}});
    const auto deep_arguments = api_error([&] { (void)parse(history); });
    failures +=
        check(deep_arguments.status == 400 && deep_arguments.code == "request_json_too_deep" &&
                  deep_arguments.param == "messages",
              "deeply nested tool-call arguments were not refused");

    // Configurable; set once at startup by the HTTP server.
    configure_output_constraint_limits({.max_depth = 8});
    failures += check(limited(schema_error(nested_items(10)), "response_format.json_schema.schema",
                              "--constraint-max-depth"),
                      "the configured depth limit was not applied");
    configure_output_constraint_limits({});

    // The exact-number check reads the raw body and searches the schema recursively: a deep schema
    // with an inexact bound must stop at the depth limit, not overflow the stack.
    std::string text =
        R"({"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":)";
    for (int i = 0; i < 100000; ++i) text += R"({"type":"array","items":)";
    text += R"({"type":"number","maximum":0.30000000000000001})";
    text += std::string(100000, '}') + "}}}";
    const auto inexact =
        api_error([&] { validate_schema_number_input(ninfer::text::parse_json_numbers(text)); });
    failures += check(limited(inexact, "response_format.json_schema.schema/items/items",
                              "--constraint-max-depth"),
                      "a deep schema with an inexact number was not stopped at the depth limit");
    return failures;
}

Json function_tool(std::string name = "weather", bool strict = false) {
    return Json{{"type", "function"},
                {"function", Json{{"name", std::move(name)},
                                  {"description", "Get weather"},
                                  {"parameters", Json{{"type", "object"}}},
                                  {"strict", strict}}}};
}

int test_tools() {
    int failures                      = 0;
    Json body                         = base_request();
    body["tools"]                     = Json::array({function_tool()});
    const OpenAIChatRequest automatic = parse(body);
    failures += check(automatic.generation.uses_tools(), "function tools default to auto");
    failures += check(prompt(automatic.generation).options.tool_jsons.size() == 1,
                      "auto tools reach PromptInput");

    body["tools"][0]["future_item_field"]                 = "ignored";
    body["tools"][0]["function"]["future_function_field"] = "ignored";
    const std::string normalized_definition = prompt(parse(body).generation).options.tool_jsons[0];
    failures += check(normalized_definition.find("future_item_field") == std::string::npos &&
                          normalized_definition.find("future_function_field") == std::string::npos,
                      "unknown tool fields do not silently alter the model prompt");

    body["tool_choice"]          = "none";
    body["parallel_tool_calls"]  = false;
    const OpenAIChatRequest none = parse(body);
    failures +=
        check(!none.generation.uses_tools() && prompt(none.generation).options.tool_jsons.empty(),
              "tool_choice none makes parallel_tool_calls neutral and removes executable tools");

    body["tool_choice"] = "required";
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_choice_not_supported",
                      "required tool choice rejected");
    body["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "weather"}}}};
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_choice_not_supported",
                      "named tool choice rejected");

    body          = base_request();
    body["tools"] = Json::array({function_tool(), function_tool("search")});
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "search"}}})}}}};
    const GenerationRequest allowed = parse(body).generation;
    failures += check(allowed.tools.size() == 1 && allowed.tools[0].name == "search" &&
                          prompt(allowed).options.tool_jsons.size() == 1,
                      "allowed_tools auto narrows the executable function set");

    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"mode", "auto"},
             {"tools", Json::array({Json{{"type", "function"}, {"name", "weather"}}})}};
    const GenerationRequest direct_allowed = parse(body).generation;
    failures += check(direct_allowed.tools.size() == 1 && direct_allowed.tools[0].name == "weather",
                      "direct allowed_tools compatibility shape is accepted");
    body["tool_choice"]["mode"]     = "required";
    const ApiError required_allowed = api_error([&] { (void)parse(body); });
    failures +=
        check(required_allowed.code == "tool_choice_not_supported" &&
                  required_allowed.message.find("at least one tool call") != std::string::npos,
              "required allowed_tools reports the unenforceable guarantee");
    body["tool_choice"]["mode"]             = "auto";
    body["tool_choice"]["tools"][0]["name"] = "missing";
    failures += check(api_error([&] { (void)parse(body); }).param == "tool_choice",
                      "allowed_tools rejects names absent from the declared tool set");

    body          = base_request();
    body["tools"] = Json::array({function_tool("weather", true)});
    failures += check(api_error([&] { (void)parse(body); }).code == "strict_tools_not_supported",
                      "strict tools rejected");
    body["tools"] = Json::array({Json{{"type", "custom"}, {"name", "shell"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_type_not_supported",
                      "custom tools rejected");

    body                        = base_request();
    body["tools"]               = Json::array({function_tool()});
    body["parallel_tool_calls"] = false;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "parallel_tool_calls_not_supported",
              "parallel_tool_calls=false rejected when tools exist");
    body.erase("tools");
    failures += check(parse(body).generation.tools.empty(),
                      "parallel_tool_calls=false is neutral without tools");
    body["tool_choice"] = "auto";
    failures +=
        check(parse(body).generation.tools.empty(), "tool_choice auto is neutral without tools");

    Json history = base_request();
    history["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "weather?"}},
                     Json{{"role", "assistant"},
                          {"content", nullptr},
                          {"tool_calls",
                           Json::array({Json{{"id", "call_1"},
                                             {"type", "function"},
                                             {"function", Json{{"name", "weather"},
                                                               {"arguments", "not-json-yet"}}}}})}},
                     Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "sunny"}}});
    failures += check(parse(history).generation.has_tool_history(),
                      "tool-call history follows wire types without inventing JSON validation");

    Json mixed_assistant        = base_request();
    mixed_assistant["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "inspect"}},
         Json{{"role", "assistant"},
              {"content", "I will inspect it"},
              {"tool_calls",
               Json::array({Json{
                   {"id", "call_2"},
                   {"type", "function"},
                   {"function", Json{{"name", "inspect"}, {"arguments", R"({"path":"a"})"}}}}})}}});
    const GenerationRequest mixed_request  = parse(mixed_assistant).generation;
    const ninfer::PromptInput mixed_prompt = prompt(mixed_request);
    failures += check(mixed_request.messages[1].cache_boundary_after &&
                          !mixed_request.messages[1].content[0].cache_boundary_after &&
                          !mixed_prompt.context_cache.markers.empty() &&
                          mixed_prompt.context_cache.markers.back().location ==
                              ninfer::PromptCacheMarkerLocation::MessageBoundary &&
                          mixed_prompt.context_cache.markers.back().after_message_count == 2,
                      "automatic caching stops after a complete assistant text/tool-call turn");

    const Json ordered = Json::parse(
        R"({"model":"qwen","messages":[{"role":"user","content":"probe"}],"tools":[{"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}}]})");
    const ninfer::PromptInput ordered_prompt = prompt(parse(ordered).generation);
    failures += check(
        ordered_prompt.options.tool_jsons.size() == 1 &&
            ordered_prompt.options.tool_jsons.front() ==
                R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}},"strict":false}})",
        "OpenAI Chat changed tool-schema member order before PromptInput");
    return failures;
}

int test_messages_and_media() {
    int failures                   = 0;
    Json body                      = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "text"}, {"text", "alpha"}}, Json{{"type", "text"}, {"text", "beta"}}});
    const ninfer::PromptInput translated = prompt(parse(body).generation);
    failures += check(translated.messages[0].parts.size() == 2 &&
                          translated.messages[0].parts[0].text == "alpha" &&
                          translated.messages[0].parts[1].text == "beta",
                      "adjacent text parts preserve exact text without inserted newline");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image_url"},
              {"image_url", Json{{"url", "https://example.test/a.png"}, {"detail", "auto"}}}},
         Json{{"type", "video_url"}, {"video_url", "https://example.test/a.mp4"}}});
    const GenerationRequest media = parse(body).generation;
    failures += check(media.media_item_count() == 2 &&
                          media.messages[0].content[0].kind == ContentKind::Image &&
                          media.messages[0].content[1].kind == ContentKind::Video,
                      "image and video compatibility inputs normalize to Engine media");

    body["messages"][0]["content"][0]["image_url"]["detail"] = "high";
    failures += check(api_error([&] { (void)parse(body); }).code == "image_detail_not_supported",
                      "explicit image preprocessing detail rejected");

    auto content_rejected = [&](const char* role, const char* type) {
        Json invalid                   = base_request();
        invalid["messages"][0]["role"] = role;
        invalid["messages"][0]["content"] =
            Json::array({Json{{"type", type}, {type, "https://example.test/x"}}});
        return api_error([&] { (void)parse(invalid); }).code == "modality_not_supported";
    };
    failures +=
        check(content_rejected("assistant", "image_url"), "assistant media history rejected");
    failures += check(content_rejected("system", "image_url"),
                      "system media rejected at protocol boundary");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "capture it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_capture"},
                                 {"type", "function"},
                                 {"function", Json{{"name", "capture"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"},
              {"tool_call_id", "call_capture"},
              {"content",
               Json::array({Json{{"type", "text"}, {"text", "captured"}},
                            Json{{"type", "image_url"},
                                 {"image_url", Json{{"url", "https://example.test/capture.png"},
                                                    {"detail", "auto"}}}}})}}});
    const GenerationRequest tool_image = parse(body).generation;
    failures += check(tool_image.messages.back().role == ninfer::ChatRole::Tool &&
                          tool_image.messages.back().tool_call_id == "call_capture" &&
                          tool_image.messages.back().content.size() == 2 &&
                          tool_image.messages.back().content[0].kind == ContentKind::Text &&
                          tool_image.messages.back().content[1].kind == ContentKind::Image,
                      "tool result text and image parts normalize to one tool turn");

    body["messages"].back()["content"] = Json::array(
        {Json{{"type", "video_url"}, {"video_url", "https://example.test/capture.mp4"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "tool result video remains outside the Chat compatibility extension");

    body = base_request();
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "input_audio"}, {"input_audio", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "input audio rejected");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "file"}, {"file", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "file input rejected");

    body                        = base_request();
    body["messages"][0]["name"] = "speaker";
    failures += check(api_error([&] { (void)parse(body); }).code == "message_name_not_supported",
                      "message name rejected");

    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls",
         Json::array({Json{{"id", "call_1"},
                           {"type", "function"},
                           {"function", Json{{"name", "get_status"}, {"arguments", "{}"}}}}})}});
    body["messages"].push_back(Json{
        {"role", "tool"}, {"name", "get_status"}, {"tool_call_id", "call_1"}, {"content", "ok"}});
    const GenerationRequest named_tool_history = parse(body).generation;
    const ChatTurn& named_tool                 = named_tool_history.messages.back();
    failures += check(named_tool.role == ninfer::ChatRole::Tool &&
                          named_tool.tool_call_id == "call_1" && !named_tool.tool_result_name &&
                          named_tool.content.size() == 1 && named_tool.content[0].text == "ok",
                      "tool message name is an ignored compatibility extension");

    body["messages"].back()["name"] = Json::array();
    failures +=
        check(api_error([&] { (void)parse(body); }).message == "message name must be a string",
              "tool message name remains type checked");

    body                           = base_request();
    body["messages"][0]["name"]    = "";
    body["messages"][0]["content"] = Json::array();
    failures += check(parse(body).generation.messages[0].content.empty(),
                      "empty names and empty content arrays remain neutral");

    body = base_request();
    body["messages"].push_back(
        Json{{"role", "assistant"},
             {"content", Json::array({Json{{"type", "refusal"}, {"refusal", "part"}}})},
             {"refusal", "top-level"}});
    const GenerationRequest refusal_history = parse(body).generation;
    const ChatTurn& refusal                 = refusal_history.messages.back();
    failures += check(refusal.content.size() == 2 && refusal.content[0].text == "part" &&
                          refusal.content[1].text == "top-level",
                      "assistant refusal history is preserved as assistant text");

    body = base_request();
    body["messages"].push_back(Json{{"role", "assistant"}});
    failures += check(parse(body).generation.messages.back().content.empty(),
                      "an empty assistant history turn is representable");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "run it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"function_call", Json{{"name", "legacy"}, {"arguments", R"({"value":1})"}}}},
         Json{{"role", "function"}, {"name", "legacy"}, {"content", "done"}}});
    const GenerationRequest legacy = parse(body).generation;
    failures += check(legacy.messages[1].tool_calls.size() == 1 &&
                          legacy.messages[1].tool_calls[0].name == "legacy" &&
                          legacy.messages[2].role == ninfer::ChatRole::Tool,
                      "legacy function-call history lowers to Engine tool history");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", ""},
                                 {"type", "function"},
                                 {"function", Json{{"name", "weather"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", ""}, {"content", "done"}}});
    failures += check(parse(body).generation.has_tool_history(),
                      "string tool-call identifiers may be empty without changing history");
    return failures;
}

int test_reasoning_and_extensions() {
    int failures = 0;
    Json body    = base_request();
    body["messages"].push_back(Json{{"role", "assistant"},
                                    {"content", "answer"},
                                    {"reasoning_content", "thought"},
                                    {"reasoning", "thought"}});
    failures += check(parse(body).generation.messages.back().reasoning_content == "thought",
                      "assistant reasoning aliases normalize");
    body["messages"].back()["reasoning"] = "different";
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting assistant reasoning aliases rejected");
    body["messages"].back()["reasoning_content"] = "";
    failures += check(parse(body).generation.messages.back().reasoning_content == "different",
                      "an empty reasoning alias does not conflict with a meaningful alias");
    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"}, {"content", nullptr}, {"reasoning_content", "unfinished thought"}});
    failures +=
        check(parse(body).generation.messages.back().reasoning_content == "unfinished thought",
              "reasoning-only assistant history is preserved");

    body                         = base_request();
    body["enable_thinking"]      = true;
    body["preserve_thinking"]    = false;
    body["chat_template_kwargs"] = Json{{"enable_thinking", true}, {"preserve_thinking", false}};
    const GenerationRequest normalized = parse(body).generation;
    failures += check(normalized.enable_thinking == true && normalized.preserve_thinking == false,
                      "Qwen/vLLM template aliases normalize");
    body["chat_template_kwargs"]["enable_thinking"] = false;
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting thinking aliases rejected");
    body                         = base_request();
    body["chat_template_kwargs"] = Json{{"future", 1}};
    failures +=
        check(Json::parse(parse(body).generation.chat_template_kwargs_json).at("future") == 1,
              "custom template keyword did not survive protocol parsing");
    body["chat_template_kwargs"] = Json{{"future", nullptr}};
    failures += check(parse(body).generation.messages.size() == 1,
                      "null unknown template option is neutral");
    body                     = base_request();
    body["reasoning_effort"] = "adaptive";
    failures += check(parse(body).generation.reasoning_effort == RequestedReasoningEffort::Adaptive,
                      "adaptive reasoning_effort did not parse");
    body["reasoning_effort"] = "high";
    failures += check(parse(body).generation.reasoning_effort == RequestedReasoningEffort::High,
                      "high reasoning_effort did not parse");
    body["reasoning_effort"] = "ultra";
    failures += check(api_error([&] { (void)parse(body); }).param == "reasoning_effort",
                      "unknown reasoning_effort accepted");

    body                        = base_request();
    body["repetition_penalty"]  = 1.0;
    body["mm_processor_kwargs"] = Json{{"max_pixels", nullptr}};
    failures +=
        check(parse(body).generation.messages.size() == 1, "neutral ecosystem defaults accepted");
    body["repetition_penalty"] = 1.1;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "repetition_penalty_not_supported",
              "non-neutral repetition penalty rejected");
    body                        = base_request();
    body["mm_processor_kwargs"] = Json{{"max_pixels", 100}};
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "mm_processor_kwargs_not_supported",
              "non-empty media processor kwargs rejected");
    return failures;
}

int test_stops_and_ranges() {
    int failures                            = 0;
    Json body                               = base_request();
    body["stop"]                            = Json::array({"A", "B"});
    const ninfer::RequestOptions translated = options(parse(body).generation);
    failures += check(translated.stop.strings.size() == 4,
                      "each stop string applies to Content and Reasoning");
    failures += check(translated.stop.strings[0].channel == ninfer::OutputChannel::Content &&
                          translated.stop.strings[1].channel == ninfer::OutputChannel::Reasoning,
                      "stop channel ordering is explicit");

    body["stop"] = Json::array({"1", "2", "3", "4", "5"});
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "more than four stop strings rejected");
    body["stop"] = "";
    failures +=
        check(api_error([&] { (void)parse(body); }).param == "stop", "empty stop string rejected");

    body                                  = base_request();
    body["top_k"]                         = 21;
    const GenerationRequest invalid_top_k = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_top_k); }).param == "top_k",
                      "Engine translator owns sampler value range");
    body["top_k"]                         = 5;
    body["min_p"]                         = 1.1;
    const GenerationRequest invalid_min_p = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_min_p); }).param == "min_p",
                      "min_p range enforced by common Engine translator");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                                = "answer";
    outcome.reasoning                           = "thought";
    outcome.prompt_tokens                       = 20;
    outcome.completion_tokens                   = 7;
    outcome.reasoning_tokens                    = 3;
    outcome.finish_reason                       = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens     = 12;
    outcome.metrics.prompt_wall_seconds         = 0.04;
    outcome.metrics.generation_wall_seconds     = 0.03;
    outcome.metrics.speculative_draft_tokens    = 9;
    outcome.metrics.speculative_accepted_tokens = 6;
    return outcome;
}

OpenAIChatResponseIdentity identity() {
    return OpenAIChatResponseIdentity{.id = "chatcmpl-test", .model = "qwen", .created = 42};
}

int test_aggregate_response() {
    int failures              = 0;
    GenerationOutcome outcome = sample_outcome();
    Json response             = Json::parse(make_chat_completion_response(identity(), outcome));
    failures += check(response["choices"][0]["message"]["content"] == "answer" &&
                          response["choices"][0]["message"]["reasoning_content"] == "thought" &&
                          response["choices"][0]["message"]["refusal"].is_null(),
                      "aggregate response separates reasoning and content");
    failures += check(response["choices"][0]["logprobs"].is_null(),
                      "aggregate choice carries nullable logprobs");
    failures += check(response["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3,
                      "aggregate usage exposes cache hits and reasoning tokens");
    failures += check(
        response["timings"]["cache_n"] == 12 && response["timings"]["prompt_n"] == 8 &&
            response["timings"]["prompt_ms"] == 40.0 &&
            response["timings"]["prompt_per_second"] == 200.0 &&
            response["timings"]["predicted_n"] == 7 &&
            response["timings"]["predicted_ms"] == 30.0 &&
            response["timings"]["predicted_per_second"] == 200.0 &&
            response["timings"]["draft_n"] == 9 && response["timings"]["draft_n_accepted"] == 6,
        "aggregate timings use exact cache and N-1 generation intervals");

    outcome.text.clear();
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit",
        .arguments_json =
            R"({"file_path":"/tmp/probe.cpp","old_string":"old","new_string":"new"})"});
    response         = Json::parse(make_chat_completion_response(identity(), outcome));
    const Json& call = response["choices"][0]["message"]["tool_calls"][0];
    failures += check(response["choices"][0]["finish_reason"] == "tool_calls" &&
                          response["choices"][0]["message"]["content"].is_null(),
                      "aggregate tool call has OpenAI terminal shape");
    failures += check(
        call["id"].get<std::string>().starts_with("call_") && call["function"]["name"] == "Edit" &&
            !Json::parse(call["function"]["arguments"].get<std::string>()).contains("replace_all"),
        "OpenAI adapter owns wire tool-call identifiers");
    return failures;
}

int test_stream_response() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true);
    Json role = parse_sse(stream.start());
    failures += check(role["choices"][0]["delta"]["role"] == "assistant" &&
                          role["choices"][0]["logprobs"].is_null() && role["usage"].is_null(),
                      "stream starts with role, nullable logprobs, and null usage");
    Json reasoning = parse_sse(stream.reasoning_delta("thought"));
    Json content   = parse_sse(stream.content_delta("ans"));
    failures += check(reasoning["choices"][0]["delta"]["reasoning_content"] == "thought" &&
                          content["choices"][0]["delta"]["content"] == "ans",
                      "stream separates reasoning and content deltas");

    GenerationOutcome outcome             = sample_outcome();
    const std::vector<std::string> events = stream.finish(outcome);
    failures +=
        check(events.size() == 4, "finish emits buffered suffix, terminal, usage, and done");
    failures += check(parse_sse(events[0])["choices"][0]["delta"]["content"] == "wer",
                      "terminal content suffix is emitted exactly once");
    failures += check(parse_sse(events[1])["choices"][0]["finish_reason"] == "stop",
                      "stream terminal finish reason emitted");
    const Json usage = parse_sse(events[2]);
    failures += check(usage["choices"].empty() &&
                          usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          usage["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3 &&
                          usage["timings"]["predicted_n"] == 7,
                      "dedicated stream usage carries token accounting and terminal timings");
    failures += check(events.back() == "data: [DONE]\n\n", "stream ends with DONE sentinel");

    OpenAIChatStream mismatch(identity(), false);
    (void)mismatch.start();
    (void)mismatch.content_delta("different");
    failures += check(throws_logic([&] { (void)mismatch.finish(outcome); }),
                      "stream encoder rejects terminal/content divergence");

    OpenAIChatStream tool_stream(identity(), false);
    (void)tool_stream.start();
    GenerationOutcome tool_outcome;
    tool_outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit", .arguments_json = R"({"file_path":"/tmp/probe.cpp"})"});
    tool_outcome.finish_reason                 = ninfer::FinishReason::StopToken;
    const std::vector<std::string> tool_events = tool_stream.finish(tool_outcome);
    const Json tool_delta                      = parse_sse(tool_events[0]);
    failures += check(
        tool_delta["choices"][0]["delta"]["tool_calls"][0]["id"].get<std::string>().starts_with(
            "call_") &&
            tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Edit" &&
            parse_sse(tool_events[1])["choices"][0]["finish_reason"] == "tool_calls",
        "stream encoder owns stable OpenAI tool-call shape");
    return failures;
}

int test_stream_observations() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true, true, true);
    const Json role = parse_sse(stream.start());
    failures += check(!role.contains("timings") && !role.contains("prompt_progress"),
                      "transport role chunk precedes Engine observations");

    stream.note_start(
        ninfer::GenerationStart{.prompt = {.prompt_tokens = 32}, .reused_prompt_tokens = 12});
    const Json initial = parse_sse(stream.initial_prompt_progress());
    failures +=
        check(initial["choices"][0]["delta"].empty() && initial["prompt_progress"]["total"] == 32 &&
                  initial["prompt_progress"]["cache"] == 12 &&
                  initial["prompt_progress"]["processed"] == 12 &&
                  initial["prompt_progress"]["time_ms"] == 0,
              "initial prompt progress begins at the admitted cache frontier");

    const Json middle = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 20,
        .elapsed_ns              = 57000000,
    }));
    failures += check(middle["prompt_progress"]["processed"] == 20 &&
                          middle["prompt_progress"]["time_ms"] == 57,
                      "prompt progress exposes a cumulative completed frontier");
    const Json complete = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 32,
        .elapsed_ns              = 100000000,
    }));
    failures +=
        check(complete["prompt_progress"]["processed"] == complete["prompt_progress"]["total"],
              "final prompt progress reaches the complete prompt");

    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens = 1, .prompt_elapsed_ns = 110000000, .generation_elapsed_ns = 0});
    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens      = 3,
        .prompt_elapsed_ns     = 110000000,
        .generation_elapsed_ns = 20000000,
    });
    const Json content = parse_sse(stream.content_delta("answer"));
    failures +=
        check(content["timings"]["prompt_n"] == 20 && content["timings"]["predicted_n"] == 3 &&
                  content["timings"]["predicted_per_second"] == 100.0,
              "visible output uses the latest independent commit observation");

    GenerationOutcome outcome = sample_outcome();
    outcome.reasoning.clear();
    const std::vector<std::string> terminal = stream.finish(outcome);
    failures += check(parse_sse(terminal[1])["timings"]["predicted_n"] == 7,
                      "terminal usage replaces live timing with exact final accounting");
    return failures;
}

// Calls the Qwen parser keeps despite their names (undeclared, catalog spellings resolved to a
// declared tool) come back in the next request's history without a 400, and a tool that
// allowed_tools filters out for the turn never becomes a call.
int test_salvaged_tool_calls() {
    const auto tool = [](const char* name) {
        return Json{
            {"type", "function"},
            {"function",
             Json{{"name", name},
                  {"parameters", Json{{"type", "object"},
                                      {"properties", Json{{"path", Json{{"type", "string"}}}}}}}}}};
    };
    Json body     = base_request();
    body["tools"] = Json::array({tool("Read"), tool("Write")});
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "Read"}}})}}}};
    const OpenAIChatRequest request = parse(body);
    const ninfer::PromptInput input = prompt(request.generation);
    const std::size_t max_name      = request.generation.tool_name_max_length;

    int failures = 0;
    for (const char* filtered : {"Write", "srv.Write", "tools[\"srv\"][\"Write\"]"}) {
        failures +=
            check(ninfer::test::parse_model_call(input, max_name, filtered).tool_calls.empty(),
                  std::string("allowed_tools-filtered call became structured: ") + filtered);
    }

    int kept = 0;
    for (const std::string& name : ninfer::test::salvage_probe_names()) {
        const auto parsed = ninfer::test::parse_model_call(input, max_name, name);
        if (parsed.tool_calls.empty()) { continue; }
        ++kept;
        GenerationOutcome outcome;
        outcome.finish_reason = ninfer::FinishReason::StopToken;
        outcome.tool_calls    = parsed.tool_calls;
        const Json message    = Json::parse(
            make_chat_completion_response(identity(), outcome))["choices"][0]["message"];
        Json next     = base_request();
        next["tools"] = body["tools"];
        next["messages"].push_back(Json{
            {"role", "assistant"}, {"content", nullptr}, {"tool_calls", message["tool_calls"]}});
        for (const Json& call : message["tool_calls"]) {
            next["messages"].push_back(
                Json{{"role", "tool"}, {"tool_call_id", call["id"]}, {"content", "unknown tool"}});
        }
        const ApiError error = api_error([&] { (void)parse(next); });
        failures +=
            check(error.message == "no exception",
                  "kept call '" + name + "' could not come back in history: " + error.message);
    }
    failures += check(kept >= 4, "the probe names no longer exercise salvaged calls");
    return failures;
}

int test_common_objects() {
    int failures      = 0;
    const Json models = Json::parse(make_models_list("qwen", 7, 240000));
    failures +=
        check(models["data"][0]["id"] == "qwen" && models["data"][0]["max_model_len"] == 240000,
              "models list advertises the configured context limit");
    const Json model = Json::parse(make_model_object("qwen", 7, 240000));
    failures += check(model["max_model_len"] == 240000,
                      "model lookup advertises the configured context limit");
    const Json error = Json::parse(make_error_body(
        ApiError{.status = 400, .message = "bad", .param = "messages", .code = "invalid"}));
    failures += check(error["error"]["param"] == "messages" && error["error"]["code"] == "invalid",
                      "OpenAI common error shape remains stable");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_request_envelope_and_sampling();
    failures += test_standard_field_policy();
    failures += test_constrained_decoding_extensions();
    failures += test_constraint_limits();
    failures += test_tools();
    failures += test_messages_and_media();
    failures += test_reasoning_and_extensions();
    failures += test_stops_and_ranges();
    failures += test_aggregate_response();
    failures += test_stream_response();
    failures += test_stream_observations();
    failures += test_salvaged_tool_calls();
    failures += test_common_objects();
    if (failures == 0) { std::cout << "OpenAI Chat protocol tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
