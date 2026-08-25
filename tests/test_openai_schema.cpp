// Contract test for the OpenAI serving layer: request parsing (string + parts
// content, unsupported-feature rejection), response/chunk/models/error
// serialization shapes, and finish_reason mapping. This is the schema boundary
// consumed by external OpenAI clients.

#include "serve/openai_schema.h"
#include "serve/request.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>

namespace {

using Json = nlohmann::json;
using namespace ninfer::serve;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

bool throws_api(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException&) { return true; } catch (...) {
        return false;
    }
    return false;
}

std::string api_code(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

RequestLimits default_limits() {
    RequestLimits limits;
    limits.default_max_tokens = 512;
    return limits;
}

ServeOptions default_server() { return ServeOptions{}; }

ninfer::PromptCapabilities effort_capabilities() {
    ninfer::PromptCapabilities capabilities;
    capabilities.enable_thinking                 = true;
    capabilities.reasoning_effort.low            = true;
    capabilities.reasoning_effort.medium         = true;
    capabilities.reasoning_effort.xhigh          = true;
    capabilities.reasoning_effort.default_effort = ninfer::ReasoningEffort::XHigh;
    return capabilities;
}

ninfer::OwnedMedia fake_media(const ContentPart& part) {
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.bytes.push_back(0);
    media.media_type = part.source.media_type;
    return media;
}

ninfer::PromptInput translate(const GenerationRequest& req) {
    const ServeOptions server = default_server();
    return to_prompt_input(req, resolve_prompt_semantics(req, server, effort_capabilities()),
                           fake_media);
}

std::string joined_text(const ninfer::ChatMessage& message) {
    std::string text;
    for (const ninfer::MessagePart& part : message.parts) {
        if (part.kind == ninfer::MessagePartKind::Text) { text += part.text; }
    }
    return text;
}

// Strip "data: " prefix and trailing blank line from an SSE event, returning the
// parsed JSON payload.
Json parse_sse(const std::string& event) {
    const std::string prefix = "data: ";
    const std::string suffix = "\n\n";
    if (event.rfind(prefix, 0) != 0 || event.size() < prefix.size() + suffix.size()) {
        throw std::runtime_error("bad SSE framing: " + event);
    }
    const std::string json =
        event.substr(prefix.size(), event.size() - prefix.size() - suffix.size());
    return Json::parse(json);
}

int test_parse_string_content() {
    int failures                = 0;
    const Json body             = {{"model", "qwen3.6-27b"},
                                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.model == "qwen3.6-27b", "model parsed");
    failures += check(req.messages.size() == 1, "one message parsed");
    failures += check(req.messages[0].role == ninfer::ChatRole::User, "role parsed");
    failures += check(req.messages[0].content.size() == 1, "one content part");
    failures += check(req.messages[0].content[0].kind == ContentKind::Text, "text part kind");
    failures += check(req.messages[0].content[0].text == "hello", "text part content");
    failures += check(!req.stream, "stream defaults false");
    failures += check(req.max_tokens == 512, "max_tokens default applied");
    failures += check(!req.max_tokens_set, "max_tokens_set false when defaulted");
    return failures;
}

int test_preserve_thinking_options() {
    const Json base = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
    };
    int failures = 0;

    Json kwargs                    = base;
    kwargs["chat_template_kwargs"] = Json{{"preserve_thinking", true}};
    const GenerationRequest kwargs_request =
        parse_chat_completion_request(kwargs, default_limits());
    failures += check(kwargs_request.preserve_thinking == true,
                      "chat_template_kwargs preserve_thinking parsed");
    failures += check(translate(kwargs_request).options.preserve_thinking,
                      "resolved preserve_thinking reached PromptInput");

    Json alias                 = base;
    alias["preserve_thinking"] = false;
    failures +=
        check(parse_chat_completion_request(alias, default_limits()).preserve_thinking == false,
              "top-level preserve_thinking alias parsed");

    Json same                 = kwargs;
    same["preserve_thinking"] = true;
    failures +=
        check(parse_chat_completion_request(same, default_limits()).preserve_thinking == true,
              "matching preserve_thinking values rejected");

    Json nulls                    = base;
    nulls["preserve_thinking"]    = nullptr;
    nulls["chat_template_kwargs"] = Json{{"preserve_thinking", nullptr}, {"future", nullptr}};
    failures +=
        check(!parse_chat_completion_request(nulls, default_limits()).preserve_thinking.has_value(),
              "null preserve_thinking did not remain omitted");

    Json conflict                 = kwargs;
    conflict["preserve_thinking"] = false;
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(conflict, default_limits()); }),
              "conflicting preserve_thinking values accepted");

    Json bad_kwargs                    = base;
    bad_kwargs["chat_template_kwargs"] = true;
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(bad_kwargs, default_limits()); }),
        "non-object chat_template_kwargs accepted");
    Json bad_value                    = base;
    bad_value["chat_template_kwargs"] = Json{{"preserve_thinking", "yes"}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(bad_value, default_limits()); }),
              "non-boolean preserve_thinking accepted");
    Json unknown                    = base;
    unknown["chat_template_kwargs"] = Json{{"preserve_thinking", true}, {"foo", 1}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(unknown, default_limits()); }),
              "unknown non-null chat template option accepted");
    return failures;
}

bool thinking_disabled(const ResolvedPromptSemantics& semantics) {
    return !semantics.enable_thinking && !semantics.reasoning_effort;
}

int test_enable_thinking_dialect() {
    const Json base = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
    };
    int failures = 0;

    const ResolvedPromptSemantics omitted =
        resolve_prompt_semantics(parse_chat_completion_request(base, default_limits()),
                                 default_server(), effort_capabilities());
    failures += check(omitted.enable_thinking && !omitted.reasoning_effort,
                      "omitted thinking did not use the server default on");

    Json none                = base;
    none["reasoning_effort"] = "none";
    const ResolvedPromptSemantics none_semantics =
        resolve_prompt_semantics(parse_chat_completion_request(none, default_limits()),
                                 default_server(), effort_capabilities());
    failures += check(thinking_disabled(none_semantics),
                      "reasoning_effort none did not disable thinking");

    Json kwargs                    = base;
    kwargs["chat_template_kwargs"] = Json{{"enable_thinking", false}};
    const GenerationRequest kwargs_request =
        parse_chat_completion_request(kwargs, default_limits());
    failures += check(kwargs_request.enable_thinking == false,
                      "chat_template_kwargs enable_thinking was not parsed");
    const ResolvedPromptSemantics kwargs_semantics =
        resolve_prompt_semantics(kwargs_request, default_server(), effort_capabilities());
    failures += check(thinking_disabled(kwargs_semantics),
                      "chat_template_kwargs enable_thinking false did not match reasoning_effort none");

    Json top                 = base;
    top["enable_thinking"]   = false;
    const GenerationRequest top_request = parse_chat_completion_request(top, default_limits());
    failures +=
        check(top_request.enable_thinking == false, "top-level enable_thinking was not parsed");
    const ResolvedPromptSemantics top_semantics =
        resolve_prompt_semantics(top_request, default_server(), effort_capabilities());
    failures += check(thinking_disabled(top_semantics),
                      "top-level enable_thinking false did not match reasoning_effort none");

    Json both               = kwargs;
    both["enable_thinking"] = false;
    failures += check(parse_chat_completion_request(both, default_limits()).enable_thinking == false,
                      "matching enable_thinking values were rejected");

    Json conflict               = kwargs;
    conflict["enable_thinking"] = true;
    failures += check(api_code([&] {
                          (void)parse_chat_completion_request(conflict, default_limits());
                      }) == "conflicting_template_option",
                      "conflicting enable_thinking values were accepted");

    Json on                    = base;
    on["chat_template_kwargs"] = Json{{"enable_thinking", true}};
    ServeOptions no_think      = default_server();
    no_think.enable_thinking   = false;
    failures += check(resolve_prompt_semantics(parse_chat_completion_request(on, default_limits()),
                                               no_think, effort_capabilities())
                          .enable_thinking,
                      "chat_template_kwargs enable_thinking true did not override server default off");

    Json misspelled                    = base;
    misspelled["chat_template_kwargs"] = Json{{"enable_thinkng", false}};
    failures += check(api_code([&] {
                          (void)parse_chat_completion_request(misspelled, default_limits());
                      }) == "chat_template_option_not_supported",
                      "misspelled enable_thinking was not rejected");

    Json bad                    = base;
    bad["chat_template_kwargs"] = Json{{"enable_thinking", "no"}};
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(bad, default_limits()); }),
        "non-boolean chat_template_kwargs.enable_thinking was accepted");

    Json combined                    = base;
    combined["chat_template_kwargs"] = Json{{"enable_thinking", false}, {"preserve_thinking", true}};
    const GenerationRequest combined_request =
        parse_chat_completion_request(combined, default_limits());
    failures += check(combined_request.enable_thinking == false &&
                          combined_request.preserve_thinking == true,
                      "enable_thinking and preserve_thinking were not accepted together");
    ServeOptions preserved     = default_server();
    preserved.preserve_thinking = true;
    const ResolvedPromptSemantics combined_semantics =
        resolve_prompt_semantics(combined_request, default_server(), effort_capabilities());
    failures += check(thinking_disabled(combined_semantics) && combined_semantics.preserve_thinking,
                      "enable_thinking false cleared request preserve_thinking");
    const ResolvedPromptSemantics server_preserved =
        resolve_prompt_semantics(kwargs_request, preserved, effort_capabilities());
    failures += check(thinking_disabled(server_preserved) && server_preserved.preserve_thinking,
                      "enable_thinking false cleared server preserve_thinking");

    Json agree               = none;
    agree["enable_thinking"] = false;
    failures += check(thinking_disabled(resolve_prompt_semantics(
                          parse_chat_completion_request(agree, default_limits()), default_server(),
                          effort_capabilities())),
                      "enable_thinking false with reasoning_effort none was rejected");

    Json effort_conflict               = none;
    effort_conflict["enable_thinking"] = true;
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(
                              parse_chat_completion_request(effort_conflict, default_limits()),
                              default_server(), effort_capabilities());
                      }) == "conflicting_template_option",
                      "enable_thinking true with reasoning_effort none was accepted");
    return failures;
}

int test_reasoning_effort() {
    const Json base = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
    };
    int failures = 0;

    Json low                            = base;
    low["reasoning_effort"]             = "low";
    const GenerationRequest low_request = parse_chat_completion_request(low, default_limits());
    failures += check(low_request.reasoning_effort == RequestedReasoningEffort::Low,
                      "Chat Completions reasoning_effort was not parsed");
    const ninfer::PromptInput low_prompt = translate(low_request);
    failures += check(low_prompt.options.enable_thinking &&
                          low_prompt.options.reasoning_effort == ninfer::ReasoningEffort::Low,
                      "Chat Completions low effort did not reach PromptInput");

    Json none                = base;
    none["reasoning_effort"] = "none";
    const ninfer::PromptInput none_prompt =
        translate(parse_chat_completion_request(none, default_limits()));
    failures += check(!none_prompt.options.enable_thinking && !none_prompt.options.reasoning_effort,
                      "Chat Completions none effort did not disable thinking");

    for (const auto& [wire, expected] :
         std::array<std::pair<const char*, RequestedReasoningEffort>, 6>{
             {{"minimal", RequestedReasoningEffort::Minimal},
              {"medium", RequestedReasoningEffort::Medium},
              {"high", RequestedReasoningEffort::High},
              {"xhigh", RequestedReasoningEffort::XHigh},
              {"max", RequestedReasoningEffort::Max},
              {"none", RequestedReasoningEffort::None}}}) {
        Json body                = base;
        body["reasoning_effort"] = wire;
        failures += check(parse_chat_completion_request(body, default_limits()).reasoning_effort ==
                              expected,
                          std::string("Chat Completions did not accept protocol effort ") + wire);
    }

    Json high                            = base;
    high["reasoning_effort"]             = "high";
    const GenerationRequest high_request = parse_chat_completion_request(high, default_limits());
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(high_request, default_server(),
                                                         effort_capabilities());
                      }) == "reasoning_effort_not_supported",
                      "protocol-valid high effort was not rejected by template capability");

    ninfer::PromptCapabilities toggle_capabilities;
    toggle_capabilities.enable_thinking = true;
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(low_request, default_server(),
                                                         toggle_capabilities);
                      }) == "reasoning_effort_not_supported",
                      "reasoning effort was accepted without template support");

    Json conflict               = low;
    conflict["enable_thinking"] = false;
    const GenerationRequest conflicting_request =
        parse_chat_completion_request(conflict, default_limits());
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(conflicting_request, default_server(),
                                                         effort_capabilities());
                      }) == "conflicting_template_option",
                      "conflicting enable_thinking and reasoning_effort were accepted");

    Json invalid                = base;
    invalid["reasoning_effort"] = "ultra";
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(invalid, default_limits()); }),
              "unknown Chat Completions reasoning effort was accepted");
    invalid["reasoning_effort"] = 1;
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(invalid, default_limits()); }),
              "non-string Chat Completions reasoning effort was accepted");
    return failures;
}

int test_parse_parts_and_flatten() {
    int failures    = 0;
    const Json body = {
        {"model", "m"},
        {"messages",
         Json::array({Json{{"role", "user"},
                           {"content", Json::array({Json{{"type", "text"}, {"text", "a"}},
                                                    Json{{"type", "text"}, {"text", "b"}}})}}})}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.messages[0].content.size() == 2, "two content parts");
    const ninfer::PromptInput prompt = translate(req);
    failures += check(prompt.messages.size() == 1, "flattened to one message");
    failures += check(joined_text(prompt.messages[0]) == "a\nb", "text parts joined");
    return failures;
}

int test_parse_media_in_translate() {
    const Json body = {
        {"model", "m"},
        {"messages",
         Json::array({Json{
             {"role", "user"},
             {"content",
              Json::array(
                  {Json{{"type", "image_url"},
                        {"image_url", Json{{"url", "data:image/png;base64,AA=="}}}},
                   Json{{"type", "video_url"},
                        {"video_url", Json{{"url", "https://example.test/clip.mp4"}}}}})}}})}};
    const GenerationRequest req      = parse_chat_completion_request(body, default_limits());
    const ninfer::PromptInput prompt = translate(req);
    int failures                     = 0;
    failures += check(req.messages[0].content[0].kind == ContentKind::Image,
                      "image content kind preserved");
    failures += check(req.messages[0].content[0].source.kind ==
                          ninfer::product::media_acquire::SourceKind::Data,
                      "image data URI source preserved");
    failures += check(prompt.messages[0].parts[0].kind == ninfer::MessagePartKind::Media &&
                          prompt.messages[0].parts[0].media.kind == ninfer::MediaKind::Image,
                      "image translated to structured chat part");
    failures += check(req.messages[0].content[1].kind == ContentKind::Video,
                      "video content kind preserved");
    failures += check(req.messages[0].content[1].source.kind ==
                          ninfer::product::media_acquire::SourceKind::Url,
                      "video URL source preserved");
    failures += check(prompt.messages[0].parts[1].kind == ninfer::MessagePartKind::Media &&
                          prompt.messages[0].parts[1].media.kind == ninfer::MediaKind::Video,
                      "video translated to structured chat part");
    return failures;
}

int test_instruction_roles_preserved() {
    const Json body = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "developer"}, {"content", "be terse"}},
                                  Json{{"role", "user"}, {"content", "hi"}},
                                  Json{{"role", "system"}, {"content", "new context"}}})}};
    const GenerationRequest req      = parse_chat_completion_request(body, default_limits());
    const ninfer::PromptInput prompt = translate(req);
    int failures                     = 0;
    failures +=
        check(req.messages.size() == 3 && req.messages[0].role == ninfer::ChatRole::Developer &&
                  req.messages[1].role == ninfer::ChatRole::User &&
                  req.messages[2].role == ninfer::ChatRole::System,
              "schema did not preserve ordered developer/system roles");
    failures += check(prompt.messages.size() == 3 &&
                          prompt.messages[0].role == ninfer::ChatRole::Developer &&
                          prompt.messages[1].role == ninfer::ChatRole::User &&
                          prompt.messages[2].role == ninfer::ChatRole::System,
                      "translation changed roles before target-specific lowering");
    return failures;
}

int test_reject_unsupported() {
    int failures    = 0;
    const Json base = {{"model", "m"},
                       {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};

    Json n2 = base;
    n2["n"] = 2;
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(n2, default_limits()); }),
              "n>1 rejected");

    Json custom_tool = base;
    custom_tool["tools"] =
        Json::array({Json{{"type", "custom"}, {"custom", Json{{"name", "search"}}}}});
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(custom_tool, default_limits()); }),
        "custom tools rejected");

    Json functions         = base;
    functions["functions"] = Json::array({Json::object()});
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(functions, default_limits()); }),
              "deprecated functions rejected");

    Json function_call             = base;
    function_call["function_call"] = "auto";
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(function_call, default_limits()); }),
        "deprecated function_call rejected");

    Json rf               = base;
    rf["response_format"] = Json{{"type", "json_object"}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(rf, default_limits()); }),
              "json response_format rejected");

    Json rf_text               = base;
    rf_text["response_format"] = Json{{"type", "text"}};
    bool text_ok               = true;
    try {
        (void)parse_chat_completion_request(rf_text, default_limits());
    } catch (...) { text_ok = false; }
    failures += check(text_ok, "text response_format accepted");

    Json no_model = {{"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(no_model, default_limits()); }),
              "missing model rejected");

    Json function_role = {
        {"model", "m"}, {"messages", Json::array({Json{{"role", "function"}, {"content", "x"}}})}};
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(function_role, default_limits()); }),
        "function role rejected");
    return failures;
}

int test_parse_function_tools_and_choices() {
    int failures = 0;
    const Json tool =
        Json{{"type", "function"},
             {"function",
              Json{{"name", "get_weather"},
                   {"description", "Fetch weather"},
                   {"parameters", Json{{"type", "object"},
                                       {"properties", Json{{"city", Json{{"type", "string"}}}}},
                                       {"required", Json::array({"city"})}}},
                   {"strict", true}}}};
    const Json base = {{"model", "m"},
                       {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                       {"tools", Json::array({tool})}};

    GenerationRequest req = parse_chat_completion_request(base, default_limits());
    failures += check(req.tools.size() == 1, "one tool parsed");
    failures += check(req.tools[0].name == "get_weather", "tool name parsed");
    failures += check(req.tools[0].description == "Fetch weather", "tool description parsed");
    failures += check(req.tools[0].strict, "tool strict metadata parsed");
    failures += check(Json::parse(req.tools[0].parameters_json).at("required").at(0) == "city",
                      "tool parameters carried");
    failures += check(Json::parse(req.tools[0].definition_json).at("type") == "function",
                      "tool definition json carried");
    failures += check(req.tool_choice.mode == ToolChoiceMode::Auto, "default tool choice is auto");
    failures += check(req.uses_tools(), "tools enabled by default");
    failures += check(to_request_options(req, default_server()).output.preserve_special_tokens,
                      "active tools preserve special tokens in Engine output");

    Json none           = base;
    none["tool_choice"] = "none";
    req                 = parse_chat_completion_request(none, default_limits());
    failures += check(req.tool_choice.mode == ToolChoiceMode::None, "tool_choice none parsed");
    failures += check(!req.uses_tools(), "tool_choice none disables tools");
    failures += check(!to_request_options(req, default_server()).output.preserve_special_tokens,
                      "disabled tools do not preserve special tokens");

    Json required           = base;
    required["tool_choice"] = "required";
    req                     = parse_chat_completion_request(required, default_limits());
    failures +=
        check(req.tool_choice.mode == ToolChoiceMode::Required, "tool_choice required parsed");

    Json named           = base;
    named["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "get_weather"}}}};
    req                  = parse_chat_completion_request(named, default_limits());
    failures += check(req.tool_choice.mode == ToolChoiceMode::Named, "named tool_choice parsed");
    failures += check(req.tool_choice.name == "get_weather", "named tool_choice name parsed");

    Json unknown           = base;
    unknown["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "missing"}}}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(unknown, default_limits()); }),
              "unknown named tool_choice rejected");

    // Duplicate function tool names are rejected (matches Responses behavior).
    Json dup           = base;
    dup["tools"]       = Json::array({tool, tool});
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(dup, default_limits()); }),
              "duplicate function tool names rejected");
    return failures;
}

int test_parse_tool_history_messages() {
    int failures    = 0;
    const Json body = {
        {"model", "m"},
        {"messages",
         Json::array(
             {Json{{"role", "user"}, {"content", "weather?"}},
              Json{{"role", "assistant"},
                   {"content", nullptr},
                   {"tool_calls",
                    Json::array({Json{{"id", "call_1"},
                                      {"type", "function"},
                                      {"function", Json{{"name", "get_weather"},
                                                        {"arguments", R"({"city":"Paris"})"}}}}})}},
              Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", R"({"temp":20})"}}})}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.messages.size() == 3, "tool history message count");
    failures += check(req.messages[1].tool_calls.size() == 1, "assistant tool call parsed");
    failures += check(req.messages[1].tool_calls[0].id == "call_1", "tool call id parsed");
    failures += check(req.messages[1].tool_calls[0].name == "get_weather", "tool call name parsed");
    failures += check(req.messages[1].tool_calls[0].arguments_json == R"({"city":"Paris"})",
                      "tool call arguments parsed");
    failures += check(req.messages[2].role == ninfer::ChatRole::Tool, "tool role parsed");
    failures += check(req.messages[2].tool_call_id == "call_1", "tool_call_id parsed");
    failures +=
        check(req.messages[2].content.at(0).text == R"({"temp":20})", "tool content parsed");
    failures += check(to_request_options(req, default_server()).output.preserve_special_tokens,
                      "tool history preserves special tokens in Engine output");

    Json bad_args                                                     = body;
    bad_args["messages"][1]["tool_calls"][0]["function"]["arguments"] = R"(["Paris"])";
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(bad_args, default_limits()); }),
              "non-object tool call arguments rejected");
    return failures;
}

int test_parse_stop_and_max_tokens() {
    int failures          = 0;
    Json body             = {{"model", "m"},
                             {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                             {"stop", Json::array({"</s>", "STOP"})},
                             {"max_completion_tokens", 42}};
    GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.stop_strings.size() == 2, "two stop strings");
    failures += check(req.stop_strings[0] == "</s>", "stop string 0");
    failures += check(req.max_tokens == 42 && req.max_tokens_set, "max_completion_tokens alias");
    const ninfer::RequestOptions options = to_request_options(req, default_server());
    failures += check(options.execution.requested_output_tokens == 42,
                      "max_completion_tokens reaches Engine options");
    failures += check(options.stop.strings.size() == 2 && options.stop.strings[0].text == "</s>" &&
                          options.stop.strings[1].text == "STOP",
                      "stop strings reach Engine options");

    Json single = {{"model", "m"},
                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                   {"stop", "END"}};
    req         = parse_chat_completion_request(single, default_limits());
    failures +=
        check(req.stop_strings.size() == 1 && req.stop_strings[0] == "END", "single stop string");
    return failures;
}

int test_parse_sampling_carried() {
    int failures                = 0;
    const Json body             = {{"model", "m"},
                                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                                   {"temperature", 0.7},
                                   {"top_p", 0.9},
                                   {"seed", 123},
                                   {"logit_bias", Json{{"5", -1.5}}}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.sampling.temperature.has_value() && *req.sampling.temperature == 0.7,
                      "temperature carried");
    failures +=
        check(req.sampling.top_p.has_value() && *req.sampling.top_p == 0.9, "top_p carried");
    failures += check(req.sampling.seed.has_value() && *req.sampling.seed == 123u, "seed carried");
    failures +=
        check(req.sampling.logit_bias.count(5) == 1 && req.sampling.logit_bias.at(5) == -1.5,
              "logit_bias carried");
    const ninfer::RequestOptions options = to_request_options(req, default_server());
    failures += check(options.execution.sampling.temperature == 0.7F,
                      "temperature reaches Engine overrides");
    failures += check(options.execution.sampling.top_p == 0.9F, "top_p reaches Engine overrides");
    failures += check(options.execution.sampling.seed == 123u, "seed reaches Engine overrides");
    failures +=
        check(!options.execution.sampling.top_k && !options.execution.sampling.presence_penalty,
              "omitted request fields unexpectedly replaced model defaults");
    return failures;
}

int test_response_serialization() {
    int failures = 0;
    const CompletionUsage usage{10, 3};
    const Json j = Json::parse(
        make_chat_completion_response("id-1", "m", 111, "hello world", "", "stop", usage));
    failures += check(j.at("object") == "chat.completion", "response object");
    failures += check(j.at("id") == "id-1", "response id");
    failures +=
        check(j.at("choices").at(0).at("message").at("role") == "assistant", "assistant role");
    failures += check(j.at("choices").at(0).at("message").at("content") == "hello world",
                      "response content");
    // Empty reasoning must not emit the reasoning_content key at all.
    failures += check(!j.at("choices").at(0).at("message").contains("reasoning_content"),
                      "no reasoning_content when reasoning empty");
    failures +=
        check(j.at("choices").at(0).at("finish_reason") == "stop", "response finish_reason");
    failures += check(j.at("usage").at("prompt_tokens") == 10, "usage prompt_tokens");
    failures += check(j.at("usage").at("completion_tokens") == 3, "usage completion_tokens");
    failures += check(j.at("usage").at("total_tokens") == 13, "usage total_tokens");
    failures += check(j.at("usage").at("prompt_tokens_details").at("cached_tokens") == 0,
                      "default cached_tokens is additive and zero");
    failures += check(j.at("usage").at("prefix_cache_hit_tokens") == 0,
                      "default prefix_cache_hit_tokens matches the log field");
    failures += check(j.at("usage").at("prefix_reuse_path") == "full_reset",
                      "default prefix_reuse_path matches the log field");

    // Non-empty reasoning is attached as message.reasoning_content, content stays answer-only.
    const Json jr = Json::parse(make_chat_completion_response("id-2", "m", 111, "the answer",
                                                              "let me think", "stop", usage));
    failures += check(jr.at("choices").at(0).at("message").at("content") == "the answer",
                      "reasoning response content is answer only");
    failures +=
        check(jr.at("choices").at(0).at("message").at("reasoning_content") == "let me think",
              "reasoning_content carried");
    return failures;
}

int test_tool_response_serialization() {
    int failures = 0;
    const CompletionUsage usage{12, 6};
    const std::vector<ToolCall> calls = {
        ToolCall{"call_abc", "get_weather", R"({"city":"Paris"})"}};
    const Json j = Json::parse(
        make_chat_completion_tool_response("id-tool", "m", 222, "", "need weather", calls, usage));

    failures += check(j.at("object") == "chat.completion", "tool response object");
    const Json& choice = j.at("choices").at(0);
    failures += check(choice.at("finish_reason") == "tool_calls", "tool finish reason");
    const Json& message = choice.at("message");
    failures += check(message.at("role") == "assistant", "tool message role");
    failures += check(message.at("content").is_null(), "empty tool content is null");
    failures += check(message.at("reasoning_content") == "need weather", "tool reasoning carried");
    const Json& call = message.at("tool_calls").at(0);
    failures += check(call.at("id") == "call_abc", "tool call id");
    failures += check(call.at("type") == "function", "tool call type");
    failures += check(call.at("function").at("name") == "get_weather", "tool function name");
    failures += check(call.at("function").at("arguments") == R"({"city":"Paris"})",
                      "tool function arguments");
    failures += check(j.at("usage").at("total_tokens") == 18, "tool usage total");

    const Json with_content = Json::parse(make_chat_completion_tool_response(
        "id-tool-2", "m", 223, "Calling weather.", "", calls, usage));
    failures +=
        check(with_content.at("choices").at(0).at("message").at("content") == "Calling weather.",
              "tool content prefix carried");
    return failures;
}

int test_chunk_serialization() {
    int failures    = 0;
    const Json role = parse_sse(make_chat_chunk_role("id", "m", 1, false));
    failures += check(role.at("object") == "chat.completion.chunk", "chunk object");
    failures += check(role.at("choices").at(0).at("delta").at("role") == "assistant", "role delta");
    failures += check(!role.contains("usage"), "no usage key when include_usage=false");

    const Json content = parse_sse(make_chat_chunk_content("id", "m", 1, "tok", false));
    failures +=
        check(content.at("choices").at(0).at("delta").at("content") == "tok", "content delta");

    // Reasoning deltas carry reasoning_content (not content) so clients render them
    // as a separate thinking channel.
    const Json reasoning = parse_sse(make_chat_chunk_reasoning("id", "m", 1, "why", false));
    failures += check(reasoning.at("choices").at(0).at("delta").at("reasoning_content") == "why",
                      "reasoning delta");
    failures += check(!reasoning.at("choices").at(0).at("delta").contains("content"),
                      "reasoning delta has no content key");

    // When usage reporting is on, content-bearing chunks carry usage: null.
    const Json role_usage = parse_sse(make_chat_chunk_role("id", "m", 1, true));
    failures += check(role_usage.contains("usage") && role_usage.at("usage").is_null(),
                      "role usage null when include_usage=true");
    const Json content_usage = parse_sse(make_chat_chunk_content("id", "m", 1, "x", true));
    failures += check(content_usage.contains("usage") && content_usage.at("usage").is_null(),
                      "content usage null when include_usage=true");

    // Final chunk carries finish_reason with an empty delta and no usage stats.
    const Json final_chunk = parse_sse(make_chat_chunk_final("id", "m", 1, "length", true));
    failures += check(final_chunk.at("choices").at(0).at("finish_reason") == "length",
                      "final finish_reason");
    failures += check(final_chunk.at("choices").at(0).at("delta").empty(), "final delta empty");
    failures += check(final_chunk.contains("usage") && final_chunk.at("usage").is_null(),
                      "final usage null (stats live on dedicated chunk)");

    const Json final_no_usage = parse_sse(make_chat_chunk_final("id", "m", 1, "stop", false));
    failures += check(!final_no_usage.contains("usage"), "no usage key when include_usage=false");

    // Dedicated usage chunk: empty choices, populated usage.
    const CompletionUsage usage{2, 5};
    const Json usage_chunk = parse_sse(make_chat_chunk_usage("id", "m", 1, usage));
    failures += check(usage_chunk.at("choices").is_array() && usage_chunk.at("choices").empty(),
                      "usage chunk has empty choices");
    failures +=
        check(usage_chunk.at("usage").at("prompt_tokens") == 2, "usage chunk prompt_tokens");
    failures += check(usage_chunk.at("usage").at("total_tokens") == 7, "usage chunk total");
    failures += check(usage_chunk.at("usage").at("prompt_tokens_details").at("cached_tokens") == 0,
                      "usage chunk cached_tokens additive");

    failures += check(sse_done() == "data: [DONE]\n\n", "done sentinel");
    return failures;
}

int test_usage_prefix_observability() {
    int failures = 0;
    CompletionUsage usage;
    usage.prompt_tokens        = 100;
    usage.completion_tokens    = 8;
    usage.cached_prompt_tokens = 60;
    usage.prefix_reuse_path    = ninfer::PrefixReusePath::SeedPrefixCache;
    const Json j =
        Json::parse(make_chat_completion_response("id", "m", 1, "pong", "", "stop", usage));
    const Json& u = j.at("usage");
    failures += check(u.at("prompt_tokens") == 100 && u.at("completion_tokens") == 8 &&
                          u.at("total_tokens") == 108,
                      "OpenAI usage totals unchanged");
    failures += check(u.at("prompt_tokens_details").at("cached_tokens") == 60,
                      "OpenAI cached_tokens is a subset of prompt_tokens");
    failures += check(u.at("prefix_cache_hit_tokens") == 60,
                      "log-named cached count matches prompt_tokens_details");
    failures += check(u.at("prefix_reuse_path") == "seed_prefix", "log-named reuse path");
    failures += check(u.at("total_tokens") == u.at("prompt_tokens").get<int>() +
                                                  u.at("completion_tokens").get<int>(),
                      "cached_tokens is not an addend of total_tokens");

    CompletionUsage clamped;
    clamped.prompt_tokens        = 10;
    clamped.cached_prompt_tokens = 99;
    clamped.prefix_reuse_path    = ninfer::PrefixReusePath::AppendAtFrontier;
    const Json c =
        Json::parse(make_chat_completion_response("id", "m", 1, "x", "", "stop", clamped)).at("usage");
    failures += check(c.at("prompt_tokens_details").at("cached_tokens") == 10 &&
                          c.at("prefix_cache_hit_tokens") == 10,
                      "cached_tokens clamped to prompt_tokens");
    failures += check(c.at("prefix_reuse_path") == "append_frontier", "append_frontier wire name");

    for (const auto& [path, name] :
         std::array<std::pair<ninfer::PrefixReusePath, const char*>, 5>{
             {{ninfer::PrefixReusePath::FullReset, "full_reset"},
              {ninfer::PrefixReusePath::AppendAtFrontier, "append_frontier"},
              {ninfer::PrefixReusePath::RestoreTurnCheckpoint, "restore_turn_checkpoint"},
              {ninfer::PrefixReusePath::RestoreResponseCheckpoint, "restore_response_checkpoint"},
              {ninfer::PrefixReusePath::SeedPrefixCache, "seed_prefix"}}}) {
        failures += check(std::string(prefix_reuse_path_name(path)) == name,
                          std::string("reuse path wire name ") + name);
    }
    return failures;
}

int test_tool_chunk_serialization() {
    int failures                      = 0;
    const std::vector<ToolCall> calls = {
        ToolCall{"call_abc", "get_weather", R"({"city":"Paris"})"}};
    const Json chunk = parse_sse(make_chat_chunk_tool_calls("id", "m", 1, calls, true));
    failures += check(chunk.at("object") == "chat.completion.chunk", "tool chunk object");
    const Json& delta = chunk.at("choices").at(0).at("delta");
    const Json& call  = delta.at("tool_calls").at(0);
    failures += check(call.at("index") == 0, "tool chunk index");
    failures += check(call.at("id") == "call_abc", "tool chunk id");
    failures += check(call.at("type") == "function", "tool chunk type");
    failures += check(call.at("function").at("name") == "get_weather", "tool chunk name");
    failures +=
        check(call.at("function").at("arguments") == R"({"city":"Paris"})", "tool chunk arguments");
    failures +=
        check(chunk.contains("usage") && chunk.at("usage").is_null(), "tool chunk usage null");

    const Json final_chunk = parse_sse(make_chat_chunk_final("id", "m", 1, "tool_calls", true));
    failures += check(final_chunk.at("choices").at(0).at("finish_reason") == "tool_calls",
                      "tool final finish reason");
    return failures;
}

int test_models_and_error() {
    int failures                               = 0;
    constexpr std::uint32_t configured_context = 131072;
    const Json list = Json::parse(make_models_list("qwen3.6-27b", 1, configured_context));
    failures += check(list.at("object") == "list", "models list object");
    failures += check(list.at("data").at(0).at("id") == "qwen3.6-27b", "models list id");
    failures += check(list.at("data").at(0).at("object") == "model", "models list entry object");
    failures += check(list.at("data").at(0).at("owned_by") == "ninfer", "models list owner");
    failures += check(list.at("data").at(0).at("max_model_len") == configured_context,
                      "models list configured context");

    const Json one = Json::parse(make_model_object("qwen3.6-27b", 1, configured_context));
    failures += check(one.at("id") == "qwen3.6-27b" && one.at("object") == "model", "model object");
    failures += check(one.at("owned_by") == "ninfer", "model owner");
    failures += check(one.at("max_model_len") == configured_context,
                      "model object configured context");

    constexpr std::uint32_t long_context = 262144;
    const Json long_list = Json::parse(make_models_list("qwen3.8-27b", 1, long_context));
    failures += check(long_list.at("data").at(0).at("max_model_len") == long_context,
                      "models list 262144 context");
    failures += check(Json::parse(make_model_object("qwen3.8-27b", 1, long_context))
                              .at("max_model_len") == long_context,
                      "model object 262144 context");

    ApiError error;
    error.status   = 400;
    error.type     = "invalid_request_error";
    error.message  = "bad";
    error.param    = "messages";
    const Json err = Json::parse(make_error_body(error));
    failures += check(err.at("error").at("message") == "bad", "error message");
    failures += check(err.at("error").at("type") == "invalid_request_error", "error type");
    failures += check(err.at("error").at("param") == "messages", "error param");
    failures += check(err.at("error").at("code").is_null(), "error code null when empty");
    return failures;
}

int test_finish_reason_wire() {
    int failures = 0;
    failures += check(std::string(finish_reason_wire(ninfer::FinishReason::StopToken)) == "stop",
                      "stop token wire");
    failures +=
        check(std::string(finish_reason_wire(ninfer::FinishReason::OutputLimit)) == "length",
               "output limit wire");
    failures += check(std::string(finish_reason_wire(ninfer::FinishReason::Cancelled)) == "stop",
                      "cancelled maps to stop");
    return failures;
}

int test_parse_content_parts_allowed_types() {
    int failures = 0;
    ChatTurn turn;
    const Json valid_text = Json::array({Json{{"type", "text"}, {"text", "hello"}}});
    parse_content_parts(valid_text, turn, 0, {"text"});
    failures += check(turn.content.size() == 1, "text parsed with allowed_types");

    ChatTurn media_turn;
    const Json media_parts = Json::array({
        Json{{"type", "text"}, {"text", "result"}},
        Json{{"type", "image_url"}, {"image_url", Json{{"url", "data:image/png;base64,AA=="}}}}
    });
    parse_content_parts(media_parts, media_turn, 0, {"text", "image_url"});
    failures += check(media_turn.content.size() == 2, "text+image parsed with allowed_types");

    bool rejected = false;
    std::string error_message;
    try {
        ChatTurn rejected_turn;
        parse_content_parts(media_parts, rejected_turn, 0, {"text"});
    } catch (const ApiException& e) {
        rejected      = true;
        error_message = e.error().message;
    }
    failures += check(rejected, "disallowed media part was rejected");
    failures += check(error_message.find("content parts must have type 'text'") != std::string::npos,
                      "error message lists allowed types");

    return failures;
}

int test_parse_tool_message_content_parts() {
    int failures = 0;
    const Json body = {
        {"model", "m"},
        {"messages", Json::array({
            Json{{"role", "user"}, {"content", "run screenshot"}},
            Json{{"role", "assistant"}, {"content", nullptr}, {"tool_calls", Json::array({
                Json{{"id", "call_1"}, {"type", "function"}, {"function", Json{{"name", "screenshot"}, {"arguments", "{}"}}}}
            })}},
            Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", Json::array({
                Json{{"type", "text"}, {"text", "captured:"}},
                Json{{"type", "image_url"}, {"image_url", Json{{"url", "data:image/png;base64,AA=="}}}}
            })}}
        })}
    };
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.messages.size() == 3, "parsed 3 messages");
    failures += check(req.messages[2].role == ninfer::ChatRole::Tool, "third message is tool role");
    failures += check(req.messages[2].content.size() == 2, "tool message has 2 content parts");
    failures += check(req.messages[2].content[0].kind == ContentKind::Text, "tool content part 0 is text");
    failures += check(req.messages[2].content[1].kind == ContentKind::Image, "tool content part 1 is image");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_string_content();
    failures += test_preserve_thinking_options();
    failures += test_enable_thinking_dialect();
    failures += test_reasoning_effort();
    failures += test_parse_parts_and_flatten();
    failures += test_instruction_roles_preserved();
    failures += test_parse_media_in_translate();
    failures += test_reject_unsupported();
    failures += test_parse_function_tools_and_choices();
    failures += test_parse_tool_history_messages();
    failures += test_parse_stop_and_max_tokens();
    failures += test_parse_sampling_carried();
    failures += test_response_serialization();
    failures += test_tool_response_serialization();
    failures += test_chunk_serialization();
    failures += test_usage_prefix_observability();
    failures += test_tool_chunk_serialization();
    failures += test_models_and_error();
    failures += test_finish_reason_wire();
    failures += test_parse_content_parts_allowed_types();
    failures += test_parse_tool_message_content_parts();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
