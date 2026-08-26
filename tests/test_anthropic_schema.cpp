// Contract test for the Anthropic Messages serving layer: request parsing
// (system field, string + block content, tool_use/tool_result round-trip,
// tool_choice/thinking/sampling mapping, unsupported-content rejection),
// non-streaming response + streaming SSE event shapes, stop_reason mapping,
// count_tokens body, and Anthropic error body. This is the schema boundary
// consumed by Claude Code / other Anthropic clients.

#include "serve/anthropic_schema.h"
#include "serve/request.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
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

// Parse an Anthropic SSE event ("event: <type>\ndata: <json>\n\n") into its JSON
// payload, optionally returning the event-type line.
Json parse_sse(const std::string& event, std::string* out_type = nullptr) {
    const std::string ev_prefix   = "event: ";
    const std::string data_marker = "\ndata: ";
    const std::string suffix      = "\n\n";
    if (event.rfind(ev_prefix, 0) != 0) { throw std::runtime_error("bad SSE framing: " + event); }
    const std::size_t data_pos = event.find(data_marker);
    if (data_pos == std::string::npos) { throw std::runtime_error("no data line: " + event); }
    if (out_type != nullptr) {
        *out_type = event.substr(ev_prefix.size(), data_pos - ev_prefix.size());
    }
    const std::size_t json_begin = data_pos + data_marker.size();
    if (event.size() < json_begin + suffix.size()) {
        throw std::runtime_error("short SSE: " + event);
    }
    return Json::parse(event.substr(json_begin, event.size() - json_begin - suffix.size()));
}

int test_parse_basic_and_system() {
    int failures                = 0;
    const Json body             = {{"model", "claude-sonnet-4-5"},
                                   {"max_tokens", 256},
                                   {"system", "be terse"},
                                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
    const GenerationRequest req = parse_messages_request(body, default_limits());
    failures += check(req.model == "claude-sonnet-4-5", "model echoed verbatim");
    failures += check(req.max_tokens == 256 && req.max_tokens_set, "max_tokens parsed");
    failures += check(req.messages.size() == 2, "system + user turns");
    failures += check(req.messages[0].role == ninfer::ChatRole::System, "system turn is first");
    failures += check(req.messages[0].content[0].text == "be terse", "system text carried");
    failures += check(req.messages[1].role == ninfer::ChatRole::User, "user turn follows system");
    failures += check(req.messages[1].content[0].text == "hello", "user text carried");
    failures += check(!req.stream, "stream defaults false");
    return failures;
}

int test_parse_system_array_and_blocks() {
    int failures    = 0;
    const Json body = {
        {"model", "m"},
        {"max_tokens", 16},
        {"system", Json::array({Json{{"type", "text"}, {"text", "a"}},
                                Json{{"type", "text"}, {"text", "b"}}})},
        {"messages",
         Json::array({Json{{"role", "user"},
                           {"content", Json::array({Json{{"type", "text"}, {"text", "x"}},
                                                    Json{{"type", "text"}, {"text", "y"}}})}}})}};
    const GenerationRequest req = parse_messages_request(body, default_limits());
    failures += check(req.messages[0].role == ninfer::ChatRole::System, "system first");
    failures += check(req.messages[0].content[0].text == "a\nb", "system blocks joined");
    const ninfer::PromptInput prompt = translate(req);
    failures += check(prompt.messages.size() == 2, "flattened system + user");
    failures += check(joined_text(prompt.messages[1]) == "x\ny", "user blocks joined");
    return failures;
}

int test_ordered_system_messages() {
    int failures    = 0;
    const Json body = {
        {"model", "m"},
        {"max_tokens", 16},
        {"system", "top-level system"},
        {"messages", Json::array({
                         Json{{"role", "user"}, {"content", "hello"}},
                         Json{{"role", "system"}, {"content", "reminder from messages"}},
                     })}};
    const GenerationRequest req = parse_messages_request(body, default_limits());
    failures += check(req.messages.size() == 3, "top-level, user, and dynamic system kept");
    failures += check(req.messages[0].role == ninfer::ChatRole::System &&
                          req.messages[0].content[0].text == "top-level system",
                      "top-level system was not preserved independently");
    failures += check(req.messages[1].role == ninfer::ChatRole::User &&
                          req.messages[1].content[0].text == "hello",
                      "user turn moved around dynamic system");
    failures += check(req.messages[2].role == ninfer::ChatRole::System &&
                          req.messages[2].content[0].text == "reminder from messages",
                      "dynamic system was not preserved at its message-array position");
    const ninfer::PromptInput prompt = translate(req);
    failures +=
        check(prompt.messages.size() == 3 && prompt.messages[0].role == ninfer::ChatRole::System &&
                  prompt.messages[1].role == ninfer::ChatRole::User &&
                  prompt.messages[2].role == ninfer::ChatRole::System,
              "translation reordered or lowered Anthropic system turns");

    const Json blocks_body = {
        {"model", "m"},
        {"max_tokens", 16},
        {"messages", Json::array({
                         Json{{"role", "user"}, {"content", "hi"}},
                         Json{{"role", "system"},
                              {"content", Json::array({Json{{"type", "text"}, {"text", "r"}}})}},
                     })}};
    const GenerationRequest breq = parse_messages_request(blocks_body, default_limits());
    failures +=
        check(breq.messages.size() == 2 && breq.messages[0].role == ninfer::ChatRole::User &&
                  breq.messages[1].role == ninfer::ChatRole::System &&
                  breq.messages[1].content[0].text == "r",
              "array-valued dynamic system did not retain its position");

    const Json consecutive = {
        {"model", "m"},
        {"max_tokens", 16},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}},
                                  Json{{"role", "system"}, {"content", "first"}},
                                  Json{{"role", "system"}, {"content", "second"}},
                                  Json{{"role", "assistant"}, {"content", "answer"}}})}};
    const GenerationRequest creq = parse_messages_request(consecutive, default_limits());
    failures +=
        check(creq.messages.size() == 4 && creq.messages[1].role == ninfer::ChatRole::System &&
                  creq.messages[1].content[0].text == "first" &&
                  creq.messages[2].role == ninfer::ChatRole::System &&
                  creq.messages[2].content[0].text == "second",
              "consecutive dynamic system turns were merged or reordered");

    const Json leading = {{"model", "m"},
                          {"max_tokens", 16},
                          {"messages", Json::array({Json{{"role", "system"}, {"content", "bad"}},
                                                    Json{{"role", "user"}, {"content", "hi"}}})}};
    failures += check(api_code([&] { (void)parse_messages_request(leading, default_limits()); }) ==
                          "invalid_message_order",
                      "leading array system was not rejected by the Anthropic schema");

    const Json followed_by_user = {
        {"model", "m"},
        {"max_tokens", 16},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "first"}},
                                  Json{{"role", "system"}, {"content", "bad"}},
                                  Json{{"role", "user"}, {"content", "second"}}})}};
    failures += check(api_code([&] {
                          (void)parse_messages_request(followed_by_user, default_limits());
                      }) == "invalid_message_order",
                      "system section followed by user was not rejected by the Anthropic schema");

    const Json tool_use          = Json{{"role", "assistant"},
                                        {"content", Json::array({Json{{"type", "tool_use"},
                                                                      {"id", "toolu_1"},
                                                                      {"name", "inspect"},
                                                                      {"input", Json::object()}}})}};
    const Json tool_result       = Json{{"role", "user"},
                                        {"content", Json::array({Json{{"type", "tool_result"},
                                                                      {"tool_use_id", "toolu_1"},
                                                                      {"content", "done"}}})}};
    const Json after_tool_result = {
        {"model", "m"},
        {"max_tokens", 16},
        {"messages",
         Json::array({Json{{"role", "user"}, {"content", "inspect"}}, tool_use, tool_result,
                      Json{{"role", "system"}, {"content", "diagnostics"}}})}};
    const GenerationRequest tool_req = parse_messages_request(after_tool_result, default_limits());
    failures += check(tool_req.messages.size() == 4 &&
                          tool_req.messages[0].role == ninfer::ChatRole::User &&
                          tool_req.messages[1].role == ninfer::ChatRole::Assistant &&
                          tool_req.messages[2].role == ninfer::ChatRole::Tool &&
                          tool_req.messages[3].role == ninfer::ChatRole::System,
                      "system after tool_result did not retain the expected semantic position");

    const Json interrupted_tool_pair = {
        {"model", "m"},
        {"max_tokens", 16},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "inspect"}}, tool_use,
                                  Json{{"role", "system"}, {"content", "bad"}}, tool_result})}};
    failures += check(api_code([&] {
                          (void)parse_messages_request(interrupted_tool_pair, default_limits());
                      }) == "invalid_message_order",
                      "system interrupting tool_use/tool_result was not rejected");
    return failures;
}

int test_user_content_block_order() {
    const Json body = {
        {"model", "m"},
        {"max_tokens", 16},
        {"messages", Json::array({Json{
                         {"role", "user"},
                         {"content", Json::array({Json{{"type", "text"}, {"text", "before"}},
                                                  Json{{"type", "tool_result"},
                                                       {"tool_use_id", "toolu_1"},
                                                       {"content", "result"}},
                                                  Json{{"type", "text"}, {"text", "after"}}})}}})}};
    const GenerationRequest req = parse_messages_request(body, default_limits());
    int failures = check(req.messages.size() == 3, "mixed user content did not expand in place");
    failures += check(req.messages[0].role == ninfer::ChatRole::User &&
                          req.messages[0].content[0].text == "before" &&
                          req.messages[1].role == ninfer::ChatRole::Tool &&
                          req.messages[1].content[0].text == "result" &&
                          req.messages[2].role == ninfer::ChatRole::User &&
                          req.messages[2].content[0].text == "after",
                      "tool_result expansion reordered surrounding user blocks");
    return failures;
}

int test_missing_and_bad_fields() {
    int failures        = 0;
    const Json no_model = {{"max_tokens", 8},
                           {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    failures += check(throws_api([&] { (void)parse_messages_request(no_model, default_limits()); }),
                      "missing model rejected");

    const Json bad_role = {
        {"model", "m"},
        {"max_tokens", 8},
        {"messages", Json::array({Json{{"role", "developer"}, {"content", "hi"}}})}};
    failures += check(throws_api([&] { (void)parse_messages_request(bad_role, default_limits()); }),
                      "unknown message role rejected");

    const Json empty_msgs = {{"model", "m"}, {"max_tokens", 8}, {"messages", Json::array()}};
    failures +=
        check(throws_api([&] { (void)parse_messages_request(empty_msgs, default_limits()); }),
              "empty messages rejected");

    const Json bad_max = {{"model", "m"},
                          {"max_tokens", 0},
                          {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    failures += check(throws_api([&] { (void)parse_messages_request(bad_max, default_limits()); }),
                      "non-positive max_tokens rejected");

    // Omitting max_tokens falls back to the server default (lenient vs the API).
    const Json no_max           = {{"model", "m"},
                                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    const GenerationRequest req = parse_messages_request(no_max, default_limits());
    failures += check(req.max_tokens == 512 && !req.max_tokens_set, "max_tokens default applied");
    return failures;
}

int test_parse_image() {
    const Json body = {
        {"model", "m"},
        {"max_tokens", 8},
        {"messages", Json::array({Json{
                         {"role", "user"},
                         {"content", Json::array({Json{{"type", "image"},
                                                       {"source", Json{{"type", "base64"},
                                                                       {"media_type", "image/png"},
                                                                       {"data", "AA=="}}}}})}}})}};
    const GenerationRequest req      = parse_messages_request(body, default_limits());
    const ninfer::PromptInput prompt = translate(req);
    int failures                     = 0;
    failures += check(req.messages[0].content[0].kind == ContentKind::Image,
                      "Anthropic image kind preserved");
    failures += check(req.messages[0].content[0].source.value == "data:image/png;base64,AA==",
                      "Anthropic base64 converted to data URI");
    failures += check(prompt.messages[0].parts[0].kind == ninfer::MessagePartKind::Media &&
                          prompt.messages[0].parts[0].media.kind == ninfer::MediaKind::Image,
                      "Anthropic image translated to structured chat part");
    return failures;
}

int test_tools_and_choice() {
    int failures = 0;
    const Json tool =
        Json{{"name", "get_weather"},
             {"description", "Look up weather"},
             {"input_schema", Json{{"type", "object"},
                                   {"properties", Json{{"city", Json{{"type", "string"}}}}},
                                   {"required", Json::array({"city"})}}}};
    Json body = {
        {"model", "m"},
        {"max_tokens", 8},
        {"tools", Json::array({tool})},
        {"tool_choice", Json{{"type", "auto"}}},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "weather in Paris?"}}})}};

    const GenerationRequest req = parse_messages_request(body, default_limits());
    failures += check(req.tools.size() == 1, "one tool parsed");
    failures += check(req.tools[0].name == "get_weather", "tool name parsed");
    failures += check(req.tool_choice.mode == ToolChoiceMode::Auto, "auto -> Auto");
    failures += check(req.uses_tools(), "uses_tools true");
    failures += check(to_request_options(req, default_server()).output.preserve_special_tokens,
                      "active tools preserve special tokens in Engine output");
    // definition_json must be a normalized OpenAI function-tool object for the
    // Qwen <tools> renderer.
    const Json def = Json::parse(req.tools[0].definition_json);
    failures += check(def.at("type") == "function", "definition type function");
    failures += check(def.at("function").at("name") == "get_weather", "definition name");
    failures += check(def.at("function").at("parameters").at("required").at(0) == "city",
                      "input_schema mapped to parameters");
    failures += check(def.at("function").at("strict") == false, "strict defaulted false");

    Json any           = body;
    any["tool_choice"] = Json{{"type", "any"}};
    failures += check(parse_messages_request(any, default_limits()).tool_choice.mode ==
                          ToolChoiceMode::Required,
                      "any -> Required");

    Json none           = body;
    none["tool_choice"] = Json{{"type", "none"}};
    failures += check(parse_messages_request(none, default_limits()).tool_choice.mode ==
                          ToolChoiceMode::None,
                      "none -> None");

    Json named                        = body;
    named["tool_choice"]              = Json{{"type", "tool"}, {"name", "get_weather"}};
    const GenerationRequest named_req = parse_messages_request(named, default_limits());
    failures += check(named_req.tool_choice.mode == ToolChoiceMode::Named &&
                          named_req.tool_choice.name == "get_weather",
                      "tool+name -> Named");

    Json unknown           = body;
    unknown["tool_choice"] = Json{{"type", "tool"}, {"name", "nope"}};
    failures += check(throws_api([&] { (void)parse_messages_request(unknown, default_limits()); }),
                      "unknown named tool rejected");

    Json server_tool     = body;
    server_tool["tools"] = Json::array({Json{{"type", "bash_20250124"}, {"name", "bash"}}});
    failures +=
        check(throws_api([&] { (void)parse_messages_request(server_tool, default_limits()); }),
              "server tool without input_schema rejected");

    const std::string max_name(128, 'a');
    Json max_tool          = tool;
    max_tool["name"]       = max_name;
    Json max_name_body     = body;
    max_name_body["tools"] = Json::array({max_tool});
    failures +=
        check(parse_messages_request(max_name_body, default_limits()).tools[0].name == max_name,
              "128-character Anthropic tool name accepted");

    Json too_long_tool     = tool;
    too_long_tool["name"]  = std::string(129, 'a');
    Json too_long_body     = body;
    too_long_body["tools"] = Json::array({too_long_tool});
    failures +=
        check(throws_api([&] { (void)parse_messages_request(too_long_body, default_limits()); }),
              "129-character Anthropic tool name rejected");
    return failures;
}

int test_duplicate_tool_name_rejected() {
    int failures = 0;
    const Json tool =
        Json{{"name", "get_weather"},
             {"input_schema", Json{{"type", "object"},
                                   {"properties", Json{{"city", Json{{"type", "string"}}}}},
                                   {"required", Json::array({"city"})}}}};
    Json body = {
        {"model", "m"},
        {"max_tokens", 8},
        {"tools", Json::array({tool, tool})},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "weather in Paris?"}}})}};
    failures += check(throws_api([&] { (void)parse_messages_request(body, default_limits()); }),
                      "duplicate tool names rejected");
    return failures;
}

int test_tool_use_result_roundtrip() {
    int failures    = 0;
    const Json tool = Json{{"name", "get_weather"}, {"input_schema", Json{{"type", "object"}}}};
    const Json assistant =
        Json{{"role", "assistant"},
             {"content", Json::array({Json{{"type", "text"}, {"text", "let me check"}},
                                      Json{{"type", "tool_use"},
                                           {"id", "toolu_1"},
                                           {"name", "get_weather"},
                                           {"input", Json{{"city", "Paris"}}}}})}};
    const Json user_result =
        Json{{"role", "user"},
             {"content",
              Json::array(
                  {Json{{"type", "tool_result"},
                        {"tool_use_id", "toolu_1"},
                        {"content", Json::array({Json{{"type", "text"}, {"text", "sunny"}},
                                                 Json{{"type", "image"},
                                                      {"source", Json{{"type", "base64"},
                                                                      {"media_type", "image/png"},
                                                                      {"data", "AA=="}}}}})}}})}};
    const Json body = {{"model", "m"},
                       {"max_tokens", 8},
                       {"tools", Json::array({tool})},
                       {"messages", Json::array({Json{{"role", "user"}, {"content", "weather?"}},
                                                 assistant, user_result})}};

    const GenerationRequest req = parse_messages_request(body, default_limits());
    // user, assistant, tool
    failures += check(req.messages.size() == 3, "user + assistant + tool turns");
    failures += check(req.messages[1].role == ninfer::ChatRole::Assistant, "assistant turn");
    failures += check(req.messages[1].tool_calls.size() == 1, "assistant tool_call parsed");
    failures += check(req.messages[1].tool_calls[0].id == "toolu_1", "tool_use id carried");
    failures += check(req.messages[1].tool_calls[0].name == "get_weather", "tool_use name carried");
    const Json args = Json::parse(req.messages[1].tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "tool_use input stringified to arguments");
    failures += check(req.messages[1].content[0].text == "let me check", "assistant text carried");
    failures += check(req.messages[2].role == ninfer::ChatRole::Tool, "tool_result -> tool turn");
    failures += check(req.messages[2].tool_call_id == "toolu_1", "tool_result tool_use_id carried");
    failures += check(req.messages[2].content[0].text == "sunny", "tool_result text carried");
    failures += check(req.messages[2].content.size() == 2 &&
                          req.messages[2].content[1].kind == ContentKind::Image,
                      "tool_result image carried");
    const ninfer::PromptInput prompt = translate(req);
    failures += check(prompt.messages[2].parts.size() == 2 &&
                          prompt.messages[2].parts[1].kind == ninfer::MessagePartKind::Media &&
                          prompt.messages[2].parts[1].media.kind == ninfer::MediaKind::Image,
                      "tool_result image translated to structured chat");
    failures += check(req.has_tool_history(), "tool history detected");
    return failures;
}

int test_thinking_and_sampling() {
    int failures                = 0;
    Json body                   = {{"model", "m"},
                                   {"max_tokens", 8},
                                   {"temperature", 0.3},
                                   {"top_p", 0.8},
                                   {"top_k", 40},
                                   {"stop_sequences", Json::array({"STOP", "END"})},
                                   {"thinking", Json{{"type", "enabled"}, {"budget_tokens", 1024}}},
                                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    const GenerationRequest req = parse_messages_request(body, default_limits());
    failures += check(req.sampling.temperature.has_value() && *req.sampling.temperature == 0.3,
                      "temperature parsed");
    failures += check(req.sampling.top_p.has_value() && *req.sampling.top_p == 0.8, "top_p parsed");
    failures += check(req.sampling.top_k.has_value() && *req.sampling.top_k == 40, "top_k parsed");
    failures += check(req.stop_strings.size() == 2 && req.stop_strings[0] == "STOP",
                      "stop_sequences parsed");
    failures += check(req.enable_thinking.has_value() && *req.enable_thinking, "thinking enabled");
    failures += check(!req.preserve_thinking.has_value(),
                      "Anthropic thinking.type unexpectedly enabled history preservation");
    const ninfer::RequestOptions options = to_request_options(req, default_server());
    failures +=
        check(options.execution.requested_output_tokens == 8, "max_tokens reaches Engine options");
    failures += check(options.execution.sampling.temperature == 0.3F &&
                          options.execution.sampling.top_p == 0.8F &&
                          options.execution.sampling.top_k == 40 &&
                          !options.execution.sampling.presence_penalty,
                      "sampling reaches Engine overrides without inventing omitted fields");
    failures += check(options.stop.strings.size() == 2 && options.stop.strings[0].text == "STOP" &&
                          options.stop.strings[1].text == "END",
                      "stop_sequences reach Engine options");

    Json disabled                = body;
    disabled["thinking"]         = Json{{"type", "disabled"}};
    const GenerationRequest dreq = parse_messages_request(disabled, default_limits());
    failures +=
        check(dreq.enable_thinking.has_value() && !*dreq.enable_thinking, "thinking disabled");

    // Claude Code sends thinking.type == "adaptive" (extended thinking); any
    // non-"disabled" mode must map to thinking-on, not a 400.
    Json adaptive                = body;
    adaptive["thinking"]         = Json{{"type", "adaptive"}};
    const GenerationRequest areq = parse_messages_request(adaptive, default_limits());
    failures +=
        check(areq.enable_thinking.has_value() && *areq.enable_thinking, "adaptive -> thinking on");

    Json preserve                 = body;
    preserve["thinking"]          = Json{{"type", "disabled"}};
    preserve["preserve_thinking"] = true;
    const GenerationRequest preq  = parse_messages_request(preserve, default_limits());
    failures += check(preq.enable_thinking == false && preq.preserve_thinking == true,
                      "Anthropic preserve_thinking was not independent from thinking.type");
    Json invalid                 = body;
    invalid["preserve_thinking"] = "yes";
    failures += check(throws_api([&] { (void)parse_messages_request(invalid, default_limits()); }),
                      "Anthropic accepted non-boolean preserve_thinking");
    return failures;
}

int test_reasoning_effort() {
    const Json base = {
        {"model", "m"},
        {"max_tokens", 8},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
    };
    int failures = 0;

    for (const auto& [wire, expected] :
         std::array<std::pair<const char*, RequestedReasoningEffort>, 5>{
             {{"low", RequestedReasoningEffort::Low},
              {"medium", RequestedReasoningEffort::Medium},
              {"high", RequestedReasoningEffort::High},
              {"xhigh", RequestedReasoningEffort::XHigh},
              {"max", RequestedReasoningEffort::Max}}}) {
        Json body                       = base;
        body["output_config"]           = Json{{"effort", wire}};
        const GenerationRequest request = parse_messages_request(body, default_limits());
        failures += check(request.reasoning_effort == expected,
                          std::string("Anthropic did not accept protocol effort ") + wire);
    }

    Json low                             = base;
    low["output_config"]                 = Json{{"effort", "low"}};
    const GenerationRequest low_request  = parse_messages_request(low, default_limits());
    const ninfer::PromptInput low_prompt = translate(low_request);
    failures += check(low_prompt.options.enable_thinking &&
                          low_prompt.options.reasoning_effort == ninfer::ReasoningEffort::Low,
                      "Anthropic low effort did not reach PromptInput");

    Json high                            = base;
    high["output_config"]                = Json{{"effort", "high"}};
    const GenerationRequest high_request = parse_messages_request(high, default_limits());
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(high_request, default_server(),
                                                         effort_capabilities());
                      }) == "reasoning_effort_not_supported",
                      "Anthropic high effort bypassed template capability validation");

    Json conflict        = low;
    conflict["thinking"] = Json{{"type", "disabled"}};
    const GenerationRequest conflicting_request =
        parse_messages_request(conflict, default_limits());
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(conflicting_request, default_server(),
                                                         effort_capabilities());
                      }) == "conflicting_template_option",
                      "Anthropic disabled thinking and effort were accepted together");

    Json invalid             = base;
    invalid["output_config"] = Json{{"effort", "minimal"}};
    failures += check(throws_api([&] { (void)parse_messages_request(invalid, default_limits()); }),
                      "Anthropic accepted an effort outside its protocol vocabulary");
    invalid["output_config"] = Json{{"effort", 1}};
    failures += check(throws_api([&] { (void)parse_messages_request(invalid, default_limits()); }),
                      "Anthropic accepted non-string output_config.effort");
    invalid["output_config"] = Json{{"format", Json{{"type", "json_schema"}}}};
    failures += check(throws_api([&] { (void)parse_messages_request(invalid, default_limits()); }),
                      "unsupported Anthropic output_config option was ignored");
    return failures;
}

int test_stop_reason_mapping() {
    int failures = 0;
    failures += check(std::string(messages_stop_reason(ninfer::FinishReason::OutputLimit, false)) ==
                          "max_tokens",
                      "output limit -> max_tokens");
    failures += check(std::string(messages_stop_reason(ninfer::FinishReason::StopToken, false)) ==
                          "end_turn",
                      "stop token -> end_turn");
    failures += check(std::string(messages_stop_reason(ninfer::FinishReason::Cancelled, false)) ==
                          "end_turn",
                      "cancelled -> end_turn");
    failures += check(std::string(messages_stop_reason(ninfer::FinishReason::StopString, true)) ==
                          "tool_use",
                      "tool calls -> tool_use");
    return failures;
}

int test_response_serialization() {
    int failures = 0;
    const CompletionUsage usage{7, 3};
    const std::vector<ToolCall> tools = {ToolCall{"toolu_9", "get_weather", R"({"city":"Paris"})"}};

    const Json resp = Json::parse(make_messages_response(
        "msg_1", "claude-x", "the answer", "the reasoning", tools, "tool_use", usage));
    failures += check(resp.at("id") == "msg_1", "id echoed");
    failures += check(resp.at("type") == "message", "type message");
    failures += check(resp.at("role") == "assistant", "role assistant");
    failures += check(resp.at("model") == "claude-x", "model echoed");
    failures += check(resp.at("stop_reason") == "tool_use", "stop_reason");
    failures += check(resp.at("stop_sequence").is_null(), "stop_sequence null");
    failures += check(resp.at("usage").at("input_tokens") == 7, "input_tokens");
    failures += check(resp.at("usage").at("output_tokens") == 3, "output_tokens");
    const Json& content = resp.at("content");
    failures += check(content.size() == 3, "thinking + text + tool_use blocks");
    failures += check(content.at(0).at("type") == "thinking" &&
                          content.at(0).at("thinking") == "the reasoning",
                      "thinking block");
    failures +=
        check(content.at(1).at("type") == "text" && content.at(1).at("text") == "the answer",
              "text block");
    failures += check(content.at(2).at("type") == "tool_use" && content.at(2).at("id") == "toolu_9",
                      "tool_use block");
    failures += check(content.at(2).at("input").at("city") == "Paris", "tool_use input is object");

    // Empty output falls back to a single empty text block.
    const Json empty = Json::parse(
        make_messages_response("msg_2", "m", "", "", {}, "end_turn", CompletionUsage{1, 0}));
    failures +=
        check(empty.at("content").size() == 1 && empty.at("content").at(0).at("type") == "text" &&
                  empty.at("content").at(0).at("text") == "",
              "empty output -> empty text block");
    return failures;
}

// Regression: a string-typed tool argument (e.g. taskId="1") must survive the
// Anthropic render path as a JSON string, not be coerced to a number. The
// parser preserves the raw text into arguments_json; make_messages_response
// forwards it verbatim into the tool_use.input object.
int test_string_typed_argument_survives_render() {
    int failures = 0;
    const CompletionUsage usage{3, 1};
    // arguments_json carries taskId as a string, exactly as the schema-aware
    // parser emits it for a string-typed parameter with a numeric-looking value.
    const std::vector<ToolCall> calls = {
        ToolCall{"toolu_1", "TaskUpdate", R"({"taskId":"1"})"}};
    const Json resp = Json::parse(
        make_messages_response("msg_s", "claude-x", "", "", calls, "tool_use", usage));
    const Json& content = resp.at("content");
    failures += check(content.size() == 1 && content.at(0).at("type") == "tool_use",
                      "render produced a single tool_use block");
    const Json& input = content.at(0).at("input");
    failures += check(input.at("taskId").is_string(),
                      "string-typed taskId rendered as a JSON string, not a number");
    failures += check(input.at("taskId") == "1", "string-typed taskId value preserved on the wire");
    failures += check(!input.at("taskId").is_number(),
                      "string-typed taskId is not a JSON number on the wire");
    return failures;
}

int test_streaming_events() {
    int failures = 0;
    std::string type;

    const Json start = parse_sse(make_message_start("msg_1", "claude-x", 11), &type);
    failures += check(type == "message_start", "message_start event type");
    failures += check(start.at("type") == "message_start", "message_start payload type");
    failures += check(start.at("message").at("id") == "msg_1", "message_start id");
    failures += check(start.at("message").at("model") == "claude-x", "message_start model");
    failures += check(start.at("message").at("usage").at("input_tokens") == 11,
                      "message_start input_tokens");
    failures += check(start.at("message").at("content").is_array() &&
                          start.at("message").at("content").empty(),
                      "message_start empty content");

    const Json tstart = parse_sse(make_content_block_start_text(1), &type);
    failures += check(type == "content_block_start" && tstart.at("index") == 1 &&
                          tstart.at("content_block").at("type") == "text",
                      "text block start");

    const Json think_start = parse_sse(make_content_block_start_thinking(0), &type);
    failures +=
        check(think_start.at("content_block").at("type") == "thinking", "thinking block start");

    const Json tdelta = parse_sse(make_content_block_delta_text(1, "hi"), &type);
    failures +=
        check(type == "content_block_delta" && tdelta.at("delta").at("type") == "text_delta" &&
                  tdelta.at("delta").at("text") == "hi",
              "text_delta");

    const Json thdelta = parse_sse(make_content_block_delta_thinking(0, "hmm"), &type);
    failures += check(thdelta.at("delta").at("type") == "thinking_delta" &&
                          thdelta.at("delta").at("thinking") == "hmm",
                      "thinking_delta");

    const ToolCall call{"toolu_2", "get_weather", R"({"city":"Paris"})"};
    const Json tustart = parse_sse(make_content_block_start_tool_use(2, call), &type);
    failures += check(tustart.at("content_block").at("type") == "tool_use" &&
                          tustart.at("content_block").at("id") == "toolu_2" &&
                          tustart.at("content_block").at("name") == "get_weather" &&
                          tustart.at("content_block").at("input").is_object() &&
                          tustart.at("content_block").at("input").empty(),
                      "tool_use block start with empty input");

    const Json ijdelta =
        parse_sse(make_content_block_delta_tool_json(2, R"({"city":"Paris"})"), &type);
    failures += check(ijdelta.at("delta").at("type") == "input_json_delta" &&
                          ijdelta.at("delta").at("partial_json") == R"({"city":"Paris"})",
                      "input_json_delta");

    const Json stop = parse_sse(make_content_block_stop(2), &type);
    failures += check(type == "content_block_stop" && stop.at("index") == 2, "content_block_stop");

    const Json mdelta = parse_sse(make_message_delta("tool_use", 5), &type);
    failures +=
        check(type == "message_delta" && mdelta.at("delta").at("stop_reason") == "tool_use" &&
                  mdelta.at("delta").at("stop_sequence").is_null() &&
                  mdelta.at("usage").at("output_tokens") == 5,
              "message_delta stop_reason + usage");

    const Json mstop = parse_sse(make_message_stop(), &type);
    failures += check(type == "message_stop" && mstop.at("type") == "message_stop", "message_stop");

    const Json ping = parse_sse(make_messages_ping(), &type);
    failures += check(type == "ping" && ping.at("type") == "ping", "ping");
    return failures;
}

int test_count_tokens_and_error() {
    int failures  = 0;
    const Json ct = Json::parse(make_count_tokens_response(123));
    failures += check(ct.at("input_tokens") == 123, "count_tokens body");

    ApiError err;
    err.status      = 401;
    err.message     = "missing or invalid API key";
    const Json body = Json::parse(make_messages_error_body(err));
    failures += check(body.at("type") == "error", "error envelope type");
    failures +=
        check(body.at("error").at("type") == "authentication_error", "401 -> authentication_error");
    failures +=
        check(body.at("error").at("message") == "missing or invalid API key", "error message");

    ApiError server;
    server.status          = 500;
    server.message         = "boom";
    const Json server_body = Json::parse(make_messages_error_body(server));
    failures += check(server_body.at("error").at("type") == "api_error", "500 -> api_error");

    std::string type;
    const Json sse_err = parse_sse(messages_sse_error_event(err), &type);
    failures += check(type == "error" && sse_err.at("type") == "error", "sse error event");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_basic_and_system();
    failures += test_parse_system_array_and_blocks();
    failures += test_ordered_system_messages();
    failures += test_user_content_block_order();
    failures += test_missing_and_bad_fields();
    failures += test_parse_image();
    failures += test_tools_and_choice();
    failures += test_duplicate_tool_name_rejected();
    failures += test_tool_use_result_roundtrip();
    failures += test_thinking_and_sampling();
    failures += test_reasoning_effort();
    failures += test_stop_reason_mapping();
    failures += test_response_serialization();
    failures += test_string_typed_argument_survives_render();
    failures += test_streaming_events();
    failures += test_count_tokens_and_error();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
