// Contract test for the Anthropic Messages serving layer: request parsing
// (system field, string + block content, tool_use/tool_result round-trip,
// tool_choice/thinking/sampling mapping, unsupported-content rejection),
// non-streaming response + streaming SSE event shapes, stop_reason mapping,
// count_tokens body, and Anthropic error body. This is the schema boundary
// consumed by Claude Code / other Anthropic clients.

#include "ninfer/serve/anthropic_schema.h"
#include "ninfer/serve/request.h"
#include "ninfer/serve/translate.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>

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

RequestLimits default_limits() {
    RequestLimits limits;
    limits.default_max_tokens = 512;
    limits.max_context        = 8192;
    return limits;
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
    failures += check(req.messages[0].role == "system", "system turn is first");
    failures += check(req.messages[0].content[0].text == "be terse", "system text carried");
    failures += check(req.messages[1].role == "user", "user turn follows system");
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
    failures += check(req.messages[0].role == "system", "system first");
    failures += check(req.messages[0].content[0].text == "a\nb", "system blocks joined");
    const auto messages = to_chat_messages(req);
    failures += check(messages.size() == 2, "flattened system + user");
    failures += check(messages[1].rendered_content() == "x\ny", "user blocks joined");
    return failures;
}

// Regression: Claude Code (v2.1.x) injects "system reminders" as system-role
// messages inside the messages array (in addition to the top-level `system`
// field). Earlier we rejected those with 400 "message role must be 'user' or
// 'assistant'". They must instead fold into the single leading system turn so the
// Qwen template (which drops non-leading system turns) keeps the content.
int test_system_role_in_messages_folds() {
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
    // Exactly one leading system turn (top-level + in-array folded), then the user.
    failures += check(req.messages.size() == 2, "system folded, user kept");
    failures += check(req.messages[0].role == "system", "single leading system turn");
    failures += check(req.messages[0].content[0].text == "top-level system\nreminder from messages",
                      "top-level + in-array system merged in order");
    failures += check(req.messages[1].role == "user" && req.messages[1].content[0].text == "hello",
                      "user turn preserved");

    // Also works with array-of-text-blocks content and no top-level system.
    const Json blocks_body = {
        {"model", "m"},
        {"max_tokens", 16},
        {"messages", Json::array({
                         Json{{"role", "user"}, {"content", "hi"}},
                         Json{{"role", "system"},
                              {"content", Json::array({Json{{"type", "text"}, {"text", "r"}}})}},
                     })}};
    const GenerationRequest breq = parse_messages_request(blocks_body, default_limits());
    failures += check(breq.messages.size() == 2 && breq.messages[0].role == "system" &&
                          breq.messages[0].content[0].text == "r",
                      "in-array system text block folded without top-level system");
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
    const GenerationRequest req = parse_messages_request(body, default_limits());
    const auto messages         = to_chat_messages(req);
    int failures                = 0;
    failures += check(req.messages[0].content[0].kind == ContentKind::Image,
                      "Anthropic image kind preserved");
    failures += check(req.messages[0].content[0].source.value == "data:image/png;base64,AA==",
                      "Anthropic base64 converted to data URI");
    failures += check(messages[0].parts[0].kind == ninfer::text::ChatPartKind::Image,
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
    failures += check(req.messages[1].role == "assistant", "assistant turn");
    failures += check(req.messages[1].tool_calls.size() == 1, "assistant tool_call parsed");
    failures += check(req.messages[1].tool_calls[0].id == "toolu_1", "tool_use id carried");
    failures += check(req.messages[1].tool_calls[0].name == "get_weather", "tool_use name carried");
    const Json args = Json::parse(req.messages[1].tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "tool_use input stringified to arguments");
    failures += check(req.messages[1].content[0].text == "let me check", "assistant text carried");
    failures += check(req.messages[2].role == "tool", "tool_result -> tool turn");
    failures += check(req.messages[2].tool_call_id == "toolu_1", "tool_result tool_use_id carried");
    failures += check(req.messages[2].content[0].text == "sunny", "tool_result text carried");
    failures += check(req.messages[2].content.size() == 2 &&
                          req.messages[2].content[1].kind == ContentKind::Image,
                      "tool_result image carried");
    const auto messages = to_chat_messages(req);
    failures += check(messages[2].parts.size() == 2 &&
                          messages[2].parts[1].kind == ninfer::text::ChatPartKind::Image,
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
    return failures;
}

int test_stop_reason_mapping() {
    int failures = 0;
    failures += check(std::string(messages_stop_reason(ninfer::text::FinishReason::Length, false)) ==
                          "max_tokens",
                      "length -> max_tokens");
    failures +=
        check(std::string(messages_stop_reason(ninfer::text::FinishReason::Stop, false)) == "end_turn",
              "stop -> end_turn");
    failures += check(
        std::string(messages_stop_reason(ninfer::text::FinishReason::Cancelled, false)) == "end_turn",
        "cancelled -> end_turn");
    failures +=
        check(std::string(messages_stop_reason(ninfer::text::FinishReason::Stop, true)) == "tool_use",
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
    failures += test_system_role_in_messages_folds();
    failures += test_missing_and_bad_fields();
    failures += test_parse_image();
    failures += test_tools_and_choice();
    failures += test_tool_use_result_roundtrip();
    failures += test_thinking_and_sampling();
    failures += test_stop_reason_mapping();
    failures += test_response_serialization();
    failures += test_streaming_events();
    failures += test_count_tokens_and_error();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
