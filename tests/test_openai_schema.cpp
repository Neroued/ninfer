// Contract test for the OpenAI serving layer: request parsing (string + parts
// content, unsupported-feature rejection), response/chunk/models/error
// serialization shapes, and finish_reason mapping. This is the schema boundary
// consumed by external OpenAI clients.

#include "serve/openai_schema.h"
#include "serve/request.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cstddef>
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
    return limits;
}

ServeOptions default_server() { return ServeOptions{}; }

ninfer::OwnedMedia fake_media(const ContentPart& part) {
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.bytes.push_back(0);
    media.media_type = part.source.media_type;
    return media;
}

ninfer::PromptInput translate(const GenerationRequest& req) {
    return to_prompt_input(req, default_server(), fake_media);
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
    failures += check(req.messages[0].role == "user", "role parsed");
    failures += check(req.messages[0].content.size() == 1, "one content part");
    failures += check(req.messages[0].content[0].kind == ContentKind::Text, "text part kind");
    failures += check(req.messages[0].content[0].text == "hello", "text part content");
    failures += check(!req.stream, "stream defaults false");
    failures += check(req.max_tokens == 512, "max_tokens default applied");
    failures += check(!req.max_tokens_set, "max_tokens_set false when defaulted");
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

int test_parse_image_in_translate() {
    const Json body = {
        {"model", "m"},
        {"messages",
         Json::array({Json{
             {"role", "user"},
             {"content",
              Json::array({Json{{"type", "image_url"},
                                {"image_url", Json{{"url", "data:image/png;base64,AA=="}}}}})}}})}};
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
    return failures;
}

int test_developer_role_mapped() {
    const Json body = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "developer"}, {"content", "be terse"}},
                                  Json{{"role", "user"}, {"content", "hi"}}})}};
    const GenerationRequest req      = parse_chat_completion_request(body, default_limits());
    const ninfer::PromptInput prompt = translate(req);
    int failures                     = 0;
    failures += check(prompt.messages.size() == 2, "developer + user parsed");
    failures += check(prompt.messages[0].role == "system", "developer role mapped to system");
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

    Json none           = base;
    none["tool_choice"] = "none";
    req                 = parse_chat_completion_request(none, default_limits());
    failures += check(req.tool_choice.mode == ToolChoiceMode::None, "tool_choice none parsed");
    failures += check(!req.uses_tools(), "tool_choice none disables tools");

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
    failures += check(req.messages[2].role == "tool", "tool role parsed");
    failures += check(req.messages[2].tool_call_id == "call_1", "tool_call_id parsed");
    failures +=
        check(req.messages[2].content.at(0).text == R"({"temp":20})", "tool content parsed");

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

    failures += check(sse_done() == "data: [DONE]\n\n", "done sentinel");
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
    int failures    = 0;
    const Json list = Json::parse(make_models_list("qwen3.6-27b", 1));
    failures += check(list.at("object") == "list", "models list object");
    failures += check(list.at("data").at(0).at("id") == "qwen3.6-27b", "models list id");
    failures += check(list.at("data").at(0).at("object") == "model", "models list entry object");
    failures += check(list.at("data").at(0).at("owned_by") == "ninfer", "models list owner");

    const Json one = Json::parse(make_model_object("qwen3.6-27b", 1));
    failures += check(one.at("id") == "qwen3.6-27b" && one.at("object") == "model", "model object");
    failures += check(one.at("owned_by") == "ninfer", "model owner");

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

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_string_content();
    failures += test_parse_parts_and_flatten();
    failures += test_developer_role_mapped();
    failures += test_parse_image_in_translate();
    failures += test_reject_unsupported();
    failures += test_parse_function_tools_and_choices();
    failures += test_parse_tool_history_messages();
    failures += test_parse_stop_and_max_tokens();
    failures += test_parse_sampling_carried();
    failures += test_response_serialization();
    failures += test_tool_response_serialization();
    failures += test_chunk_serialization();
    failures += test_tool_chunk_serialization();
    failures += test_models_and_error();
    failures += test_finish_reason_wire();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
