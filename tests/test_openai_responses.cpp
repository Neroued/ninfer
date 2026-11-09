// Host-only contract tests for the OpenAI Responses adapter. These tests keep request parsing,
// previous_response_id state reconstruction, output encoding, and SSE sequencing independent of
// an Engine instance.

#include "serve/generation_service.h"
#include "serve/openai_responses.h"
#include "serve/translate.h"
#include "tool_call_salvage_probe.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

RequestLimits limits() {
    RequestLimits value;
    value.default_max_tokens = 256;
    return value;
}

std::string api_code(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& exception) { return exception.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

ApiError api_error(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& exception) { return exception.error(); } catch (...) {
        return ApiError{.status = 0, .message = "wrong exception"};
    }
    return ApiError{.status = 0, .message = "no exception"};
}

Json parse_event(const std::string& wire) {
    const std::size_t newline = wire.find('\n');
    if (!wire.starts_with("event: ") || newline == std::string::npos || !wire.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    const std::string type   = wire.substr(7, newline - 7);
    const std::string prefix = "data: ";
    if (wire.compare(newline + 1, prefix.size(), prefix) != 0) {
        throw std::runtime_error("missing SSE data field");
    }
    const std::size_t begin = newline + 1 + prefix.size();
    Json payload            = Json::parse(wire.substr(begin, wire.size() - begin - 2));
    if (payload.at("type") != type) { throw std::runtime_error("SSE type mismatch"); }
    return payload;
}

ChatTurn text_turn(ninfer::ChatRole role, std::string text) {
    ChatTurn turn;
    turn.role = role;
    ContentPart part;
    part.kind     = ContentKind::Text;
    part.type_raw = role == ninfer::ChatRole::Assistant ? "output_text" : "input_text";
    part.text     = std::move(text);
    turn.content.push_back(std::move(part));
    return turn;
}

ChatTurn call_turn(std::initializer_list<std::pair<const char*, const char*>> calls) {
    ChatTurn turn;
    turn.role = ninfer::ChatRole::Assistant;
    for (const auto& [id, name] : calls) {
        turn.tool_calls.push_back(
            ToolCall{.id = id, .name = name, .arguments_json = R"({"value":1})"});
    }
    return turn;
}

StoredOpenAIResponse stored_parent(OpenAIResponseContext context) {
    StoredOpenAIResponse record;
    record.id                = "resp_parent";
    record.session_key       = "responses-session";
    record.response          = Json{{"id", record.id}, {"object", "response"}};
    record.context           = std::move(context);
    record.preserve_thinking = true;
    return record;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                            = "answer";
    outcome.reasoning                       = "thought";
    outcome.prompt_tokens                   = 11;
    outcome.completion_tokens               = 7;
    outcome.reasoning_tokens                = 3;
    outcome.finish_reason                   = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens = 4;
    return outcome;
}

int test_basic_request_and_resolution() {
    const Json body = {{"model", "qwen3.6-27b"},
                       {"input", "hello"},
                       {"instructions", "be concise"},
                       {"max_output_tokens", 64},
                       {"temperature", 0.3},
                       {"top_p", 0.8},
                       {"reasoning", Json{{"effort", "medium"}}},
                       {"metadata", Json{{"trace", "abc"}}}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());

    int failures = 0;
    failures += check(request.prompt.model == "qwen3.6-27b", "model parsed");
    failures += check(request.prompt.input_turns.size() == 1 &&
                          request.prompt.input_turns[0].role == ninfer::ChatRole::User &&
                          request.prompt.input_turns[0].content[0].text == "hello",
                      "string input normalized to a user turn");
    failures +=
        check(request.prompt.input_items.size() == 1 &&
                  request.prompt.input_items[0].at("type") == "message" &&
                  request.prompt.input_items[0].at("content")[0].at("type") == "input_text" &&
                  request.prompt.input_items[0].contains("id"),
              "canonical input Item built");
    failures += check(request.prompt.instructions && *request.prompt.instructions == "be concise",
                      "instructions retained outside current input state");
    failures += check(request.requested_max_output_tokens == 64 &&
                          request.prompt.generation.max_tokens == 64,
                      "explicit output budget reaches generation request");
    failures +=
        check(request.prompt.generation.reasoning_effort == RequestedReasoningEffort::Medium,
              "reasoning effort parsed");
    failures += check(request.store && !request.stream && request.parallel_tool_calls,
                      "Responses defaults applied");

    OpenAIResponsesStore store(8, 1ULL << 20);
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request.prompt, store, "resp_current", true);
    failures += check(resolved.generation.messages.size() == 2 &&
                          resolved.generation.messages[0].role == ninfer::ChatRole::Developer &&
                          resolved.generation.messages[0].content[0].text == "be concise" &&
                          resolved.generation.messages[1].content[0].text == "hello",
                      "instructions and current input composed in model order");
    failures +=
        check(resolved.session_key == "resp_current" &&
                  resolved.cache_hints.session_key == "resp_current" &&
                  resolved.cache_hints.retention == ninfer::CacheRetentionHint::LiveSession &&
                  resolved.cache_hints.update_session_index,
              "stored root response receives one live Engine session");
    return failures;
}

int test_budgets_and_nonsemantic_hints() {
    const Json base = {{"model", "m"}, {"input", "hello"}};
    int failures    = 0;

    const OpenAIResponsesCreateRequest omitted =
        parse_openai_responses_create_request(base, limits());
    failures +=
        check(!omitted.requested_max_output_tokens && omitted.prompt.generation.max_tokens == 256,
              "omitted output budget uses the server default without changing its echo");
    for (const int budget : {0, 1, 15}) {
        Json body                 = base;
        body["max_output_tokens"] = budget;
        const OpenAIResponsesCreateRequest parsed =
            parse_openai_responses_create_request(body, limits());
        failures += check(parsed.requested_max_output_tokens == budget &&
                              parsed.prompt.generation.max_tokens == budget,
                          "non-negative output budget accepted");
    }

    Json hints = base;
    hints.update({{"background", false},
                  {"client_metadata", Json{{"session_id", "session-1"}, {"trace", Json::array()}}},
                  {"include", Json::array()},
                  {"max_tool_calls", 0},
                  {"prompt_cache_key", "stable-prefix"},
                  {"prompt_cache_retention", "24h"},
                  {"prompt_cache_options",
                   Json{{"mode", "explicit"}, {"ttl", "30m"}, {"future_hint", true}}},
                  {"safety_identifier", "local-user"},
                  {"service_tier", "auto"},
                  {"stream_options", Json{{"include_obfuscation", false}}},
                  {"text", Json{{"format", Json{{"type", "text"}}}, {"verbosity", "medium"}}},
                  {"top_logprobs", 0},
                  {"truncation", "disabled"},
                  {"user", "legacy-user"}});
    const OpenAIResponsesCreateRequest accepted =
        parse_openai_responses_create_request(hints, limits());
    failures += check(accepted.max_tool_calls == 0,
                      "typed nonsemantic hints are accepted without changing generation");

    hints["client_metadata"] = nullptr;
    failures += check(parse_openai_responses_create_request(hints, limits()).prompt.model == "m",
                      "null client_metadata is neutral");
    return failures;
}

int test_typed_items_and_cache_markers() {
    const Json body = {
        {"model", "m"},
        {"input",
         Json::array(
             {Json{{"id", "rs_1"},
                   {"type", "reasoning"},
                   {"summary", Json::array()},
                   {"content",
                    Json::array({Json{{"type", "reasoning_text"}, {"text", "use tools"}}})}},
              Json{{"id", "fc_1"},
                   {"type", "function_call"},
                   {"call_id", "call_1"},
                   {"name", "weather"},
                   {"arguments", R"({"city":"Paris"})"}},
              Json{{"id", "fco_1"},
                   {"type", "function_call_output"},
                   {"call_id", "call_1"},
                   {"output",
                    Json::array({Json{{"type", "input_text"},
                                      {"text", "20C"},
                                      {"prompt_cache_breakpoint", Json{{"mode", "explicit"}}}},
                                 Json{{"type", "input_image"},
                                      {"image_url", "data:image/png;base64,AA=="},
                                      {"detail", "auto"}}})}},
              Json{{"id", "msg_refusal"},
                   {"type", "message"},
                   {"role", "assistant"},
                   {"status", "incomplete"},
                   {"phase", "commentary"},
                   {"content",
                    Json::array({Json{{"type", "refusal"}, {"refusal", "cannot answer that"}}})}},
              Json{{"id", "msg_1"},
                   {"type", "message"},
                   {"role", "user"},
                   {"content", Json::array({Json{
                                   {"type", "input_text"},
                                   {"text", "describe"},
                                   {"prompt_cache_breakpoint", Json{{"mode", "explicit"}}}}})}}})}};

    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());
    int failures = 0;
    failures += check(request.prompt.input_turns.size() == 4 &&
                          request.prompt.input_turns[0].role == ninfer::ChatRole::Assistant &&
                          request.prompt.input_turns[0].reasoning_content == "use tools" &&
                          request.prompt.input_turns[0].tool_calls.size() == 1,
                      "reasoning and function call form one assistant turn");
    failures += check(request.prompt.input_turns[1].role == ninfer::ChatRole::Tool &&
                          request.prompt.input_turns[1].content.size() == 2 &&
                          request.prompt.input_turns[1].content[0].cache_boundary_after &&
                          request.prompt.input_turns[1].content[0].cache_boundary_after->kind ==
                              ninfer::PromptCacheMarkerKind::SharedStablePrefix &&
                          request.prompt.input_turns[1].content[1].kind == ContentKind::Image,
                      "typed multimodal tool output and explicit cache marker preserved");
    failures += check(request.prompt.input_turns[2].role == ninfer::ChatRole::Assistant &&
                          request.prompt.input_turns[2].content[0].text == "cannot answer that" &&
                          request.prompt.input_items[3].at("status") == "incomplete" &&
                          request.prompt.input_items[3].at("phase") == "commentary",
                      "refusal text and harmless assistant metadata are accepted");
    failures += check(request.prompt.input_turns[3].content[0].cache_boundary_after &&
                          request.prompt.input_turns[3].content[0].cache_boundary_after->kind ==
                              ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                      "message cache marker preserved");

    OpenAIResponsesStore store(8, 1ULL << 20);
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request.prompt, store, "resp_typed", true);
    failures += check(resolved.generation.messages.size() == 4 &&
                          resolved.generation.messages[1].tool_call_id == "call_1",
                      "typed Items survive call-graph normalization");
    const ninfer::PromptInput translated = to_prompt_input(
        resolved.generation, ResolvedPromptSemantics{}, [](const ContentPart& part) {
            ninfer::OwnedMedia media;
            media.kind  = part.kind == ContentKind::Image ? ninfer::MediaKind::Image
                                                          : ninfer::MediaKind::Video;
            media.bytes = {1};
            return media;
        });
    failures += check(translated.context_cache.markers.size() == 2 &&
                          translated.context_cache.markers[0].kind ==
                              ninfer::PromptCacheMarkerKind::SharedStablePrefix &&
                          translated.context_cache.markers[0].location ==
                              ninfer::PromptCacheMarkerLocation::MessagePartBoundary &&
                          translated.context_cache.markers[1].kind ==
                              ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                      "Responses breakpoints become shared Engine part boundaries");
    return failures;
}

int test_contiguous_assistant_items() {
    const Json body = {
        {"model", "m"},
        {"input", Json::array({Json{{"id", "msg_user"},
                                    {"type", "message"},
                                    {"role", "user"},
                                    {"content", "Inspect the file"}},
                               Json{{"id", "msg_preamble"},
                                    {"type", "message"},
                                    {"role", "assistant"},
                                    {"status", "incomplete"},
                                    {"phase", "commentary"},
                                    {"content", Json::array({Json{{"type", "output_text"},
                                                                  {"text", "Let me check:"},
                                                                  {"prompt_cache_breakpoint",
                                                                   Json{{"mode", "explicit"}}}}})}},
                               Json{{"id", "fc_read"},
                                    {"type", "function_call"},
                                    {"call_id", "call_read"},
                                    {"name", "read_file"},
                                    {"arguments", R"({"path":"a"})"}},
                               Json{{"id", "fco_read"},
                                    {"type", "function_call_output"},
                                    {"call_id", "call_read"},
                                    {"output", "contents"}},
                               Json{{"id", "msg_continue"},
                                    {"type", "message"},
                                    {"role", "user"},
                                    {"content", "Continue"}}})}};

    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());
    int failures = 0;
    failures += check(request.prompt.input_turns.size() == 4,
                      "contiguous assistant message and call created an extra turn");
    const ChatTurn& assistant = request.prompt.input_turns[1];
    failures +=
        check(assistant.role == ninfer::ChatRole::Assistant && assistant.content.size() == 1 &&
                  assistant.content[0].text == "Let me check:" &&
                  assistant.content[0].cache_boundary_after &&
                  assistant.content[0].cache_boundary_after->kind ==
                      ninfer::PromptCacheMarkerKind::SharedStablePrefix &&
                  assistant.tool_calls.size() == 1 && assistant.tool_calls[0].id == "call_read",
              "assistant preamble, cache marker and function call were not coalesced");
    failures += check(request.prompt.input_turns[2].role == ninfer::ChatRole::Tool &&
                          request.prompt.input_turns[2].tool_call_id == "call_read",
                      "function result did not end the assistant Item group");
    failures += check(request.prompt.input_items.size() == 5 &&
                          request.prompt.input_items[1].at("id") == "msg_preamble" &&
                          request.prompt.input_items[1].at("phase") == "commentary" &&
                          request.prompt.input_items[2].at("type") == "function_call",
                      "semantic grouping changed canonical Responses Item order or metadata");

    OpenAIResponsesStore store(8, 1ULL << 20);
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request.prompt, store, "resp_grouped", true);
    failures += check(resolved.generation.messages.size() == 4 &&
                          resolved.generation.messages[1].content.size() == 1 &&
                          resolved.generation.messages[1].tool_calls.size() == 1,
                      "call-graph normalization split the coalesced assistant turn");
    return failures;
}

bool same_assistant_turn(const ChatTurn& left, const ChatTurn& right) {
    if (left.role != ninfer::ChatRole::Assistant || right.role != ninfer::ChatRole::Assistant ||
        left.reasoning_content != right.reasoning_content ||
        left.content.size() != right.content.size() ||
        left.tool_calls.size() != right.tool_calls.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.content.size(); ++index) {
        if (left.content[index].kind != right.content[index].kind ||
            left.content[index].type_raw != right.content[index].type_raw ||
            left.content[index].text != right.content[index].text ||
            left.content[index].cache_boundary_after != right.content[index].cache_boundary_after) {
            return false;
        }
    }
    for (std::size_t index = 0; index < left.tool_calls.size(); ++index) {
        if (left.tool_calls[index].id != right.tool_calls[index].id ||
            left.tool_calls[index].name != right.tool_calls[index].name ||
            left.tool_calls[index].arguments_json != right.tool_calls[index].arguments_json) {
            return false;
        }
    }
    return true;
}

int test_response_output_history_round_trip() {
    const OpenAIResponsesCreateRequest source = parse_openai_responses_create_request(
        Json{{"model", "m"}, {"input", "Inspect both files"}, {"store", false}}, limits());
    GenerationOutcome outcome;
    outcome.reasoning     = "I should inspect both paths.";
    outcome.text          = "Let me check:";
    outcome.finish_reason = ninfer::FinishReason::StopToken;
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit",
        .arguments_json =
            R"({"file_path":"/tmp/probe.cpp","old_string":"old","new_string":"new"})"});
    outcome.tool_calls.push_back(
        ninfer::GeneratedToolCall{.name = "read_file", .arguments_json = R"({"path":"b"})"});
    const BuiltOpenAIResponse built =
        make_openai_response_object("resp_round_trip", 1, source, {}, outcome);

    Json input = Json::array(
        {Json{{"type", "message"}, {"role", "user"}, {"content", "Inspect both files"}}});
    for (const Json& item : built.output_items) { input.push_back(item); }
    for (const Json& item : built.output_items) {
        if (item.at("type") == "function_call") {
            input.push_back(Json{{"type", "function_call_output"},
                                 {"call_id", item.at("call_id")},
                                 {"output", "contents"}});
        }
    }
    input.push_back(Json{{"type", "message"}, {"role", "user"}, {"content", "Continue"}});
    const OpenAIResponsesCreateRequest replay = parse_openai_responses_create_request(
        Json{{"model", "m"}, {"input", std::move(input)}, {"store", false}}, limits());

    int failures = 0;
    failures +=
        check(built.output_history.size() == 1 && replay.prompt.input_turns.size() == 5 &&
                  same_assistant_turn(replay.prompt.input_turns[1], built.output_history[0]),
              "Responses output Items did not round-trip to their stored assistant history");
    const auto edit_item =
        std::find_if(built.output_items.begin(), built.output_items.end(), [](const Json& item) {
            return item.at("type") == "function_call" && item.at("name") == "Edit";
        });
    failures += check(
        edit_item != built.output_items.end() &&
            !Json::parse(edit_item->at("arguments").get<std::string>()).contains("replace_all"),
        "Responses output changed an omitted optional tool argument");

    GenerationOutcome incomplete;
    incomplete.reasoning     = "unfinished reasoning";
    incomplete.finish_reason = ninfer::FinishReason::OutputLimit;
    const BuiltOpenAIResponse reasoning_only =
        make_openai_response_object("resp_reasoning_only", 1, source, {}, incomplete);
    const Json reasoning_replay = {
        {"model", "m"},
        {"input",
         Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", "Start"}},
                      reasoning_only.output_items[0],
                      Json{{"type", "message"}, {"role", "user"}, {"content", "Continue"}}})}};
    const OpenAIResponsesCreateRequest parsed_reasoning =
        parse_openai_responses_create_request(reasoning_replay, limits());
    failures += check(reasoning_only.output_history.size() == 1 &&
                          parsed_reasoning.prompt.input_turns.size() == 3 &&
                          same_assistant_turn(parsed_reasoning.prompt.input_turns[1],
                                              reasoning_only.output_history[0]),
                      "reasoning-only incomplete output could not be replayed");
    return failures;
}

int test_assistant_item_boundaries_and_errors() {
    const Json first  = Json{{"type", "message"}, {"role", "assistant"}, {"content", "first "}};
    const Json second = Json{{"type", "message"}, {"role", "assistant"}, {"content", "second"}};
    const OpenAIResponsesCreateRequest grouped = parse_openai_responses_create_request(
        Json{{"model", "m"},
             {"input",
              Json::array({first, second,
                           Json{{"type", "message"}, {"role", "user"}, {"content", "boundary"}},
                           first,
                           Json{{"type", "message"}, {"role", "user"}, {"content", "done"}}})}},
        limits());
    int failures = 0;
    failures += check(grouped.prompt.input_turns.size() == 4 &&
                          grouped.prompt.input_turns[0].content.size() == 2 &&
                          grouped.prompt.input_turns[0].content[0].text == "first " &&
                          grouped.prompt.input_turns[0].content[1].text == "second" &&
                          grouped.prompt.input_turns[2].content.size() == 1,
                      "assistant content Items were not grouped or user boundary was ignored");

    const Json reasoning =
        Json{{"type", "reasoning"},
             {"content", Json::array({Json{{"type", "reasoning_text"}, {"text", "thought"}}})}};
    const Json call = Json{
        {"type", "function_call"}, {"call_id", "call_1"}, {"name", "lookup"}, {"arguments", "{}"}};
    const auto rejected = [&](Json input) {
        return api_error([&] {
            (void)parse_openai_responses_create_request(
                Json{{"model", "m"}, {"input", std::move(input)}}, limits());
        });
    };
    const ApiError reasoning_after_content = rejected(Json::array({first, reasoning}));
    failures +=
        check(reasoning_after_content.status == 400 &&
                  reasoning_after_content.type == "invalid_request_error" &&
                  reasoning_after_content.code == "invalid_assistant_history" &&
                  reasoning_after_content.param == "input" &&
                  reasoning_after_content.message.find(
                      "reasoning cannot follow assistant message content") != std::string::npos,
              "reasoning after assistant content did not fail precisely");
    const ApiError content_after_call = rejected(Json::array({call, first}));
    failures += check(content_after_call.code == "invalid_assistant_history" &&
                          content_after_call.message.find(
                              "assistant message content cannot follow function_call Items") !=
                              std::string::npos,
                      "assistant content after calls was silently reordered");
    const ApiError duplicate_reasoning = rejected(Json::array({reasoning, reasoning}));
    failures +=
        check(duplicate_reasoning.code == "invalid_assistant_history" &&
                  duplicate_reasoning.message.find(
                      "reasoning cannot follow another reasoning Item") != std::string::npos,
              "duplicate reasoning Items were silently overwritten");
    return failures;
}

int test_tools_and_effective_subset() {
    const Json weather = {{"type", "function"},
                          {"name", "weather"},
                          {"description", "Get weather"},
                          {"parameters", Json{{"type", "object"}}},
                          {"strict", false}};
    const Json clock   = {{"type", "function"},
                          {"name", "clock"},
                          {"allowed_callers", Json::array({"direct"})},
                          {"defer_loading", false}};
    const Json body    = {
        {"model", "m"},
        {"input", "time"},
        {"tools", Json::array({weather, clock})},
        {"tool_choice",
            Json{{"type", "allowed_tools"},
                 {"mode", "auto"},
                 {"tools", Json::array({Json{{"type", "function"}, {"name", "clock"}}})}}}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());
    int failures = 0;
    failures += check(request.tools.size() == 2 && request.prompt.generation.tools.size() == 1 &&
                          request.prompt.generation.tools[0].name == "clock",
                      "wire tool list and effective callable subset remain distinct");

    Json none           = body;
    none["tool_choice"] = "none";
    const OpenAIResponsesCreateRequest disabled =
        parse_openai_responses_create_request(none, limits());
    failures += check(disabled.prompt.generation.tool_choice.mode == ToolChoiceMode::None &&
                          !disabled.prompt.generation.uses_tools(),
                      "tool_choice none disables generation tools without deleting their echo");

    const Json ordered = Json::parse(
        R"({"model":"m","input":"probe","tools":[{"type":"function","name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}]})");
    const OpenAIResponsesCreateRequest ordered_request =
        parse_openai_responses_create_request(ordered, limits());
    const ninfer::PromptInput ordered_prompt =
        to_prompt_input(ordered_request.prompt.generation, ResolvedPromptSemantics{}, {});
    failures += check(
        ordered_prompt.options.tool_jsons.size() == 1 &&
            ordered_prompt.options.tool_jsons.front() ==
                R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}},"strict":false}})",
        "OpenAI Responses changed tool-schema member order before PromptInput");
    return failures;
}

int test_namespace_tools() {
    const Json clock_namespace = {
        {"type", "namespace"},
        {"name", "mcp__clock"},
        {"description", "Clock service"},
        {"tools", Json::array({Json{{"type", "function"},
                                    {"name", "now"},
                                    {"description", "Read the current time"},
                                    {"parameters", Json{{"type", "object"}}},
                                    {"strict", false},
                                    {"allowed_callers", Json::array({"direct"})},
                                    {"defer_loading", false}}})}};
    const Json weather_namespace = {
        {"type", "namespace"},
        {"name", "mcp__weather"},
        {"description", "Weather service"},
        {"tools", Json::array({Json{{"type", "function"}, {"name", "now"}}})}};
    const Json body = {{"model", "m"},
                       {"input", "time"},
                       {"tools", Json::array({clock_namespace, weather_namespace})},
                       {"tool_choice", Json{{"type", "allowed_tools"},
                                            {"mode", "auto"},
                                            {"tools", Json::array({Json{{"type", "function"},
                                                                        {"namespace", "mcp__clock"},
                                                                        {"name", "now"}}})}}}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());

    int failures = 0;
    failures += check(request.tools.size() == 2 && request.tools[0].at("type") == "namespace" &&
                          request.tools[0].at("tools")[0].at("name") == "now",
                      "wire namespace grouping is retained in the response echo");
    failures += check(request.prompt.generation.tools.size() == 1 &&
                          request.prompt.generation.tools[0].name == "mcp__clock__now" &&
                          request.prompt.generation.tools[0].description ==
                              "Clock service\n\nRead the current time",
                      "allowed namespace function lowers to one Engine tool with shared context");
    failures +=
        check(request.tool_identities.at("mcp__clock__now").name == "now" &&
                  request.tool_identities.at("mcp__clock__now").wire_namespace == "mcp__clock",
              "Engine identity has an explicit reversible wire mapping");

    GenerationOutcome outcome;
    outcome.finish_reason = ninfer::FinishReason::StopToken;
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "mcp__clock__now", .arguments_json = R"({"timezone":"UTC"})"});
    const BuiltOpenAIResponse built =
        make_openai_response_object("resp_namespace", 1, request, {}, outcome);
    const Json& output = built.body.at("output").at(0);
    failures += check(output.at("name") == "now" && output.at("namespace") == "mcp__clock" &&
                          built.output_history[0].tool_calls[0].name == "mcp__clock__now",
                      "terminal response restores wire identity while stored history stays flat");

    OpenAIResponsesCreateRequest stream_request = request;
    stream_request.stream                       = true;
    OpenAIResponsesEventStream stream("resp_namespace_stream", 1, stream_request, {});
    (void)stream.start();
    const OpenAIResponsesStreamFinish streamed = stream.finish(outcome);
    bool saw_added                             = false;
    bool saw_done                              = false;
    bool saw_item_done                         = false;
    for (const std::string& wire : streamed.events_before_terminal) {
        const Json event = parse_event(wire);
        if (event.at("type") == "response.output_item.added" &&
            event.at("item").at("type") == "function_call") {
            saw_added = event.at("item").at("name") == "now" &&
                        event.at("item").at("namespace") == "mcp__clock";
        } else if (event.at("type") == "response.function_call_arguments.done") {
            saw_done = event.at("name") == "now" && event.at("namespace") == "mcp__clock";
        } else if (event.at("type") == "response.output_item.done" &&
                   event.at("item").at("type") == "function_call") {
            saw_item_done = event.at("item").at("name") == "now" &&
                            event.at("item").at("namespace") == "mcp__clock";
        }
    }
    failures += check(saw_added && saw_done && saw_item_done,
                      "every named SSE function-call event restores the namespace identity");

    const Json history = {
        {"model", "m"},
        {"input", Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", "time"}},
                               Json{{"type", "function_call"},
                                    {"call_id", "call_clock"},
                                    {"namespace", "mcp__clock"},
                                    {"name", "now"},
                                    {"arguments", "{}"}},
                               Json{{"type", "function_call_output"},
                                    {"call_id", "call_clock"},
                                    {"namespace", "mcp__clock"},
                                    {"name", "now"},
                                    {"output", "12:00"}}})}};
    const OpenAIResponsesCreateRequest replay =
        parse_openai_responses_create_request(history, limits());
    OpenAIResponsesStore store(8, 1ULL << 20);
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(replay.prompt, store, "resp_replay", true);
    failures += check(resolved.generation.messages[1].tool_calls[0].name == "mcp__clock__now" &&
                          resolved.generation.messages[2].tool_result_name == "mcp__clock__now" &&
                          replay.prompt.input_items[1].at("namespace") == "mcp__clock" &&
                          replay.prompt.input_items[2].at("namespace") == "mcp__clock",
                      "stateless namespaced call history validates and preserves wire identity");

    Json mismatch                = history;
    mismatch["input"][2]["name"] = "later";
    const OpenAIResponsesCreateRequest mismatched =
        parse_openai_responses_create_request(mismatch, limits());
    failures += check(api_code([&] {
                          (void)resolve_openai_responses_prompt(mismatched.prompt, store,
                                                                "resp_mismatch", true);
                      }) == "invalid_tool_history",
                      "tool output identity must agree with its call_id");

    store.put(stored_parent(append_openai_response_context(
        {}, {text_turn(ninfer::ChatRole::User, "time"),
             call_turn({{"call_stored_clock", "mcp__clock__now"}})})));
    const OpenAIResponsesCreateRequest stored_replay = parse_openai_responses_create_request(
        Json{{"model", "m"},
             {"previous_response_id", "resp_parent"},
             {"input", Json::array({Json{{"type", "function_call_output"},
                                         {"call_id", "call_stored_clock"},
                                         {"namespace", "mcp__clock"},
                                         {"name", "now"},
                                         {"output", "12:00"}}})}},
        limits());
    const OpenAIResponsesResolvedPrompt stored_resolved =
        resolve_openai_responses_prompt(stored_replay.prompt, store, "resp_stored_replay", true);
    failures +=
        check(stored_resolved.generation.messages.back().tool_call_id == "call_stored_clock" &&
                  stored_resolved.generation.messages.back().tool_result_name == "mcp__clock__now",
              "namespaced tool result validates against previous_response_id history");

    Json collision           = body;
    collision["tool_choice"] = "auto";
    collision["tools"].push_back(Json{{"type", "function"}, {"name", "mcp__clock__now"}});
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(collision, limits());
                      }) == "duplicate_tool_name",
                      "plain and namespaced Engine identities cannot collide");

    Json nested_custom                 = body;
    nested_custom["tool_choice"]       = "auto";
    nested_custom["tools"][0]["tools"] = Json::array({Json{{"type", "custom"}, {"name", "raw"}}});
    const OpenAIResponsesCreateRequest lowered_custom =
        parse_openai_responses_create_request(nested_custom, limits());
    failures += check(lowered_custom.prompt.generation.tools.size() == 2 &&
                          lowered_custom.prompt.generation.tools[0].name == "mcp__clock__raw" &&
                          lowered_custom.prompt.generation.tools[1].name == "mcp__weather__now" &&
                          lowered_custom.prompt.generation.tools[0].input_schema_json.find(
                              "\"input\"") != std::string::npos,
                      "namespace custom tools lower to a single-input Engine function");
    failures += check(lowered_custom.tools[0]["tools"][0].at("type") == "custom",
                      "the wire tool list keeps the nested custom tool shape");

    Json oversized           = body;
    oversized["tool_choice"] = "auto";
    oversized["tools"] =
        Json::array({Json{{"type", "namespace"},
                          {"name", std::string(60, 'n')},
                          {"tools", Json::array({Json{{"type", "function"}, {"name", "tool"}}})}}});
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(oversized, limits());
                      }) == "invalid_tool_name",
                      "flattened identities must fit the Engine tool-name contract");
    return failures;
}

// Calls the Qwen parser keeps despite their names come back as function_call input Items without a
// 400, and a tool that allowed_tools filters out for the turn never becomes a call.
int test_salvaged_tool_calls() {
    const auto tool = [](const char* name) {
        return Json{{"type", "function"},
                    {"name", name},
                    {"parameters", Json{{"type", "object"},
                                        {"properties", Json{{"path", Json{{"type", "string"}}}}}}}};
    };
    const Json tools = Json::array({tool("Read"), tool("Write")});
    const Json body  = {
        {"model", "m"},
        {"input", "read a.txt"},
        {"store", false},
        {"tools", tools},
        {"tool_choice",
          Json{{"type", "allowed_tools"},
               {"mode", "auto"},
               {"tools", Json::array({Json{{"type", "function"}, {"name", "Read"}}})}}}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());
    const ninfer::PromptInput input =
        to_prompt_input(request.prompt.generation, ResolvedPromptSemantics{}, {});
    const std::size_t max_name = request.prompt.generation.tool_name_max_length;

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
        const BuiltOpenAIResponse built =
            make_openai_response_object("resp_salvaged", 1, request, {}, outcome);
        Json history =
            Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", "read a.txt"}}});
        for (const Json& item : built.output_items) {
            history.push_back(item);
            if (item.at("type") == "function_call") {
                history.push_back(Json{{"type", "function_call_output"},
                                       {"call_id", item.at("call_id")},
                                       {"output", "unknown tool"}});
            }
        }
        const ApiError error = api_error([&] {
            (void)parse_openai_responses_create_request(
                Json{{"model", "m"}, {"input", history}, {"store", false}, {"tools", tools}},
                limits());
        });
        failures += check(error.message == "no exception", "kept call '" + name +
                                                               "' could not come back as an "
                                                               "input Item: " +
                                                               error.message);
    }
    failures += check(kept >= 4, "the probe names no longer exercise salvaged calls");
    return failures;
}

// The model spells a namespace member as a catalog reference (srv.Read): the call reaches the
// client as that member, with its namespace, not as the plain tool that shares its name, and comes
// back in the next request's history.
int test_namespace_member_catalog_spelling() {
    const Json read = {
        {"type", "function"},
        {"name", "Read"},
        {"parameters",
         Json{{"type", "object"}, {"properties", Json{{"path", Json{{"type", "string"}}}}}}}};
    const Json tools                           = Json::array({read, Json{{"type", "namespace"},
                                                                         {"name", "srv"},
                                                                         {"description", "Server files"},
                                                                         {"tools", Json::array({read})}}});
    const OpenAIResponsesCreateRequest request = parse_openai_responses_create_request(
        Json{{"model", "m"}, {"input", "read a.txt"}, {"store", false}, {"tools", tools}},
        limits());
    const ninfer::PromptInput input =
        to_prompt_input(request.prompt.generation, ResolvedPromptSemantics{}, {});
    const std::size_t max_name = request.prompt.generation.tool_name_max_length;

    int failures = 0;
    for (const char* spelling : {"srv.Read", "srv/Read", "tools[\"srv\"][\"Read\"]"}) {
        const auto parsed = ninfer::test::parse_model_call(input, max_name, spelling);
        if (parsed.tool_calls.size() != 1) {
            failures +=
                check(false, std::string("namespace member spelling was not kept: ") + spelling);
            continue;
        }
        GenerationOutcome outcome;
        outcome.finish_reason = ninfer::FinishReason::StopToken;
        outcome.tool_calls    = parsed.tool_calls;
        const BuiltOpenAIResponse built =
            make_openai_response_object("resp_member", 1, request, {}, outcome);
        const Json& output = built.body.at("output").at(0);
        failures += check(output.at("name") == "Read" && output.value("namespace", "") == "srv",
                          std::string("namespace member spelling reached another tool: ") +
                              spelling + " -> " + output.dump());
        Json history =
            Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", "x"}}, output,
                         Json{{"type", "function_call_output"},
                              {"call_id", output.at("call_id")},
                              {"output", "a"}}});
        const ApiError error = api_error([&] {
            (void)parse_openai_responses_create_request(
                Json{{"model", "m"}, {"input", history}, {"store", false}, {"tools", tools}},
                limits());
        });
        failures += check(error.message == "no exception",
                          std::string("namespace member call did not come back: ") + error.message);
    }
    return failures;
}

int test_explicit_rejections() {
    const Json base = {{"model", "m"}, {"input", "hello"}};
    int failures    = 0;

    Json value     = base;
    value["tools"] = Json::array({Json{{"type", "function"}, {"name", "f"}, {"strict", true}}});
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "strict_tools_not_supported",
                      "strict function schema is rejected explicitly");

    value         = base;
    value["text"] = Json{{"format", Json{{"type", "json_schema"}}}};
    failures += check(api_error([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }).param == "text.format.name",
                      "malformed output schema is rejected");

    value               = base;
    value["background"] = true;
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "background_not_supported",
                      "background execution is rejected explicitly");

    value                 = base;
    value["conversation"] = "conv_1";
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "conversations_not_supported",
                      "unavailable platform state is rejected explicitly");

    value               = base;
    value["truncation"] = "auto";
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "truncation_not_supported",
                      "lossy server truncation is rejected explicitly");

    value                  = base;
    value["made_up_field"] = 1;
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(value, limits());
                      }) == "unknown_parameter",
                      "unknown request parameter is rejected");

    for (const Json invalid : {Json("trace"), Json::array(), Json(7)}) {
        value                    = base;
        value["client_metadata"] = invalid;
        failures += check(api_code([&] {
                              (void)parse_openai_responses_create_request(value, limits());
                          }) == "invalid_type",
                          "client_metadata rejects malformed top-level shapes");
    }
    return failures;
}

int test_previous_response_call_graph() {
    OpenAIResponsesStore store(16, 1ULL << 20);
    const OpenAIResponseContext context =
        append_openai_response_context({}, {text_turn(ninfer::ChatRole::User, "run both"),
                                            call_turn({{"call_a", "alpha"}, {"call_b", "beta"}})});
    store.put(stored_parent(context));

    const Json reordered_body = {
        {"model", "m"},
        {"previous_response_id", "resp_parent"},
        {"input",
         Json::array(
             {Json{{"type", "function_call_output"}, {"call_id", "call_b"}, {"output", "B"}},
              Json{{"type", "function_call_output"}, {"call_id", "call_a"}, {"output", "A"}}})}};
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(reordered_body, limits());
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request.prompt, store, "resp_child", true);
    int failures = 0;
    failures += check(resolved.generation.messages.size() == 4 &&
                          resolved.generation.messages[2].tool_call_id == "call_a" &&
                          resolved.generation.messages[3].tool_call_id == "call_b",
                      "complete tool results are normalized to declaration order");
    failures += check(resolved.session_key == "responses-session" &&
                          resolved.generation.preserve_thinking == true &&
                          !resolved.preserve_thinking_semantic_change,
                      "parent continuation inherits session and prompt semantics");

    const OpenAIResponsesResolvedPrompt disposable =
        resolve_openai_responses_prompt(request.prompt, store, "resp_disposable", false);
    failures +=
        check(disposable.session_key == "responses-session" &&
                  disposable.cache_hints.retention == ninfer::CacheRetentionHint::Disposable &&
                  !disposable.cache_hints.update_session_index,
              "store=false consumes parent session without advancing it");

    Json partial     = reordered_body;
    partial["input"] = Json::array(
        {Json{{"type", "function_call_output"}, {"call_id", "call_b"}, {"output", "B"}}});
    const OpenAIResponsesCreateRequest invalid =
        parse_openai_responses_create_request(partial, limits());
    failures += check(api_code([&] {
                          (void)resolve_openai_responses_prompt(invalid.prompt, store,
                                                                "resp_invalid", true);
                      }) == "invalid_tool_history",
                      "non-prefix partial tool results are rejected");

    Json unknown     = reordered_body;
    unknown["input"] = Json::array(
        {Json{{"type", "function_call_output"}, {"call_id", "call_unknown"}, {"output", "x"}}});
    const OpenAIResponsesCreateRequest unknown_request =
        parse_openai_responses_create_request(unknown, limits());
    failures += check(api_code([&] {
                          (void)resolve_openai_responses_prompt(unknown_request.prompt, store,
                                                                "resp_unknown", true);
                      }) == "invalid_tool_history",
                      "unknown tool result call_id is rejected");

    OpenAIResponsesCreateRequest missing_parent = request;
    missing_parent.prompt.previous_response_id  = "resp_missing";
    failures += check(api_code([&] {
                          (void)resolve_openai_responses_prompt(missing_parent.prompt, store,
                                                                "resp_new", true);
                      }) == "response_not_found",
                      "missing parent response is reported precisely");
    return failures;
}

int test_response_object() {
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(Json{{"model", "m"},
                                                   {"input", "hello"},
                                                   {"reasoning", Json{{"effort", "low"}}},
                                                   {"store", false}},
                                              limits());
    OpenAIResponsesRuntimeValues runtime;
    runtime.temperature = 0.6F;
    runtime.top_p       = 0.95F;
    const BuiltOpenAIResponse built =
        make_openai_response_object("resp_test", 123, request, runtime, sample_outcome());
    const Json& response = built.body;
    int failures         = 0;
    failures += check(response.at("object") == "response" && response.at("status") == "completed" &&
                          response.at("completed_at").is_number_integer(),
                      "completed response has a completion timestamp");
    failures += check(response.at("max_output_tokens").is_null(),
                      "omitted output budget remains null in the response");
    failures += check(response.at("output").size() == 2 &&
                          response.at("output")[0].at("type") == "reasoning" &&
                          response.at("output")[1].at("type") == "message",
                      "reasoning and message are emitted as typed output Items");
    failures +=
        check(response.at("usage").at("input_tokens_details").at("cached_tokens") == 4 &&
                  response.at("usage").at("output_tokens_details").at("reasoning_tokens") == 3 &&
                  response.at("usage").at("total_tokens") == 18,
              "usage and cached token details serialized");

    GenerationOutcome incomplete = sample_outcome();
    incomplete.text.clear();
    incomplete.finish_reason = ninfer::FinishReason::OutputLimit;
    const BuiltOpenAIResponse limited =
        make_openai_response_object("resp_limit", 123, request, runtime, incomplete);
    failures += check(limited.body.at("status") == "incomplete" &&
                          limited.body.at("completed_at").is_null() &&
                          limited.body.at("output").size() == 1 &&
                          limited.body.at("output")[0].at("type") == "reasoning",
                      "reasoning-only incomplete output does not invent an empty message");

    GenerationOutcome tools = sample_outcome();
    tools.text.clear();
    tools.reasoning.clear();
    tools.tool_calls.push_back(
        ninfer::GeneratedToolCall{.name = "weather", .arguments_json = R"({"city":"Paris"})"});
    const BuiltOpenAIResponse tool_response =
        make_openai_response_object("resp_tool", 123, request, runtime, tools);
    const Json& item = tool_response.body.at("output").at(0);
    failures += check(item.at("type") == "function_call" &&
                          item.at("call_id").get<std::string>().starts_with("call_") &&
                          tool_response.output_history[0].tool_calls[0].id ==
                              item.at("call_id").get<std::string>(),
                      "wire and continuation history share one stable function call_id");
    return failures;
}

int test_sse_sequence_and_failures() {
    OpenAIResponsesCreateRequest request = parse_openai_responses_create_request(
        Json{{"model", "m"}, {"input", "hello"}, {"stream", true}}, limits());
    OpenAIResponsesEventStream encoder("resp_stream", 123, request, {});
    std::vector<std::string> wire = encoder.start();
    std::vector<std::string> next = encoder.reasoning_delta("thought");
    wire.insert(wire.end(), next.begin(), next.end());
    next = encoder.content_delta("ans");
    wire.insert(wire.end(), next.begin(), next.end());
    OpenAIResponsesStreamFinish finish = encoder.finish(sample_outcome());
    wire.insert(wire.end(), finish.events_before_terminal.begin(),
                finish.events_before_terminal.end());
    wire.push_back(encoder.terminal(finish.response));

    int failures                    = 0;
    std::uint64_t expected_sequence = 0;
    std::string text_deltas;
    for (const std::string& event : wire) {
        failures += check(event.find("[DONE]") == std::string::npos,
                          "Responses stream does not use Chat [DONE]");
        const Json payload = parse_event(event);
        failures += check(payload.at("sequence_number") == expected_sequence++,
                          "SSE sequence numbers are contiguous");
        if (payload.at("type") == "response.output_text.delta") {
            text_deltas += payload.at("delta").get<std::string>();
        }
    }
    failures += check(parse_event(wire.front()).at("type") == "response.created" &&
                          parse_event(wire.back()).at("type") == "response.completed" &&
                          text_deltas == "answer",
                      "SSE starts, reconstructs output, and terminates canonically");

    OpenAIResponsesEventStream failed("resp_failed", 123, std::move(request), {});
    (void)failed.start();
    const Json failure = parse_event(failed.failed(
        ApiError{.status = 500, .type = "server_error", .message = "stopped", .code = "stopped"}));
    failures += check(failure.at("type") == "response.failed" &&
                          failure.at("response").at("status") == "failed" &&
                          failure.at("response").at("completed_at").is_null(),
                      "runtime failure uses response.failed with no completion timestamp");

    OpenAIResponsesCreateRequest cancelled_request = parse_openai_responses_create_request(
        Json{{"model", "m"}, {"input", "hello"}, {"stream", true}}, limits());
    OpenAIResponsesEventStream cancelled("resp_cancelled", 123, cancelled_request, {});
    (void)cancelled.start();
    GenerationOutcome cancelled_outcome;
    cancelled_outcome.finish_reason                    = ninfer::FinishReason::Cancelled;
    const OpenAIResponsesStreamFinish cancelled_finish = cancelled.finish(cancelled_outcome);
    const Json cancelled_terminal = parse_event(cancelled.terminal(cancelled_finish.response));
    failures += check(cancelled_terminal.at("type") == "response.failed" &&
                          cancelled_terminal.at("response").at("status") == "cancelled",
                      "cancelled generation uses the protocol terminal failure event");
    return failures;
}

int test_input_tokens_uses_shared_state_path() {
    OpenAIResponsesStore store(8, 1ULL << 20);
    store.put(stored_parent(
        append_openai_response_context({}, {text_turn(ninfer::ChatRole::User, "parent"),
                                            text_turn(ninfer::ChatRole::Assistant, "answer")})));

    const OpenAIResponsesPromptRequest request = parse_openai_responses_input_tokens_request(
        Json{{"model", "m"},
             {"previous_response_id", "resp_parent"},
             {"instructions", "count this"},
             {"input", "next"},
             {"tools",
              Json::array(
                  {Json{{"type", "namespace"},
                        {"name", "mcp__clock"},
                        {"tools", Json::array({Json{{"type", "function"}, {"name", "now"}}})}}})}},
        limits());
    const OpenAIResponsesResolvedPrompt resolved =
        resolve_openai_responses_prompt(request, store, std::nullopt, false);
    int failures = 0;
    failures += check(resolved.generation.messages.size() == 4 &&
                          resolved.generation.messages[0].role == ninfer::ChatRole::Developer &&
                          resolved.generation.messages.back().content[0].text == "next",
                      "input token counting resolves instructions, parent, and current input");
    failures += check(resolved.generation.tools.size() == 1 &&
                          resolved.generation.tools[0].name == "mcp__clock__now",
                      "input token counting uses the namespace tool translation path");
    failures += check(nlohmann::json::parse(make_openai_response_input_tokens_body(9)) ==
                          nlohmann::json{{"object", "response.input_tokens"}, {"input_tokens", 9}},
                      "input token count response shape");
    return failures;
}

int test_custom_tool_round_trip() {
    Json patch_tool = {
        {"type", "custom"},
        {"name", "apply_patch"},
        {"description", "Apply a patch"},
        {"format", Json{{"type", "grammar"}, {"syntax", "lark"}, {"definition", "start: /.+/"}}}};
    Json source_body = {{"model", "m"},
                        {"input", "patch it"},
                        {"store", false},
                        {"tool_choice", "auto"},
                        {"tools", Json::array({patch_tool})}};
    const OpenAIResponsesCreateRequest source =
        parse_openai_responses_create_request(source_body, limits());
    int failures = 0;
    failures += check(source.prompt.generation.tools.size() == 1 &&
                          source.prompt.generation.tools[0].name == "apply_patch" &&
                          source.prompt.generation.tools[0].input_schema_json.find("\"input\"") !=
                              std::string::npos,
                      "custom tool lowers to a one-parameter Engine function");
    failures += check(source.prompt.generation.tools[0].description.find("lark grammar") !=
                          std::string::npos,
                      "grammar definition reaches the model as prompt text");
    failures += check(source.tools.size() == 1 && source.tools[0].at("type") == "custom" &&
                          source.tools[0].contains("format"),
                      "wire tool list echoes the custom entry unchanged");

    GenerationOutcome outcome;
    outcome.finish_reason = ninfer::FinishReason::StopToken;
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "apply_patch", .arguments_json = R"({"input":"*** Begin Patch"})"});
    const BuiltOpenAIResponse built =
        make_openai_response_object("resp_custom", 1, source, {}, outcome);
    failures += check(built.output_items.size() == 1 &&
                          built.output_items[0].at("type") == "custom_tool_call" &&
                          built.output_items[0].at("id").get<std::string>().starts_with("ctc") &&
                          built.output_items[0].at("input") == "*** Begin Patch",
                      "the Engine's function-path call leaves as a custom_tool_call Item");

    OpenAIResponsesEventStream encoder("resp_custom_stream", 1, source, {});
    (void)encoder.start();
    const OpenAIResponsesStreamFinish stream_finish = encoder.finish(outcome);
    bool saw_delta                                  = false;
    bool saw_done                                   = false;
    for (const std::string& event : stream_finish.events_before_terminal) {
        const Json payload = parse_event(event);
        if (payload.at("type") == "response.custom_tool_call_input.delta") {
            saw_delta = payload.at("delta") == "*** Begin Patch";
        }
        if (payload.at("type") == "response.custom_tool_call_input.done") {
            saw_done = payload.at("input") == "*** Begin Patch";
        }
    }
    failures += check(saw_delta && saw_done,
                      "custom input streams as one whole-input delta followed by done");

    const std::string call_id = built.output_items[0].at("call_id").get<std::string>();
    Json result_item          = {
        {"type", "custom_tool_call_output"}, {"call_id", call_id}, {"output", "ok"}};
    Json input = Json::array({Json{{"type", "message"}, {"role", "user"}, {"content", "patch it"}},
                              built.output_items[0], result_item});
    input.push_back(Json{{"type", "message"}, {"role", "user"}, {"content", "Continue"}});
    Json replay_body = {{"model", "m"}, {"input", std::move(input)}, {"store", false}};
    const OpenAIResponsesCreateRequest replay =
        parse_openai_responses_create_request(replay_body, limits());
    failures += check(replay.prompt.input_turns.size() == 4 &&
                          replay.prompt.input_turns[1].tool_calls[0].name == "apply_patch" &&
                          replay.prompt.input_turns[1].tool_calls[0].arguments_json ==
                              R"({"input":"*** Begin Patch"})" &&
                          replay.prompt.input_turns[2].tool_call_id == call_id,
                      "custom_tool_call and its output replay as function-path history turns");
    failures += check(replay.prompt.input_items[1].at("type") == "custom_tool_call" &&
                          replay.prompt.input_items[2].at("type") == "custom_tool_call_output",
                      "canonical Items keep the custom wire types");

    for (const Json& format :
         {Json("lark"), Json{{"type", "regex"}}, Json{{"type", "text"}, {"extra", 1}},
          Json{{"type", "grammar"}, {"syntax", "ebnf"}, {"definition", "x"}},
          Json{{"type", "grammar"}, {"syntax", "lark"}, {"definition", 5}},
          Json{{"type", "grammar"}, {"syntax", "lark"}}}) {
        Json odd_format                  = source_body;
        odd_format["tools"][0]["format"] = format;
        failures += check(api_error([&] {
                              (void)parse_openai_responses_create_request(odd_format, limits());
                          }).status == 400,
                          "custom tool format " + format.dump() + " is rejected");
    }
    Json text_format                  = source_body;
    text_format["tools"][0]["format"] = Json{{"type", "text"}};
    failures += check(parse_openai_responses_create_request(text_format, limits())
                              .prompt.generation.tools[0]
                              .description == "Apply a patch",
                      "a text format adds nothing to the description");

    Json broken = {
        {"type", "custom_tool_call"}, {"call_id", "c1"}, {"name", "apply_patch"}, {"input", 3}};
    Json broken_body = {{"model", "m"}, {"input", Json::array({broken})}, {"store", false}};
    failures += check(api_error([&] {
                          (void)parse_openai_responses_create_request(broken_body, limits());
                      }).status == 400,
                      "non-string custom_tool_call input is rejected");
    return failures;
}

int test_codex_cli_compatibility() {
    // The shape Codex CLI sends on every request: encrypted-reasoning include, a summary tier,
    // and hosted tools it attaches whether the gateway can run them or not.
    const Json body = {{"model", "qwen3.6-27b"},
                       {"input", "hi"},
                       {"store", false},
                       {"stream", true},
                       {"include", Json::array({"reasoning.encrypted_content"})},
                       {"reasoning", Json{{"effort", "medium"}, {"summary", "auto"}}},
                       {"prompt_cache_key", "pcache"},
                       {"client_metadata", Json{{"session_id", "s1"}}},
                       {"tool_choice", "auto"},
                       {"tools", Json::array({Json{{"type", "web_search"}, {"name", "ws"}},
                                              Json{{"type", "function"},
                                                   {"name", "shell"},
                                                   {"parameters", Json{{"type", "object"}}}}})}};
    int failures    = 0;
    const OpenAIResponsesCreateRequest request =
        parse_openai_responses_create_request(body, limits());
    failures += check(request.prompt.generation.tools.size() == 1 &&
                          request.prompt.generation.tools[0].name == "shell",
                      "hosted web_search tool is dropped, function tools survive");
    failures += check(request.tools.size() == 1 && request.tools[0].at("name") == "shell",
                      "wire tool list reports only the function tool");
    failures +=
        check(request.prompt.generation.reasoning_effort == RequestedReasoningEffort::Medium,
              "reasoning effort honored next to an ignored summary");

    Json hosted_only = body;
    hosted_only.erase("tool_choice");
    hosted_only["tools"] = Json::array({Json{{"type", "web_search_preview"}}});
    const OpenAIResponsesCreateRequest only_hosted =
        parse_openai_responses_create_request(hosted_only, limits());
    failures += check(only_hosted.prompt.generation.tools.empty() && only_hosted.tools.empty(),
                      "an all-web-search tool list parses to no Engine tools");
    for (const char* kept :
         {"local_shell", "computer_use", "file_search", "code_interpreter", "image_generation"}) {
        Json executor_backed     = body;
        executor_backed["tools"] = Json::array({Json{{"type", kept}}});
        failures +=
            check(api_code([&] {
                      (void)parse_openai_responses_create_request(executor_backed, limits());
                  }) == "tool_type_not_supported",
                  std::string(kept) + " is not dropped silently");
    }

    Json logprobs       = body;
    logprobs["include"] = Json::array({"message.output_text.logprobs"});
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(logprobs, limits());
                      }) == "include_not_supported",
                      "include values beyond encrypted_content remain unsupported");
    Json mixed = body;
    mixed["include"] =
        Json::array({"reasoning.encrypted_content", "web_search_call.action.sources"});
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(mixed, limits());
                      }) == "include_not_supported",
                      "one unusable include value rejects the whole request");

    Json odd_summary         = body;
    odd_summary["reasoning"] = Json{{"summary", "yolo"}};
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(odd_summary, limits());
                      }) == "reasoning_option_not_supported",
                      "unknown reasoning.summary tier is rejected");
    Json typed_summary         = body;
    typed_summary["reasoning"] = Json{{"summary", true}};
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(typed_summary, limits());
                      }) == "reasoning_option_not_supported",
                      "non-string reasoning.summary is rejected");

    Json unknown_executor     = body;
    unknown_executor["tools"] = Json::array({Json{{"type", "browser_invoker"}, {"name", "b"}}});
    failures += check(api_code([&] {
                          (void)parse_openai_responses_create_request(unknown_executor, limits());
                      }) == "tool_type_not_supported",
                      "executor-backed tool types outside the hosted list still 400");
    return failures;
}

// Structured output, ported from Neroued/ninfer 08902f73, eb6696ae, 8ffa3131 and 2734a56e
// (test_openai_responses.cpp), adapted to Tappa A (no JSON output with active tools yet).
int test_constrained_decoding() {
    int failures = 0;
    OpenAIResponsesStore store(8, 1024 * 1024);
    const auto engine_constraint = [&](const Json& body) {
        const auto parsed = parse_openai_responses_create_request(body, limits());
        const auto resolved =
            resolve_openai_responses_prompt(parsed.prompt, store, std::nullopt, false);
        return to_request_options(resolved.generation, {}, {}, true).constraint;
    };
    Json body{{"model", "qwen"}, {"input", "hello"}};
    failures += check(!engine_constraint(body), "unconstrained request gained a constraint");
    body["text"]     = Json{{"format", {{"type", "text"}}}};
    const auto plain = parse_openai_responses_create_request(body, limits());
    failures += check(!engine_constraint(body) && !plain.prompt.generation.constraint &&
                          make_openai_response_object("resp_plain", 1, plain, {}, sample_outcome())
                                  .body["text"]["format"] == nlohmann::json{{"type", "text"}},
                      "text format changed the request or its echo");

    for (const auto& value : {Json{{"choice", {"yes", "no"}}}, Json{{"regex", "[a-z]+"}},
                              Json{{"grammar", "root ::= \"yes\""}}}) {
        body                       = Json{{"model", "qwen"}, {"input", "hello"}};
        body["structured_outputs"] = value;
        const auto expected =
            value.contains("choice")  ? ninfer::OutputConstraint::choice({"yes", "no"})
            : value.contains("regex") ? ninfer::OutputConstraint::regex("[a-z]+")
                                      : ninfer::OutputConstraint::grammar("root ::= \"yes\"");
        failures += check(engine_constraint(body) == expected,
                          "Responses structured_outputs lost through prompt resolution");
    }
    body["structured_outputs"] = Json{{"grammar", ""}};
    failures += check(api_error([&] {
                          (void)parse_openai_responses_create_request(body, limits());
                      }).param == "structured_outputs.grammar",
                      "empty grammar was accepted");

    body = Json{
        {"model", "qwen"}, {"input", "hello"}, {"text", {{"format", {{"type", "json_object"}}}}}};
    failures += check(engine_constraint(body) == ninfer::OutputConstraint::json_object(),
                      "Responses json_object lost");

    const Json schema = {
        {"type", "object"},
        {"properties", {{"z", {{"type", "string"}}}, {"a", {{"type", "integer"}}}}}};
    body = Json{
        {"model", "qwen"},
        {"input", "hello"},
        {"stream", true},
        {"text",
         {{"format",
           {{"type", "json_schema"}, {"name", "answer"}, {"strict", true}, {"schema", schema}}}}}};
    const auto schema_request = parse_openai_responses_create_request(body, limits());
    failures += check(schema_request.stream &&
                          schema_request.prompt.generation.constraint ==
                              ninfer::OutputConstraint::json_schema(schema.dump()) &&
                          schema_request.prompt.generation.constraint_param == "text.format.schema",
                      "Responses JSON schema source/order or path lost");
    const auto response =
        make_openai_response_object("resp_test", 1, schema_request, {}, sample_outcome());
    failures += check(response.body["text"]["format"] == schema_request.text_format &&
                          schema_request.text_format["name"] == "answer" &&
                          schema_request.text_format["schema"]["properties"].contains("z"),
                      "Responses format echo lost");
    OpenAIResponsesEventStream stream("resp_stream", 1, schema_request, {});
    const auto started = stream.start();
    failures +=
        check(!started.empty() && parse_event(started.front())["response"]["text"]["format"] ==
                                      Json::parse(schema_request.text_format.dump()),
              "Responses stream did not echo text.format in response.created");

    // Malformed formats and combinations: 400 on the field.
    const auto param_of = [&](Json changed) {
        return api_error([&] { (void)parse_openai_responses_create_request(changed, limits()); });
    };
    auto bad = body;
    bad["text"]["format"].erase("name");
    failures += check(param_of(bad).param == "text.format.name", "schema without name accepted");
    bad                           = body;
    bad["text"]["format"]["type"] = "xml";
    failures += check(param_of(bad).param == "text.format.type", "unknown format type accepted");
    bad                   = body;
    bad["text"]["format"] = Json{{"type", "text"}, {"x", 1}};
    failures += check(param_of(bad).param == "text.format", "text format with options accepted");
    bad                       = body;
    bad["structured_outputs"] = Json{{"regex", "a"}};
    failures += check(param_of(bad).param == "structured_outputs.regex",
                      "conflicting output formats accepted");
    bad                   = body;
    bad["tools"]          = Json::array({Json{{"type", "function"}, {"name", "lookup"}}});
    const auto with_tools = param_of(bad);
    failures += check(with_tools.param == "text.format.schema" &&
                          with_tools.code == "unsupported_output_combination",
                      "JSON output with active tools accepted before the tool grammar exists");
    bad["tool_choice"] = "none";
    failures += check(parse_openai_responses_create_request(bad, limits())
                          .prompt.generation.constraint.has_value(),
                      "inactive tools blocked a constraint");
    bad                = body;
    bad["guided_json"] = Json::object();
    failures += check(param_of(bad).code == "unknown_parameter", "guided_json accepted");

    // Token counting accepts the format and ignores it.
    failures += check(api_code([&] {
                          (void)parse_openai_responses_input_tokens_request(
                              Json{{"model", "qwen"}, {"input", "hello"}, {"text", body["text"]}},
                              limits());
                      }).empty(),
                      "input_tokens rejected a text.format schema");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_constrained_decoding();
    failures += test_basic_request_and_resolution();
    failures += test_budgets_and_nonsemantic_hints();
    failures += test_typed_items_and_cache_markers();
    failures += test_contiguous_assistant_items();
    failures += test_response_output_history_round_trip();
    failures += test_assistant_item_boundaries_and_errors();
    failures += test_tools_and_effective_subset();
    failures += test_namespace_tools();
    failures += test_salvaged_tool_calls();
    failures += test_namespace_member_catalog_spelling();
    failures += test_explicit_rejections();
    failures += test_custom_tool_round_trip();
    failures += test_codex_cli_compatibility();
    failures += test_previous_response_call_graph();
    failures += test_response_object();
    failures += test_sse_sequence_and_failures();
    failures += test_input_tokens_uses_shared_state_path();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
