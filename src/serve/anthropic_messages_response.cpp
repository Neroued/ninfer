#include "serve/anthropic_messages.h"

#include "serve/generation_service.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::string random_identifier(const char* prefix) {
    static thread_local std::mt19937_64 random{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> distribution;
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx",
                  static_cast<unsigned long long>(distribution(random)));
    return std::string(prefix) + buffer.data();
}

std::string event(const char* type, Json payload) {
    return std::string("event: ") + type + "\ndata: " + payload.dump() + "\n\n";
}

std::vector<ToolCall> materialize_tool_calls(const GenerationOutcome& outcome) {
    std::vector<ToolCall> result;
    result.reserve(outcome.tool_calls.size());
    for (const ninfer::GeneratedToolCall& call : outcome.tool_calls) {
        result.push_back(ToolCall{.id             = random_identifier("toolu_"),
                                  .name           = call.name,
                                  .arguments_json = call.arguments_json});
    }
    return result;
}

Json parse_tool_input(const ToolCall& call) {
    Json input = Json::parse(call.arguments_json, nullptr, false);
    if (input.is_discarded() || !input.is_object()) {
        throw std::logic_error("Engine produced non-object Anthropic tool input");
    }
    return input;
}

struct StopPresentation {
    const char* reason = "end_turn";
    Json sequence      = nullptr;
};

StopPresentation stop_presentation(const GenerationOutcome& outcome) {
    if (!outcome.tool_calls.empty()) { return StopPresentation{.reason = "tool_use"}; }
    switch (outcome.finish_reason) {
    case ninfer::FinishReason::OutputLimit:
        return StopPresentation{.reason = "max_tokens"};
    case ninfer::FinishReason::ContextCapacity:
        return StopPresentation{.reason = "model_context_window_exceeded"};
    case ninfer::FinishReason::StopString:
        if (!outcome.matched_stop_string) {
            throw std::logic_error("stop-string terminal result has no matched declaration");
        }
        return StopPresentation{.reason   = "stop_sequence",
                                .sequence = *outcome.matched_stop_string};
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::None:
        return StopPresentation{.reason = "end_turn"};
    case ninfer::FinishReason::Cancelled:
        throw std::logic_error("cancelled generation cannot be serialized as an Anthropic message");
    }
    throw std::logic_error("unknown Engine finish reason");
}

Json final_usage(const GenerationOutcome& outcome) {
    const int prompt = std::max(0, outcome.prompt_tokens);
    const int cached = static_cast<int>(std::min<std::uint64_t>(
        outcome.metrics.prefix_cache_hit_tokens, static_cast<std::uint64_t>(prompt)));
    return Json{
        {"input_tokens", prompt - cached},
        {"cache_creation_input_tokens", nullptr},
        {"cache_read_input_tokens", cached},
        {"cache_creation", nullptr},
        {"output_tokens", std::max(0, outcome.completion_tokens)},
        {"output_tokens_details", Json{{"thinking_tokens", std::max(0, outcome.reasoning_tokens)}}},
        {"server_tool_use", Json{{"web_search_requests", 0}, {"web_fetch_requests", 0}}},
        {"service_tier", nullptr},
        {"inference_geo", nullptr}};
}

Json streaming_start_usage(int input_tokens, std::optional<int> cache_read_input_tokens) {
    const int prompt = std::max(0, input_tokens);
    std::optional<int> cached;
    if (cache_read_input_tokens) { cached = std::clamp(*cache_read_input_tokens, 0, prompt); }
    return Json{{"input_tokens", cached ? prompt - *cached : prompt},
                {"cache_creation_input_tokens", nullptr},
                {"cache_read_input_tokens", cached ? Json(*cached) : Json(nullptr)},
                {"cache_creation", nullptr},
                {"output_tokens", 0},
                {"server_tool_use", Json{{"web_search_requests", 0}, {"web_fetch_requests", 0}}},
                {"service_tier", nullptr},
                {"inference_geo", nullptr}};
}

void append(std::vector<std::string>& target, std::vector<std::string> source) {
    target.insert(target.end(), std::make_move_iterator(source.begin()),
                  std::make_move_iterator(source.end()));
}

void require_prefix(std::string_view complete, std::string_view streamed, const char* channel) {
    if (!complete.starts_with(streamed)) {
        throw std::logic_error(std::string("streamed Anthropic ") + channel +
                               " does not match terminal output");
    }
}

} // namespace

std::string new_anthropic_request_id() { return random_identifier("req_"); }

AnthropicResponseIdentity make_anthropic_response_identity(std::string request_id,
                                                           std::string model) {
    return AnthropicResponseIdentity{.request_id = std::move(request_id),
                                     .message_id = random_identifier("msg_"),
                                     .model      = std::move(model)};
}

ApiError normalize_anthropic_error(ApiError error) {
    if (error.param == "reasoning_effort") { error.param = "output_config.effort"; }
    if (error.code == "server_overloaded" || error.status == 429) {
        error.status = 529;
        error.type   = "overloaded_error";
        return error;
    }
    if (error.code == "request_queue_timeout" || error.code == "media_fetch_timeout" ||
        error.status == 504) {
        error.status = 504;
        error.type   = "timeout_error";
        return error;
    }
    switch (error.status) {
    case 400:
        error.type = "invalid_request_error";
        break;
    case 401:
        error.type = "authentication_error";
        break;
    case 403:
        error.type = "permission_error";
        break;
    case 404:
        error.type = "not_found_error";
        break;
    case 413:
        error.type = "request_too_large";
        break;
    default:
        if (error.status >= 500) { error.type = "api_error"; }
        break;
    }
    return error;
}

std::string make_anthropic_error_body(const ApiError& api_error, const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    return Json{{"type", "error"},
                {"error", Json{{"type", error.type}, {"message", error.message}}},
                {"request_id", request_id}}
        .dump();
}

std::string make_anthropic_sse_error(const ApiError& error, const std::string& request_id) {
    return std::string("event: error\ndata: ") + make_anthropic_error_body(error, request_id) +
           "\n\n";
}

std::string make_anthropic_messages_response(const AnthropicResponseIdentity& identity,
                                             const GenerationOutcome& outcome,
                                             const AnthropicThinkingSigner& signer) {
    Json content = Json::array();
    if (!outcome.reasoning.empty()) {
        content.push_back(Json{{"type", "thinking"},
                               {"thinking", outcome.reasoning},
                               {"signature", signer.sign(outcome.reasoning, 0)}});
    }
    if (!outcome.text.empty()) {
        content.push_back(Json{{"type", "text"}, {"text", outcome.text}});
    }
    for (const ToolCall& call : materialize_tool_calls(outcome)) {
        content.push_back(Json{{"type", "tool_use"},
                               {"id", call.id},
                               {"name", call.name},
                               {"input", parse_tool_input(call)}});
    }
    const StopPresentation stop = stop_presentation(outcome);
    return Json{{"id", identity.message_id},
                {"type", "message"},
                {"role", "assistant"},
                {"model", identity.model},
                {"content", std::move(content)},
                {"stop_reason", stop.reason},
                {"stop_sequence", stop.sequence},
                {"usage", final_usage(outcome)}}
        .dump();
}

std::string make_anthropic_count_tokens_response(int input_tokens) {
    return Json{{"input_tokens", input_tokens}}.dump();
}

AnthropicMessagesStream::AnthropicMessagesStream(AnthropicResponseIdentity identity,
                                                 int input_tokens, AnthropicThinkingSigner signer)
    : identity_(std::move(identity)), signer_(std::move(signer)), input_tokens_(input_tokens) {}

std::string AnthropicMessagesStream::start() { return start_with_cache(std::nullopt); }

std::string AnthropicMessagesStream::start(const ninfer::GenerationStart& generation) {
    if (generation.prompt.prompt_tokens != static_cast<std::uint32_t>(input_tokens_)) {
        throw std::logic_error("Anthropic stream prompt count differs from Engine start");
    }
    return start_with_cache(static_cast<int>(generation.reused_prompt_tokens));
}

std::string AnthropicMessagesStream::start_with_cache(std::optional<int> cache_read_input_tokens) {
    if (started_ || finished_) { throw std::logic_error("Anthropic stream already started"); }
    started_ = true;
    const Json message{{"id", identity_.message_id},
                       {"type", "message"},
                       {"role", "assistant"},
                       {"model", identity_.model},
                       {"content", Json::array()},
                       {"stop_reason", nullptr},
                       {"stop_sequence", nullptr},
                       {"usage", streaming_start_usage(input_tokens_, cache_read_input_tokens)}};
    return event("message_start", Json{{"type", "message_start"}, {"message", message}});
}

std::vector<std::string> AnthropicMessagesStream::reasoning_delta(const std::string& text) {
    if (!started_ || finished_ || text_open_) {
        throw std::logic_error("invalid Anthropic reasoning delta state");
    }
    std::vector<std::string> events;
    if (!thinking_open_) {
        thinking_index_ = next_index_++;
        thinking_open_  = true;
        events.push_back(
            event("content_block_start",
                  Json{{"type", "content_block_start"},
                       {"index", thinking_index_},
                       {"content_block",
                        Json{{"type", "thinking"}, {"thinking", ""}, {"signature", ""}}}}));
    }
    reasoning_ += text;
    if (!text.empty()) {
        events.push_back(
            event("content_block_delta",
                  Json{{"type", "content_block_delta"},
                       {"index", thinking_index_},
                       {"delta", Json{{"type", "thinking_delta"}, {"thinking", text}}}}));
    }
    return events;
}

std::vector<std::string> AnthropicMessagesStream::close_thinking() {
    std::vector<std::string> events;
    if (!thinking_open_) { return events; }
    if (!signature_sent_) {
        events.push_back(event(
            "content_block_delta",
            Json{{"type", "content_block_delta"},
                 {"index", thinking_index_},
                 {"delta", Json{{"type", "signature_delta"},
                                {"signature", signer_.sign(reasoning_, static_cast<std::size_t>(
                                                                           thinking_index_))}}}}));
        signature_sent_ = true;
    }
    events.push_back(event("content_block_stop",
                           Json{{"type", "content_block_stop"}, {"index", thinking_index_}}));
    thinking_open_ = false;
    return events;
}

std::vector<std::string> AnthropicMessagesStream::content_delta(const std::string& text) {
    if (!started_ || finished_) { throw std::logic_error("invalid Anthropic text delta state"); }
    std::vector<std::string> events = close_thinking();
    if (!text_open_) {
        text_index_ = next_index_++;
        text_open_  = true;
        events.push_back(event("content_block_start",
                               Json{{"type", "content_block_start"},
                                    {"index", text_index_},
                                    {"content_block", Json{{"type", "text"}, {"text", ""}}}}));
    }
    content_ += text;
    if (!text.empty()) {
        events.push_back(event("content_block_delta",
                               Json{{"type", "content_block_delta"},
                                    {"index", text_index_},
                                    {"delta", Json{{"type", "text_delta"}, {"text", text}}}}));
    }
    return events;
}

std::vector<std::string> AnthropicMessagesStream::close_text() {
    if (!text_open_) { return {}; }
    text_open_ = false;
    return {
        event("content_block_stop", Json{{"type", "content_block_stop"}, {"index", text_index_}})};
}

std::vector<std::string> AnthropicMessagesStream::finish(const GenerationOutcome& outcome) {
    if (!started_ || finished_) { throw std::logic_error("invalid Anthropic stream finish state"); }
    require_prefix(outcome.reasoning, reasoning_, "reasoning");
    require_prefix(outcome.text, content_, "content");

    std::vector<std::string> events;
    const std::string reasoning_suffix = outcome.reasoning.substr(reasoning_.size());
    if (!reasoning_suffix.empty()) {
        if (text_open_) {
            throw std::logic_error("terminal Anthropic reasoning appeared after streamed text");
        }
        append(events, reasoning_delta(reasoning_suffix));
    }
    const std::string content_suffix = outcome.text.substr(content_.size());
    if (!content_suffix.empty()) { append(events, content_delta(content_suffix)); }
    append(events, close_thinking());
    append(events, close_text());

    for (const ToolCall& call : materialize_tool_calls(outcome)) {
        (void)parse_tool_input(call);
        const int index = next_index_++;
        events.push_back(
            event("content_block_start", Json{{"type", "content_block_start"},
                                              {"index", index},
                                              {"content_block", Json{{"type", "tool_use"},
                                                                     {"id", call.id},
                                                                     {"name", call.name},
                                                                     {"input", Json::object()}}}}));
        events.push_back(event("content_block_delta",
                               Json{{"type", "content_block_delta"},
                                    {"index", index},
                                    {"delta", Json{{"type", "input_json_delta"},
                                                   {"partial_json", call.arguments_json}}}}));
        events.push_back(
            event("content_block_stop", Json{{"type", "content_block_stop"}, {"index", index}}));
    }

    const StopPresentation stop = stop_presentation(outcome);
    events.push_back(event("message_delta", Json{{"type", "message_delta"},
                                                 {"delta", Json{{"stop_reason", stop.reason},
                                                                {"stop_sequence", stop.sequence}}},
                                                 {"usage", final_usage(outcome)}}));
    events.push_back(event("message_stop", Json{{"type", "message_stop"}}));
    finished_ = true;
    return events;
}

std::string AnthropicMessagesStream::error(const ApiError& api_error) const {
    return make_anthropic_sse_error(api_error, identity_.request_id);
}

} // namespace ninfer::serve
