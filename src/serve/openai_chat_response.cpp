#include "serve/openai_chat.h"

#include "serve/generation_service.h"
#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

struct CompletionUsage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int cached_tokens     = 0;
    int reasoning_tokens  = 0;
};

const char* finish_reason(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "length";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::StopString:
    case ninfer::FinishReason::Cancelled:
        return "stop";
    }
    return "stop";
}

std::vector<ToolCall>
materialize_tool_calls(const std::vector<ninfer::GeneratedToolCall>& generated) {
    std::vector<ToolCall> calls;
    calls.reserve(generated.size());
    for (const ninfer::GeneratedToolCall& call : generated) {
        calls.push_back(ToolCall{.id             = new_openai_chat_tool_call_id(),
                                 .name           = call.name,
                                 .arguments_json = call.arguments_json});
    }
    return calls;
}

Json tool_calls_json(const std::vector<ToolCall>& calls, bool include_index) {
    Json output = Json::array();
    for (std::size_t index = 0; index < calls.size(); ++index) {
        const ToolCall& call = calls[index];
        Json value           = {{"id", call.id},
                                {"type", "function"},
                                {"function", Json{{"name", call.name}, {"arguments", call.arguments_json}}}};
        if (include_index) { value["index"] = static_cast<int>(index); }
        output.push_back(std::move(value));
    }
    return output;
}

Json usage_json(const CompletionUsage& usage) {
    const int cached_tokens = std::clamp(usage.cached_tokens, 0, usage.prompt_tokens);
    return Json{{"prompt_tokens", usage.prompt_tokens},
                {"prompt_tokens_details", Json{{"cached_tokens", cached_tokens}}},
                {"completion_tokens", usage.completion_tokens},
                {"completion_tokens_details",
                 Json{{"reasoning_tokens", std::max(0, usage.reasoning_tokens)}}},
                {"total_tokens", usage.prompt_tokens + usage.completion_tokens}};
}

CompletionUsage usage_from(const GenerationOutcome& outcome) {
    return CompletionUsage{
        .prompt_tokens     = outcome.prompt_tokens,
        .completion_tokens = outcome.completion_tokens,
        .cached_tokens     = static_cast<int>(outcome.metrics.prefix_cache_hit_tokens),
        .reasoning_tokens  = outcome.reasoning_tokens,
    };
}

Json base_payload(const OpenAIChatResponseIdentity& identity, const char* object) {
    return Json{{"id", identity.id},
                {"object", object},
                {"created", identity.created},
                {"model", identity.model}};
}

Json stream_choice(Json delta, Json finish_reason = nullptr) {
    return Json{{"index", 0},
                {"delta", std::move(delta)},
                {"logprobs", nullptr},
                {"finish_reason", std::move(finish_reason)}};
}

std::string event(Json payload) { return "data: " + payload.dump() + "\n\n"; }

std::string chunk(const OpenAIChatResponseIdentity& identity, Json delta, Json finish_reason,
                  bool include_usage) {
    Json payload       = base_payload(identity, "chat.completion.chunk");
    payload["choices"] = Json::array({stream_choice(std::move(delta), std::move(finish_reason))});
    if (include_usage) { payload["usage"] = nullptr; }
    return event(std::move(payload));
}

std::string usage_chunk(const OpenAIChatResponseIdentity& identity, const CompletionUsage& usage) {
    Json payload       = base_payload(identity, "chat.completion.chunk");
    payload["choices"] = Json::array();
    payload["usage"]   = usage_json(usage);
    return event(std::move(payload));
}

void require_prefix(std::string_view complete, std::string_view streamed, const char* channel) {
    if (!complete.starts_with(streamed)) {
        throw std::logic_error(std::string("streamed ") + channel +
                               " does not match terminal output");
    }
}

} // namespace

OpenAIChatResponseIdentity make_openai_chat_response_identity(std::string model) {
    return OpenAIChatResponseIdentity{
        .id      = new_openai_chat_completion_id(),
        .model   = std::move(model),
        .created = unix_time_now(),
    };
}

std::string make_chat_completion_response(const OpenAIChatResponseIdentity& identity,
                                          const GenerationOutcome& outcome) {
    Json message = {{"role", "assistant"}, {"content", outcome.text}, {"refusal", nullptr}};
    const bool has_tool_calls = !outcome.tool_calls.empty();
    // vLLM/SGLang-compatible reasoning_content preserves the Engine's Reasoning/Content split.
    if (!outcome.reasoning.empty()) { message["reasoning_content"] = outcome.reasoning; }
    if (has_tool_calls) {
        const std::vector<ToolCall> calls = materialize_tool_calls(outcome.tool_calls);
        message["content"]    = outcome.text.empty() ? Json(nullptr) : Json(outcome.text);
        message["tool_calls"] = tool_calls_json(calls, false);
    }

    Json payload       = base_payload(identity, "chat.completion");
    payload["choices"] = Json::array(
        {Json{{"index", 0},
              {"message", std::move(message)},
              {"logprobs", nullptr},
              {"finish_reason",
               has_tool_calls ? Json("tool_calls") : Json(finish_reason(outcome.finish_reason))}}});
    payload["usage"] = usage_json(usage_from(outcome));
    return payload.dump();
}

OpenAIChatStream::OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage)
    : identity_(std::move(identity)), include_usage_(include_usage) {}

std::string OpenAIChatStream::start() {
    if (started_ || finished_) { throw std::logic_error("OpenAI Chat stream already started"); }
    started_ = true;
    return chunk(identity_, Json{{"role", "assistant"}, {"content", ""}}, nullptr, include_usage_);
}

std::string OpenAIChatStream::reasoning_delta(const std::string& text) {
    if (!started_ || finished_ || content_started_) {
        throw std::logic_error("invalid OpenAI Chat reasoning delta state");
    }
    reasoning_ += text;
    return chunk(identity_, Json{{"reasoning_content", text}}, nullptr, include_usage_);
}

std::string OpenAIChatStream::content_delta(const std::string& text) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid OpenAI Chat content delta state");
    }
    content_started_ = true;
    content_ += text;
    return chunk(identity_, Json{{"content", text}}, nullptr, include_usage_);
}

std::vector<std::string> OpenAIChatStream::finish(const GenerationOutcome& outcome) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid OpenAI Chat stream finish state");
    }
    finished_ = true;
    require_prefix(outcome.reasoning, reasoning_, "reasoning");
    require_prefix(outcome.text, content_, "content");

    std::vector<std::string> events;
    const std::string reasoning_suffix = outcome.reasoning.substr(reasoning_.size());
    if (!reasoning_suffix.empty()) {
        if (content_started_) {
            throw std::logic_error("terminal reasoning appeared after streamed content");
        }
        events.push_back(chunk(identity_, Json{{"reasoning_content", reasoning_suffix}}, nullptr,
                               include_usage_));
    }
    const std::string content_suffix = outcome.text.substr(content_.size());
    if (!content_suffix.empty()) {
        events.push_back(
            chunk(identity_, Json{{"content", content_suffix}}, nullptr, include_usage_));
    }

    if (!outcome.tool_calls.empty()) {
        const std::vector<ToolCall> calls = materialize_tool_calls(outcome.tool_calls);
        events.push_back(chunk(identity_, Json{{"tool_calls", tool_calls_json(calls, true)}},
                               nullptr, include_usage_));
        events.push_back(chunk(identity_, Json::object(), "tool_calls", include_usage_));
    } else {
        events.push_back(
            chunk(identity_, Json::object(), finish_reason(outcome.finish_reason), include_usage_));
    }
    if (include_usage_) { events.push_back(usage_chunk(identity_, usage_from(outcome))); }
    events.emplace_back("data: [DONE]\n\n");
    return events;
}

} // namespace ninfer::serve
