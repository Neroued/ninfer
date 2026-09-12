#include "serve/openai_chat.h"

#include "serve/generation_service.h"
#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

struct CompletionTimings {
    std::uint32_t cache_n          = 0;
    std::uint32_t prompt_n         = 0;
    double prompt_ms               = 0.0;
    double prompt_per_token_ms     = 0.0;
    double prompt_per_second       = 0.0;
    std::uint32_t predicted_n      = 0;
    double predicted_ms            = 0.0;
    double predicted_per_token_ms  = 0.0;
    double predicted_per_second    = 0.0;
    std::uint64_t draft_n          = 0;
    std::uint64_t draft_n_accepted = 0;
};

double finite_nonnegative(double value) {
    return std::isfinite(value) && value > 0.0 ? value : 0.0;
}

double milliseconds_per_token(std::uint32_t tokens, double milliseconds) {
    return tokens == 0 ? 0.0 : milliseconds / static_cast<double>(tokens);
}

double tokens_per_second(std::uint32_t tokens, double milliseconds) {
    return milliseconds > 0.0 ? 1000.0 * static_cast<double>(tokens) / milliseconds : 0.0;
}

CompletionTimings make_completion_timings(std::uint32_t prompt_tokens, std::uint32_t cached_tokens,
                                          std::uint32_t generated_tokens, double prompt_ms,
                                          double generation_ms, std::uint64_t draft_tokens = 0,
                                          std::uint64_t accepted_draft_tokens = 0) {
    CompletionTimings timings;
    timings.cache_n             = std::min(cached_tokens, prompt_tokens);
    timings.prompt_n            = prompt_tokens - timings.cache_n;
    timings.prompt_ms           = finite_nonnegative(prompt_ms);
    timings.prompt_per_token_ms = milliseconds_per_token(timings.prompt_n, timings.prompt_ms);
    timings.prompt_per_second   = tokens_per_second(timings.prompt_n, timings.prompt_ms);

    timings.predicted_n                  = generated_tokens;
    timings.predicted_ms                 = finite_nonnegative(generation_ms);
    const std::uint32_t decode_intervals = generated_tokens > 0 ? generated_tokens - 1 : 0;
    timings.predicted_per_token_ms = milliseconds_per_token(decode_intervals, timings.predicted_ms);
    timings.predicted_per_second   = tokens_per_second(decode_intervals, timings.predicted_ms);
    timings.draft_n                = draft_tokens;
    timings.draft_n_accepted       = accepted_draft_tokens;
    return timings;
}

Json timings_json(const CompletionTimings& timings) {
    Json output = {{"cache_n", timings.cache_n},
                   {"prompt_n", timings.prompt_n},
                   {"prompt_ms", timings.prompt_ms},
                   {"prompt_per_token_ms", timings.prompt_per_token_ms},
                   {"prompt_per_second", timings.prompt_per_second},
                   {"predicted_n", timings.predicted_n},
                   {"predicted_ms", timings.predicted_ms},
                   {"predicted_per_token_ms", timings.predicted_per_token_ms},
                   {"predicted_per_second", timings.predicted_per_second}};
    if (timings.draft_n != 0) {
        output["draft_n"]          = timings.draft_n;
        output["draft_n_accepted"] = timings.draft_n_accepted;
    }
    return output;
}

CompletionTimings outcome_timings(const GenerationOutcome& outcome) {
    return make_completion_timings(
        static_cast<std::uint32_t>(std::max(0, outcome.prompt_tokens)),
        outcome.metrics.prefix_cache_hit_tokens,
        static_cast<std::uint32_t>(std::max(0, outcome.completion_tokens)),
        outcome.metrics.prompt_wall_seconds * 1000.0,
        outcome.metrics.generation_wall_seconds * 1000.0, outcome.metrics.speculative_draft_tokens,
        outcome.metrics.speculative_accepted_tokens);
}

CompletionTimings observation_timings(std::uint32_t prompt_tokens, std::uint32_t cached_tokens,
                                      const ninfer::GenerationTimingObservation& observation) {
    constexpr double kNanosecondsToMilliseconds = 1.0e-6;
    return make_completion_timings(
        prompt_tokens, cached_tokens, observation.generated_tokens,
        static_cast<double>(observation.prompt_elapsed_ns) * kNanosecondsToMilliseconds,
        static_cast<double>(observation.generation_elapsed_ns) * kNanosecondsToMilliseconds);
}

Json prompt_progress_json(std::uint32_t total, std::uint32_t cached, std::uint32_t processed,
                          std::uint64_t elapsed_ns) {
    return Json{{"total", total},
                {"cache", cached},
                {"processed", processed},
                {"time_ms", elapsed_ns / 1000000ULL}};
}

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
                  bool include_usage, Json timings = nullptr) {
    Json payload       = base_payload(identity, "chat.completion.chunk");
    payload["choices"] = Json::array({stream_choice(std::move(delta), std::move(finish_reason))});
    if (include_usage) { payload["usage"] = nullptr; }
    if (!timings.is_null()) { payload["timings"] = std::move(timings); }
    return event(std::move(payload));
}

std::string usage_chunk(const OpenAIChatResponseIdentity& identity, const CompletionUsage& usage,
                        Json timings) {
    Json payload       = base_payload(identity, "chat.completion.chunk");
    payload["choices"] = Json::array();
    payload["usage"]   = usage_json(usage);
    payload["timings"] = std::move(timings);
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
    payload["usage"]   = usage_json(usage_from(outcome));
    payload["timings"] = timings_json(outcome_timings(outcome));
    return payload.dump();
}

OpenAIChatStream::OpenAIChatStream(OpenAIChatResponseIdentity identity, bool include_usage,
                                   bool timings_per_token, bool return_progress)
    : identity_(std::move(identity)), include_usage_(include_usage),
      timings_per_token_(timings_per_token), return_progress_(return_progress) {}

std::string OpenAIChatStream::start() {
    if (started_ || finished_) { throw std::logic_error("OpenAI Chat stream already started"); }
    started_ = true;
    return chunk(identity_, Json{{"role", "assistant"}, {"content", ""}}, nullptr, include_usage_);
}

void OpenAIChatStream::note_start(const ninfer::GenerationStart& start) {
    if (!started_ || admitted_ || finished_ ||
        start.reused_prompt_tokens > start.prompt.prompt_tokens) {
        throw std::logic_error("invalid OpenAI Chat generation-start state");
    }
    admitted_      = true;
    prompt_tokens_ = start.prompt.prompt_tokens;
    cached_tokens_ = start.reused_prompt_tokens;
}

std::string OpenAIChatStream::initial_prompt_progress() {
    if (!return_progress_ || !started_ || !admitted_ || progress_started_ || finished_) {
        throw std::logic_error("invalid OpenAI Chat initial prompt-progress state");
    }
    progress_started_         = true;
    last_progress_tokens_     = cached_tokens_;
    last_progress_elapsed_ns_ = 0;
    Json payload              = base_payload(identity_, "chat.completion.chunk");
    payload["choices"]        = Json::array({stream_choice(Json::object())});
    if (include_usage_) { payload["usage"] = nullptr; }
    payload["prompt_progress"] =
        prompt_progress_json(prompt_tokens_, cached_tokens_, cached_tokens_, 0);
    return event(std::move(payload));
}

std::string OpenAIChatStream::prompt_progress(const ninfer::PromptProgress& progress) {
    if (!return_progress_ || !started_ || !admitted_ || !progress_started_ || finished_ ||
        progress.total_prompt_tokens != prompt_tokens_ ||
        progress.reused_prompt_tokens != cached_tokens_ ||
        progress.processed_prompt_tokens < last_progress_tokens_ ||
        progress.processed_prompt_tokens > prompt_tokens_ ||
        progress.elapsed_ns < last_progress_elapsed_ns_) {
        throw std::logic_error("invalid OpenAI Chat prompt-progress state");
    }
    last_progress_tokens_     = progress.processed_prompt_tokens;
    last_progress_elapsed_ns_ = progress.elapsed_ns;
    Json payload              = base_payload(identity_, "chat.completion.chunk");
    payload["choices"]        = Json::array({stream_choice(Json::object())});
    if (include_usage_) { payload["usage"] = nullptr; }
    payload["prompt_progress"] =
        prompt_progress_json(progress.total_prompt_tokens, progress.reused_prompt_tokens,
                             progress.processed_prompt_tokens, progress.elapsed_ns);
    return event(std::move(payload));
}

void OpenAIChatStream::note_timing(const ninfer::GenerationTimingObservation& timing) {
    if (!timings_per_token_ || !started_ || !admitted_ || finished_ ||
        timing.generated_tokens == 0) {
        throw std::logic_error("invalid OpenAI Chat live-timing state");
    }
    if (live_timing_ && (timing.generated_tokens < live_timing_->generated_tokens ||
                         timing.prompt_elapsed_ns != live_timing_->prompt_elapsed_ns ||
                         timing.generation_elapsed_ns < live_timing_->generation_elapsed_ns)) {
        throw std::logic_error("OpenAI Chat live timings are not cumulative");
    }
    live_timing_ = timing;
}

Json OpenAIChatStream::live_timings_json() const {
    if (!timings_per_token_) { return nullptr; }
    if (!live_timing_ || !admitted_) {
        throw std::logic_error("OpenAI Chat output has no committed timing observation");
    }
    return timings_json(observation_timings(prompt_tokens_, cached_tokens_, *live_timing_));
}

std::string OpenAIChatStream::reasoning_delta(const std::string& text) {
    if (!started_ || finished_ || content_started_) {
        throw std::logic_error("invalid OpenAI Chat reasoning delta state");
    }
    reasoning_ += text;
    return chunk(identity_, Json{{"reasoning_content", text}}, nullptr, include_usage_,
                 live_timings_json());
}

std::string OpenAIChatStream::content_delta(const std::string& text) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid OpenAI Chat content delta state");
    }
    content_started_ = true;
    content_ += text;
    return chunk(identity_, Json{{"content", text}}, nullptr, include_usage_, live_timings_json());
}

std::vector<std::string> OpenAIChatStream::finish(const GenerationOutcome& outcome) {
    if (!started_ || finished_) {
        throw std::logic_error("invalid OpenAI Chat stream finish state");
    }
    finished_ = true;
    require_prefix(outcome.reasoning, reasoning_, "reasoning");
    require_prefix(outcome.text, content_, "content");

    const Json final_timings  = timings_json(outcome_timings(outcome));
    const Json output_timings = timings_per_token_ ? final_timings : Json(nullptr);
    std::vector<std::string> events;
    const std::string reasoning_suffix = outcome.reasoning.substr(reasoning_.size());
    if (!reasoning_suffix.empty()) {
        if (content_started_) {
            throw std::logic_error("terminal reasoning appeared after streamed content");
        }
        events.push_back(chunk(identity_, Json{{"reasoning_content", reasoning_suffix}}, nullptr,
                               include_usage_, output_timings));
    }
    const std::string content_suffix = outcome.text.substr(content_.size());
    if (!content_suffix.empty()) {
        events.push_back(chunk(identity_, Json{{"content", content_suffix}}, nullptr,
                               include_usage_, output_timings));
    }

    if (!outcome.tool_calls.empty()) {
        const std::vector<ToolCall> calls = materialize_tool_calls(outcome.tool_calls);
        events.push_back(chunk(identity_, Json{{"tool_calls", tool_calls_json(calls, true)}},
                               nullptr, include_usage_, output_timings));
        events.push_back(chunk(identity_, Json::object(), "tool_calls", include_usage_,
                               include_usage_ ? Json(nullptr) : final_timings));
    } else {
        events.push_back(chunk(identity_, Json::object(), finish_reason(outcome.finish_reason),
                               include_usage_, include_usage_ ? Json(nullptr) : final_timings));
    }
    if (include_usage_) {
        events.push_back(usage_chunk(identity_, usage_from(outcome), final_timings));
    }
    events.emplace_back("data: [DONE]\n\n");
    return events;
}

} // namespace ninfer::serve
