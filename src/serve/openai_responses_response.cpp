#include "serve/openai_responses.h"
#include "product/constraint_observation.h"

#include "serve/generation_service.h"
#include "serve/openai_common.h"
#include "serve/tool_stream_state.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::string response_status(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::OutputLimit:
    case ninfer::FinishReason::ContextCapacity:
        return "incomplete";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    case ninfer::FinishReason::None:
    case ninfer::FinishReason::StopToken:
    case ninfer::FinishReason::StopString:
        return "completed";
    }
    return "failed";
}

struct ItemIds {
    std::string reasoning;
    std::string message;
    std::vector<std::string> function_calls;
    std::vector<std::string> call_ids;
};

void add_wire_function_identity(Json& object, const OpenAIResponsesCreateRequest& request,
                                std::string_view engine_name) {
    const auto position = request.tool_identities.find(std::string(engine_name));
    if (position == request.tool_identities.end()) {
        object["name"] = engine_name;
        return;
    }
    object["name"] = position->second.name;
    if (position->second.wire_namespace) { object["namespace"] = *position->second.wire_namespace; }
}

Json reasoning_part(std::string_view text) { return {{"type", "reasoning_text"}, {"text", text}}; }

Json message_part(std::string_view text) {
    return {{"type", "output_text"}, {"annotations", Json::array()}, {"text", text}};
}

Json reasoning_item(std::string_view id, std::string_view status, Json content) {
    return {{"id", id},
            {"type", "reasoning"},
            {"status", status},
            {"summary", Json::array()},
            {"content", std::move(content)}};
}

Json message_item(std::string_view id, std::string_view status, Json content) {
    return {{"id", id},
            {"type", "message"},
            {"status", status},
            {"role", "assistant"},
            {"content", std::move(content)}};
}

Json function_call_item(const OpenAIResponsesCreateRequest& request, std::string_view id,
                        std::string_view call_id, const GeneratedToolCall& call,
                        std::string_view status) {
    Json item{{"id", id},
              {"type", "function_call"},
              {"status", status},
              {"call_id", call_id},
              {"arguments", call.arguments_json}};
    add_wire_function_identity(item, request, call.name);
    return item;
}

Json response_common(const std::string& id, std::int64_t created_at,
                     const OpenAIResponsesCreateRequest& request,
                     const OpenAIResponsesRuntimeValues& runtime) {
    const Json reasoning = {{"effort", request.prompt.generation.reasoning_effort
                                           ? Json(requested_reasoning_effort_name(
                                                 *request.prompt.generation.reasoning_effort))
                                           : Json(nullptr)},
                            {"summary", nullptr}};
    return Json{
        {"id", id},
        {"object", "response"},
        {"created_at", created_at},
        {"background", false},
        {"instructions",
         request.prompt.instructions ? Json(*request.prompt.instructions) : Json(nullptr)},
        {"max_output_tokens", request.requested_max_output_tokens
                                  ? Json(*request.requested_max_output_tokens)
                                  : Json(nullptr)},
        {"max_tool_calls", request.max_tool_calls ? Json(*request.max_tool_calls) : Json(nullptr)},
        {"metadata", request.metadata},
        {"model", request.prompt.model},
        {"parallel_tool_calls", request.parallel_tool_calls},
        {"previous_response_id", request.prompt.previous_response_id
                                     ? Json(*request.prompt.previous_response_id)
                                     : Json(nullptr)},
        {"reasoning", reasoning},
        {"service_tier", "default"},
        {"store", request.store},
        {"temperature", runtime.temperature},
        {"text", Json{{"format", request.text_format}}},
        {"tool_choice", request.tool_choice},
        {"tools", request.tools},
        {"top_logprobs", 0},
        {"top_p", runtime.top_p},
        {"truncation", "disabled"}};
}

bool needs_message_item(const GenerationOutcome& outcome, const std::string& status) {
    if (!outcome.text.empty()) { return true; }
    if (!outcome.tool_calls.empty() || !outcome.reasoning.empty()) { return false; }
    return status == "completed";
}

BuiltOpenAIResponse build_response(const std::string& id, std::int64_t created_at,
                                   const OpenAIResponsesCreateRequest& request,
                                   const OpenAIResponsesRuntimeValues& runtime,
                                   const GenerationOutcome& outcome, ItemIds ids) {
    BuiltOpenAIResponse built;
    const std::string status      = response_status(outcome.finish_reason);
    const std::string item_status = status == "completed" ? "completed" : "incomplete";

    if (!outcome.reasoning.empty()) {
        if (ids.reasoning.empty()) { ids.reasoning = new_openai_response_item_id("rs"); }
        const char* reasoning_status = (!outcome.text.empty() || !outcome.tool_calls.empty())
                                           ? "completed"
                                           : item_status.c_str();
        built.output_items.push_back(reasoning_item(
            ids.reasoning, reasoning_status, Json::array({reasoning_part(outcome.reasoning)})));
    }

    if (needs_message_item(outcome, status)) {
        if (ids.message.empty()) { ids.message = new_openai_response_item_id("msg"); }
        const auto& message_status = outcome.tool_calls.empty() ? item_status : "completed";
        built.output_items.push_back(
            message_item(ids.message, message_status, Json::array({message_part(outcome.text)})));
    }

    ids.function_calls.resize(outcome.tool_calls.size());
    ids.call_ids.resize(outcome.tool_calls.size());
    for (std::size_t index = 0; index < outcome.tool_calls.size(); ++index) {
        if (ids.function_calls[index].empty()) {
            ids.function_calls[index] = new_openai_response_item_id("fc");
        }
        if (ids.call_ids[index].empty()) {
            ids.call_ids[index] = new_openai_response_item_id("call");
        }
        const ninfer::GeneratedToolCall& call = outcome.tool_calls[index];
        built.output_items.push_back(function_call_item(
            request, ids.function_calls[index], ids.call_ids[index], call,
            call.status == ToolCallStatus::Complete ? "completed" : "incomplete"));
    }

    if (!outcome.reasoning.empty() || !outcome.text.empty() || !outcome.tool_calls.empty() ||
        status == "completed") {
        ChatTurn history;
        history.role              = ChatRole::Assistant;
        history.reasoning_content = outcome.reasoning;
        history.tool_calls.reserve(outcome.tool_calls.size());
        for (std::size_t index = 0; index < outcome.tool_calls.size(); ++index) {
            const ninfer::GeneratedToolCall& call = outcome.tool_calls[index];
            if (call.status != ToolCallStatus::Complete) continue;
            history.tool_calls.push_back(ToolCall{.id             = ids.call_ids[index],
                                                  .name           = call.name,
                                                  .arguments_json = call.arguments_json});
        }
        if (!outcome.text.empty()) {
            ContentPart part;
            part.kind     = ContentKind::Text;
            part.type_raw = "output_text";
            part.text     = outcome.text;
            history.content.push_back(std::move(part));
        }
        if (!history.reasoning_content.empty() || !history.content.empty() ||
            !history.tool_calls.empty() || status == "completed")
            built.output_history.push_back(std::move(history));
    }

    Json response            = response_common(id, created_at, request, runtime);
    response["status"]       = status;
    response["completed_at"] = status == "completed" ? Json(unix_time_now()) : Json(nullptr);
    response["error"]        = nullptr;
    response["output"]       = built.output_items;
    response["incomplete_details"] =
        status == "incomplete" ? Json{{"reason", "max_output_tokens"}} : Json(nullptr);
    const int observed_cached =
        std::max(runtime.cached_input_tokens, static_cast<int>(outcome.stats.reused_prompt_tokens));
    const int cached_tokens = std::clamp(observed_cached, 0, outcome.prompt_tokens);
    response["usage"] =
        Json{{"input_tokens", outcome.prompt_tokens},
             {"input_tokens_details", Json{{"cached_tokens", cached_tokens}}},
             {"output_tokens", outcome.completion_tokens},
             {"output_tokens_details", Json{{"reasoning_tokens", outcome.reasoning_tokens}}},
             {"total_tokens", outcome.prompt_tokens + outcome.completion_tokens}};
    built.body = std::move(response);
    if (outcome.stats.constraint)
        built.body["constraint"] = product::constraint_observation_json(outcome.stats.constraint);
    return built;
}

std::string sse(const Json& event) {
    return "event: " + event.at("type").get<std::string>() + "\n" + "data: " + event.dump() +
           "\n\n";
}

Json in_progress_response(const std::string& id, std::int64_t created_at,
                          const OpenAIResponsesCreateRequest& request,
                          const OpenAIResponsesRuntimeValues& runtime) {
    Json response                  = response_common(id, created_at, request, runtime);
    response["status"]             = "in_progress";
    response["completed_at"]       = nullptr;
    response["error"]              = nullptr;
    response["incomplete_details"] = nullptr;
    response["output"]             = Json::array();
    response["usage"]              = nullptr;
    return response;
}

} // namespace

BuiltOpenAIResponse make_openai_response_object(const std::string& id, std::int64_t created_at,
                                                const OpenAIResponsesCreateRequest& request,
                                                const OpenAIResponsesRuntimeValues& runtime,
                                                const GenerationOutcome& outcome) {
    return build_response(id, created_at, request, runtime, outcome, {});
}

std::string make_openai_response_input_tokens_body(int input_tokens) {
    return Json{{"object", "response.input_tokens"}, {"input_tokens", input_tokens}}.dump();
}

class OpenAIResponsesEventStream::Impl {
public:
    Impl(std::string response_id, std::int64_t created_at_, OpenAIResponsesCreateRequest request_,
         OpenAIResponsesRuntimeValues runtime_)
        : id(std::move(response_id)), created_at(created_at_), request(std::move(request_)),
          runtime(runtime_) {}

    Json event(std::string type, Json fields = Json::object()) {
        fields["type"]            = std::move(type);
        fields["sequence_number"] = sequence++;
        return fields;
    }

    std::vector<std::string> ensure_reasoning() {
        if (reasoning_started) { return {}; }
        reasoning_started = true;
        ids.reasoning     = new_openai_response_item_id("rs");
        reasoning_index   = next_output_index++;
        const Json item   = reasoning_item(ids.reasoning, "in_progress", Json::array());
        const Json part   = reasoning_part("");
        return {sse(event("response.output_item.added",
                          Json{{"output_index", reasoning_index}, {"item", item}})),
                sse(event("response.content_part.added", Json{{"item_id", ids.reasoning},
                                                              {"output_index", reasoning_index},
                                                              {"content_index", 0},
                                                              {"part", part}}))};
    }

    std::vector<std::string> close_reasoning(const std::string& final_text,
                                             const char* item_status = "completed") {
        if (!reasoning_started || reasoning_done) { return {}; }
        reasoning_done   = true;
        reasoning_status = item_status;
        reasoning_text   = final_text;
        const Json part  = reasoning_part(reasoning_text);
        const Json item  = reasoning_item(ids.reasoning, item_status, Json::array({part}));
        return {sse(event("response.reasoning_text.done", Json{{"item_id", ids.reasoning},
                                                               {"output_index", reasoning_index},
                                                               {"content_index", 0},
                                                               {"text", reasoning_text}})),
                sse(event("response.content_part.done", Json{{"item_id", ids.reasoning},
                                                             {"output_index", reasoning_index},
                                                             {"content_index", 0},
                                                             {"part", part}})),
                sse(event("response.output_item.done",
                          Json{{"output_index", reasoning_index}, {"item", item}}))};
    }

    std::vector<std::string> ensure_message() {
        if (message_started) { return {}; }
        message_started = true;
        ids.message     = new_openai_response_item_id("msg");
        message_index   = next_output_index++;
        const Json item = message_item(ids.message, "in_progress", Json::array());
        const Json part = message_part("");
        return {sse(event("response.output_item.added",
                          Json{{"output_index", message_index}, {"item", item}})),
                sse(event("response.content_part.added", Json{{"item_id", ids.message},
                                                              {"output_index", message_index},
                                                              {"content_index", 0},
                                                              {"part", part}}))};
    }

    std::vector<std::string> close_message(const std::string& final_text,
                                           const char* item_status = "completed") {
        if (!message_started || message_done) { return {}; }
        message_done    = true;
        message_status  = item_status;
        content_text    = final_text;
        const Json part = message_part(content_text);
        const Json item = message_item(ids.message, item_status, Json::array({part}));
        return {sse(event("response.output_text.done", Json{{"item_id", ids.message},
                                                            {"output_index", message_index},
                                                            {"content_index", 0},
                                                            {"text", content_text},
                                                            {"logprobs", Json::array()}})),
                sse(event("response.content_part.done", Json{{"item_id", ids.message},
                                                             {"output_index", message_index},
                                                             {"content_index", 0},
                                                             {"part", part}})),
                sse(event("response.output_item.done",
                          Json{{"output_index", message_index}, {"item", item}}))};
    }

    Json output_snapshot() const {
        Json output = Json::array();
        for (int i = 0; i < next_output_index; ++i) output.push_back(nullptr);
        if (reasoning_started)
            output[reasoning_index] =
                reasoning_item(ids.reasoning, reasoning_done ? reasoning_status : "incomplete",
                               Json::array({reasoning_part(reasoning_text)}));
        if (message_started)
            output[message_index] =
                message_item(ids.message, message_done ? message_status : "incomplete",
                             Json::array({message_part(content_text)}));
        for (std::size_t i = 0; i < tools.calls().size(); ++i) {
            const auto& call        = tools.at(i);
            output[tool_indices[i]] = function_call_item(
                request, ids.function_calls[i], ids.call_ids[i], call,
                call.status == ToolCallStatus::Complete ? "completed" : "incomplete");
        }
        return output;
    }

    std::string id;
    std::int64_t created_at = 0;
    OpenAIResponsesCreateRequest request;
    OpenAIResponsesRuntimeValues runtime;
    std::uint64_t sequence = 0;
    int next_output_index  = 0;
    int reasoning_index    = -1;
    int message_index      = -1;
    bool started           = false;
    bool reasoning_started = false;
    bool reasoning_done    = false;
    bool message_started   = false;
    bool message_done      = false;
    bool finish_built      = false;
    bool terminal_emitted  = false;
    std::string reasoning_status;
    std::string message_status;
    std::string reasoning_text;
    std::string content_text;
    ToolStreamState tools;
    std::vector<int> tool_indices;
    ItemIds ids;
};

OpenAIResponsesEventStream::OpenAIResponsesEventStream(std::string response_id,
                                                       std::int64_t created_at,
                                                       OpenAIResponsesCreateRequest request,
                                                       OpenAIResponsesRuntimeValues runtime)
    : impl_(std::make_unique<Impl>(std::move(response_id), created_at, std::move(request),
                                   runtime)) {}

OpenAIResponsesEventStream::~OpenAIResponsesEventStream() = default;
OpenAIResponsesEventStream::OpenAIResponsesEventStream(OpenAIResponsesEventStream&&) noexcept =
    default;
OpenAIResponsesEventStream&
OpenAIResponsesEventStream::operator=(OpenAIResponsesEventStream&&) noexcept = default;

std::vector<std::string> OpenAIResponsesEventStream::start() {
    if (impl_->started) { throw std::logic_error("Responses event stream already started"); }
    impl_->started = true;
    const Json response =
        in_progress_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime);
    return {sse(impl_->event("response.created", Json{{"response", response}})),
            sse(impl_->event("response.in_progress", Json{{"response", response}}))};
}

std::vector<std::string> OpenAIResponsesEventStream::reasoning_delta(const std::string& text) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid reasoning delta event state");
    }
    if (text.empty()) { return {}; }
    std::vector<std::string> events = impl_->ensure_reasoning();
    impl_->reasoning_text += text;
    events.push_back(sse(
        impl_->event("response.reasoning_text.delta", Json{{"item_id", impl_->ids.reasoning},
                                                           {"output_index", impl_->reasoning_index},
                                                           {"content_index", 0},
                                                           {"delta", text}})));
    return events;
}

std::vector<std::string> OpenAIResponsesEventStream::content_delta(const std::string& text) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid content delta event state");
    }
    if (text.empty()) { return {}; }
    std::vector<std::string> events = impl_->close_reasoning(impl_->reasoning_text);
    std::vector<std::string> added  = impl_->ensure_message();
    events.insert(events.end(), std::make_move_iterator(added.begin()),
                  std::make_move_iterator(added.end()));
    impl_->content_text += text;
    events.push_back(
        sse(impl_->event("response.output_text.delta", Json{{"item_id", impl_->ids.message},
                                                            {"output_index", impl_->message_index},
                                                            {"content_index", 0},
                                                            {"delta", text},
                                                            {"logprobs", Json::array()}})));
    return events;
}

std::vector<std::string>
OpenAIResponsesEventStream::tool_start(const ninfer::ToolCallStart& value) {
    if (!impl_->started || impl_->finish_built)
        throw std::logic_error("invalid Responses tool start state");
    impl_->tools.start(value);
    auto events = impl_->close_reasoning(impl_->reasoning_text);
    auto closed = impl_->close_message(impl_->content_text);
    events.insert(events.end(), std::make_move_iterator(closed.begin()),
                  std::make_move_iterator(closed.end()));
    const auto item_id = new_openai_response_item_id("fc");
    const auto call_id = new_openai_response_item_id("call");
    impl_->ids.function_calls.push_back(item_id);
    impl_->ids.call_ids.push_back(call_id);
    const int output_index = impl_->next_output_index++;
    impl_->tool_indices.push_back(output_index);
    const Json item = function_call_item(impl_->request, item_id, call_id,
                                         impl_->tools.at(value.index), "in_progress");
    events.push_back(sse(impl_->event("response.output_item.added",
                                      Json{{"output_index", output_index}, {"item", item}})));
    return events;
}

std::vector<std::string>
OpenAIResponsesEventStream::tool_arguments(const ninfer::ToolArgumentsDelta& value) {
    if (!impl_->started || impl_->finish_built)
        throw std::logic_error("invalid Responses tool delta state");
    impl_->tools.append(value);
    if (value.text.empty()) return {};
    return {sse(impl_->event("response.function_call_arguments.delta",
                             Json{{"item_id", impl_->ids.function_calls.at(value.index)},
                                  {"output_index", impl_->tool_indices.at(value.index)},
                                  {"delta", value.text}}))};
}

std::vector<std::string> OpenAIResponsesEventStream::tool_end(const ninfer::ToolCallEnd& value) {
    if (!impl_->started || impl_->finish_built)
        throw std::logic_error("invalid Responses tool end state");
    impl_->tools.end(value);
    const auto& call       = impl_->tools.at(value.index);
    const auto& item_id    = impl_->ids.function_calls.at(value.index);
    const int output_index = impl_->tool_indices.at(value.index);
    Json arguments{
        {"item_id", item_id}, {"output_index", output_index}, {"arguments", call.arguments_json}};
    add_wire_function_identity(arguments, impl_->request, call.name);
    const Json item =
        function_call_item(impl_->request, item_id, impl_->ids.call_ids.at(value.index), call,
                           value.status == ToolCallStatus::Complete ? "completed" : "incomplete");
    return {sse(impl_->event("response.function_call_arguments.done", std::move(arguments))),
            sse(impl_->event("response.output_item.done",
                             Json{{"output_index", output_index}, {"item", item}}))};
}

OpenAIResponsesStreamFinish OpenAIResponsesEventStream::finish(const GenerationOutcome& outcome) {
    if (!impl_->started || impl_->finish_built) {
        throw std::logic_error("invalid Responses stream finish state");
    }
    impl_->finish_built = true;
    OpenAIResponsesStreamFinish finished;
    const std::string status = response_status(outcome.finish_reason);
    const char* item_status  = status == "completed" ? "completed" : "incomplete";

    auto append = [&](std::vector<std::string> events) {
        finished.events_before_terminal.insert(finished.events_before_terminal.end(),
                                               std::make_move_iterator(events.begin()),
                                               std::make_move_iterator(events.end()));
    };
    if (!outcome.reasoning.empty() && !impl_->reasoning_started) {
        append(impl_->ensure_reasoning());
        impl_->reasoning_text = outcome.reasoning;
        finished.events_before_terminal.push_back(sse(impl_->event(
            "response.reasoning_text.delta", Json{{"item_id", impl_->ids.reasoning},
                                                  {"output_index", impl_->reasoning_index},
                                                  {"content_index", 0},
                                                  {"delta", outcome.reasoning}})));
    }
    const char* reasoning_status =
        (!outcome.text.empty() || !outcome.tool_calls.empty()) ? "completed" : item_status;
    append(impl_->close_reasoning(outcome.reasoning, reasoning_status));

    if (needs_message_item(outcome, status)) {
        append(impl_->ensure_message());
        if (outcome.text != impl_->content_text) {
            if (!outcome.text.starts_with(impl_->content_text)) {
                throw std::logic_error("streamed content does not match terminal content");
            }
            const std::string suffix = outcome.text.substr(impl_->content_text.size());
            if (!suffix.empty()) {
                impl_->content_text += suffix;
                finished.events_before_terminal.push_back(sse(impl_->event(
                    "response.output_text.delta", Json{{"item_id", impl_->ids.message},
                                                       {"output_index", impl_->message_index},
                                                       {"content_index", 0},
                                                       {"delta", suffix},
                                                       {"logprobs", Json::array()}})));
            }
        }
        append(impl_->close_message(outcome.text,
                                    outcome.tool_calls.empty() ? item_status : "completed"));
    }

    impl_->tools.verify(outcome.tool_calls);

    finished.response = build_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime,
                                       outcome, impl_->ids);
    return finished;
}

std::string OpenAIResponsesEventStream::terminal(const BuiltOpenAIResponse& response) {
    if (!impl_->finish_built || impl_->terminal_emitted) {
        throw std::logic_error("invalid Responses terminal event state");
    }
    impl_->terminal_emitted  = true;
    const std::string status = response.body.at("status").get<std::string>();
    const std::string type   = status == "completed"    ? "response.completed"
                               : status == "incomplete" ? "response.incomplete"
                                                        : "response.failed";
    return sse(impl_->event(type, Json{{"response", response.body}}));
}

std::string OpenAIResponsesEventStream::failed(const ApiError& error) {
    if (!impl_->started || impl_->terminal_emitted) {
        throw std::logic_error("invalid Responses failed event state");
    }
    impl_->terminal_emitted = true;
    Json response =
        in_progress_response(impl_->id, impl_->created_at, impl_->request, impl_->runtime);
    response["status"]       = "failed";
    response["output"]       = impl_->output_snapshot();
    response["completed_at"] = nullptr;
    response["error"]        = Json{{"code", error.code.empty() ? Json(nullptr) : Json(error.code)},
                                    {"message", error.message}};
    return sse(impl_->event("response.failed", Json{{"response", response}}));
}

} // namespace ninfer::serve
