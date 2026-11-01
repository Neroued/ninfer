#include "output_fixture.h"
#include "models/glm5next/frontend/prepared_prompt.h"
#include "serve/anthropic_messages.h"
#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"
#include "serve/translate.h"

#include <iostream>
#include <type_traits>

namespace {
using namespace ninfer;
using namespace ninfer::serve;
using namespace glm_fixture;

GenerationRequest responses_request(const Json& body) {
    const auto parsed = parse_openai_responses_create_request(body, {});
    OpenAIResponsesStore store(8, 1ULL << 20);
    return resolve_openai_responses_prompt(parsed.prompt, store, {}, false).generation;
}

PromptInput translate(const GenerationRequest& request, const MediaAcquirer& media = {}) {
    return to_prompt_input(request, collect_prompt_options(request, {}), media);
}

void roundtrip(const glm::Frontend& frontend, const GenerationRequest& request) {
    const auto input    = translate(request);
    const auto prepared = frontend.prepare(input);
    const auto& data    = glm::PreparedPromptAccess::view(prepared);
    require(frontend.count_tokens(input) == data.token_ids.size(), "history count differs");
    std::string history;
    for (const auto& message : input.messages) {
        for (const auto& tool : message.tool_calls) {
            require(Json::parse(tool.arguments_json) == Json{{"x", "123"}},
                    "protocol history changed argument type");
            history = tool.id;
        }
        if (message.role == ChatRole::Tool)
            require(!history.empty() && message.tool_call_id == history,
                    "protocol history lost call/result association");
    }
    require(!history.empty(), "tool history was omitted");
    require(data.output.starts_in_reasoning, "tool response did not start a new thinking turn");
}

struct Decoded {
    GenerationOutcome outcome;
    std::vector<OutputEvent> events;
};

Decoded decode(const glm::Frontend& frontend, const text::Tokenizer& tokenizer,
               const GenerationRequest& request) {
    const auto prompt  = frontend.prepare(translate(request));
    const auto options = to_request_options(request, {}, true);
    auto session       = frontend.make_output_session(prompt, options.stop, options.output,
                                                      options.execution.thinking, options.constraint,
                                                      options.tool_choice);
    const auto ids     = tokenizer.encode(
        "reason</think>answer<tool_call>call<arg_key>x</arg_key><arg_value>123</arg_value>"
            "</tool_call><|observation|>");
    Decoded decoded;
    Output aggregate;
    for (const auto id : ids) {
        (void)session.preview_model(std::span(&id, 1), 4096, FinishReason::OutputLimit);
        const auto events = session.commit_preview();
        aggregate.append(events);
        decoded.events.insert(decoded.events.end(), events.begin(), events.end());
    }
    if (const auto error = session.output_error()) std::rethrow_exception(error->exception());
    require(aggregate.reasoning == "reason" && aggregate.content == "answer" &&
                aggregate.calls.size() == 1 &&
                Json::parse(aggregate.calls[0].arguments_json) == Json{{"x", "123"}},
            "protocol request changed GLM output semantics");
    decoded.outcome = {.text                = aggregate.content,
                       .reasoning           = aggregate.reasoning,
                       .generated_token_ids = ids,
                       .tool_calls          = session.take_tool_calls(),
                       .prompt_tokens       = static_cast<int>(prompt.summary().prompt_tokens),
                       .completion_tokens   = static_cast<int>(ids.size()),
                       .reasoning_tokens    = static_cast<int>(session.reasoning_tokens()),
                       .finish_reason       = FinishReason::StopToken};
    require(decoded.outcome.tool_calls == aggregate.calls, "committed tool aggregate differs");
    return decoded;
}

Json event_json(const std::string& wire) {
    const auto data = wire.find("data: ");
    require(data != std::string::npos, "missing SSE data");
    return Json::parse(wire.substr(data + 6));
}

template <class Stream>
std::vector<Json> emit(Stream& stream, const std::vector<OutputEvent>& events) {
    std::vector<Json> wire;
    const auto append = [&](const auto& next) {
        if constexpr (std::is_same_v<std::decay_t<decltype(next)>, std::string>) {
            if (!next.empty()) wire.push_back(event_json(next));
        } else {
            for (const auto& item : next) wire.push_back(event_json(item));
        }
    };
    for (const auto& event : events)
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, OutputDelta>) {
                    if (value.channel == OutputChannel::Reasoning)
                        append(stream.reasoning_delta(value.text));
                    else
                        append(stream.content_delta(value.text));
                } else if constexpr (std::is_same_v<T, ToolCallStart>)
                    append(stream.tool_start(value));
                else if constexpr (std::is_same_v<T, ToolArgumentsDelta>)
                    append(stream.tool_arguments(value));
                else
                    append(stream.tool_end(value));
            },
            event);
    return wire;
}

void tools(const glm::Frontend& frontend, const text::Tokenizer& tokenizer) {
    auto schema             = tool_schema({{"x", {{"$ref", "#/$defs/text"}}}}, {"x"});
    schema["$defs"]["text"] = {{"enum", {"123", "456"}}};
    const Json tool         = Json::parse(declaration(schema));
    Json chat{{"model", "glm"},
              {"messages", {{{"role", "user"}, {"content", "question"}}}},
              {"tools", {tool}}};
    Json responses{
        {"model", "glm"},
        {"input", "question"},
        {"tools",
         {{{"type", "function"}, {"name", "call"}, {"parameters", schema}, {"strict", false}}}}};
    Json anthropic{{"model", "glm"},
                   {"max_tokens", 4096},
                   {"messages", chat["messages"]},
                   {"tools", {{{"name", "call"}, {"input_schema", schema}}}}};

    const auto c = decode(frontend, tokenizer, parse_chat_completion_request(chat, {}).generation);
    const auto r = decode(frontend, tokenizer, responses_request(responses));
    const auto a =
        decode(frontend, tokenizer, parse_anthropic_messages_request(anthropic, {}).generation);
    require(c.outcome.tool_calls == r.outcome.tool_calls &&
                c.outcome.tool_calls == a.outcome.tool_calls,
            "protocols produced different tool contracts");

    const auto ci = make_openai_chat_response_identity("glm");
    OpenAIChatStream cs(ci, false);
    (void)cs.start();
    std::string content, reasoning, arguments;
    for (const auto& event : emit(cs, c.events)) {
        const auto& delta = event["choices"][0]["delta"];
        content += delta.value("content", "");
        reasoning += delta.value("reasoning_content", "");
        if (delta.contains("tool_calls"))
            arguments += delta["tool_calls"][0]["function"].value("arguments", "");
    }
    require(content == "answer" && reasoning == "reason" &&
                arguments == c.outcome.tool_calls[0].arguments_json,
            "Chat SSE differs from committed output");
    const auto cf = cs.finish(c.outcome);
    require(event_json(cf[0])["choices"][0]["finish_reason"] == "tool_calls",
            "Chat finish reason differs");
    const Json cm =
        Json::parse(make_chat_completion_response(ci, c.outcome))["choices"][0]["message"];
    chat["messages"].push_back(cm);
    chat["messages"].push_back(
        {{"role", "tool"}, {"tool_call_id", cm["tool_calls"][0]["id"]}, {"content", "result"}});
    roundtrip(frontend, parse_chat_completion_request(chat, {}).generation);

    const auto rr = parse_openai_responses_create_request(responses, {});
    OpenAIResponsesEventStream rs("resp_glm", 1, rr, {});
    (void)rs.start();
    arguments.clear();
    for (const auto& event : emit(rs, r.events))
        if (event["type"] == "response.function_call_arguments.delta")
            arguments += event["delta"].get<std::string>();
    const auto rf = rs.finish(r.outcome);
    require(arguments == r.outcome.tool_calls[0].arguments_json, "Responses SSE arguments differ");
    responses["input"] = Json::array({{{"role", "user"}, {"content", "question"}}});
    std::string call_id;
    for (const auto& item : rf.response.output_items) {
        responses["input"].push_back(Json(item));
        if (item["type"] == "function_call") call_id = item["call_id"];
    }
    responses["input"].push_back(
        {{"type", "function_call_output"}, {"call_id", call_id}, {"output", "result"}});
    roundtrip(frontend, responses_request(responses));

    const auto ai = make_anthropic_response_identity("req_glm", "glm");
    AnthropicMessagesStream as(ai, a.outcome.prompt_tokens, {{"call", true}});
    (void)as.start();
    arguments.clear();
    for (const auto& event : emit(as, a.events))
        if (event["type"] == "content_block_delta" && event["delta"]["type"] == "input_json_delta")
            arguments += event["delta"]["partial_json"].get<std::string>();
    (void)as.finish(a.outcome);
    const Json am = Json::parse(make_anthropic_messages_response(ai, a.outcome));
    require(am["stop_reason"] == "tool_use" && Json::parse(arguments) == Json{{"x", "123"}},
            "Anthropic tool result differs");
    anthropic["messages"].push_back({{"role", "assistant"}, {"content", am["content"]}});
    for (const auto& item : am["content"])
        if (item["type"] == "tool_use") call_id = item["id"];
    anthropic["messages"].push_back(
        {{"role", "user"},
         {"content",
          {{{"type", "tool_result"}, {"tool_use_id", call_id}, {"content", "result"}}}}});
    roundtrip(frontend, parse_anthropic_messages_request(anthropic, {}).generation);
}

template <class Action>
void error(Action action, RequestErrorKind kind, std::string_view code, std::string_view param) {
    try {
        action();
    } catch (const RequestError& e) {
        require(e.kind() == kind, "wrong Frontend request error");
        const auto mapped = request_error_to_api_error(e);
        require(mapped.status == 400 && mapped.code == code && mapped.param == param,
                "wrong public error mapping");
        require(normalize_anthropic_error(mapped).type == "invalid_request_error",
                "Anthropic error category differs");
        return;
    }
    throw std::runtime_error("expected Frontend request error");
}

void media_and_errors(const glm::Frontend& frontend) {
    const Json chat{{"model", "glm"},
                    {"messages",
                     {{{"role", "user"},
                       {"content",
                        {{{"type", "text"}, {"text", "describe"}},
                         {{"type", "image_url"},
                          {"image_url", {{"url", "https://example.test/image.png"}}}}}}}}}};
    const Json responses{
        {"model", "glm"},
        {"input",
         {{{"role", "user"},
           {"content",
            {{{"type", "input_text"}, {"text", "describe"}},
             {{"type", "input_image"}, {"image_url", "https://example.test/image.png"}}}}}}}};
    const Json anthropic{
        {"model", "glm"},
        {"max_tokens", 4096},
        {"messages",
         {{{"role", "user"},
           {"content",
            {{{"type", "text"}, {"text", "describe"}},
             {{"type", "image"},
              {"source", {{"type", "url"}, {"url", "https://example.test/image.png"}}}}}}}}}};
    const auto file    = read(fixtures / "image-aligned.png");
    const auto acquire = [&](const ContentPart& part) {
        require(part.kind == ContentKind::Image, "media kind changed in translation");
        return OwnedMedia{.bytes = {file.begin(), file.end()}, .media_type = "image/png"};
    };
    std::uint32_t count = 0;
    for (const auto& request :
         {parse_chat_completion_request(chat, {}).generation, responses_request(responses),
          parse_anthropic_messages_request(anthropic, {}).generation}) {
        const auto input   = translate(request, acquire);
        const auto current = frontend.count_tokens(input);
        require(!count || count == current, "protocol media token counts differ");
        count            = current;
        const auto ready = frontend.prepare(input);
        require(ready.summary().prompt_tokens == count && ready.summary().has_media,
                "media counting differs from expanded prompt");
        const auto& data = glm::PreparedPromptAccess::view(ready);
        const auto cache = frontend.media_cache_summary();
        require(cache.entries == 1 &&
                    cache.live_bytes == data.media_payloads[0]->patch_elements * 2 &&
                    cache.retained_bytes == cache.live_bytes,
                "media cache public accounting differs");
        auto invalid                             = input;
        invalid.messages[0].parts[1].media.bytes = {1, 2, 3};
        error([&] { (void)frontend.prepare(invalid); }, RequestErrorKind::InvalidMedia,
              "invalid_media", "messages");
        error([&] { (void)frontend.count_tokens(invalid); }, RequestErrorKind::InvalidMedia,
              "invalid_media", "messages");
    }
    const auto cache = frontend.media_cache_summary();
    require(cache.hits >= 2 && cache.misses == 1 && cache.retained_bytes > 0 && cache.inflight == 0,
            "media cache did not retain shared preprocessing after request release");

    OpenAIResponsesStore store(8, 1ULL << 20);
    const auto count_request = parse_openai_responses_input_tokens_request(responses, {});
    const auto resolved      = resolve_openai_responses_prompt(count_request, store, {}, false);
    require(frontend.count_tokens(translate(resolved.generation, acquire)) == count &&
                frontend.count_tokens(translate(
                    parse_anthropic_count_tokens_request(anthropic).generation, acquire)) == count,
            "count-tokens endpoints changed media semantics");

    Json conflict{{"model", "glm"},
                  {"messages", {{{"role", "user"}, {"content", "question"}}}},
                  {"enable_thinking", false},
                  {"reasoning_effort", "high"}};
    const auto conflicting = translate(parse_chat_completion_request(conflict, {}).generation);
    error([&] { (void)frontend.prepare(conflicting); }, RequestErrorKind::ConflictingPromptOptions,
          "conflicting_template_option", "reasoning_effort");
    error([&] { (void)frontend.count_tokens(conflicting); },
          RequestErrorKind::ConflictingPromptOptions, "conflicting_template_option",
          "reasoning_effort");
    auto direct                              = input();
    direct.options.preserve_thinking         = true;
    direct.options.chat_template_kwargs_json = R"({"clear_thinking":true})";
    error([&] { (void)frontend.prepare(direct); }, RequestErrorKind::ConflictingPromptOptions,
          "conflicting_template_option", "clear_thinking");
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::shared_ptr<const text::Tokenizer> tokenizer;
        const auto processor =
            Json::parse(read(fixtures / "media-golden.json"))["cases"][0]["config"];
        const auto frontend = output_frontend(argc > 1 ? std::filesystem::path(argv[1])
                                                       : fixtures / "media-tokenizer.json",
                                              tokenizer, processor);
        tools(frontend, *tokenizer);
        media_and_errors(frontend);
        std::cout << "GLM Chat, Responses, Anthropic, media counts and errors passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
