// Contract tests for the implemented OpenAI Responses Core surface: typed
// input Items, explicit capability rejection, terminal objects, and semantic
// SSE event sequencing.

#include "serve/generation_service.h"
#include "serve/responses_schema.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
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

ninfer::PromptCapabilities effort_capabilities() {
    ninfer::PromptCapabilities capabilities;
    capabilities.enable_thinking                 = true;
    capabilities.reasoning_effort.low            = true;
    capabilities.reasoning_effort.medium         = true;
    capabilities.reasoning_effort.xhigh          = true;
    capabilities.reasoning_effort.default_effort = ninfer::ReasoningEffort::XHigh;
    return capabilities;
}

bool throws_api(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const ApiException&) { return true; } catch (...) {
        return false;
    }
    return false;
}

std::string api_code(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

Json parse_event(const std::string& event) {
    const std::size_t newline = event.find('\n');
    if (!event.starts_with("event: ") || newline == std::string::npos || !event.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    const std::string type   = event.substr(7, newline - 7);
    const std::string prefix = "data: ";
    if (event.compare(newline + 1, prefix.size(), prefix) != 0) {
        throw std::runtime_error("missing SSE data field");
    }
    const std::size_t begin = newline + 1 + prefix.size();
    Json payload            = Json::parse(event.substr(begin, event.size() - begin - 2));
    if (payload.at("type") != type) { throw std::runtime_error("SSE type mismatch"); }
    return payload;
}

int test_basic_request() {
    const Json body                = {{"model", "qwen3.6-27b"},
                                      {"input", "hello"},
                                      {"instructions", "be concise"},
                                      {"previous_response_id", "resp_previous"},
                                      {"max_output_tokens", 64},
                                      {"temperature", 0.3},
                                      {"top_p", 0.8},
                                      {"reasoning", Json{{"effort", "medium"}}},
                                      {"metadata", Json{{"trace", "abc"}}}};
    const ResponsesRequest request = parse_responses_request(body, limits());
    int failures                   = 0;
    failures += check(request.generation.model == "qwen3.6-27b", "model parsed");
    failures += check(request.input_turns.size() == 1 &&
                          request.input_turns[0].role == ninfer::ChatRole::User &&
                          request.input_turns[0].content[0].text == "hello",
                      "string input normalized to a user turn");
    failures += check(request.input_items[0].at("type") == "message" &&
                          request.input_items[0].at("content")[0].at("type") == "input_text" &&
                          request.input_items[0].contains("id"),
                      "canonical input Item built with an id");
    failures += check(request.instructions && *request.instructions == "be concise",
                      "instructions parsed separately");
    failures +=
        check(request.previous_response_id && *request.previous_response_id == "resp_previous",
              "previous response id parsed");
    failures += check(request.generation.max_tokens == 64 && request.generation.max_tokens_set,
                      "max_output_tokens reaches generation request");
    failures += check(request.generation.reasoning_effort == RequestedReasoningEffort::Medium,
                      "medium reasoning effort was not parsed");
    failures += check(request.store && !request.stream, "Responses defaults applied");
    ResponsesRequest composed = request;
    ChatTurn previous;
    previous.role = ninfer::ChatRole::Assistant;
    ContentPart previous_text;
    previous_text.kind = ContentKind::Text;
    previous_text.text = "old answer";
    previous.content.push_back(std::move(previous_text));
    compose_responses_generation_messages(composed, {previous});
    failures += check(composed.generation.messages.size() == 3 &&
                          composed.generation.messages[0].role == ninfer::ChatRole::Developer &&
                          composed.generation.messages[1].content[0].text == "old answer" &&
                          composed.generation.messages[2].content[0].text == "hello",
                      "instructions, previous context, and current input composed in order");
    return failures;
}

int test_instruction_message_order() {
    ResponsesRequest request = parse_responses_request(
        Json{{"model", "m"},
             {"input",
              Json::array(
                  {Json{{"type", "message"}, {"role", "system"}, {"content", "initial"}},
                   Json{{"type", "message"}, {"role", "user"}, {"content", "work"}},
                   Json{{"type", "message"}, {"role", "developer"}, {"content", "current"}}})},
             {"max_output_tokens", 32}},
        limits());
    int failures = check(request.input_turns.size() == 3 &&
                             request.input_turns[0].role == ninfer::ChatRole::System &&
                             request.input_turns[1].role == ninfer::ChatRole::User &&
                             request.input_turns[2].role == ninfer::ChatRole::Developer,
                         "Responses input instruction roles were lowered or reordered");
    compose_responses_generation_messages(request, {});
    failures += check(request.generation.messages.size() == 3 &&
                          request.generation.messages[0].role == ninfer::ChatRole::System &&
                          request.generation.messages[1].role == ninfer::ChatRole::User &&
                          request.generation.messages[2].role == ninfer::ChatRole::Developer,
                      "Responses composition changed input instruction order");
    return failures;
}

int test_reasoning_effort() {
    const Json base = {{"model", "m"}, {"input", "hello"}, {"max_output_tokens", 32}};
    int failures    = 0;

    for (const auto& [wire, expected] :
         std::array<std::pair<const char*, RequestedReasoningEffort>, 7>{
             {{"none", RequestedReasoningEffort::None},
              {"minimal", RequestedReasoningEffort::Minimal},
              {"low", RequestedReasoningEffort::Low},
              {"medium", RequestedReasoningEffort::Medium},
              {"high", RequestedReasoningEffort::High},
              {"xhigh", RequestedReasoningEffort::XHigh},
              {"max", RequestedReasoningEffort::Max}}}) {
        Json body         = base;
        body["reasoning"] = Json{{"effort", wire}};
        failures +=
            check(parse_responses_request(body, limits()).generation.reasoning_effort == expected,
                  std::string("Responses did not accept protocol effort ") + wire);
    }

    Json low                            = base;
    low["reasoning"]                    = Json{{"effort", "low"}};
    const GenerationRequest low_request = parse_responses_request(low, limits()).generation;
    const ResolvedPromptSemantics low_semantics =
        resolve_prompt_semantics(low_request, ServeOptions{}, effort_capabilities());
    failures += check(low_semantics.enable_thinking &&
                          low_semantics.reasoning_effort == ninfer::ReasoningEffort::Low,
                      "Responses low effort did not resolve through template capabilities");

    Json none                                    = base;
    none["reasoning"]                            = Json{{"effort", "none"}};
    const ResolvedPromptSemantics none_semantics = resolve_prompt_semantics(
        parse_responses_request(none, limits()).generation, ServeOptions{}, effort_capabilities());
    failures += check(!none_semantics.enable_thinking && !none_semantics.reasoning_effort,
                      "Responses none effort did not disable thinking");

    Json high                            = base;
    high["reasoning"]                    = Json{{"effort", "high"}};
    const GenerationRequest high_request = parse_responses_request(high, limits()).generation;
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(high_request, ServeOptions{},
                                                         effort_capabilities());
                      }) == "reasoning_effort_not_supported",
                      "Responses high effort bypassed template capability validation");

    Json invalid         = base;
    invalid["reasoning"] = Json{{"effort", "ultra"}};
    failures += check(throws_api([&] { (void)parse_responses_request(invalid, limits()); }),
                      "unknown Responses reasoning effort was accepted");
    invalid["reasoning"] = Json{{"effort", 1}};
    failures += check(throws_api([&] { (void)parse_responses_request(invalid, limits()); }),
                      "non-string Responses reasoning effort was accepted");
    return failures;
}

int test_preserve_thinking_options_and_inheritance() {
    const Json base = {{"model", "m"}, {"input", "hello"}};
    int failures    = 0;

    Json kwargs                    = base;
    kwargs["chat_template_kwargs"] = Json{{"preserve_thinking", true}};
    ResponsesRequest request       = parse_responses_request(kwargs, limits());
    failures += check(request.generation.preserve_thinking == true,
                      "Responses chat_template_kwargs preserve_thinking parsed");

    ResponsesRequest inherited = parse_responses_request(base, limits());
    inherit_responses_preserve_thinking(inherited, true);
    failures += check(inherited.generation.preserve_thinking == true &&
                          !inherited.generation.preserve_thinking_semantic_change,
                      "Responses child did not inherit parent preserve_thinking");

    ResponsesRequest unchanged = request;
    inherit_responses_preserve_thinking(unchanged, true);
    failures += check(!unchanged.generation.preserve_thinking_semantic_change,
                      "equal explicit preserve value marked a semantic change");

    Json explicit_false                 = base;
    explicit_false["preserve_thinking"] = false;
    ResponsesRequest changed            = parse_responses_request(explicit_false, limits());
    inherit_responses_preserve_thinking(changed, true);
    failures += check(changed.generation.preserve_thinking == false &&
                          changed.generation.preserve_thinking_semantic_change,
                      "explicit Responses preserve branch was not marked");

    Json conflict                 = kwargs;
    conflict["preserve_thinking"] = false;
    failures += check(api_code([&] { (void)parse_responses_request(conflict, limits()); }) ==
                          "conflicting_template_option",
                      "Responses conflicting preserve values were accepted");

    Json unknown                    = base;
    unknown["chat_template_kwargs"] = Json{{"unknown", 1}};
    failures += check(api_code([&] { (void)parse_responses_request(unknown, limits()); }) ==
                          "chat_template_option_not_supported",
                      "Responses unknown chat template option was accepted");
    return failures;
}

int test_typed_items_and_tools() {
    const Json function = {{"type", "function"},
                           {"name", "weather"},
                           {"description", "Get weather"},
                           {"parameters", Json{{"type", "object"}, {"properties", Json::object()}}},
                           {"strict", false}};
    const Json body     = {
        {"model", "qwen3.6-27b"},
        {"input",
             Json::array({Json{{"id", "rs_old"},
                               {"type", "reasoning"},
                               {"summary", Json::array()},
                               {"content", Json::array({Json{{"type", "reasoning_text"},
                                                             {"text", "need tools"}}})}},
                          Json{{"id", "fc_old_1"},
                               {"type", "function_call"},
                               {"call_id", "call_1"},
                               {"name", "weather"},
                               {"arguments", R"({"city":"Paris"})"}},
                          Json{{"id", "fc_old_2"},
                               {"type", "function_call"},
                               {"call_id", "call_2"},
                               {"name", "weather"},
                               {"arguments", R"({"city":"Rome"})"}},
                          Json{{"id", "fco_old"},
                               {"type", "function_call_output"},
                               {"call_id", "call_1"},
                               {"output", R"({"temp":20})"}},
                          Json{{"type", "message"},
                               {"role", "user"},
                               {"content",
                                Json::array({Json{{"type", "input_image"},
                                                  {"image_url", "data:image/png;base64,AA=="},
                                                  {"detail", "auto"}},
                                             Json{{"type", "input_text"}, {"text", "describe"}}})}}})},
        {"tools", Json::array({function})},
        {"tool_choice", "auto"},
        {"parallel_tool_calls", true},
        {"max_output_tokens", 32}};

    const ResponsesRequest request = parse_responses_request(body, limits());
    int failures                   = 0;
    failures += check(request.input_turns.size() == 3, "typed Items grouped into three turns");
    failures += check(request.input_turns[0].role == ninfer::ChatRole::Assistant &&
                          request.input_turns[0].reasoning_content == "need tools" &&
                          request.input_turns[0].tool_calls.size() == 2,
                      "reasoning and adjacent function calls grouped into one assistant turn");
    failures += check(request.input_turns[1].role == ninfer::ChatRole::Tool &&
                          request.input_turns[1].tool_call_id == "call_1",
                      "function output translated to tool turn");
    failures += check(request.input_turns[2].content[0].kind == ContentKind::Image,
                      "input image translated to media part");
    failures += check(request.generation.tools.size() == 1 &&
                          request.generation.tools[0].name == "weather" &&
                          !request.generation.tools[0].strict,
                      "flat Responses function converted to internal tool");
    const Json nested = Json::parse(request.generation.tools[0].definition_json);
    failures += check(nested.at("function").at("name") == "weather",
                      "Qwen prompt receives normalized nested function definition");
    return failures;
}

int test_explicit_rejections() {
    const Json base = {{"model", "qwen3.6-27b"}, {"input", "hello"}, {"max_output_tokens", 32}};
    int failures    = 0;

    Json strict     = base;
    strict["tools"] = Json::array({Json{
        {"type", "function"}, {"name", "f"}, {"parameters", Json::object()}, {"strict", true}}});
    failures += check(api_code([&] { (void)parse_responses_request(strict, limits()); }) ==
                          "strict_tools_not_supported",
                      "strict tools rejected explicitly");

    Json required           = base;
    required["tools"]       = Json::array({Json{{"type", "function"}, {"name", "f"}}});
    required["tool_choice"] = "required";
    failures += check(api_code([&] { (void)parse_responses_request(required, limits()); }) ==
                          "tool_choice_not_supported",
                      "required tool choice rejected explicitly");

    Json structured    = base;
    structured["text"] = Json{{"format", Json{{"type", "json_schema"}}}};
    failures += check(api_code([&] { (void)parse_responses_request(structured, limits()); }) ==
                          "structured_outputs_not_supported",
                      "structured output rejected");

    Json background          = base;
    background["background"] = true;
    failures += check(api_code([&] { (void)parse_responses_request(background, limits()); }) ==
                          "background_not_supported",
                      "background rejected");

    Json unknown       = base;
    unknown["made_up"] = 1;
    failures += check(api_code([&] { (void)parse_responses_request(unknown, limits()); }) ==
                          "unknown_parameter",
                      "unknown parameter rejected");

    Json too_small                 = base;
    too_small["max_output_tokens"] = 15;
    failures += check(api_code([&] { (void)parse_responses_request(too_small, limits()); }) ==
                          "invalid_value",
                      "OpenAI minimum max_output_tokens enforced");
    return failures;
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

int test_response_object() {
    ResponsesRequest request = parse_responses_request(Json{{"model", "qwen3.6-27b"},
                                                            {"input", "hello"},
                                                            {"max_output_tokens", 32},
                                                            {"reasoning", Json{{"effort", "low"}}},
                                                            {"store", false}},
                                                       limits());
    ResponsesRuntimeValues runtime;
    runtime.temperature = 0.6F;
    runtime.top_p       = 0.95F;
    const BuiltResponse built =
        make_response_object("resp_test", 123, request, runtime, sample_outcome());
    const Json& response = built.body;
    int failures         = 0;
    failures += check(response.at("object") == "response" && response.at("status") == "completed",
                      "completed response envelope");
    failures += check(response.at("output").size() == 2 &&
                          response.at("output")[0].at("type") == "reasoning" &&
                          response.at("output")[1].at("type") == "message",
                      "typed reasoning and message output Items");
    failures += check(!response.contains("output_text"),
                      "SDK-only output_text helper absent from wire body");
    failures += check(response.at("reasoning").at("effort") == "low",
                      "Responses object did not echo reasoning effort");
    failures +=
        check(response.at("usage").at("input_tokens_details").at("cached_tokens") == 4 &&
                  response.at("usage").at("output_tokens_details").at("reasoning_tokens") == 3 &&
                  response.at("usage").at("total_tokens") == 18,
              "Responses usage details serialized");
    failures += check(built.output_history.size() == 1 &&
                          built.output_history[0].reasoning_content == "thought" &&
                          built.output_history[0].content[0].text == "answer",
                      "terminal output converted to continuation history");

    GenerationOutcome incomplete = sample_outcome();
    incomplete.finish_reason     = ninfer::FinishReason::OutputLimit;
    const Json limited = make_response_object("resp_limit", 123, request, runtime, incomplete).body;
    failures += check(limited.at("status") == "incomplete" &&
                          limited.at("incomplete_details").at("reason") == "max_output_tokens",
                      "output limit mapped to incomplete response");

    GenerationOutcome tools = sample_outcome();
    tools.text.clear();
    tools.tool_calls.push_back(ToolCall{"call_weather", "weather", R"({"city":"Paris"})"});
    const Json tool_response = make_response_object("resp_tool", 123, request, runtime, tools).body;
    failures += check(tool_response.at("output").back().at("type") == "function_call" &&
                          tool_response.at("output").back().at("call_id") == "call_weather" &&
                          !tool_response.at("output").back().at("id").get<std::string>().empty(),
                      "function call has distinct Item id and call_id");
    return failures;
}

int test_sse_sequence() {
    ResponsesRequest request =
        parse_responses_request(Json{{"model", "qwen3.6-27b"},
                                     {"input", "hello"},
                                     {"instructions", "be concise"},
                                     {"max_output_tokens", 32},
                                     {"metadata", Json{{"trace", "original"}}},
                                     {"stream", true}},
                                limits());
    ResponsesEventStream encoder("resp_stream", 123, request, {});
    request.generation.model      = "mutated";
    request.instructions          = "mutated";
    request.metadata              = Json{{"trace", "mutated"}};
    std::vector<std::string> wire = encoder.start();
    std::vector<std::string> more = encoder.reasoning_delta("thought");
    wire.insert(wire.end(), more.begin(), more.end());
    more = encoder.content_delta("ans");
    wire.insert(wire.end(), more.begin(), more.end());
    GenerationOutcome outcome    = sample_outcome();
    ResponsesStreamFinish finish = encoder.finish(outcome);
    wire.insert(wire.end(), finish.events_before_terminal.begin(),
                finish.events_before_terminal.end());
    wire.push_back(encoder.terminal(finish.response));

    int failures                    = 0;
    std::uint64_t expected_sequence = 0;
    std::string text_deltas;
    bool saw_reasoning_delta = false;
    for (const std::string& event : wire) {
        failures += check(event.find("[DONE]") == std::string::npos,
                          "Responses stream must not emit Chat [DONE]");
        const Json payload = parse_event(event);
        failures += check(payload.at("sequence_number") == expected_sequence++,
                          "sequence_number is contiguous");
        if (payload.at("type") == "response.output_text.delta") {
            text_deltas += payload.at("delta").get<std::string>();
        }
        if (payload.at("type") == "response.reasoning_text.delta") { saw_reasoning_delta = true; }
    }
    failures += check(parse_event(wire.front()).at("type") == "response.created",
                      "stream starts with response.created");
    const Json created = parse_event(wire.front()).at("response");
    failures +=
        check(created.at("model") == "qwen3.6-27b" && created.at("instructions") == "be concise" &&
                  created.at("metadata").at("trace") == "original",
              "stream owns stable response fields");
    failures += check(parse_event(wire.back()).at("type") == "response.completed",
                      "stream ends with response.completed");
    failures += check(text_deltas == "answer", "text deltas reconstruct terminal output");
    failures += check(saw_reasoning_delta, "raw reasoning delta emitted");
    return failures;
}

int test_sse_function_call() {
    ResponsesRequest request = parse_responses_request(Json{{"model", "qwen3.6-27b"},
                                                            {"input", "weather"},
                                                            {"max_output_tokens", 32},
                                                            {"stream", true}},
                                                       limits());
    ResponsesEventStream encoder("resp_tool_stream", 123, request, {});
    std::vector<std::string> wire = encoder.start();
    GenerationOutcome outcome;
    outcome.prompt_tokens     = 8;
    outcome.completion_tokens = 4;
    outcome.finish_reason     = ninfer::FinishReason::StopToken;
    outcome.tool_calls.push_back(ToolCall{"call_weather", "weather", R"({"city":"Paris"})"});
    ResponsesStreamFinish finish = encoder.finish(outcome);
    wire.insert(wire.end(), finish.events_before_terminal.begin(),
                finish.events_before_terminal.end());
    wire.push_back(encoder.terminal(finish.response));

    int failures = 0;
    std::string arguments;
    std::string item_id;
    for (const std::string& event : wire) {
        const Json payload = parse_event(event);
        if (payload.at("type") == "response.function_call_arguments.delta") {
            arguments += payload.at("delta").get<std::string>();
            item_id = payload.at("item_id").get<std::string>();
        }
    }
    failures +=
        check(arguments == R"({"city":"Paris"})", "function argument deltas reconstruct arguments");
    const Json& item = finish.response.body.at("output").at(0);
    failures += check(item.at("type") == "function_call" && item.at("id") == item_id &&
                          item.at("call_id") == "call_weather" && item_id != "call_weather",
                      "function stream preserves distinct stable Item id and call_id");
    return failures;
}

int test_input_tokens_schema() {
    const ResponsesRequest request = parse_response_input_tokens_request(
        Json{{"model", "qwen3.6-27b"}, {"input", "hello"}}, limits());
    int failures = 0;
    failures += check(!request.store && !request.stream, "input_tokens request is stateless");
    failures += check(Json::parse(make_response_input_tokens_body(9)) ==
                          Json{{"object", "response.input_tokens"}, {"input_tokens", 9}},
                      "input_tokens response shape");
    failures +=
        check(api_code([&] {
                  (void)parse_response_input_tokens_request(
                      Json{{"model", "qwen3.6-27b"}, {"input", "hello"}, {"instructions", "x"}},
                      limits());
              }) == "unknown_parameter",
              "input_tokens accepts only model and input");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_basic_request();
    failures += test_instruction_message_order();
    failures += test_preserve_thinking_options_and_inheritance();
    failures += test_reasoning_effort();
    failures += test_typed_items_and_tools();
    failures += test_explicit_rejections();
    failures += test_response_object();
    failures += test_sse_sequence();
    failures += test_sse_function_call();
    failures += test_input_tokens_schema();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
