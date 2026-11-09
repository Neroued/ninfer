#include "serve/anthropic_messages.h"

#include "serve/generation_service.h"
#include "serve/translate.h"
#include "tool_call_salvage_probe.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

Json base_request() {
    return Json{{"model", "claude-local"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
                {"max_tokens", 4096}};
}

RequestLimits limits() {
    RequestLimits value;
    value.default_max_tokens = 8192;
    return value;
}

AnthropicMessagesRequest parse(const Json& body) {
    return parse_anthropic_messages_request(body, limits());
}

std::string api_code(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

std::string api_param(const std::function<void()>& action) {
    try {
        action();
    } catch (const ApiException& error) { return error.error().param; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

ResolvedPromptSemantics semantics(const GenerationRequest& request, bool default_thinking = true) {
    ServeOptions options;
    options.enable_thinking = default_thinking;
    return resolve_prompt_semantics(request, options);
}

ninfer::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), [](const ContentPart& part) {
        ninfer::OwnedMedia media;
        media.kind =
            part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
        media.bytes               = {1};
        media.image_resize_policy = part.image_resize_policy;
        return media;
    });
}

Json parse_event(const std::string& value) {
    const std::size_t newline = value.find('\n');
    if (!value.starts_with("event: ") || newline == std::string::npos || !value.ends_with("\n\n")) {
        throw std::runtime_error("invalid Anthropic SSE framing");
    }
    const std::string type   = value.substr(7, newline - 7);
    const std::string prefix = "data: ";
    if (value.compare(newline + 1U, prefix.size(), prefix) != 0) {
        throw std::runtime_error("missing Anthropic SSE data");
    }
    Json payload = Json::parse(value.substr(newline + 1U + prefix.size(),
                                            value.size() - newline - 1U - prefix.size() - 2U));
    if (payload.at("type") != type) { throw std::runtime_error("Anthropic SSE type mismatch"); }
    return payload;
}

int test_envelope_and_field_policy() {
    Json body             = base_request();
    body["stream"]        = true;
    body["temperature"]   = 0.7;
    body["top_p"]         = 0.8;
    body["top_k"]         = 20;
    body["metadata"]      = Json{{"user_id", "u"}};
    body["service_tier"]  = "auto";
    body["inference_geo"] = "anywhere";
    body["future_field"]  = Json{{"unknown", true}};
    body["output_config"] = Json{{"effort", "low"}, {"future_option", true}};

    const AnthropicMessagesRequest request = parse(body);
    int failures =
        check(request.model == "claude-local" && request.stream && request.output_tokens_explicit &&
                  request.generation.max_tokens == 4096,
              "Messages envelope fields were not preserved");
    failures += check(request.generation.sampling.temperature == 0.7 &&
                          request.generation.sampling.top_p == 0.8 &&
                          request.generation.sampling.top_k == 20 &&
                          request.generation.reasoning_effort == RequestedReasoningEffort::Low,
                      "executable Anthropic generation fields were not lowered");

    body.erase("max_tokens");
    const AnthropicMessagesRequest defaulted = parse(body);
    failures += check(!defaulted.output_tokens_explicit && defaulted.generation.max_tokens == 8192,
                      "omitted max_tokens did not use the server default");
    body["max_tokens"] = 0;
    failures += check(api_code([&] { (void)parse(body); }) == "cache_prewarm_not_supported",
                      "max_tokens=0 was accepted as a false cache prewarm");

    body                = base_request();
    body["temperature"] = 1.01;
    failures += check(api_param([&] { (void)parse(body); }) == "temperature",
                      "Anthropic temperature range was not enforced");
    body          = base_request();
    body["top_k"] = 21;
    failures += check(api_param([&] { (void)parse(body); }) == "top_k",
                      "Engine top_k range was not enforced");
    body                  = base_request();
    body["output_config"] = Json{{"format", Json{{"type", "json_schema"}}}};
    failures += check(api_param([&] { (void)parse(body); }) == "output_config.format.schema",
                      "malformed output schema was accepted");
    body              = base_request();
    body["container"] = "container_1";
    failures += check(api_code([&] { (void)parse(body); }) == "container_not_supported",
                      "container execution was silently ignored");
    return failures;
}

int test_message_normalization() {
    Json body        = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    body["system"] =
        Json::array({Json{{"type", "text"}, {"text", "A"}},
                     Json{{"type", "text"},
                          {"text", "B"},
                          {"cache_control", Json{{"type", "ephemeral"}, {"ttl", "1h"}}}}});
    body["messages"] =
        Json::array({Json{{"role", "user"},
                          {"content", Json::array({Json{{"type", "text"}, {"text", "one"}},
                                                   Json{{"type", "text"}, {"text", "two"}}})}},
                     Json{{"role", "user"}, {"content", "three"}},
                     Json{{"role", "assistant"}, {"content", "prefix"}}});

    const GenerationRequest request = parse(body).generation;
    int failures                    = check(request.messages.size() == 3 &&
                                                request.messages[0].role == ninfer::ChatRole::System &&
                                                request.messages[1].role == ninfer::ChatRole::User &&
                                                request.messages[1].content.size() == 3,
                                            "consecutive Anthropic roles were not merged by block concatenation");
    failures += check(request.messages[0].content[0].text == "A" &&
                          request.messages[0].content[1].text == "B" &&
                          request.messages[1].content[0].text == "one" &&
                          request.messages[1].content[1].text == "two" &&
                          request.messages[1].content[2].text == "three",
                      "message normalization inserted or removed text");
    failures +=
        check(request.continuation == ninfer::PromptContinuationMode::ContinueFinalAssistant,
              "final assistant text did not select continuation mode");
    const ninfer::PromptInput translated = prompt(request);
    failures += check(translated.options.continuation ==
                              ninfer::PromptContinuationMode::ContinueFinalAssistant &&
                          translated.context_cache.markers.size() == 1 &&
                          translated.context_cache.markers[0].location ==
                              ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
                      "assistant continuation or system block cache boundary was lost");
    return failures;
}

int test_attribution_system_block() {
    const auto attributed = [](std::string fingerprint) {
        Json body        = base_request();
        body["thinking"] = Json{{"type", "disabled"}};
        body["system"]   = Json::array(
            {Json{{"type", "text"},
                    {"text", "x-anthropic-billing-header: cc_version=2.1; cch=" + fingerprint}},
               Json{{"type", "text"},
                    {"text", "real system"},
                    {"cache_control", Json{{"type", "ephemeral"}}}}});
        return body;
    };

    const Json first_body                 = attributed(".89c");
    const GenerationRequest first_request = parse(first_body).generation;
    const ninfer::PromptInput first       = prompt(first_request);
    const ninfer::PromptInput second      = prompt(parse(attributed(".8f7")).generation);
    const ninfer::PromptInput counted =
        prompt(parse_anthropic_count_tokens_request(first_body).generation);

    const auto has_real_system = [](const ninfer::PromptInput& input) {
        return input.messages.size() == 2 &&
               input.messages.front().role == ninfer::ChatRole::System &&
               input.messages.front().parts.size() == 1 &&
               input.messages.front().parts.front().text == "real system" &&
               input.context_cache.markers.size() == 1 &&
               input.context_cache.markers.front().location ==
                   ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary;
    };
    int failures =
        check(has_real_system(first) && has_real_system(second) && has_real_system(counted) &&
                  first.messages.front().parts.front().text ==
                      second.messages.front().parts.front().text &&
                  first.context_cache.markers == second.context_cache.markers &&
                  first.context_cache.markers == counted.context_cache.markers,
              "Anthropic attribution changed model input or Count Tokens lowering");

    Json attribution_only        = base_request();
    attribution_only["thinking"] = Json{{"type", "disabled"}};
    attribution_only["system"]   = Json::array(
        {Json{{"type", "text"}, {"text", "x-anthropic-billing-header: cc_version=2.1"}}});
    const GenerationRequest without_empty = parse(attribution_only).generation;
    failures += check(without_empty.messages.size() == 1 &&
                          without_empty.messages.front().role == ninfer::ChatRole::User,
                      "attribution-only system produced an empty model turn");

    Json non_first        = base_request();
    non_first["thinking"] = Json{{"type", "disabled"}};
    non_first["system"] =
        Json::array({Json{{"type", "text"}, {"text", "real system"}},
                     Json{{"type", "text"}, {"text", "x-anthropic-billing-header: user text"}}});
    const GenerationRequest preserved_non_first = parse(non_first).generation;
    failures += check(preserved_non_first.messages.size() == 2 &&
                          preserved_non_first.messages.front().content.size() == 2 &&
                          preserved_non_first.messages.front().content.back().text ==
                              "x-anthropic-billing-header: user text",
                      "non-first attribution-like system text was consumed");

    Json string_system                       = base_request();
    string_system["thinking"]                = Json{{"type", "disabled"}};
    string_system["system"]                  = "x-anthropic-billing-header: ordinary string";
    const GenerationRequest preserved_string = parse(string_system).generation;
    failures += check(preserved_string.messages.size() == 2 &&
                          preserved_string.messages.front().content.size() == 1 &&
                          preserved_string.messages.front().content.front().text ==
                              "x-anthropic-billing-header: ordinary string",
                      "ordinary string system content was consumed as attribution");
    return failures;
}

Json tool_use(std::string id, std::string name = "lookup") {
    return Json{{"type", "tool_use"},
                {"id", std::move(id)},
                {"name", std::move(name)},
                {"input", Json::object()}};
}

Json tool_result(std::string id, std::string content, bool is_error = false) {
    return Json{{"type", "tool_result"},
                {"tool_use_id", std::move(id)},
                {"content", std::move(content)},
                {"is_error", is_error}};
}

int test_tool_history() {
    Json body        = base_request();
    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", Json::array({tool_use("toolu_a"), tool_use("toolu_b")})}},
         Json{{"role", "user"},
              {"content", Json::array({Json{{"type", "tool_result"},
                                            {"tool_use_id", "toolu_b"},
                                            {"content", "result B"},
                                            {"is_error", true},
                                            {"cache_control", Json{{"type", "ephemeral"}}}},
                                       tool_result("toolu_a", "result A"),
                                       Json{{"type", "text"}, {"text", "continue"}}})}}});
    const GenerationRequest normalized          = parse(body).generation;
    int failures                                = check(normalized.messages.size() == 4 &&
                                                            normalized.messages[0].role == ninfer::ChatRole::Assistant &&
                                                            normalized.messages[1].role == ninfer::ChatRole::Tool &&
                                                            normalized.messages[1].tool_call_id == "toolu_a" &&
                                                            normalized.messages[1].content[0].text == "result A" &&
                                                            normalized.messages[2].role == ninfer::ChatRole::Tool &&
                                                            normalized.messages[2].tool_call_id == "toolu_b" &&
                                                            normalized.messages[2].content[0].text == "result B" &&
                                                            normalized.messages[2].tool_result_is_error &&
                                                            normalized.messages[2].cache_boundary_after &&
                                                            normalized.messages[2].cache_boundary_after->kind ==
                                                                ninfer::PromptCacheMarkerKind::SharedStablePrefix &&
                                                            normalized.messages[3].role == ninfer::ChatRole::User &&
                                                            normalized.messages[3].content[0].text == "continue",
                                                        "valid out-of-order tool results were not associated by ID");
    const ninfer::PromptInput normalized_prompt = prompt(normalized);
    failures += check(normalized_prompt.messages[2].parts.size() == 2 &&
                          normalized_prompt.messages[2].parts[0].text == "[tool_error]\n" &&
                          normalized_prompt.messages[2].parts[1].text == "result B",
                      "tool result error state did not follow its ID during normalization");

    body["messages"][1]["content"] =
        Json::array({tool_result("toolu_unknown", "wrong"), tool_result("toolu_a", "A")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history" &&
                          api_param([&] { (void)parse(body); }) == "messages",
                      "unknown tool_result ID was accepted");

    body["messages"][1]["content"] =
        Json::array({tool_result("toolu_a", "first"), tool_result("toolu_a", "duplicate")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "duplicate tool_result ID was accepted");

    body["messages"][1]["content"] = Json::array({tool_result("toolu_a", "only one")});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "missing tool_result was accepted");

    body["messages"] =
        Json::array({Json{{"role", "assistant"}, {"content", Json::array({tool_use("toolu_a")})}},
                     Json{{"role", "user"},
                          {"content", Json::array({Json{{"type", "text"}, {"text", "before"}},
                                                   tool_result("toolu_a", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "tool_result after ordinary user content was accepted");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"}, {"content", Json::array({tool_use("toolu_a")})}},
         Json{{"role", "user"}, {"content", "before"}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_a", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "same-role message joining bypassed tool_result ordering");

    body["messages"] = Json::array(
        {Json{{"role", "user"},
              {"content", Json::array({tool_result("toolu_external", "imported"),
                                       Json{{"type", "text"}, {"text", "continue"}}})}}});
    const GenerationRequest truncated = parse(body).generation;
    failures += check(truncated.messages.size() == 2 &&
                          truncated.messages[0].role == ninfer::ChatRole::Tool &&
                          truncated.messages[0].tool_call_id == "toolu_external" &&
                          truncated.messages[1].role == ninfer::ChatRole::User,
                      "leading tool_result from a truncated history was rejected or reordered");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "first"}},
         Json{{"role", "assistant"}, {"content", "ordinary"}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_orphan", "late")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "orphan tool_result inside a visible history was accepted");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", Json::array({tool_use("toolu_same"), tool_use("toolu_same", "other")})}},
         Json{{"role", "user"}, {"content", Json::array({tool_result("toolu_same", "result")})}}});
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_history",
                      "duplicate tool_use ID was accepted");
    failures += check(api_code([&] { (void)parse_anthropic_count_tokens_request(body); }) ==
                          "invalid_tool_history",
                      "Count Tokens did not share Messages tool-history validation");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", Json::array({tool_use("toolu_mixed"), tool_use("toolu_only")})}},
         Json{{"role", "user"},
              {"content",
               Json::array({Json{{"type", "tool_result"},
                                 {"tool_use_id", "toolu_mixed"},
                                 {"content", Json::array({Json{{"type", "text"}, {"text", "hit"}},
                                                          Json{{"type", "tool_reference"},
                                                               {"tool_name", "mcp__x"}}})}},
                            Json{{"type", "tool_result"},
                                 {"tool_use_id", "toolu_only"},
                                 {"content", Json::array({Json{{"type", "tool_reference"},
                                                               {"tool_name", "mcp__y"}}})}}})}}});
    const Json schema = Json{{"type", "object"}};
    body["tools"] =
        Json::array({Json{{"name", "mcp__x"}, {"input_schema", schema}, {"defer_loading", true}},
                     Json{{"name", "mcp__y"}, {"input_schema", schema}, {"defer_loading", true}}});
    const GenerationRequest pointers = parse(body).generation;
    failures += check(prompt(pointers).options.tool_jsons.size() == 2,
                      "tool_reference blocks did not load both deferred tools");
    failures += check(pointers.messages.size() == 3 && pointers.messages[1].content.size() == 1 &&
                          pointers.messages[1].content[0].text == "hit" &&
                          pointers.messages[2].content.size() == 1 &&
                          pointers.messages[2].content[0].text.empty(),
                      "tool_reference blocks did not drop out of tool_result content");
    return failures;
}

Json ordinary_tool(bool strict = false) {
    return Json{{"name", "weather"},
                {"description", "Get weather"},
                {"input_schema", Json{{"type", "object"},
                                      {"properties", Json{{"city", Json{{"type", "string"}}}}}}},
                {"input_examples", Json::array({Json{{"city", "Paris"}}})},
                {"strict", strict}};
}

int test_tools() {
    Json body                       = base_request();
    body["tools"]                   = Json::array({ordinary_tool()});
    body["tool_choice"]             = Json{{"type", "auto"}, {"disable_parallel_tool_use", false}};
    const GenerationRequest request = parse(body).generation;
    const ninfer::PromptInput translated = prompt(request);
    const Json rendered                  = Json::parse(translated.options.tool_jsons.at(0));
    int failures = check(request.uses_tools() && rendered["function"]["name"] == "weather" &&
                             rendered["function"]["input_examples"].is_array(),
                         "Anthropic tool schema/examples did not reach the Qwen prompt");

    body["tools"] = Json::array({ordinary_tool(true)});
    failures += check(api_code([&] { (void)parse(body); }) == "strict_tools_not_supported",
                      "active strict tool was accepted without constrained decoding");
    body["tool_choice"]               = Json{{"type", "none"}, {"disable_parallel_tool_use", true}};
    body["tools"][0]["defer_loading"] = true;
    body["tools"][0]["allowed_callers"] = Json::array({"code_execution"});
    const GenerationRequest disabled    = parse(body).generation;
    failures += check(!disabled.uses_tools() && prompt(disabled).options.tool_jsons.empty(),
                      "tool_choice:none did not neutralize inactive tool guarantees");

    body                              = base_request();
    body["tools"]                     = Json::array({ordinary_tool()});
    body["tools"][0]["defer_loading"] = true;
    const GenerationRequest deferred  = parse(body).generation;
    failures += check(prompt(deferred).options.tool_jsons.empty(),
                      "a deferred tool without any tool_reference reached the prompt");
    body["messages"]               = Json::array({Json{
                      {"role", "user"},
                      {"content",
                       Json::array({Json{{"type", "tool_result"},
                                         {"tool_use_id", "toolu_search"},
                                         {"content", Json::array({Json{{"type", "text"}, {"text", "1 match"}},
                                                                  Json{{"type", "tool_reference"},
                                                                       {"tool_name", "weather"}}})}}})}}});
    const GenerationRequest loaded = parse(body).generation;
    failures += check(prompt(loaded).options.tool_jsons.size() == 1,
                      "a tool_reference did not bring its deferred tool into the prompt");
    failures += check(api_code([&] { (void)parse_anthropic_count_tokens_request(body); }).empty(),
                      "Count Tokens rejected a request carrying tool_reference history");

    // A loaded deferred tool renders exactly like the same tool declared up front.
    Json eager_body                         = body;
    eager_body["tools"][0]["defer_loading"] = false;
    failures += check(prompt(parse(eager_body).generation).options.tool_jsons ==
                          prompt(loaded).options.tool_jsons,
                      "a loaded deferred tool rendered differently from an eager one");

    Json& reference        = body["messages"][0]["content"][0]["content"][1];
    reference["tool_name"] = "mcp__unknown";
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_reference",
                      "a tool_reference to an undeclared tool was accepted");
    failures += check(api_code([&] { (void)parse_anthropic_count_tokens_request(body); }) ==
                          "invalid_tool_reference",
                      "Count Tokens accepted a tool_reference to an undeclared tool");
    reference["tool_name"] = 7;
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_reference",
                      "a tool_reference with a non-string tool_name was accepted");
    reference.erase("tool_name");
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_reference",
                      "a tool_reference without tool_name was accepted");
    reference["tool_name"] = std::string(129, 'a');
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_tool_reference",
                      "a tool_reference with an over-long tool_name was accepted");

    // Only tool_result content loads a deferred tool; a stray content array elsewhere does not.
    body["messages"] = Json::array(
        {Json{{"role", "user"},
              {"content",
               Json::array({Json{{"type", "text"},
                                 {"text", "hi"},
                                 {"content", Json::array({Json{{"type", "tool_reference"},
                                                               {"tool_name", "weather"}}})}}})}}});
    failures += check(prompt(parse(body).generation).options.tool_jsons.empty(),
                      "a tool_reference outside tool_result content loaded a deferred tool");

    // allowed_callers is validated whether or not the deferred tool is loaded.
    body                                = base_request();
    body["tools"]                       = Json::array({ordinary_tool()});
    body["tools"][0]["defer_loading"]   = true;
    body["tools"][0]["allowed_callers"] = Json::array({"code_execution"});
    failures += check(api_code([&] { (void)parse(body); }) == "tool_caller_not_supported",
                      "a deferred tool skipped allowed_callers validation");

    body                = base_request();
    body["tools"]       = Json::array({ordinary_tool()});
    body["tool_choice"] = Json{{"type", "any"}};
    failures += check(api_code([&] { (void)parse(body); }) == "tool_choice_not_supported",
                      "forced any-tool choice was silently downgraded");
    body["tool_choice"] = Json{{"type", "tool"}, {"name", "weather"}};
    failures += check(api_code([&] { (void)parse(body); }) == "tool_choice_not_supported",
                      "named tool choice was silently downgraded");
    body["tool_choice"] = Json{{"type", "auto"}, {"disable_parallel_tool_use", true}};
    failures += check(api_code([&] { (void)parse(body); }) == "parallel_tool_use_not_supported",
                      "active single-tool-call guarantee was silently downgraded");

    body          = base_request();
    body["tools"] = Json::array({Json{{"type", "web_search_20250305"}, {"name", "web_search"}}});
    failures += check(api_code([&] { (void)parse(body); }) == "anthropic_tools_not_supported",
                      "Anthropic-provided tool was treated as a user tool");
    body["tool_choice"] = Json{{"type", "none"}};
    failures += check(api_code([&] { (void)parse(body); }).empty(),
                      "inactive Anthropic-provided tool unnecessarily blocked generation");

    body          = base_request();
    body["tools"] = Json::array({ordinary_tool(), ordinary_tool()});
    failures += check(api_param([&] { (void)parse(body); }) == "tools",
                      "duplicate tool names were accepted");
    return failures;
}

int test_prompt_object_order() {
    const auto parse_raw = [](std::string_view input) {
        const std::string body =
            R"({"model":"claude-local","thinking":{"type":"disabled"},"tools":[{"name":"probe","input_schema":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}}}],"messages":[{"role":"user","content":"run probe"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_order","name":"probe","input":)" +
            std::string(input) +
            R"(}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_order","content":"done"}]}],"max_tokens":32})";
        return parse(Json::parse(body)).generation;
    };

    const std::string original      = R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})";
    const GenerationRequest request = parse_raw(original);
    const auto assistant =
        std::find_if(request.messages.begin(), request.messages.end(),
                     [](const ChatTurn& turn) { return turn.role == ninfer::ChatRole::Assistant; });
    const ninfer::PromptInput translated = prompt(request);
    const std::string expected_tool =
        R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"object","properties":{"yankee":{"type":"integer"},"bravo":{"type":"boolean"}}}}},"strict":false}})";
    int failures = check(assistant != request.messages.end() && assistant->tool_calls.size() == 1 &&
                             assistant->tool_calls.front().arguments_json == original &&
                             translated.options.tool_jsons.size() == 1 &&
                             translated.options.tool_jsons.front() == expected_tool,
                         "Anthropic request changed prompt-bearing object member order");

    const GenerationRequest reordered =
        parse_raw(R"({"alpha":{"yankee":2,"bravo":true},"zeta":"last"})");
    const auto reordered_assistant =
        std::find_if(reordered.messages.begin(), reordered.messages.end(),
                     [](const ChatTurn& turn) { return turn.role == ninfer::ChatRole::Assistant; });
    failures +=
        check(reordered_assistant != reordered.messages.end() &&
                  reordered_assistant->tool_calls.size() == 1 &&
                  reordered_assistant->tool_calls.front().arguments_json != original,
              "Anthropic request treated reordered tool input as the original representation");
    return failures;
}

int test_thinking_and_count_tokens() {
    Json body = base_request();
    body["thinking"] =
        Json{{"type", "enabled"}, {"budget_tokens", 1024}, {"display", "summarized"}};
    const GenerationRequest enabled = parse(body).generation;
    ServeOptions server;
    server.default_thinking_budget = 777;
    const ninfer::RequestOptions options =
        to_request_options(enabled, server, semantics(enabled), true);
    int failures = check(enabled.enable_thinking == true && enabled.thinking_budget == 1024 &&
                             options.execution.thinking.budget == 1024,
                         "request Thinking budget did not reach Engine options");

    body["thinking"]["budget_tokens"] = 4096;
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "Thinking budget equal to max_tokens was accepted");
    body["thinking"] = Json{{"type", "future"}};
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "unknown Thinking mode defaulted to enabled");
    body["thinking"]                      = Json{{"type", "adaptive"}, {"display", "omitted"}};
    const AnthropicMessagesRequest hidden = parse(body);
    failures += check(hidden.hide_thinking && hidden.generation.enable_thinking == true &&
                          !parse(base_request()).hide_thinking,
                      "display:omitted was not parsed as hidden adaptive Thinking");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "summarized"}};
    failures += check(!parse(body).hide_thinking, "display:summarized hid the Thinking text");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "future"}};
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "unknown Thinking display was accepted");
    body["thinking"] = Json{{"type", "disabled"}, {"display", "omitted"}};
    failures += check(api_param([&] { (void)parse(body); }) == "thinking",
                      "display was accepted with disabled Thinking");
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "omitted"}};

    body["max_tokens"]                        = 0;
    body["temperature"]                       = "ignored for counting";
    body["output_config"]                     = Json{{"format", Json{{"type", "json_schema"}}}};
    const AnthropicCountTokensRequest counted = parse_anthropic_count_tokens_request(body);
    failures += check(counted.generation.enable_thinking == true,
                      "Count Tokens did not share prompt-affecting Thinking parsing");

    body             = base_request();
    body["thinking"] = Json{{"type", "disabled"}};
    body["messages"].push_back(Json{{"role", "assistant"}, {"content", "prefix"}});
    failures += check(api_code([&] { (void)semantics(parse(body).generation); }).empty(),
                      "disabled-Thinking assistant prefill was rejected");
    body.erase("thinking");
    failures += check(api_code([&] { (void)semantics(parse(body).generation); }) ==
                          "assistant_prefill_not_supported",
                      "Thinking-on assistant prefill was not rejected at capability resolution");
    return failures;
}

int test_thinking_history_transport_metadata() {
    Json body = base_request();
    body["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "before"}},
                     Json{{"role", "assistant"},
                          {"content", Json::array({Json{{"type", "thinking"},
                                                        {"thinking", "thought"},
                                                        {"signature", "opaque-from-stopped-serve"}},
                                                   Json{{"type", "text"}, {"text", "answer"}}})}},
                     Json{{"role", "user"}, {"content", "after"}}});
    const GenerationRequest accepted = parse(body).generation;
    int failures                     = check(accepted.messages.size() == 3 &&
                                                 accepted.messages[1].reasoning_content == "thought" &&
                                                 accepted.messages[1].content[0].text == "answer",
                                             "Thinking history with opaque transport metadata was not lowered");

    Json changed_signature                                      = body;
    changed_signature["messages"][1]["content"][0]["signature"] = "different";
    const GenerationRequest changed_metadata = parse(changed_signature).generation;
    failures += check(changed_metadata.messages[1].reasoning_content == "thought",
                      "opaque Thinking signature changed prompt semantics");

    Json missing_signature = body;
    missing_signature["messages"][1]["content"][0].erase("signature");
    const GenerationRequest missing_metadata = parse(missing_signature).generation;
    failures += check(missing_metadata.messages[1].reasoning_content == "thought",
                      "missing Thinking signature blocked visible history");

    Json structured_signature                                      = body;
    structured_signature["messages"][1]["content"][0]["signature"] = Json{{"provider", "external"}};
    const GenerationRequest structured_metadata = parse(structured_signature).generation;
    failures += check(structured_metadata.messages[1].reasoning_content == "thought",
                      "non-semantic Thinking metadata was interpreted by request lowering");

    Json changed_thinking                                     = body;
    changed_thinking["messages"][1]["content"][0]["thinking"] = "changed";
    const GenerationRequest changed_semantics                 = parse(changed_thinking).generation;
    failures += check(changed_semantics.messages[1].reasoning_content == "changed",
                      "visible Thinking text did not own prompt semantics");

    const AnthropicCountTokensRequest counted =
        parse_anthropic_count_tokens_request(missing_signature);
    failures += check(counted.generation.messages[1].reasoning_content == "thought",
                      "Count Tokens assigned semantics to Thinking signature metadata");
    return failures;
}

int test_content_and_cache_hints() {
    Json body                       = base_request();
    body["cache_control"]           = Json{{"type", "ephemeral"}, {"ttl", "5m"}};
    body["messages"]                = Json::array({Json{
                       {"role", "user"},
                       {"content",
                        Json::array({Json{{"type", "text"},
                                          {"text", "look"},
                                          {"cache_control", Json{{"type", "ephemeral"}}}},
                                     Json{{"type", "image"},
                                          {"source", Json{{"type", "url"}, {"url", "https://example/image.png"}}},
                                          {"transformations", Json{{"oversized_image", "error"}}}}})}}});
    const GenerationRequest request = parse(body).generation;
    int failures                    = check(request.messages[0].content[1].cache_boundary_after &&
                                                ninfer::has_shared_candidate_evidence(
                                 request.messages[0].content[1].cache_boundary_after->evidence,
                                 ninfer::SharedCandidateEvidence::RequestedAutomatic) &&
                                                request.media_item_count() == 1 &&
                                                request.messages[0].content[1].image_resize_policy ==
                                                    ninfer::ImageResizePolicy::RejectOversized,
                                            "automatic caching or image transformation policy was lost");
    const ninfer::PromptInput translated = prompt(request);
    failures += check(translated.context_cache.markers.size() == 2 &&
                          translated.context_cache.markers[1].location ==
                              ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
                      "message-part cache boundary was not represented in PromptInput");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image"}, {"source", Json{{"type", "file"}, {"file_id", "file_1"}}}}});
    failures += check(api_code([&] { (void)parse(body); }) == "files_not_supported",
                      "Files image source was flattened or ignored");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "document"}, {"source", Json::object()}}});
    failures += check(api_code([&] { (void)parse(body); }) == "documents_not_supported",
                      "document citation semantics were silently discarded");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                            = "answer";
    outcome.reasoning                       = "thought";
    outcome.prompt_tokens                   = 100;
    outcome.completion_tokens               = 12;
    outcome.reasoning_tokens                = 5;
    outcome.finish_reason                   = ninfer::FinishReason::StopString;
    outcome.matched_stop_string             = "STOP";
    outcome.metrics.prefix_cache_hit_tokens = 60;
    return outcome;
}

int test_aggregate_and_errors() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_test", "claude-local");
    GenerationOutcome outcome = sample_outcome();
    const Json response       = Json::parse(make_anthropic_messages_response(identity, outcome));
    int failures =
        check(response["content"].size() == 2 && response["content"][0]["type"] == "thinking" &&
                  response["content"][0]["signature"] == identity.message_id &&
                  response["stop_reason"] == "stop_sequence" && response["stop_sequence"] == "STOP",
              "aggregate Thinking or stop-sequence presentation is incomplete");
    failures += check(response["usage"]["input_tokens"] == 40 &&
                          response["usage"]["cache_read_input_tokens"] == 60 &&
                          response["usage"]["cache_creation_input_tokens"].is_null() &&
                          response["usage"]["output_tokens_details"]["thinking_tokens"] == 5,
                      "aggregate cache/reasoning usage was fabricated or lost");

    outcome.prompt_tokens                   = 10;
    outcome.metrics.prefix_cache_hit_tokens = 1000;
    const Json clamped = Json::parse(make_anthropic_messages_response(identity, outcome));
    failures += check(clamped["usage"]["input_tokens"] == 0 &&
                          clamped["usage"]["cache_read_input_tokens"] == 10,
                      "cache-read usage was not clamped to the prompt token count");

    outcome               = {};
    outcome.finish_reason = ninfer::FinishReason::ContextCapacity;
    const Json empty      = Json::parse(make_anthropic_messages_response(identity, outcome));
    failures +=
        check(empty["content"].empty() && empty["stop_reason"] == "model_context_window_exceeded",
              "empty output was fabricated or context capacity was misclassified");

    ApiError overloaded;
    overloaded.status         = 429;
    overloaded.code           = "server_overloaded";
    overloaded.message        = "full";
    const ApiError normalized = normalize_anthropic_error(overloaded);
    const Json error          = Json::parse(make_anthropic_error_body(overloaded, "req_error"));
    failures += check(normalized.status == 529 && normalized.type == "overloaded_error" &&
                          error["request_id"] == "req_error" &&
                          error["error"]["type"] == "overloaded_error",
                      "Anthropic overload or request-id error mapping is wrong");
    return failures;
}

// Calls the Qwen parser keeps despite their names come back as tool_use history blocks without a
// 400.
int test_salvaged_tool_calls() {
    Json body     = base_request();
    body["tools"] = Json::array(
        {Json{{"name", "Read"},
              {"input_schema", Json{{"type", "object"},
                                    {"properties", Json{{"path", Json{{"type", "string"}}}}}}}}});
    const AnthropicMessagesRequest request = parse(body);
    const ninfer::PromptInput input        = prompt(request.generation);
    const std::size_t max_name             = request.generation.tool_name_max_length;
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_salvaged", "claude-local");

    int failures = 0;
    int kept     = 0;
    for (const std::string& name : ninfer::test::salvage_probe_names()) {
        const auto parsed = ninfer::test::parse_model_call(input, max_name, name);
        if (parsed.tool_calls.empty()) { continue; }
        ++kept;
        GenerationOutcome outcome;
        outcome.finish_reason = ninfer::FinishReason::StopToken;
        outcome.tool_calls    = parsed.tool_calls;
        const Json response   = Json::parse(make_anthropic_messages_response(identity, outcome));
        Json results          = Json::array();
        for (const Json& block : response.at("content")) {
            if (block.at("type") == "tool_use") {
                results.push_back(Json{{"type", "tool_result"},
                                       {"tool_use_id", block.at("id")},
                                       {"content", "unknown tool"},
                                       {"is_error", true}});
            }
        }
        Json next = body;
        next["messages"].push_back(
            Json{{"role", "assistant"}, {"content", response.at("content")}});
        next["messages"].push_back(Json{{"role", "user"}, {"content", results}});
        std::string error;
        try {
            (void)parse(next);
        } catch (const ApiException& exception) {
            error = exception.error().message;
        } catch (const std::exception& exception) { error = exception.what(); }
        failures += check(error.empty(),
                          "kept call '" + name + "' could not come back as tool_use: " + error);
    }
    failures += check(kept >= 4, "the probe names no longer exercise salvaged calls");
    return failures;
}

int test_tool_call_presentation() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_tool", "claude-local");
    GenerationOutcome outcome;
    outcome.text                = "I need one more check.";
    outcome.finish_reason       = ninfer::FinishReason::StopToken;
    const std::string arguments = R"({"zeta":"last","alpha":{"yankee":2,"bravo":true}})";
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name           = "Edit",
        .arguments_json = arguments,
    });

    const Json aggregate = Json::parse(make_anthropic_messages_response(identity, outcome));
    int failures =
        check(aggregate.at("stop_reason") == "tool_use" && aggregate.at("content").size() == 2 &&
                  aggregate.at("content").at(0).at("type") == "text" &&
                  aggregate.at("content").at(1).at("type") == "tool_use" &&
                  aggregate.at("content").at(1).at("name") == "Edit" &&
                  aggregate.at("content").at(1).at("input").dump() == arguments &&
                  !aggregate.at("content").at(1).at("input").contains("replace_all"),
              "aggregate Anthropic response changed ordered tool input");

    AnthropicMessagesStream stream(identity, 10);
    std::vector<std::string> events{stream.start()};
    std::vector<std::string> content = stream.content_delta(outcome.text);
    events.insert(events.end(), std::make_move_iterator(content.begin()),
                  std::make_move_iterator(content.end()));
    std::vector<std::string> terminal = stream.finish(outcome);
    events.insert(events.end(), std::make_move_iterator(terminal.begin()),
                  std::make_move_iterator(terminal.end()));

    bool saw_edit_start = false;
    bool saw_arguments  = false;
    bool saw_tool_stop  = false;
    for (const std::string& wire : events) {
        const Json event = parse_event(wire);
        if (event.at("type") == "content_block_start" &&
            event.at("content_block").at("type") == "tool_use") {
            saw_edit_start = event.at("content_block").at("name") == "Edit";
        } else if (event.at("type") == "content_block_delta" &&
                   event.at("delta").at("type") == "input_json_delta") {
            saw_arguments = event.at("delta").at("partial_json").get<std::string>() == arguments;
        } else if (event.at("type") == "message_delta") {
            saw_tool_stop = event.at("delta").at("stop_reason") == "tool_use";
        }
    }
    failures += check(saw_edit_start && saw_arguments && saw_tool_stop,
                      "Anthropic stream did not terminate the recovered Edit as tool_use");
    return failures;
}

int test_stream() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_stream", "claude-local");
    AnthropicMessagesStream stream(identity, 100);
    const ninfer::GenerationStart warm_start{
        .prompt               = ninfer::PromptSummary{.prompt_tokens = 100},
        .reused_prompt_tokens = 60,
    };
    std::vector<std::string> events{stream.start(warm_start)};
    auto append_events = [&](std::vector<std::string> values) {
        events.insert(events.end(), std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    };
    append_events(stream.reasoning_delta("thought"));
    append_events(stream.content_delta("answer"));
    append_events(stream.finish(sample_outcome()));

    std::vector<std::string> types;
    bool saw_signature        = false;
    bool start_usage_is_exact = false;
    Json terminal_usage;
    for (const std::string& value : events) {
        const Json parsed = parse_event(value);
        types.push_back(parsed.at("type").get<std::string>());
        if (parsed.at("type") == "message_start") {
            start_usage_is_exact = parsed["message"]["usage"]["input_tokens"] == 40 &&
                                   parsed["message"]["usage"]["cache_read_input_tokens"] == 60;
        }
        if (parsed.at("type") == "content_block_delta" &&
            parsed["delta"]["type"] == "signature_delta") {
            const std::string signature = parsed["delta"]["signature"].get<std::string>();
            saw_signature               = signature == identity.message_id;
        }
        if (parsed.at("type") == "message_delta") { terminal_usage = parsed.at("usage"); }
    }
    int failures = check(types.front() == "message_start" && types.back() == "message_stop" &&
                             saw_signature && start_usage_is_exact,
                         "Anthropic stream lifecycle/signature/start usage is incomplete");
    const auto signature_position =
        std::find_if(events.begin(), events.end(), [](const auto& value) {
            return value.find("signature_delta") != std::string::npos;
        });
    const auto text_position = std::find_if(events.begin(), events.end(), [](const auto& value) {
        return value.find("text_delta") != std::string::npos;
    });
    failures += check(signature_position < text_position,
                      "Thinking signature was emitted after the text block began");
    const Json aggregate =
        Json::parse(make_anthropic_messages_response(identity, sample_outcome()));
    failures +=
        check(terminal_usage == aggregate.at("usage") && terminal_usage["input_tokens"] == 40 &&
                  terminal_usage["cache_read_input_tokens"] == 60,
              "terminal stream usage did not match aggregate cache usage");

    AnthropicMessagesStream cold_stream(identity, 25);
    (void)cold_stream.start(ninfer::GenerationStart{
        .prompt               = ninfer::PromptSummary{.prompt_tokens = 25},
        .reused_prompt_tokens = 0,
    });
    GenerationOutcome cold;
    cold.prompt_tokens                         = 25;
    cold.completion_tokens                     = 3;
    const std::vector<std::string> cold_events = cold_stream.finish(cold);
    const Json cold_delta = parse_event(cold_events.at(cold_events.size() - 2U));
    failures += check(cold_delta["usage"]["input_tokens"] == 25 &&
                          cold_delta["usage"]["cache_read_input_tokens"] == 0 &&
                          cold_delta["usage"]["output_tokens"] == 3,
                      "cold terminal stream usage was not cumulative and exact");

    AnthropicMessagesStream pre_admission_error(identity, 25);
    const Json provisional = parse_event(pre_admission_error.start());
    failures += check(provisional["message"]["usage"]["input_tokens"] == 25 &&
                          provisional["message"]["usage"]["cache_read_input_tokens"].is_null(),
                      "pre-admission stream error prefix fabricated cache usage");
    return failures;
}

// Every remainder class of the Base64 codec, plus multi-byte UTF-8 and control bytes.
int test_hidden_reasoning_signature() {
    int failures = 0;
    for (const std::string reasoning :
         {std::string(), std::string("a"), std::string("ab"), std::string("abc"),
          std::string("abcd"),
          std::string("na\xC3\xAFve \xE2\x80\x94 \xE6\x80\x9D\xE8\x80\x83\n\t\"quoted\"") +
              std::string(1, '\0') + "nul"}) {
        const std::string signature = encode_thinking_signature(reasoning);
        failures += check(decode_thinking_signature(signature) == reasoning,
                          "hidden-reasoning signature did not round-trip");
    }
    // RFC 4648 test vectors pin the alphabet and padding independently of the decoder.
    failures += check(encode_thinking_signature("foobar") == "ninfer-reasoning.v1:Zm9vYmFy" &&
                          encode_thinking_signature("fooba") == "ninfer-reasoning.v1:Zm9vYmE=" &&
                          encode_thinking_signature("foob") == "ninfer-reasoning.v1:Zm9vYg==",
                      "hidden-reasoning signature is not standard Base64");
    failures +=
        check(!decode_thinking_signature("msg_0123456789abcdef") &&
                  !decode_thinking_signature("") && !decode_thinking_signature("EuYBCkQIARgCKkD"),
              "a foreign signature was interpreted as hidden reasoning");
    for (const char* malformed :
         {"ninfer-reasoning.v1:Zm9", "ninfer-reasoning.v1:Zm9v!mFy", "ninfer-reasoning.v1:Zg=A",
          "ninfer-reasoning.v1:=m9v", "ninfer-reasoning.v1:Zm9vYg==Zm9v"}) {
        bool threw = false;
        try {
            (void)decode_thinking_signature(malformed);
        } catch (const std::invalid_argument&) { threw = true; }
        failures += check(threw, std::string("malformed signature was accepted: ") + malformed);
    }
    return failures;
}

int test_hidden_reasoning_round_trip() {
    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity("req_hidden", "claude-local");
    const GenerationOutcome outcome = sample_outcome();

    const Json aggregate =
        Json::parse(make_anthropic_messages_response(identity, outcome, /*hide_thinking=*/true));
    const Json& block = aggregate["content"][0];
    int failures      = check(block["type"] == "thinking" && block["thinking"] == "" &&
                                  decode_thinking_signature(block["signature"].get<std::string>()) ==
                                      outcome.reasoning &&
                                  aggregate["content"][1]["text"] == "answer",
                              "aggregate hidden Thinking leaked text or lost its signature");

    AnthropicMessagesStream stream(identity, 100, /*hide_thinking=*/true);
    std::vector<std::string> events{stream.start()};
    const auto append = [&](std::vector<std::string> values) {
        events.insert(events.end(), std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    };
    append(stream.reasoning_delta("tho"));
    append(stream.reasoning_delta("ught"));
    append(stream.content_delta("answer"));
    append(stream.finish(outcome));
    bool saw_thinking_delta = false;
    std::optional<std::string> streamed;
    for (const std::string& wire : events) {
        const Json parsed = parse_event(wire);
        if (parsed.at("type") != "content_block_delta") { continue; }
        if (parsed["delta"]["type"] == "thinking_delta") { saw_thinking_delta = true; }
        if (parsed["delta"]["type"] == "signature_delta") {
            streamed = decode_thinking_signature(parsed["delta"]["signature"].get<std::string>());
        }
    }
    failures += check(!saw_thinking_delta && streamed == outcome.reasoning,
                      "streamed hidden Thinking leaked text or lost its signature");

    // The client returns the block it received; lowering restores the reasoning from it.
    Json body        = base_request();
    body["thinking"] = Json{{"type", "adaptive"}, {"display", "omitted"}};
    body["messages"].push_back(
        Json{{"role", "assistant"},
             {"content", Json::array({block, Json{{"type", "text"}, {"text", "answer"}}})}});
    body["messages"].push_back(Json{{"role", "user"}, {"content", "next"}});
    const auto assistant_reasoning = [](const GenerationRequest& request) {
        const auto assistant = std::ranges::find_if(request.messages, [](const ChatTurn& turn) {
            return turn.role == ninfer::ChatRole::Assistant;
        });
        return assistant == request.messages.end() ? std::optional<std::string>()
                                                   : assistant->reasoning_content;
    };
    failures += check(assistant_reasoning(parse(body).generation) == outcome.reasoning,
                      "hidden reasoning was not restored from the returned block");

    // Visible text wins, a foreign signature stays metadata, a corrupt NInfer one is a 400.
    body["messages"][1]["content"][0] =
        Json{{"type", "thinking"}, {"thinking", "visible"}, {"signature", block["signature"]}};
    failures += check(assistant_reasoning(parse(body).generation) == "visible",
                      "visible Thinking text lost to its signature");
    body["messages"][1]["content"][0] =
        Json{{"type", "thinking"}, {"thinking", ""}, {"signature", "EuYBCkQIARgCKkD"}};
    failures += check(assistant_reasoning(parse(body).generation) == std::string(),
                      "a foreign signature produced reasoning");
    body["messages"][1]["content"][0] =
        Json{{"type", "thinking"}, {"thinking", ""}, {"signature", "ninfer-reasoning.v1:Zm9"}};
    failures += check(api_code([&] { (void)parse(body); }) == "invalid_thinking_signature",
                      "a corrupt hidden-reasoning signature was accepted");
    return failures;
}

// Structured output, ported from Neroued/ninfer 08902f73, eb6696ae, 8ffa3131 and 2734a56e
// (test_anthropic_schema.cpp), adapted to Tappa A (no JSON output with active tools yet).
int test_constrained_decoding() {
    int failures = 0;
    auto options = [](const GenerationRequest& request) {
        return to_request_options(request, {}, semantics(request), true);
    };
    auto body = base_request();
    failures += check(!options(parse(body).generation).constraint &&
                          !options(parse(body).generation).execution.sampling.presence_penalty,
                      "unconstrained Anthropic request changed its Engine options");
    for (const auto& value : {Json{{"choice", {"yes", "no"}}}, Json{{"regex", "[a-z]+"}}}) {
        body["structured_outputs"] = value;
        const auto request         = parse(body).generation;
        failures +=
            check(options(request).constraint ==
                      (value.contains("choice") ? ninfer::OutputConstraint::choice({"yes", "no"})
                                                : ninfer::OutputConstraint::regex("[a-z]+")),
                  "Anthropic choice/regex lost in Engine translation");
    }
    body["structured_outputs"] = Json{{"grammar", "root ::= \"yes\""}};
    const auto request         = parse(body);
    failures += check(options(request.generation).constraint ==
                          ninfer::OutputConstraint::grammar("root ::= \"yes\""),
                      "Anthropic GBNF extension was lost in Engine translation");
    body["tools"] = Json::array({ordinary_tool()});
    failures += check(api_param([&] { (void)parse(body); }) == "structured_outputs.grammar",
                      "grammar admitted active tools");
    body["tool_choice"] = Json{{"type", "none"}};
    failures += check(parse(body).generation.constraint->source == "root ::= \"yes\"",
                      "inactive tools blocked grammar");
    body["stop_sequences"] = Json::array({"yes"});
    failures += check(api_param([&] { (void)parse(body); }) == "structured_outputs.grammar",
                      "grammar admitted custom stop sequences");

    // output_config.format json_schema (streaming included), alongside effort.
    const Json schema = {
        {"type", "object"},
        {"properties", {{"z", {{"type", "string"}}}, {"a", {{"type", "integer"}}}}},
        {"required", {"z", "a"}},
        {"additionalProperties", false}};
    body           = base_request();
    body["stream"] = true;
    body["output_config"] =
        Json{{"effort", "low"}, {"format", {{"type", "json_schema"}, {"schema", schema}}}};
    const auto schema_request = parse(body);
    failures +=
        check(schema_request.stream &&
                  options(schema_request.generation).constraint ==
                      ninfer::OutputConstraint::json_schema(schema.dump()) &&
                  schema_request.generation.constraint_param == "output_config.format.schema" &&
                  schema_request.generation.reasoning_effort == RequestedReasoningEffort::Low &&
                  options(schema_request.generation).execution.sampling.presence_penalty ==
                      std::optional<float>(0.0F),
              "Anthropic JSON schema source, effort or D6 default lost");
    auto bad                               = body;
    bad["output_config"]["format"]["type"] = "json_object";
    failures += check(api_param([&] { (void)parse(bad); }) == "output_config.format.type",
                      "Anthropic accepted the OpenAI-only json_object format");
    bad                                    = body;
    bad["output_config"]["format"]["name"] = "answer";
    failures += check(api_param([&] { (void)parse(bad); }) == "output_config.format.name",
                      "Anthropic accepted an OpenAI-only schema option");
    bad                       = body;
    bad["structured_outputs"] = Json{{"grammar", "root ::= \"x\""}};
    failures += check(api_param([&] { (void)parse(bad); }) == "structured_outputs.grammar",
                      "Anthropic conflicting output formats accepted");
    bad          = body;
    bad["tools"] = Json::array({ordinary_tool()});
    failures += check(api_code([&] { (void)parse(bad); }) == "unsupported_output_combination",
                      "JSON output with active tools accepted before the tool grammar exists");
    bad                = body;
    bad["tool_choice"] = Json{{"type", "any"}};
    bad["tools"]       = Json::array({ordinary_tool()});
    failures += check(api_code([&] { (void)parse(bad); }) == "tool_choice_not_supported",
                      "forced tool_choice is still rejected in Tappa A");

    // D7: assistant prefill with a constraint is rejected before the Engine.
    bad             = body;
    bad["thinking"] = Json{{"type", "disabled"}};
    bad["messages"].push_back(Json{{"role", "assistant"}, {"content", "{\"z\":"}});
    failures += check(api_code([&] { (void)parse(bad); }) == "assistant_prefill_not_supported",
                      "assistant prefill admitted an output constraint");

    // Count Tokens ignores the format.
    failures += check(!parse_anthropic_count_tokens_request(body).generation.constraint,
                      "Count Tokens parsed an output format");

    // The error object has no param: structured-output errors keep their location in the message.
    ApiError schema_error;
    schema_error.status  = 400;
    schema_error.code    = "unsupported_json_schema";
    schema_error.param   = "output_config.format.schema/properties/date/format";
    schema_error.message = "unsupported schema keyword: format";
    const Json located   = Json::parse(make_anthropic_error_body(schema_error, "req_error"));
    failures +=
        check(located["error"]["message"] == schema_error.param + ": " + schema_error.message,
              "Anthropic schema error lost its request field location");
    ApiError other;
    other.status  = 400;
    other.param   = "messages";
    other.message = "bad";
    failures +=
        check(Json::parse(make_anthropic_error_body(other, "r"))["error"]["message"] == "bad",
              "Anthropic non-constraint error message changed");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_constrained_decoding();
    failures += test_envelope_and_field_policy();
    failures += test_message_normalization();
    failures += test_attribution_system_block();
    failures += test_tool_history();
    failures += test_tools();
    failures += test_prompt_object_order();
    failures += test_thinking_and_count_tokens();
    failures += test_thinking_history_transport_metadata();
    failures += test_content_and_cache_hints();
    failures += test_aggregate_and_errors();
    failures += test_tool_call_presentation();
    failures += test_salvaged_tool_calls();
    failures += test_stream();
    failures += test_hidden_reasoning_signature();
    failures += test_hidden_reasoning_round_trip();
    if (failures != 0) {
        std::cerr << failures << " Anthropic adapter checks failed\n";
        return 1;
    }
    std::cout << "Anthropic adapter checks passed\n";
    return 0;
}
