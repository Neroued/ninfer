// Contract test for the Ollama serving layer: /api/chat request parsing
// (messages/images/thinking/tool_calls, options mapping and rejection, think,
// num_ctx/num_predict, format rejection, stream default), preload detection and
// model-name matching (`name:latest`), non-streaming response
// and NDJSON frame shapes including the terminal timing fields, inventory
// endpoint bodies, and the Ollama error body. This is the schema boundary
// consumed by Open WebUI and other Ollama clients.

#include "serve/ollama_schema.h"
#include "serve/request.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <optional>
#include <string>

namespace {

using Json = nlohmann::json;
using namespace ninfer::serve;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

std::string api_code(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

std::string api_param(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException& error) { return error.error().param; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

RequestLimits default_limits() {
    RequestLimits limits;
    limits.default_max_tokens = 512;
    limits.max_context        = 4096;
    return limits;
}

Json user_only(const char* text) {
    return Json{{"model", "m"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", text}}})}};
}

// One NDJSON frame is a JSON object followed by exactly one newline.
Json parse_frame(const std::string& frame) {
    if (frame.empty() || frame.back() != '\n' || frame.find('\n') != frame.size() - 1) {
        throw std::runtime_error("bad NDJSON framing: " + frame);
    }
    return Json::parse(frame.substr(0, frame.size() - 1));
}

int test_parse_basic() {
    int failures                = 0;
    const GenerationRequest req = parse_ollama_chat_request(user_only("hello"), default_limits());
    failures += check(req.model == "m", "model parsed");
    failures += check(req.stream, "stream defaults true");
    failures += check(req.max_tokens == 512 && !req.max_tokens_set, "server default budget");
    failures +=
        check(req.messages.size() == 1 && req.messages[0].role == ninfer::ChatRole::User &&
                  req.messages[0].content.size() == 1 && req.messages[0].content[0].text == "hello",
              "user text carried");

    Json body                  = user_only("x");
    body["stream"]             = false;
    body["keep_alive"]         = "5m";
    body["metadata"]           = Json{{"chat_id", "abc"}};
    const GenerationRequest r2 = parse_ollama_chat_request(body, default_limits());
    failures += check(!r2.stream, "stream false honored");

    Json no_model = Json{{"messages", Json::array()}};
    failures +=
        check(api_param([&] { parse_ollama_chat_request(no_model, default_limits()); }) == "model",
              "missing model rejected");
    // Empty messages is the preload shape, never a generation request.
    Json empty        = user_only("x");
    empty["messages"] = Json::array();
    failures +=
        check(api_param([&] { parse_ollama_chat_request(empty, default_limits()); }) == "messages",
              "empty messages rejected by the generation parser");
    failures += check(ollama_preload_model(empty) == std::optional<std::string>("m"),
                      "empty messages detected as preload");
    failures += check(!ollama_preload_model(user_only("x")).has_value(),
                      "non-empty messages is not preload");
    failures += check(api_param([&] { ollama_preload_model(no_model); }) == "model",
                      "preload still requires model");
    Json missing_messages = Json{{"model", "m"}};
    failures += check(!ollama_preload_model(missing_messages).has_value(),
                      "missing messages is not preload");
    failures += check(api_param([&] {
                          parse_ollama_chat_request(missing_messages, default_limits());
                      }) == "messages",
                      "missing messages rejected");
    const Json load = Json::parse(make_ollama_chat_load_response("m", "2026-01-01T00:00:00Z"));
    failures += check(load.at("model") == "m" && load.at("done") == true &&
                          load.at("done_reason") == "load" &&
                          load.at("message").at("role") == "assistant" &&
                          load.at("message").at("content") == "" && !load.contains("eval_count"),
                      "load response shape");

    failures += check(ollama_model_matches("m", "m"), "bare model name matches");
    failures += check(ollama_model_matches("m:latest", "m"), "name:latest matches");
    failures += check(ollama_model_matches("m:latest", "m:latest"), "exact tagged id matches");
    failures += check(!ollama_model_matches("m:v2", "m") && !ollama_model_matches("mm", "m") &&
                          !ollama_model_matches("m:latest:latest", "m") &&
                          !ollama_model_matches("x:latest", "m"),
                      "other tags and names do not match");
    Json bad_role =
        Json{{"model", "m"}, {"messages", Json::array({Json{{"role", "bot"}, {"content", "x"}}})}};
    failures += check(api_code([&] { parse_ollama_chat_request(bad_role, default_limits()); }) ==
                          "unsupported_role",
                      "unknown role rejected");
    Json empty_user =
        Json{{"model", "m"}, {"messages", Json::array({Json{{"role", "user"}, {"content", ""}}})}};
    failures += check(api_param([&] { parse_ollama_chat_request(empty_user, default_limits()); }) ==
                          "messages",
                      "empty user content rejected");
    return failures;
}

int test_parse_history_thinking_tools_images() {
    int failures    = 0;
    const Json body = {
        {"model", "m"},
        {"tools", Json::array({Json{
                      {"type", "function"},
                      {"function",
                       Json{{"name", "get_weather"},
                            {"description", "weather"},
                            {"parameters",
                             Json{{"type", "object"},
                                  {"properties", Json{{"city", Json{{"type", "string"}}}}}}}}}}})},
        {"messages",
         Json::array(
             {Json{{"role", "system"}, {"content", "be terse"}},
              Json{{"role", "user"}, {"content", "look"}, {"images", Json::array({"aGVsbG8="})}},
              Json{
                  {"role", "assistant"},
                  {"content", ""},
                  {"thinking", "I should call the tool"},
                  {"tool_calls",
                   Json::array({Json{{"function", Json{{"name", "get_weather"},
                                                       {"arguments", Json{{"city", "Oslo"}}}}}}})}},
              Json{{"role", "tool"}, {"content", "{\"temp\": 5}"}, {"tool_name", "get_weather"}},
              Json{{"role", "user"}, {"content", "thanks"}}})}};
    const GenerationRequest req = parse_ollama_chat_request(body, default_limits());
    failures += check(req.tools.size() == 1 && req.tools[0].name == "get_weather", "tool parsed");
    const Json definition = Json::parse(req.tools[0].definition_json);
    failures += check(definition.at("type") == "function" &&
                          definition.at("function").at("parameters").at("type") == "object",
                      "tool normalized to OpenAI function object");
    failures += check(req.uses_tools() && req.has_tool_history(), "tool history detected");
    failures += check(req.messages.size() == 5, "all turns kept");
    failures += check(req.messages[0].role == ninfer::ChatRole::System, "system turn first");
    const ChatTurn& user = req.messages[1];
    failures += check(user.content.size() == 2 && user.content[0].kind == ContentKind::Text &&
                          user.content[1].kind == ContentKind::Image,
                      "text then image parts");
    failures +=
        check(user.content[1].source.kind == ninfer::product::media_acquire::SourceKind::Data &&
                  user.content[1].source.value.ends_with(";base64,aGVsbG8="),
              "image is a base64 data source");
    const ChatTurn& assistant = req.messages[2];
    failures += check(assistant.reasoning_content == "I should call the tool",
                      "assistant thinking round-tripped");
    failures +=
        check(assistant.tool_calls.size() == 1 && assistant.tool_calls[0].name == "get_weather" &&
                  Json::parse(assistant.tool_calls[0].arguments_json).at("city") == "Oslo",
              "tool call arguments serialized to JSON text");
    failures += check(assistant.content.empty(), "empty assistant content dropped");
    failures += check(req.messages[3].role == ninfer::ChatRole::Tool &&
                          req.messages[3].content[0].text == "{\"temp\": 5}",
                      "tool turn carried");

    // The translated prompt keeps roles and reasoning for the template.
    ServeOptions server;
    ninfer::PromptCapabilities capabilities;
    const ninfer::PromptInput input = to_prompt_input(
        req, resolve_prompt_semantics(req, server, capabilities), [](const ContentPart& part) {
            ninfer::OwnedMedia media;
            media.kind = ninfer::MediaKind::Image;
            media.bytes.push_back(0);
            media.source_name = part.type_raw;
            return media;
        });
    failures += check(input.messages.size() == 5 &&
                          input.messages[2].reasoning_content == "I should call the tool",
                      "reasoning reaches the prompt input");
    failures += check(input.options.tool_jsons.size() == 1, "tool json reaches the prompt");

    // An empty tool result is a legal turn (matches the OpenAI surface); a tool
    // turn without content is not.
    Json empty_tool                      = body;
    empty_tool["messages"][3]["content"] = "";
    const GenerationRequest empty_tool_req =
        parse_ollama_chat_request(empty_tool, default_limits());
    failures += check(empty_tool_req.messages[3].role == ninfer::ChatRole::Tool &&
                          empty_tool_req.messages[3].content.size() == 1 &&
                          empty_tool_req.messages[3].content[0].text.empty(),
                      "empty tool content accepted as an empty text part");
    Json no_tool_content = body;
    no_tool_content["messages"][3].erase("content");
    failures += check(api_param([&] {
                          parse_ollama_chat_request(no_tool_content, default_limits());
                      }) == "messages",
                      "tool turn without content rejected");

    // Missing parameters default to the same empty object schema as the OpenAI
    // surfaces, so the template sees one shape per tool.
    Json bare_tool = body;
    bare_tool["tools"][0]["function"].erase("parameters");
    const GenerationRequest bare_tool_req = parse_ollama_chat_request(bare_tool, default_limits());
    const Json bare_definition            = Json::parse(bare_tool_req.tools[0].definition_json);
    failures += check(bare_definition.at("function").at("parameters") ==
                              Json{{"type", "object"}, {"properties", Json::object()}} &&
                          bare_definition.at("function").at("strict") == false,
                      "missing parameters default to the OpenAI empty schema");

    Json bad_args                                                     = body;
    bad_args["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{\"city\":\"Oslo\"}";
    failures += check(api_param([&] { parse_ollama_chat_request(bad_args, default_limits()); }) ==
                          "messages",
                      "string tool arguments rejected");
    Json assistant_images                     = body;
    assistant_images["messages"][2]["images"] = Json::array({"aGVsbG8="});
    failures += check(api_param([&] {
                          parse_ollama_chat_request(assistant_images, default_limits());
                      }) == "messages",
                      "assistant images rejected");
    Json bad_tool                = body;
    bad_tool["tools"][0]["type"] = "code_interpreter";
    failures += check(api_code([&] { parse_ollama_chat_request(bad_tool, default_limits()); }) ==
                          "tool_type_not_supported",
                      "non-function tools rejected");
    return failures;
}

int test_options_and_think() {
    int failures                = 0;
    Json body                   = user_only("x");
    body["options"]             = Json{{"temperature", 0.6},
                                       {"top_p", 0.95},
                                       {"top_k", 20},
                                       {"min_p", 0.05},
                                       {"seed", 7},
                                       {"num_predict", 128},
                                       {"num_ctx", 4096},
                                       {"presence_penalty", 0.5},
                                       {"stop", Json::array({"<end>", ""})},
                                       {"num_gpu", 99},
                                       {"use_mmap", false}};
    body["think"]               = false;
    const GenerationRequest req = parse_ollama_chat_request(body, default_limits());
    failures += check(req.sampling.temperature && *req.sampling.temperature == 0.6, "temperature");
    failures += check(req.sampling.top_p && *req.sampling.top_p == 0.95, "top_p");
    failures += check(req.sampling.top_k && *req.sampling.top_k == 20, "top_k");
    failures += check(req.sampling.min_p && *req.sampling.min_p == 0.05, "min_p");
    failures += check(req.sampling.seed && *req.sampling.seed == 7, "seed");
    failures += check(req.sampling.presence_penalty && *req.sampling.presence_penalty == 0.5,
                      "presence_penalty");
    failures += check(req.max_tokens == 128 && req.max_tokens_set, "num_predict -> max_tokens");
    failures += check(req.stop_strings.size() == 1 && req.stop_strings[0] == "<end>",
                      "stop strings; empty entries dropped");
    failures += check(req.enable_thinking && !*req.enable_thinking, "think false");
    failures += check(!req.reasoning_effort, "no effort from boolean think");

    // The Engine sampler is reachable through translation, including min_p.
    ServeOptions server;
    const ninfer::RequestOptions options = to_request_options(req, server);
    failures +=
        check(options.execution.sampling.min_p && *options.execution.sampling.min_p == 0.05F,
              "min_p reaches the sampler overrides");
    failures += check(options.execution.requested_output_tokens == 128, "output budget applied");
    failures += check(options.stop.strings.size() == 1, "stop reaches request options");

    Json infinite                 = user_only("x");
    infinite["options"]           = Json{{"num_predict", -1}};
    const GenerationRequest r_inf = parse_ollama_chat_request(infinite, default_limits());
    failures += check(r_inf.max_tokens == 512 && !r_inf.max_tokens_set,
                      "num_predict -1 keeps server default");
    Json zero       = user_only("x");
    zero["options"] = Json{{"num_predict", 0}};
    failures += check(api_param([&] { parse_ollama_chat_request(zero, default_limits()); }) ==
                          "options.num_predict",
                      "num_predict 0 rejected");

    Json big_ctx       = user_only("x");
    big_ctx["options"] = Json{{"num_ctx", 8192}};
    failures += check(api_code([&] { parse_ollama_chat_request(big_ctx, default_limits()); }) ==
                          "context_length_exceeded",
                      "num_ctx above the server window rejected");
    Json small_ctx       = user_only("x");
    small_ctx["options"] = Json{{"num_ctx", 1024}};
    parse_ollama_chat_request(small_ctx, default_limits());

    for (const char* key : {"repeat_penalty", "mirostat", "typical_p", "bogus"}) {
        Json rejected       = user_only("x");
        rejected["options"] = Json{{key, 1}};
        failures += check(api_code([&] {
                              parse_ollama_chat_request(rejected, default_limits());
                          }) == "option_not_supported",
                          std::string("options.") + key + " rejected");
    }
    Json null_option       = user_only("x");
    null_option["options"] = Json{{"repeat_penalty", nullptr}};
    parse_ollama_chat_request(null_option, default_limits());

    Json format      = user_only("x");
    format["format"] = "json";
    failures += check(api_code([&] { parse_ollama_chat_request(format, default_limits()); }) ==
                          "format_not_supported",
                      "format rejected");

    Json think_high                = user_only("x");
    think_high["think"]            = "high";
    const GenerationRequest r_high = parse_ollama_chat_request(think_high, default_limits());
    failures += check(r_high.reasoning_effort &&
                          *r_high.reasoning_effort == RequestedReasoningEffort::High &&
                          r_high.reasoning_effort_param == "think",
                      "think level maps to reasoning effort");
    Json think_bad     = user_only("x");
    think_bad["think"] = "ultra";
    failures +=
        check(api_param([&] { parse_ollama_chat_request(think_bad, default_limits()); }) == "think",
              "unknown think level rejected");
    return failures;
}

int test_response_and_frames() {
    int failures = 0;
    OllamaTimings timings;
    timings.total_seconds       = 1.25;
    timings.prompt_eval_seconds = 0.5;
    timings.eval_seconds        = 0.75;
    timings.prompt_eval_count   = 40;
    timings.eval_count          = 30;

    std::vector<ToolCall> calls;
    calls.push_back(ToolCall{"call_1", "get_weather", "{\"city\":\"Oslo\"}"});
    const Json body = Json::parse(make_ollama_chat_response(
        "m", "2026-08-15T10:00:00.000000000Z", "answer", "thought", calls, "stop", timings));
    failures += check(body.at("model") == "m" && body.at("created_at").is_string(), "envelope");
    failures += check(body.at("done") == true && body.at("done_reason") == "stop", "done fields");
    failures += check(body.at("message").at("role") == "assistant" &&
                          body.at("message").at("content") == "answer" &&
                          body.at("message").at("thinking") == "thought",
                      "message content and thinking");
    const Json& call = body.at("message").at("tool_calls").at(0);
    failures += check(call.at("id") == "call_1" && call.at("function").at("index") == 0 &&
                          call.at("function").at("name") == "get_weather" &&
                          call.at("function").at("arguments").at("city") == "Oslo",
                      "tool call arguments are a JSON object");
    failures += check(body.at("total_duration") == 1250000000 && body.at("load_duration") == 0 &&
                          body.at("prompt_eval_count") == 40 &&
                          body.at("prompt_eval_duration") == 500000000 &&
                          body.at("eval_count") == 30 && body.at("eval_duration") == 750000000,
                      "timings in integer nanoseconds");

    const Json content = parse_frame(make_ollama_chat_content_frame("m", "t", "hi"));
    failures +=
        check(content.at("done") == false && content.at("message").at("content") == "hi" &&
                  !content.at("message").contains("thinking") && !content.contains("eval_count"),
              "content frame");
    const Json thinking = parse_frame(make_ollama_chat_thinking_frame("m", "t", "hmm"));
    failures += check(thinking.at("message").at("content") == "" &&
                          thinking.at("message").at("thinking") == "hmm",
                      "thinking frame");
    const Json tools = parse_frame(make_ollama_chat_tool_calls_frame("m", "t", calls));
    failures += check(tools.at("done") == false && tools.at("message").at("tool_calls").size() == 1,
                      "tool call frame");
    const Json done = parse_frame(make_ollama_chat_done_frame("m", "t", "length", timings));
    failures += check(done.at("done") == true && done.at("done_reason") == "length" &&
                          done.at("message").at("content") == "" && done.at("eval_count") == 30 &&
                          done.at("eval_duration") == 750000000,
                      "done frame carries timings");

    // A minimal non-streaming reply never emits thinking/tool_calls keys.
    const Json plain =
        Json::parse(make_ollama_chat_response("m", "t", "", "", {}, "stop", OllamaTimings{}));
    failures += check(
        plain.at("message").at("content") == "" && !plain.at("message").contains("thinking") &&
            !plain.at("message").contains("tool_calls") && plain.at("eval_duration") == 0,
        "empty outcome shape");
    return failures;
}

int test_inventory_and_errors() {
    int failures = 0;
    OllamaModelFacts facts;
    facts.model_id        = "qwen3.8-27b/groupwise-int";
    facts.target          = "qwen3_8_27b";
    facts.weights_id      = "groupwise-int";
    facts.modified_at     = ollama_timestamp(1755000000123456789LL);
    facts.size_bytes      = 18210531328ULL;
    facts.size_vram_bytes = 17000000000ULL;
    facts.max_context     = 16384;
    facts.vision          = true;
    facts.thinking        = true;

    failures += check(facts.modified_at == "2025-08-12T12:00:00.123456789Z", "RFC 3339 timestamp");

    const Json tags = Json::parse(make_ollama_tags(facts));
    const Json& tag = tags.at("models").at(0);
    failures += check(tags.at("models").size() == 1 && tag.at("model") == facts.model_id &&
                          tag.at("name") == facts.model_id,
                      "tags lists the one resident model");
    failures +=
        check(tag.at("size") == 18210531328ULL && tag.at("modified_at") == facts.modified_at,
              "tags size and timestamp");
    failures += check(tag.at("details").at("family") == "qwen3_8_27b" &&
                          tag.at("details").at("quantization_level") == "groupwise-int" &&
                          tag.at("details").at("format") == "ninfer",
                      "tags details");

    const Json show = Json::parse(make_ollama_show(facts));
    failures += check(show.at("details").at("family") == "qwen3_8_27b", "show details");
    failures += check(show.at("model_info").at("qwen3_8_27b.context_length") == 16384,
                      "show context length");
    const Json& capabilities = show.at("capabilities");
    failures += check(capabilities.size() == 4 && capabilities[0] == "completion" &&
                          capabilities[1] == "tools" && capabilities[2] == "vision" &&
                          capabilities[3] == "thinking",
                      "show capabilities");
    facts.vision = false;
    failures += check(Json::parse(make_ollama_show(facts)).at("capabilities").size() == 3,
                      "vision capability follows the server flag");

    const Json ps = Json::parse(make_ollama_ps(facts));
    failures += check(ps.at("models").at(0).at("size_vram") == 17000000000ULL &&
                          ps.at("models").at(0).at("model") == facts.model_id,
                      "ps reports the resident model");

    const Json version = Json::parse(make_ollama_version());
    failures += check(version.at("version") == kOllamaCompatVersion, "version body");

    ApiError err;
    err.status  = 404;
    err.message = "model 'x' not found";
    failures += check(Json::parse(make_ollama_error_body(err)).at("error") == "model 'x' not found",
                      "error body");
    const Json frame = parse_frame(make_ollama_error_frame(err));
    failures += check(frame.at("error") == "model 'x' not found", "error frame");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_basic();
    failures += test_parse_history_thinking_tools_images();
    failures += test_options_and_think();
    failures += test_response_and_frames();
    failures += test_inventory_and_errors();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
