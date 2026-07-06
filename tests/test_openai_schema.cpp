// Contract test for the OpenAI serving layer: request parsing (string + parts
// content, unsupported-feature rejection), response/chunk/models/error
// serialization shapes, and finish_reason mapping. This is the schema boundary
// consumed by external OpenAI clients.

#include "qus/serve/openai_schema.h"
#include "qus/serve/request.h"
#include "qus/serve/translate.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;
using namespace qus::serve;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

bool throws_api(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

RequestLimits default_limits() {
    RequestLimits limits;
    limits.default_max_tokens = 512;
    limits.max_context        = 8192;
    return limits;
}

// Strip "data: " prefix and trailing blank line from an SSE event, returning the
// parsed JSON payload.
Json parse_sse(const std::string& event) {
    const std::string prefix = "data: ";
    const std::string suffix = "\n\n";
    if (event.rfind(prefix, 0) != 0 || event.size() < prefix.size() + suffix.size()) {
        throw std::runtime_error("bad SSE framing: " + event);
    }
    const std::string json = event.substr(prefix.size(), event.size() - prefix.size() - suffix.size());
    return Json::parse(json);
}

int test_parse_string_content() {
    int failures = 0;
    const Json body = {{"model", "qwen3.6-27b"},
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
    int failures = 0;
    const Json body = {
        {"model", "m"},
        {"messages",
         Json::array({Json{{"role", "user"},
                           {"content", Json::array({Json{{"type", "text"}, {"text", "a"}},
                                                    Json{{"type", "text"}, {"text", "b"}}})}}})}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.messages[0].content.size() == 2, "two content parts");
    const auto messages = to_chat_messages(req);
    failures += check(messages.size() == 1, "flattened to one message");
    failures += check(messages[0].content == "a\nb", "text parts joined");
    return failures;
}

int test_reject_image_in_translate() {
    const Json body = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "user"},
                                       {"content", Json::array({Json{{"type", "image_url"},
                                                                     {"image_url", Json::object()}}})}}})}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    return check(throws_api([&] { (void)to_chat_messages(req); }),
                 "image content rejected by translate");
}

int test_reject_unsupported() {
    int failures = 0;
    const Json base = {{"model", "m"},
                       {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};

    Json n2 = base;
    n2["n"] = 2;
    failures += check(throws_api([&] { (void)parse_chat_completion_request(n2, default_limits()); }),
                      "n>1 rejected");

    Json tools = base;
    tools["tools"] = Json::array({Json::object()});
    failures += check(throws_api([&] { (void)parse_chat_completion_request(tools, default_limits()); }),
                      "tools rejected");

    Json rf     = base;
    rf["response_format"] = Json{{"type", "json_object"}};
    failures += check(throws_api([&] { (void)parse_chat_completion_request(rf, default_limits()); }),
                      "json response_format rejected");

    Json rf_text          = base;
    rf_text["response_format"] = Json{{"type", "text"}};
    bool text_ok               = true;
    try {
        (void)parse_chat_completion_request(rf_text, default_limits());
    } catch (...) {
        text_ok = false;
    }
    failures += check(text_ok, "text response_format accepted");

    Json no_model = {{"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    failures += check(throws_api([&] { (void)parse_chat_completion_request(no_model, default_limits()); }),
                      "missing model rejected");

    Json tool_role = {{"model", "m"},
                      {"messages", Json::array({Json{{"role", "tool"}, {"content", "x"}}})}};
    failures += check(throws_api([&] { (void)parse_chat_completion_request(tool_role, default_limits()); }),
                      "tool role rejected");
    return failures;
}

int test_parse_stop_and_max_tokens() {
    int failures = 0;
    Json body = {{"model", "m"},
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
    req = parse_chat_completion_request(single, default_limits());
    failures += check(req.stop_strings.size() == 1 && req.stop_strings[0] == "END",
                      "single stop string");
    return failures;
}

int test_parse_sampling_carried() {
    int failures = 0;
    const Json body = {{"model", "m"},
                       {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                       {"temperature", 0.7},
                       {"top_p", 0.9},
                       {"seed", 123},
                       {"logit_bias", Json{{"5", -1.5}}}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.sampling.temperature.has_value() && *req.sampling.temperature == 0.7,
                      "temperature carried");
    failures += check(req.sampling.top_p.has_value() && *req.sampling.top_p == 0.9, "top_p carried");
    failures += check(req.sampling.seed.has_value() && *req.sampling.seed == 123u, "seed carried");
    failures += check(req.sampling.logit_bias.count(5) == 1 && req.sampling.logit_bias.at(5) == -1.5,
                      "logit_bias carried");
    return failures;
}

int test_response_serialization() {
    int failures = 0;
    const CompletionUsage usage{10, 3};
    const Json j = Json::parse(
        make_chat_completion_response("id-1", "m", 111, "hello world", "stop", usage));
    failures += check(j.at("object") == "chat.completion", "response object");
    failures += check(j.at("id") == "id-1", "response id");
    failures += check(j.at("choices").at(0).at("message").at("role") == "assistant", "assistant role");
    failures += check(j.at("choices").at(0).at("message").at("content") == "hello world",
                      "response content");
    failures += check(j.at("choices").at(0).at("finish_reason") == "stop", "response finish_reason");
    failures += check(j.at("usage").at("prompt_tokens") == 10, "usage prompt_tokens");
    failures += check(j.at("usage").at("completion_tokens") == 3, "usage completion_tokens");
    failures += check(j.at("usage").at("total_tokens") == 13, "usage total_tokens");
    return failures;
}

int test_chunk_serialization() {
    int failures = 0;
    const Json role = parse_sse(make_chat_chunk_role("id", "m", 1));
    failures += check(role.at("object") == "chat.completion.chunk", "chunk object");
    failures += check(role.at("choices").at(0).at("delta").at("role") == "assistant", "role delta");

    const Json content = parse_sse(make_chat_chunk_content("id", "m", 1, "tok"));
    failures += check(content.at("choices").at(0).at("delta").at("content") == "tok", "content delta");

    const CompletionUsage usage{2, 5};
    const Json final_chunk = parse_sse(make_chat_chunk_final("id", "m", 1, "length", &usage));
    failures += check(final_chunk.at("choices").at(0).at("finish_reason") == "length",
                      "final finish_reason");
    failures += check(final_chunk.contains("usage"), "final usage present when requested");
    failures += check(final_chunk.at("usage").at("total_tokens") == 7, "final usage total");

    const Json final_no_usage = parse_sse(make_chat_chunk_final("id", "m", 1, "stop", nullptr));
    failures += check(!final_no_usage.contains("usage"), "no usage when not requested");

    failures += check(sse_done() == "data: [DONE]\n\n", "done sentinel");
    return failures;
}

int test_models_and_error() {
    int failures = 0;
    const Json list = Json::parse(make_models_list("qwen3.6-27b", 1));
    failures += check(list.at("object") == "list", "models list object");
    failures += check(list.at("data").at(0).at("id") == "qwen3.6-27b", "models list id");
    failures += check(list.at("data").at(0).at("object") == "model", "models list entry object");

    const Json one = Json::parse(make_model_object("qwen3.6-27b", 1));
    failures += check(one.at("id") == "qwen3.6-27b" && one.at("object") == "model", "model object");

    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = "bad";
    error.param   = "messages";
    const Json err = Json::parse(make_error_body(error));
    failures += check(err.at("error").at("message") == "bad", "error message");
    failures += check(err.at("error").at("type") == "invalid_request_error", "error type");
    failures += check(err.at("error").at("param") == "messages", "error param");
    failures += check(err.at("error").at("code").is_null(), "error code null when empty");
    return failures;
}

int test_finish_reason_wire() {
    int failures = 0;
    failures += check(std::string(finish_reason_wire(qus::text::FinishReason::Stop)) == "stop",
                      "stop wire");
    failures += check(std::string(finish_reason_wire(qus::text::FinishReason::Length)) == "length",
                      "length wire");
    failures += check(std::string(finish_reason_wire(qus::text::FinishReason::Cancelled)) == "stop",
                      "cancelled maps to stop");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_parse_string_content();
    failures += test_parse_parts_and_flatten();
    failures += test_reject_image_in_translate();
    failures += test_reject_unsupported();
    failures += test_parse_stop_and_max_tokens();
    failures += test_parse_sampling_carried();
    failures += test_response_serialization();
    failures += test_chunk_serialization();
    failures += test_models_and_error();
    failures += test_finish_reason_wire();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
