// capy-flash: client JSON nested as deep as --max-request-json-depth allows must cross the whole
// prompt path (body check, parse, protocol lowering, prompt translation, chat-template rendering
// with the real Qwen3.8 template and tokenizer) without exhausting the stack; one level more is a
// 400 before parsing. Every place where a client can nest JSON is covered: tool schemas, tool-call
// arguments (JSON in a string), tool inputs and results, template kwargs. Needs
// NINFER_TEST_TOKENIZER_DIR (a Qwen3.8 checkpoint directory); skips without it.
//
// Diagnostic: `ninfer_request_json_depth_test --unguarded N` runs every case at nesting N without
// the limit, to measure how far the recursive consumers go before the stack runs out.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "serve/anthropic_messages.h"
#include "serve/openai_chat.h"
#include "serve/openai_responses.h"
#include "serve/openai_responses_store.h"
#include "serve/request_validation.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace q35 = ninfer::models::qwen3_5;
namespace fi  = ninfer::models::qwen3_5::frontend;
using namespace ninfer::serve;

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

q35::Frontend make_frontend(const std::string& directory) {
    const std::string tokenizer_json = read_file(directory + "/tokenizer.json");
    const std::string config_json    = read_file(directory + "/tokenizer_config.json");
    const std::string generation     = read_file(directory + "/generation_config.json");
    const std::string chat_template =
        read_file(NINFER_SOURCE_DIR "/tools/chat_templates/qwen3_8.jinja");
    auto tokenizer = std::make_shared<const fi::Tokenizer>(
        fi::TokenizerResources{tokenizer_json, config_json, generation});
    q35::FrontendResources resources{tokenizer_json, config_json, chat_template, generation};
    resources.tokenizer          = tokenizer;
    resources.public_token_count = static_cast<std::uint32_t>(tokenizer->vocab_size());
    q35::FrontendOptions options;
    options.vision_enabled = false;
    options.max_context    = 1U << 20;
    return q35::make_frontend(resources, options);
}

// `[[...[1]...]]` nested `levels` deep.
std::string nested(std::size_t levels) {
    return std::string(levels, '[') + "1" + std::string(levels, ']');
}

// Text with the value at nesting `levels` below the placeholder's own nesting, so that the whole
// body nests exactly `total` levels: measure the frame, then fill it.
std::string at_total_depth(const std::string& frame, std::size_t total) {
    const auto at     = frame.find("@");
    std::size_t depth = 0, here = 0;
    bool in_string = false, escaped = false;
    for (std::size_t i = 0; i < at; ++i) {
        const char c = frame[i];
        if (in_string) {
            if (escaped)
                escaped = false;
            else if (c == '\\')
                escaped = true;
            else if (c == '"')
                in_string = false;
        } else if (c == '"')
            in_string = true;
        else if (c == '{' || c == '[')
            depth = std::max(depth, ++here);
        else if (c == '}' || c == ']')
            --here;
    }
    if (here >= total) throw std::logic_error("frame deeper than the target");
    return frame.substr(0, at) + nested(total - here) + frame.substr(at + 1);
}

// Tool-call arguments are JSON in a string: escaped, and checked on their own.
std::string arguments_string(std::size_t total) {
    return "\"{\\\"x\\\":" + nested(total - 1) + "}\"";
}

struct Case {
    const char* name;
    std::string frame; // `@` = the deep value; `$ARGS` = deep arguments string
    std::function<ninfer::PromptInput(const RequestJson&)> lower;
};

ninfer::PromptInput lower_chat(const RequestJson& body) {
    const auto request = parse_chat_completion_request(body, RequestLimits{});
    const ServeOptions server;
    return to_prompt_input(request.generation, resolve_prompt_semantics(request.generation, server),
                           {});
}

ninfer::PromptInput lower_anthropic(const RequestJson& body) {
    const auto request = parse_anthropic_messages_request(body, RequestLimits{});
    const ServeOptions server;
    return to_prompt_input(request.generation, resolve_prompt_semantics(request.generation, server),
                           {});
}

ninfer::PromptInput lower_responses(const RequestJson& body) {
    const auto request = parse_openai_responses_create_request(body, RequestLimits{});
    OpenAIResponsesStore store(8, 1ULL << 20);
    const auto resolved = resolve_openai_responses_prompt(request.prompt, store, "resp_x", false);
    const ServeOptions server;
    return to_prompt_input(resolved.generation,
                           resolve_prompt_semantics(resolved.generation, server), {});
}

std::vector<Case> cases() {
    const std::string chat_head = R"({"model":"qwen","max_tokens":8,)";
    return {
        {"chat/tool_schema",
         chat_head + R"("messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function",)"
                     R"("function":{"name":"f","parameters":{"type":"object","properties":{"x":)"
                     R"({"type":"array","default":@}}}}}]})",
         lower_chat},
        {"chat/tool_call_arguments",
         chat_head +
             R"("messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null,)"
             R"("tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":$ARGS}}]},)"
             R"({"role":"tool","tool_call_id":"c1","content":"ok"},{"role":"user","content":"go"}],)"
             R"("tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]})",
         lower_chat},
        {"chat/template_kwargs",
         chat_head +
             R"("messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"x":@}})",
         lower_chat},
        {"responses/tool_schema",
         R"({"model":"qwen","input":"hi","tools":[{"type":"function","name":"f","parameters":)"
         R"({"type":"object","properties":{"x":{"type":"array","default":@}}}}]})",
         lower_responses},
        {"responses/function_call_arguments",
         R"({"model":"qwen","input":[{"role":"user","content":"hi"},{"type":"function_call",)"
         R"("call_id":"c1","name":"f","arguments":$ARGS},{"type":"function_call_output",)"
         R"("call_id":"c1","output":"ok"}],"tools":[{"type":"function","name":"f",)"
         R"("parameters":{"type":"object"}}]})",
         lower_responses},
        {"anthropic/tool_schema",
         R"({"model":"qwen","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"tools":[)"
         R"({"name":"f","input_schema":{"type":"object","properties":{"x":{"type":"array",)"
         R"("default":@}}}}]})",
         lower_anthropic},
        {"anthropic/tool_use_input",
         R"({"model":"qwen","max_tokens":8,"messages":[{"role":"user","content":"hi"},)"
         R"({"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{"x":@}}]},)"
         R"({"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}],)"
         R"("tools":[{"name":"f","input_schema":{"type":"object"}}]})",
         lower_anthropic},
    };
}

std::string body_text(const Case& c, std::size_t total) {
    std::string frame = c.frame;
    if (const auto at = frame.find("$ARGS"); at != std::string::npos) {
        frame.replace(at, 5, arguments_string(total));
        return frame;
    }
    return at_total_depth(frame, total);
}

} // namespace

int main(int argc, char** argv) {
    const char* directory = std::getenv("NINFER_TEST_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "SKIP: NINFER_TEST_TOKENIZER_DIR is not set\n";
        return 77;
    }
    const bool unguarded = argc == 3 && std::string(argv[1]) == "--unguarded";
    const std::size_t depth =
        unguarded ? std::stoul(argv[2]) : static_cast<std::size_t>(max_request_json_depth());
    const auto frontend = make_frontend(directory);
    int failures        = 0;
    for (const auto& c : cases()) {
        const std::string text = body_text(c, depth);
        try {
            if (!unguarded) check_request_json_nesting(text, {});
            RequestJson body  = RequestJson::parse(text);
            const auto prompt = frontend.prepare(c.lower(body));
            std::cout << c.name << " at " << depth << " levels: " << prompt.token_ids().size()
                      << " prompt tokens\n";
        } catch (const ApiException& error) {
            std::cerr << "FAIL: " << c.name
                      << " at the limit was rejected: " << error.error().message << '\n';
            ++failures;
        }
        if (unguarded) continue;
        // One level more is refused before parsing (body) or where the arguments are read.
        bool refused = false;
        try {
            const std::string over = body_text(c, depth + 1);
            check_request_json_nesting(over, {});
            (void)c.lower(RequestJson::parse(over)); // arguments: checked where they are read
        } catch (const ApiException& error) {
            refused = error.error().status == 400 && error.error().code == "request_json_too_deep";
        }
        if (!refused) {
            std::cerr << "FAIL: " << c.name << " one level over the limit was not refused\n";
            ++failures;
        }
    }
    if (failures == 0) std::cout << "request JSON depth tests passed\n";
    return failures == 0 ? 0 : 1;
}
