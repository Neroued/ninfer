#include "qus/text/chat_template.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path repo_file(std::string_view relative) {
#ifdef QUS_SOURCE_DIR
    return std::filesystem::path(QUS_SOURCE_DIR) / relative;
#else
    return std::filesystem::current_path() / relative;
#endif
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check(bool condition, const char* message) { return condition ? 0 : fail(message); }

qus::text::ChatMessage text_message(std::string role, std::string text) {
    qus::text::ChatMessage message;
    message.role = std::move(role);
    message.parts.push_back(qus::text::ChatPart::text_part(std::move(text)));
    return message;
}

int expect_invalid(void (*fn)(), const char* label) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& ex) {
        std::cerr << label << " threw wrong exception: " << ex.what() << '\n';
        return 1;
    }
    std::cerr << label << " did not throw\n";
    return 1;
}

struct TempDir {
    std::filesystem::path path;

    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int i = 0; i < 16; ++i) {
            path = std::filesystem::temp_directory_path() /
                   ("qus_chat_template_" + std::to_string(stamp) + "_" + std::to_string(i));
            if (std::filesystem::create_directory(path)) { return; }
        }
        throw std::runtime_error("failed to create unique chat template temp directory");
    }

    ~TempDir() { std::filesystem::remove_all(path); }
};

std::filesystem::path write_temp_json(const std::filesystem::path& dir, const std::string& name,
                                      const std::string& text) {
    const std::filesystem::path path = dir / name;
    std::ofstream out(path);
    out << text;
    return path;
}

struct TempJson {
    TempDir dir;
    std::filesystem::path path;

    TempJson(const std::string& name, const std::string& text)
        : path(write_temp_json(dir.path, name, text)) {}
};

int test_prompt_renders_qwen_chat() {
    // Default is thinking-ON, matching the Qwen3.6 template's default prompt.
    const std::vector<qus::text::ChatMessage> messages = qus::text::messages_from_prompt("你好");
    const std::string rendered                         = qus::text::render_qwen_chat(messages);
    return check(rendered == "<|im_start|>user\n你好<|im_end|>\n"
                             "<|im_start|>assistant\n<think>\n",
                 "prompt render mismatch");
}

int test_prompt_renders_no_thinking_prefix() {
    const std::vector<qus::text::ChatMessage> messages = qus::text::messages_from_prompt("你好");
    const std::string rendered                         = qus::text::render_qwen_chat(
        messages, qus::text::ChatRenderOptions{.enable_thinking = false});
    return check(rendered == "<|im_start|>user\n你好<|im_end|>\n"
                             "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                 "no-thinking prompt render mismatch");
}

int test_json_messages_render_prefixes() {
    const TempJson json(
        "messages.json",
        R"([{"role":"system","content":"be direct"},{"role":"user","content":"hi"}])");
    const std::vector<qus::text::ChatMessage> messages = qus::text::read_messages_json(json.path);
    const std::string rendered                         = qus::text::render_qwen_chat(messages);

    int failures = 0;
    failures += check(rendered.find("<|im_start|>system\nbe direct<|im_end|>\n") == 0,
                      "system prefix mismatch");
    failures += check(rendered.find("<|im_start|>user\nhi<|im_end|>\n") != std::string::npos,
                      "user prefix mismatch");
    failures += check(rendered.find("<|im_start|>assistant\n<think>") != std::string::npos,
                      "assistant generation prefix mismatch");
    return failures;
}

int test_tool_system_block_merges_system_content() {
    std::vector<qus::text::ChatMessage> messages;
    messages.push_back(text_message("system", "be direct"));
    messages.push_back(text_message("user", "weather?"));

    qus::text::ChatRenderOptions options;
    options.tool_jsons.push_back(
        R"({"function":{"description":"Fetch weather","name":"get_weather","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},"strict":false},"type":"function"})");
    const std::string rendered = qus::text::render_qwen_chat(messages, options);

    int failures = 0;
    failures +=
        check(rendered.find("<|im_start|>system\n# Tools\n\n"
                            "You have access to the following functions:\n\n<tools>\n") == 0,
              "tool system block prefix mismatch");
    failures +=
        check(rendered.find("\"name\":\"get_weather\"") != std::string::npos, "tool json missing");
    failures +=
        check(rendered.find("</tools>\n\nIf you choose to call a function") != std::string::npos,
              "tool instructions missing");
    failures += check(rendered.find("\n\nbe direct<|im_end|>\n") != std::string::npos,
                      "system content not merged into tool block");
    failures += check(rendered.find("<|im_start|>user\nweather?<|im_end|>\n") != std::string::npos,
                      "user message missing after tool block");
    return failures;
}

int test_assistant_tool_call_history_renders_qwen_xml() {
    std::vector<qus::text::ChatMessage> messages;
    messages.push_back(text_message("user", "weather?"));
    qus::text::ChatMessage assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back(
        qus::text::ToolCall{"call_1", "get_weather", R"({"city":"Paris","days":2})"});
    messages.push_back(std::move(assistant));

    const std::string rendered = qus::text::render_qwen_chat(messages);
    int failures               = 0;
    // The tool-call turn is after the last user query, so the (empty) think block
    // is kept ahead of the <tool_call> XML, matching the official template.
    failures += check(rendered.find("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n"
                                    "<function=get_weather>\n") != std::string::npos,
                      "assistant tool call prefix missing");
    failures += check(rendered.find("<parameter=city>\nParis\n</parameter>\n") != std::string::npos,
                      "string parameter rendering mismatch");
    failures += check(rendered.find("<parameter=days>\n2\n</parameter>\n") != std::string::npos,
                      "number parameter rendering mismatch");
    failures += check(rendered.find("</function>\n</tool_call><|im_end|>\n") != std::string::npos,
                      "assistant tool call suffix missing");
    return failures;
}

int test_tool_role_renders_tool_response() {
    std::vector<qus::text::ChatMessage> messages;
    messages.push_back(text_message("user", "weather?"));
    qus::text::ChatMessage assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back(
        qus::text::ToolCall{"call_1", "get_weather", R"({"city":"Paris"})"});
    messages.push_back(std::move(assistant));
    qus::text::ChatMessage tool;
    tool.role         = "tool";
    tool.tool_call_id = "call_1";
    tool.parts.push_back(qus::text::ChatPart::text_part(R"({"temp":20})"));
    messages.push_back(std::move(tool));

    const std::string rendered = qus::text::render_qwen_chat(messages);
    return check(rendered.find("<|im_start|>user\n<tool_response>\n"
                               R"({"temp":20})"
                               "\n</tool_response><|im_end|>\n") != std::string::npos,
                 "tool response render mismatch");
}

int test_rejections() {
    int failures = 0;
    failures +=
        expect_invalid([]() { (void)qus::text::render_qwen_chat({}); }, "empty render messages");
    {
        const TempJson json(
            "qus_chat_multimodal.json",
            R"({"messages":[{"role":"user","content":[{"type":"image","image":"test.png"},{"type":"text","text":"hi"},{"type":"video","video":"test.mp4"}]}]})");
        const auto messages = qus::text::read_messages_json(json.path);
        failures += check(messages.size() == 1 && messages[0].parts.size() == 3 &&
                              messages[0].parts[0].kind == qus::text::ChatPartKind::Image &&
                              messages[0].parts[2].kind == qus::text::ChatPartKind::Video,
                          "structured multimodal content parse mismatch");
        failures += check(qus::text::render_qwen_chat(messages).find(
                              "<|vision_start|><|image_pad|><|vision_end|>hi"
                              "<|vision_start|><|video_pad|><|vision_end|>") != std::string::npos,
                          "structured multimodal content render mismatch");
    }
    failures += expect_invalid(
        []() {
            const TempJson json("qus_chat_tool_calls.json",
                                R"([{"role":"assistant","content":"hi","tool_calls":[]}])");
            (void)qus::text::read_messages_json(json.path);
        },
        "assistant tool calls");
    return failures;
}

// Byte-for-byte parity against the HF-generated golden fixture. This is the
// authoritative correctness gate for the faithful render_qwen_chat port; it needs
// only the committed fixture (no tokenizer/model), so it runs anywhere.
int test_hf_golden_render_parity() {
    const auto fixture_path = repo_file("tests/fixtures/text/qwen36_text_golden.json");
    std::ifstream in(fixture_path);
    if (!in) {
        std::cerr << "golden render parity: cannot open fixture " << fixture_path << '\n';
        return 1;
    }
    const nlohmann::json fixture = nlohmann::json::parse(in);
    int failures                 = 0;
    for (const auto& item : fixture.at("message_cases")) {
        const std::string name = item.at("name").get<std::string>();
        std::vector<qus::text::ChatMessage> messages;
        for (const auto& msg : item.at("messages")) {
            qus::text::ChatMessage cm;
            cm.role = msg.at("role").get<std::string>();
            cm.parts.push_back(
                qus::text::ChatPart::text_part(msg.at("content").get<std::string>()));
            if (msg.contains("reasoning_content")) {
                cm.reasoning_content = msg.at("reasoning_content").get<std::string>();
            }
            if (msg.contains("tool_calls")) {
                for (const auto& tc : msg.at("tool_calls")) {
                    cm.tool_calls.push_back(
                        qus::text::ToolCall{"", tc.at("name").get<std::string>(),
                                            tc.at("arguments_json").get<std::string>()});
                }
            }
            messages.push_back(std::move(cm));
        }
        qus::text::ChatRenderOptions options;
        options.enable_thinking   = item.at("enable_thinking").get<bool>();
        options.preserve_thinking = item.at("preserve_thinking").get<bool>();
        options.tool_jsons        = item.at("tool_jsons").get<std::vector<std::string>>();

        const std::string actual   = qus::text::render_qwen_chat(messages, options);
        const std::string expected = item.at("rendered").get<std::string>();
        if (actual != expected) {
            std::cerr << "golden render parity mismatch for " << name << "\nexpected: " << expected
                      << "\nactual:   " << actual << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_prompt_renders_qwen_chat();
    failures += test_prompt_renders_no_thinking_prefix();
    failures += test_json_messages_render_prefixes();
    failures += test_tool_system_block_merges_system_content();
    failures += test_assistant_tool_call_history_renders_qwen_xml();
    failures += test_tool_role_renders_tool_response();
    failures += test_rejections();
    failures += test_hf_golden_render_parity();
    return failures == 0 ? 0 : fail("qwen chat template test failed");
}
