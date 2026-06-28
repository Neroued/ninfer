#include "qus/text/chat_template.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check(bool condition, const char* message) {
    return condition ? 0 : fail(message);
}

int expect_invalid(void (*fn)(), const char* label) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return 0;
    } catch (const std::exception& ex) {
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

std::filesystem::path write_temp_json(const std::filesystem::path& dir,
                                      const std::string& name,
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
    const std::vector<qus::text::ChatMessage> messages = qus::text::messages_from_prompt("你好");
    const std::string rendered                         = qus::text::render_qwen_chat(messages);
    return check(rendered == "<|im_start|>user\n你好<|im_end|>\n"
                             "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                 "prompt render mismatch");
}

int test_json_messages_render_prefixes() {
    const TempJson json("messages.json",
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

int test_rejections() {
    int failures = 0;
    failures += expect_invalid(
        []() { (void)qus::text::render_qwen_chat({}); }, "empty render messages");
    failures += expect_invalid(
        []() {
            (void)qus::text::render_qwen_chat({qus::text::ChatMessage{"tool", "result"}});
        },
        "tool role");
    failures += expect_invalid(
        []() {
            const TempJson json("qus_chat_multimodal.json",
                                R"([{"role":"user","content":[{"type":"text","text":"hi"}]}])");
            (void)qus::text::read_messages_json(json.path);
        },
        "multimodal content");
    failures += expect_invalid(
        []() {
            const TempJson json("qus_chat_tool_calls.json",
                                R"([{"role":"assistant","content":"hi","tool_calls":[]}])");
            (void)qus::text::read_messages_json(json.path);
        },
        "assistant tool calls");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_prompt_renders_qwen_chat();
    failures += test_json_messages_render_prefixes();
    failures += test_rejections();
    return failures == 0 ? 0 : fail("qwen chat template test failed");
}
