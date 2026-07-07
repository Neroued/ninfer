#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace qus::text {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct ChatMessage {
    std::string role;
    std::string content;
    // Assistant reasoning carried across turns. When non-empty it is used verbatim
    // as the <think> block; when empty the reasoning is derived by splitting the
    // content on </think> (matching the official Qwen3.6 chat template).
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

struct ChatRenderOptions {
    bool add_generation_prompt = true;
    // Default thinking-ON, matching the Qwen3.6 template's default generation prompt.
    bool enable_thinking       = true;
    // Force-keep the <think> block on every historical assistant turn (the
    // template's preserve_thinking flag); otherwise history is stripped except
    // for turns after the last user query.
    bool preserve_thinking     = false;
    std::vector<std::string> tool_jsons;
};

std::vector<ChatMessage> messages_from_prompt(std::string prompt);
std::vector<ChatMessage> read_messages_json(const std::filesystem::path& path);
std::string render_qwen_chat(const std::vector<ChatMessage>& messages,
                             ChatRenderOptions options = {});

} // namespace qus::text
