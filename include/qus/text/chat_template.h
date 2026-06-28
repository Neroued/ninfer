#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace qus::text {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ChatRenderOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = false;
};

std::vector<ChatMessage> messages_from_prompt(std::string prompt);
std::vector<ChatMessage> read_messages_json(const std::filesystem::path& path);
std::string render_qwen_chat(const std::vector<ChatMessage>& messages,
                             ChatRenderOptions options = {});

} // namespace qus::text
