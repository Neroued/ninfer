#include "qus/text/chat_template.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace qus::text {
namespace {

using Json = nlohmann::json;

bool is_allowed_role(const std::string& role) {
    return role == "system" || role == "user" || role == "assistant";
}

std::string trim_ascii_whitespace(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() &&
           std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return text.substr(begin, end - begin);
}

ChatMessage parse_message(const Json& item, std::size_t index) {
    if (!item.is_object()) {
        throw std::invalid_argument("message " + std::to_string(index) + " must be an object");
    }
    if (item.contains("tool_calls")) {
        throw std::invalid_argument("message " + std::to_string(index) + " contains tool_calls");
    }
    if (item.contains("reasoning_content")) {
        throw std::invalid_argument("message " + std::to_string(index) +
                                    " contains reasoning_content");
    }
    if (item.contains("name")) {
        throw std::invalid_argument("message " + std::to_string(index) + " contains name");
    }
    if (item.size() != 2 || !item.contains("role") || !item.contains("content")) {
        throw std::invalid_argument("message " + std::to_string(index) +
                                    " must contain exactly role and content");
    }
    if (!item.at("role").is_string()) {
        throw std::invalid_argument("message " + std::to_string(index) + " role must be string");
    }
    const std::string role = item.at("role").get<std::string>();
    if (!is_allowed_role(role)) {
        throw std::invalid_argument("unsupported chat role: " + role);
    }
    if (!item.at("content").is_string()) {
        throw std::invalid_argument("message " + std::to_string(index) +
                                    " content must be a non-empty string");
    }
    const std::string content = item.at("content").get<std::string>();
    if (content.empty()) {
        throw std::invalid_argument("message " + std::to_string(index) +
                                    " content must be a non-empty string");
    }
    return ChatMessage{role, content};
}

} // namespace

std::vector<ChatMessage> messages_from_prompt(std::string prompt) {
    if (prompt.empty()) { throw std::invalid_argument("--prompt text is empty"); }
    return {ChatMessage{"user", std::move(prompt)}};
}

std::vector<ChatMessage> read_messages_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("failed to open messages JSON: " + path.string()); }

    Json root;
    try {
        in >> root;
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument(std::string("failed to parse messages JSON: ") + ex.what());
    }

    if (!root.is_array()) { throw std::invalid_argument("messages JSON must be an array"); }
    if (root.empty()) { throw std::invalid_argument("messages JSON must not be empty"); }

    std::vector<ChatMessage> messages;
    messages.reserve(root.size());
    for (std::size_t i = 0; i < root.size(); ++i) {
        messages.push_back(parse_message(root.at(i), i));
    }
    return messages;
}

std::string render_qwen_chat(const std::vector<ChatMessage>& messages,
                             ChatRenderOptions options) {
    if (messages.empty()) { throw std::invalid_argument("chat messages must not be empty"); }

    std::string rendered;
    for (const ChatMessage& message : messages) {
        if (!is_allowed_role(message.role)) {
            throw std::invalid_argument("unsupported chat role: " + message.role);
        }
        rendered += "<|im_start|>";
        rendered += message.role;
        rendered += '\n';
        rendered += trim_ascii_whitespace(message.content);
        rendered += "<|im_end|>\n";
    }

    if (options.add_generation_prompt) {
        rendered += "<|im_start|>assistant\n";
        if (options.enable_thinking) {
            rendered += "<think>\n";
        } else {
            rendered += "<think>\n\n</think>\n\n";
        }
    }
    return rendered;
}

} // namespace qus::text
