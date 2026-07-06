#include "qus/text/chat_template.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace qus::text {
namespace {

using Json = nlohmann::json;

bool is_allowed_role(const std::string& role) {
    return role == "system" || role == "user" || role == "assistant" || role == "tool";
}

bool is_allowed_file_role(const std::string& role) {
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

constexpr std::string_view kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

std::string parameter_text(const Json& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return value.dump();
}

std::string render_tool_call(const ToolCall& call) {
    Json args = Json::parse(call.arguments_json);
    if (!args.is_object()) {
        throw std::invalid_argument("tool call arguments must be a JSON object");
    }

    std::string rendered;
    rendered += "<tool_call>\n<function=";
    rendered += call.name;
    rendered += ">\n";
    for (auto it = args.begin(); it != args.end(); ++it) {
        rendered += "<parameter=";
        rendered += it.key();
        rendered += ">\n";
        rendered += parameter_text(it.value());
        rendered += "\n</parameter>\n";
    }
    rendered += "</function>\n</tool_call>";
    return rendered;
}

std::string render_tools_system_block(const std::vector<std::string>& tool_jsons,
                                      const std::string& merged_system) {
    std::string rendered;
    rendered += "<|im_start|>system\n";
    rendered += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const std::string& tool : tool_jsons) {
        rendered += "\n";
        rendered += tool;
    }
    rendered += "\n</tools>";
    rendered += std::string(kToolInstructions);
    if (!merged_system.empty()) {
        rendered += "\n\n";
        rendered += merged_system;
    }
    rendered += "<|im_end|>\n";
    return rendered;
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
    if (!is_allowed_file_role(role)) {
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
    const bool has_tools = !options.tool_jsons.empty();
    std::size_t skip_system = 0;
    if (has_tools && (messages[0].role == "system")) {
        std::string merged_system = trim_ascii_whitespace(messages[0].content);
        skip_system               = 1;
        if (messages.size() > 1 && messages[1].role == "system") {
            const std::string second = trim_ascii_whitespace(messages[1].content);
            if (!second.empty()) {
                if (!merged_system.empty()) { merged_system += '\n'; }
                merged_system += second;
            }
            skip_system = 2;
        }
        rendered += render_tools_system_block(options.tool_jsons, merged_system);
    } else if (has_tools) {
        rendered += render_tools_system_block(options.tool_jsons, "");
    }

    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (has_tools && i < skip_system) { continue; }
        if (has_tools && message.role == "system") { continue; }
        if (!is_allowed_role(message.role)) {
            throw std::invalid_argument("unsupported chat role: " + message.role);
        }
        if (message.role == "tool") {
            const bool opens_group = i == 0 || messages[i - 1].role != "tool";
            const bool closes_group = i + 1 == messages.size() || messages[i + 1].role != "tool";
            if (opens_group) { rendered += "<|im_start|>user"; }
            rendered += "\n<tool_response>\n";
            rendered += trim_ascii_whitespace(message.content);
            rendered += "\n</tool_response>";
            if (closes_group) { rendered += "<|im_end|>\n"; }
            continue;
        }
        rendered += "<|im_start|>";
        rendered += message.role;
        rendered += '\n';
        const std::string content = trim_ascii_whitespace(message.content);
        rendered += content;
        if (!message.tool_calls.empty()) {
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    if (!content.empty()) { rendered += "\n\n"; }
                } else {
                    rendered += "\n";
                }
                rendered += render_tool_call(message.tool_calls[call_index]);
            }
        }
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
