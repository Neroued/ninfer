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

bool starts_with(const std::string& text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string lstrip_newlines(std::string text) {
    std::size_t begin = 0;
    while (begin < text.size() && text[begin] == '\n') { ++begin; }
    return text.substr(begin);
}

std::string rstrip_newlines(std::string text) {
    std::size_t end = text.size();
    while (end > 0 && text[end - 1] == '\n') { --end; }
    return text.substr(0, end);
}

// Split an assistant turn into (reasoning, content) exactly as the Qwen3.6 jinja
// does when reasoning_content is not provided: reasoning is the text between the
// last <think> and the first </think>; content is everything after the last
// </think>. When there is no </think> the whole thing is content and reasoning is
// empty.
struct ThinkParts {
    std::string reasoning;
    std::string content;
};

ThinkParts derive_think_parts(const std::string& content) {
    ThinkParts parts;
    const std::size_t first_close = content.find("</think>");
    if (first_close == std::string::npos) {
        parts.content = content;
        return parts;
    }
    // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
    std::string before = rstrip_newlines(content.substr(0, first_close));
    const std::size_t last_open = before.rfind("<think>");
    std::string reasoning =
        (last_open == std::string::npos) ? before
                                         : before.substr(last_open + std::string("<think>").size());
    parts.reasoning = lstrip_newlines(std::move(reasoning));
    // content = content.split('</think>')[-1].lstrip('\n')
    const std::size_t last_close = content.rfind("</think>");
    parts.content = lstrip_newlines(content.substr(last_close + std::string("</think>").size()));
    return parts;
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
    if (item.contains("name")) {
        throw std::invalid_argument("message " + std::to_string(index) + " contains name");
    }
    for (auto it = item.begin(); it != item.end(); ++it) {
        if (it.key() != "role" && it.key() != "content" && it.key() != "reasoning_content") {
            throw std::invalid_argument("message " + std::to_string(index) +
                                        " contains unsupported field: " + it.key());
        }
    }
    if (!item.contains("role") || !item.contains("content")) {
        throw std::invalid_argument("message " + std::to_string(index) +
                                    " must contain role and content");
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
    ChatMessage message{role, content};
    if (item.contains("reasoning_content")) {
        if (!item.at("reasoning_content").is_string()) {
            throw std::invalid_argument("message " + std::to_string(index) +
                                        " reasoning_content must be a string");
        }
        message.reasoning_content = item.at("reasoning_content").get<std::string>();
    }
    return message;
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

    const auto is_system_role = [](const std::string& role) {
        return role == "system" || role == "developer";
    };

    // Merge up to the first two leading system/developer messages; any later ones
    // are dropped. developer folds into system. (jinja lines 45-57)
    std::size_t num_sys = 0;
    std::string merged_system;
    if (is_system_role(messages[0].role)) {
        const std::string first = trim_ascii_whitespace(messages[0].content);
        if (messages.size() > 1 && is_system_role(messages[1].role)) {
            const std::string second = trim_ascii_whitespace(messages[1].content);
            merged_system = first + "\n" + second;
            num_sys       = 2;
        } else {
            merged_system = first;
            num_sys       = 1;
        }
    }

    std::string rendered;
    const bool has_tools = !options.tool_jsons.empty();
    if (has_tools) {
        rendered += render_tools_system_block(options.tool_jsons, merged_system);
    } else if (!merged_system.empty()) {
        rendered += "<|im_start|>system\n";
        rendered += merged_system;
        rendered += "<|im_end|>\n";
    }

    // last_query_index = index of the last user message whose (trimmed) content is
    // not a bare <tool_response>...</tool_response> wrapper. (jinja lines 76-86)
    long last_query_index = static_cast<long>(messages.size()) - 1;
    bool multi_step_tool  = true;
    for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
        if (!multi_step_tool) { break; }
        const ChatMessage& message = messages[static_cast<std::size_t>(i)];
        if (message.role == "user") {
            const std::string content = trim_ascii_whitespace(message.content);
            if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
                multi_step_tool  = false;
                last_query_index = i;
            }
        }
    }

    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (i < num_sys) { continue; }
        if (is_system_role(message.role)) { continue; }
        if (!is_allowed_role(message.role)) {
            throw std::invalid_argument("unsupported chat role: " + message.role);
        }
        const std::string content = trim_ascii_whitespace(message.content);
        if (message.role == "user") {
            rendered += "<|im_start|>user\n";
            rendered += content;
            rendered += "<|im_end|>\n";
            continue;
        }
        if (message.role == "tool") {
            const bool opens_group  = i > 0 && messages[i - 1].role != "tool";
            const bool closes_group = i + 1 == messages.size() || messages[i + 1].role != "tool";
            if (opens_group) { rendered += "<|im_start|>user"; }
            rendered += "\n<tool_response>\n";
            rendered += content;
            rendered += "\n</tool_response>";
            if (closes_group) { rendered += "<|im_end|>\n"; }
            continue;
        }

        // assistant
        std::string reasoning;
        std::string body = content;
        if (!message.reasoning_content.empty()) {
            reasoning = message.reasoning_content;
        } else {
            ThinkParts parts = derive_think_parts(content);
            reasoning        = std::move(parts.reasoning);
            body             = std::move(parts.content);
        }
        reasoning = trim_ascii_whitespace(reasoning);

        const bool keep_thinking =
            options.preserve_thinking || (static_cast<long>(i) > last_query_index);
        rendered += "<|im_start|>assistant\n";
        if (keep_thinking) {
            rendered += "<think>\n";
            rendered += reasoning;
            rendered += "\n</think>\n\n";
        }
        rendered += body;
        if (!message.tool_calls.empty()) {
            const bool body_has_text = !trim_ascii_whitespace(body).empty();
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    if (body_has_text) { rendered += "\n\n"; }
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
