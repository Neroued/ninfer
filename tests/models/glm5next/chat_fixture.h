#pragma once

#include "ninfer/types.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace glm_fixture {
using namespace ninfer;
using Json = nlohmann::ordered_json;

inline std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}

inline void require(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}

template <class Action>
void rejects(Action action, std::string_view reason) {
    try {
        action();
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()).find(reason) != std::string_view::npos,
                "unexpected rejection: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected rejection: " + std::string(reason));
}

inline ChatRole role(std::string_view name) {
    if (name == "system") return ChatRole::System;
    if (name == "developer") return ChatRole::Developer;
    if (name == "user") return ChatRole::User;
    if (name == "assistant") return ChatRole::Assistant;
    if (name == "tool") return ChatRole::Tool;
    throw std::invalid_argument("invalid fixture role");
}

inline ChatMessage message(ChatRole role, std::string content) {
    return {.role = role, .parts = {{.kind = MessagePartKind::Text, .text = std::move(content)}}};
}

inline PromptInput prompt(const Json& input) {
    PromptInput result;
    for (const auto& item : input.at("messages")) {
        ChatMessage m;
        m.role              = role(item.at("role").get<std::string>());
        const auto& content = item.at("content");
        if (content.is_string()) {
            m.parts.push_back({.kind = MessagePartKind::Text, .text = content.get<std::string>()});
        } else {
            for (const auto& part : content) {
                const auto type = part.at("type").get<std::string>();
                if (type == "text")
                    m.parts.push_back({.kind = MessagePartKind::Text,
                                       .text = part.at("text").get<std::string>()});
                else {
                    require(type == "image" || type == "video", "invalid fixture part");
                    m.parts.push_back(
                        {.kind  = MessagePartKind::Media,
                         .media = {.kind = type == "image" ? MediaKind::Image : MediaKind::Video}});
                }
            }
        }
        m.reasoning_content    = item.value("reasoning_content", std::string{});
        m.tool_call_id         = item.value("tool_call_id", std::string{});
        m.tool_result_is_error = item.value("tool_result_is_error", false);
        for (const auto& call : item.value("tool_calls", Json::array()))
            m.tool_calls.push_back({.id   = call.at("id").get<std::string>(),
                                    .name = call.at("function").at("name").get<std::string>(),
                                    .arguments_json = call.at("function").at("arguments").dump()});
        result.messages.push_back(std::move(m));
    }
    const auto& opts = input.at("options");
    auto& options    = result.options;
    if (opts.value("continuation", false))
        options.continuation = PromptContinuationMode::ContinueFinalAssistant;
    if (opts.contains("enable_thinking"))
        options.enable_thinking = opts.at("enable_thinking").get<bool>();
    if (opts.contains("preserve_thinking"))
        options.preserve_thinking = opts.at("preserve_thinking").get<bool>();
    if (opts.contains("reasoning_effort")) {
        bool found = false;
        for (auto effort : {ReasoningEffort::None, ReasoningEffort::Minimal, ReasoningEffort::Low,
                            ReasoningEffort::Medium, ReasoningEffort::High, ReasoningEffort::XHigh,
                            ReasoningEffort::Max}) {
            if (reasoning_effort_name(effort) == opts["reasoning_effort"].get<std::string>()) {
                options.reasoning_effort = effort;
                found                    = true;
            }
        }
        require(found, "invalid fixture effort");
    }
    if (opts.contains("defaults")) {
        const auto& defaults = opts["defaults"];
        if (defaults.contains("enable_thinking"))
            options.defaults.enable_thinking = defaults["enable_thinking"].get<bool>();
        if (defaults.contains("preserve_thinking"))
            options.defaults.preserve_thinking = defaults["preserve_thinking"].get<bool>();
    }
    if (opts.contains("chat_template_kwargs"))
        options.chat_template_kwargs_json = opts.at("chat_template_kwargs").dump();
    for (const auto& tool : opts.value("tools", Json::array()))
        options.tool_jsons.push_back(tool.dump());
    return result;
}

} // namespace glm_fixture
