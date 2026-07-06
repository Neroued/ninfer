#include "qus/serve/translate.h"

#include "qus/serve/openai_schema.h"

namespace qus::serve {
namespace {

std::vector<std::string> effective_tool_jsons(const GenerationRequest& req) {
    std::vector<std::string> out;
    if (!req.uses_tools()) { return out; }
    if (req.tool_choice.mode == ToolChoiceMode::Named) {
        for (const ToolDefinition& tool : req.tools) {
            if (tool.name == req.tool_choice.name) {
                out.push_back(tool.definition_json);
                return out;
            }
        }
        return out;
    }
    out.reserve(req.tools.size());
    for (const ToolDefinition& tool : req.tools) { out.push_back(tool.definition_json); }
    return out;
}

} // namespace

std::vector<qus::text::ChatMessage> to_chat_messages(const GenerationRequest& req) {
    std::vector<qus::text::ChatMessage> out;
    out.reserve(req.messages.size());
    for (const ChatTurn& turn : req.messages) {
        // OpenAI's newer schema uses `developer` for instruction messages; the Qwen
        // template has no such role, so fold it into `system`.
        std::string role = turn.role == "developer" ? "system" : turn.role;
        if (role != "system" && role != "user" && role != "assistant" && role != "tool") {
            ApiError error;
            error.message = "unsupported role: " + turn.role;
            error.param   = "messages";
            error.code    = "unsupported_role";
            throw ApiException(std::move(error));
        }
        std::string text;
        for (const ContentPart& part : turn.content) {
            if (part.kind != ContentKind::Text) {
                ApiError error;
                error.message = "content type '" + part.type_raw + "' is not supported yet";
                error.param   = "messages";
                error.code    = "modality_not_supported";
                throw ApiException(std::move(error));
            }
            if (!text.empty() && !part.text.empty()) { text += '\n'; }
            text += part.text;
        }
        qus::text::ChatMessage message;
        message.role         = std::move(role);
        message.content      = std::move(text);
        message.tool_call_id = turn.tool_call_id;
        message.tool_calls.reserve(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            message.tool_calls.push_back(
                qus::text::ToolCall{call.id, call.name, call.arguments_json});
        }
        out.push_back(std::move(message));
    }
    return out;
}

qus::text::TextGenerationOptions to_generation_options(const GenerationRequest& req,
                                                       bool default_enable_thinking) {
    qus::text::TextGenerationOptions options;
    options.max_new_tokens  = req.max_tokens;
    options.raw_output      = false;
    options.enable_thinking = req.enable_thinking.value_or(default_enable_thinking);
    options.preserve_special_tokens = req.uses_tools() || req.has_tool_history();
    options.render_options.tool_jsons = effective_tool_jsons(req);
    options.render_options.enable_thinking = options.enable_thinking;
    options.stop_token_ids.clear();  // runner resolves tokenizer default stop ids
    return options;
}

const char* finish_reason_wire(qus::text::FinishReason reason) {
    switch (reason) {
    case qus::text::FinishReason::Length:
        return "length";
    case qus::text::FinishReason::Stop:
    case qus::text::FinishReason::Cancelled:
    default:
        return "stop";
    }
}

} // namespace qus::serve
