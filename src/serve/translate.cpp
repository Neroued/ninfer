#include "qus/serve/translate.h"

#include "qus/serve/openai_schema.h"

namespace qus::serve {

std::vector<qus::text::ChatMessage> to_chat_messages(const GenerationRequest& req) {
    std::vector<qus::text::ChatMessage> out;
    out.reserve(req.messages.size());
    for (const ChatTurn& turn : req.messages) {
        if (turn.role != "system" && turn.role != "user" && turn.role != "assistant") {
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
        out.push_back(qus::text::ChatMessage{turn.role, std::move(text)});
    }
    return out;
}

qus::text::TextGenerationOptions to_generation_options(const GenerationRequest& req,
                                                       bool default_enable_thinking) {
    qus::text::TextGenerationOptions options;
    options.max_new_tokens  = req.max_tokens;
    options.raw_output      = false;
    options.enable_thinking = req.enable_thinking.value_or(default_enable_thinking);
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
