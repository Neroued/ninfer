#include "qus/serve/translate.h"

#include "qus/serve/openai_schema.h"

#include <random>

namespace qus::serve {
namespace {

// A fresh 64-bit seed for requests that omit `seed` and where the operator did
// not pin one with --seed, so successive regenerations of the same prompt differ.
std::uint64_t random_seed() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng();
}

// Resolve the effective sampler: request fields override server defaults; omitted
// fields fall back to the Qwen3 thinking defaults the server was configured with.
// --greedy forces exact argmax (temperature 0) regardless of the request. The
// engine fills token_counts for the penalty path itself, so it stays null here.
qus::kernels::SamplingConfig resolve_sampling(const SamplingParams& req,
                                              const ServeOptions& server) {
    qus::kernels::SamplingConfig cfg;  // default is greedy (temperature 0)
    if (server.greedy) { return cfg; }
    cfg.temperature =
        static_cast<float>(req.temperature.value_or(server.sampling_temperature));
    cfg.top_p = static_cast<float>(req.top_p.value_or(server.sampling_top_p));
    cfg.top_k = req.top_k.value_or(server.sampling_top_k);
    cfg.min_p = 0.0f;
    cfg.presence_penalty =
        static_cast<float>(req.presence_penalty.value_or(server.sampling_presence_penalty));
    cfg.frequency_penalty =
        static_cast<float>(req.frequency_penalty.value_or(server.sampling_frequency_penalty));
    if (req.seed.has_value()) {
        cfg.seed = *req.seed;
    } else if (server.sampling_seed.has_value()) {
        cfg.seed = *server.sampling_seed;
    } else {
        cfg.seed = random_seed();
    }
    return cfg;
}

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
        message.role              = std::move(role);
        message.content           = std::move(text);
        message.reasoning_content = turn.reasoning_content;
        message.tool_call_id      = turn.tool_call_id;
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
                                                       const ServeOptions& server) {
    qus::text::TextGenerationOptions options;
    options.max_new_tokens  = req.max_tokens;
    options.raw_output      = false;
    options.enable_thinking = req.enable_thinking.value_or(server.enable_thinking);
    options.preserve_special_tokens = req.uses_tools() || req.has_tool_history();
    options.render_options.tool_jsons = effective_tool_jsons(req);
    options.render_options.enable_thinking = options.enable_thinking;
    options.stop_token_ids.clear();  // runner resolves tokenizer default stop ids
    options.sampling = resolve_sampling(req.sampling, server);
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
