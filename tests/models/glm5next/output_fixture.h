#pragma once

#include "chat_fixture.h"
#include "models/glm5next/frontend/frontend.h"
#include "models/glm5next/frontend/resources.h"
#include "text/tokenizer.h"

namespace glm_fixture {
namespace glm = ninfer::models::glm5next;

inline const auto fixtures =
    std::filesystem::path(NINFER_SOURCE_DIR) / "tests/models/glm5next/fixtures";

inline glm::Frontend output_frontend(const std::filesystem::path& path,
                                     std::shared_ptr<const ninfer::text::Tokenizer>& tokenizer) {
    const auto encoded = read(path);
    const auto source  = read(fixtures / "chat_template.jinja");
    const auto json    = Json::parse(encoded);
    std::uint32_t rows = 0;
    std::vector<TokenId> eos;
    for (const auto& item : json["model"]["vocab"].items())
        rows = std::max(rows, item.value().get<std::uint32_t>() + 1);
    for (const auto& token : json["added_tokens"]) {
        rows             = std::max(rows, token["id"].get<std::uint32_t>() + 1);
        const auto value = token["content"].get<std::string>();
        if (value == "<|endoftext|>" || value == "<|user|>" || value == "<|observation|>")
            eos.push_back(token["id"].get<TokenId>());
    }
    const auto generation = Json{{"eos_token_id", eos}}.dump();
    glm::FrontendResources resources{.text = {.tokenizer_json         = encoded,
                                              .tokenizer_config_json  = "{}",
                                              .chat_template_jinja    = source,
                                              .generation_config_json = generation}};
    glm::parse_resources(resources, rows);
    tokenizer = resources.text.tokenizer;
    return glm::make_frontend(resources, {.vision_enabled = false, .max_context = 16384});
}

inline Json tool_schema(Json properties, std::vector<std::string> required = {}) {
    return {{"type", "object"},
            {"properties", std::move(properties)},
            {"required", std::move(required)},
            {"additionalProperties", false}};
}

inline std::string declaration(const Json& schema, bool strict = false, std::string name = "call") {
    return Json{{"type", "function"},
                {"function", {{"name", name}, {"parameters", schema}, {"strict", strict}}}}
        .dump();
}

inline PromptInput input(std::vector<std::string> tools = {}) {
    PromptInput p;
    p.messages.push_back(message(ChatRole::User, "question"));
    p.options.tool_jsons = std::move(tools);
    return p;
}

struct Output {
    std::string reasoning, content;
    std::vector<GeneratedToolCall> calls;

    void append(const std::vector<OutputEvent>& events) {
        for (const auto& event : events) {
            if (const auto* text = std::get_if<OutputDelta>(&event))
                (text->channel == OutputChannel::Reasoning ? reasoning : content) += text->text;
            else if (const auto* call = std::get_if<ToolCallStart>(&event)) {
                require(call->index == calls.size(), "call event index is not append-only");
                calls.push_back({call->name, {}, ToolCallStatus::InProgress});
            } else if (const auto* args = std::get_if<ToolArgumentsDelta>(&event))
                calls.at(args->index).arguments_json += args->text;
            else if (const auto* end = std::get_if<ToolCallEnd>(&event))
                calls.at(end->index).status = end->status;
        }
    }
};

inline void commit(models::frontend::OutputSession& session, std::span<const TokenId> ids,
                   Output& result, std::uint32_t budget = 100000) {
    (void)session.preview_model(ids, budget, FinishReason::OutputLimit);
    result.append(session.commit_preview());
    if (auto error = session.output_error()) std::rethrow_exception(error->exception());
}
} // namespace glm_fixture
