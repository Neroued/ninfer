#include "models/inkling2/frontend/resources.h"

#include "artifact/binder.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::models::inkling2 {

std::uint32_t FrontendResources::control_token_limit() const {
    return static_cast<std::uint32_t>(std::max({
        text.control_tokens ? text.control_tokens->size() : std::size_t{0},
        control_to_text ? control_to_text->size() : std::size_t{0},
        control_to_header ? control_to_header->size() : std::size_t{0},
    }));
}

namespace {

TokenId require_single_token(const text::Tokenizer& tokenizer, std::string_view piece) {
    const auto ids = tokenizer.encode(piece);
    if (ids.size() != 1 || tokenizer.decode(ids) != piece) {
        throw artifact::ArtifactError("Inkling2 tokenizer must expose single framing token: " +
                                      std::string(piece));
    }
    return ids.front();
}

std::optional<TokenId> optional_single_token(const text::Tokenizer& tokenizer,
                                             std::string_view piece) {
    const auto ids = tokenizer.encode(piece);
    if (ids.size() == 1 && tokenizer.decode(ids) == piece) return ids.front();
    return std::nullopt;
}

void reject_stop_collision(const std::vector<TokenId>& stops,
                           const std::vector<TokenId>& control_tokens) {
    for (const TokenId id : control_tokens) {
        if (std::find(stops.begin(), stops.end(), id) != stops.end()) {
            throw artifact::ArtifactError("Inkling2 thinking control contains a terminal token");
        }
    }
}

} // namespace

void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows) {
    models::frontend::parse_text_resources(resources.text, embedding_rows);
    const auto& tokenizer = *resources.text.tokenizer;

    resources.processor =
        resources.processor_config_json.empty()
            ? std::nullopt
            : std::optional(parse_processor_config(resources.processor_config_json));

    FramingTokens framing;
    framing.message_model            = require_single_token(tokenizer, kMessageModel);
    framing.content_thinking         = require_single_token(tokenizer, kContentThinking);
    framing.content_text             = require_single_token(tokenizer, kContentText);
    framing.content_invoke_tool_json = require_single_token(tokenizer, kContentInvokeToolJson);
    framing.end_message              = require_single_token(tokenizer, kEndMessage);
    framing.content_image            = require_single_token(tokenizer, kContentImage);
    const std::string_view image_tok =
        resources.processor ? std::string_view(resources.processor->image_token)
                            : kDefaultImagePadToken;
    framing.image_pad                = require_single_token(tokenizer, image_tok);
    framing.content_tool_error       = optional_single_token(tokenizer, kContentToolError);
    framing.content_invoke_tool_text = optional_single_token(tokenizer, kContentInvokeToolText);
    resources.framing                = framing;

    auto to_text = std::make_shared<std::vector<TokenId>>(
        std::initializer_list<TokenId>{framing.end_message, framing.message_model,
                                       framing.content_text});
    auto to_header = std::make_shared<std::vector<TokenId>>(
        std::initializer_list<TokenId>{framing.end_message, framing.message_model});
    reject_stop_collision(resources.text.stop_token_ids, *to_text);
    reject_stop_collision(resources.text.stop_token_ids, *to_header);
    resources.control_to_text     = to_text;
    resources.control_to_header   = to_header;
    resources.text.control_tokens = std::move(to_text);

    const auto generation =
        artifact::parse_json(resources.text.generation_config_json, "generation_config.json");
    SamplingPreset sampling;
    try {
        sampling.temperature = generation.value("temperature", 1.0F);
        sampling.top_p       = generation.value("top_p", sampling.top_p);
        if (const auto top_k = generation.find("top_k"); top_k != generation.end()) {
            if (!top_k->is_number_integer() || *top_k < std::numeric_limits<std::int32_t>::min() ||
                *top_k > std::numeric_limits<std::int32_t>::max())
                throw artifact::ArtifactError("generation_config.json top_k must fit int32");
            sampling.top_k = top_k->get<std::int32_t>();
        }
        sampling.min_p            = generation.value("min_p", sampling.min_p);
        sampling.presence_penalty = generation.value("presence_penalty", sampling.presence_penalty);
        sampling.frequency_penalty =
            generation.value("frequency_penalty", sampling.frequency_penalty);
        if (!generation.value("do_sample", true)) sampling.temperature = 0.0F;
    } catch (const nlohmann::json::exception& error) {
        throw artifact::ArtifactError(std::string("invalid generation defaults: ") + error.what());
    }
    for (const float value : {sampling.temperature, sampling.top_p, sampling.min_p,
                              sampling.presence_penalty, sampling.frequency_penalty}) {
        if (!std::isfinite(value))
            throw artifact::ArtifactError("generation defaults must fit finite float values");
    }
    resources.sampling = sampling;
}

FrontendResources bind_resources(artifact::Binder& binder, std::uint32_t embedding_rows,
                                 bool vision) {
    const auto bytes = [&](std::string_view component, std::string_view role) {
        const auto data = binder.resource_bytes(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(data.data()), data.size());
    };
    FrontendResources resources;
    resources.text.tokenizer_json         = bytes("text", "tokenizer.json");
    resources.text.tokenizer_config_json  = bytes("text", "tokenizer_config.json");
    resources.text.chat_template_jinja    = bytes("text", "chat_template.jinja");
    resources.text.generation_config_json = bytes("text", "generation_config.json");
    if (vision) resources.processor_config_json = bytes("vision", "processor_config.json");
    parse_resources(resources, embedding_rows);
    return resources;
}

} // namespace ninfer::models::inkling2
