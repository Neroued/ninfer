#include "models/common/frontend/resources.h"
#include "artifact/schema.h"
#include <limits>
#include <utility>

namespace ninfer::models::frontend {

std::vector<TokenId> parse_stop_tokens(std::string_view generation_config,
                                       const text::Tokenizer& tokenizer) {
    const auto generation = artifact::parse_json(generation_config, "generation_config.json");
    if (!generation.is_object() || !generation.contains("eos_token_id"))
        throw artifact::ArtifactError("missing field eos_token_id in generation_config.json");
    std::vector<TokenId> stops;
    const auto append = [&](const auto& value) {
        const auto id = artifact::require_u64(value, "eos_token_id");
        if (id > std::uint64_t(std::numeric_limits<TokenId>::max()) ||
            !tokenizer.is_valid_token(static_cast<TokenId>(id)))
            throw artifact::ArtifactError("stop token is outside the public token domain");
        stops.push_back(static_cast<TokenId>(id));
    };
    const auto& eos = generation.at("eos_token_id");
    if (eos.is_array()) {
        if (eos.empty()) throw artifact::ArtifactError("eos_token_id must not be empty");
        for (const auto& id : eos) append(id);
    } else
        append(eos);
    return stops;
}

void parse_text_resources(TextResources& resources, std::uint32_t embedding_rows) {
    auto tokenizer = std::make_shared<const text::Tokenizer>(
        text::TokenizerResources{resources.tokenizer_json, resources.tokenizer_config_json});
    const auto count = tokenizer->vocab_size();
    if (!count || count > embedding_rows || !tokenizer->has_exact_token_domain(count))
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within embedding rows");
    auto stops                   = parse_stop_tokens(resources.generation_config_json, *tokenizer);
    resources.tokenizer          = std::move(tokenizer);
    resources.public_token_count = static_cast<std::uint32_t>(count);
    resources.stop_token_ids     = std::move(stops);
}

} // namespace ninfer::models::frontend
