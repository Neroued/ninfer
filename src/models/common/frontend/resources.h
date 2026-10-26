#pragma once

#include "ninfer/types.h"
#include "text/tokenizer.h"

#include <memory>
#include <string_view>
#include <vector>

namespace ninfer::models::frontend {

// Resource views borrow ModelDefinition storage; parsed vocabulary and policies own their data.
struct TextResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
    std::shared_ptr<const text::Tokenizer> tokenizer;
    std::shared_ptr<const std::vector<TokenId>> control_tokens;
    std::uint32_t public_token_count = 0;
    std::vector<TokenId> stop_token_ids;
};

std::vector<TokenId> parse_stop_tokens(std::string_view generation_config,
                                       const text::Tokenizer& tokenizer);

void parse_text_resources(TextResources& resources, std::uint32_t embedding_rows);

void prepare_control_tokens(TextResources&, std::string_view text, std::string_view close);

} // namespace ninfer::models::frontend
