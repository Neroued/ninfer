#pragma once

#include "models/common/qwen/frontend/tokenizer.h"

#include <memory>
#include <optional>
#include <string_view>

namespace ninfer::models::qwen {

// Byte views borrow the immutable ModelDefinition's resource storage. Tokenizer owns its
// decoded vocabulary and tables; loading and request preparation share one interpretation.
struct FrontendResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
    std::string_view preprocessor_config_json;
    std::string_view video_preprocessor_config_json;
    std::shared_ptr<const frontend::Tokenizer> tokenizer;
    std::uint32_t public_token_count = 0;
};

struct MediaGeometry {
    std::uint32_t patch_size;
    std::uint32_t temporal_patch_size;
    std::uint32_t spatial_merge_size;
};

void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows,
                     std::optional<MediaGeometry> vision);

} // namespace ninfer::models::qwen
