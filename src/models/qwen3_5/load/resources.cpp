#include "models/qwen3_5/load/bindings.h"
#include "models/common/qwen/frontend/resources.h"

namespace ninfer::models::qwen3_5::loading {

qwen::FrontendResources bind_resources(artifact::Binder& binder, const Config& config) {
    const auto resource = [&](std::string_view component, std::string_view role) {
        const auto bytes = binder.resource_bytes(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    qwen::FrontendResources out;
    out.text.tokenizer_json         = resource("text", "tokenizer.json");
    out.text.tokenizer_config_json  = resource("text", "tokenizer_config.json");
    out.text.chat_template_jinja    = resource("text", "chat_template.jinja");
    out.text.generation_config_json = resource("text", "generation_config.json");
    if (config.vision) {
        out.preprocessor_config_json       = resource("vision", "preprocessor_config.json");
        out.video_preprocessor_config_json = resource("vision", "video_preprocessor_config.json");
    }
    std::optional<qwen::MediaGeometry> vision;
    if (config.vision) {
        const auto& v = *config.vision;
        vision        = {v.patch_size, v.temporal_patch_size, v.spatial_merge_size};
    }
    qwen::parse_resources(out, config.text.vocab_size, vision);
    if (config.draft && config.draft->dflash2 &&
        config.draft->dflash2->selector_top_k > out.text.public_token_count) {
        throw artifact::ArtifactError("DFlash2 selector_top_k exceeds public token domain");
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
