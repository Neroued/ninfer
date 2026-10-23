#include "models/glm5next/frontend/processor_config.h"
#include "artifact/schema.h"

#include <cmath>
#include <limits>

namespace ninfer::models::glm5next {
namespace {
using Json = nlohmann::ordered_json;

std::uint32_t positive(const Json& json, const char* key, std::uint32_t fallback) {
    const auto value = json.find(key);
    if (value == json.end()) return fallback;
    if (!value->is_number_integer() || *value <= 0 ||
        *value > std::numeric_limits<std::uint32_t>::max())
        throw artifact::ArtifactError(std::string("invalid GLM processor ") + key);
    return value->get<std::uint32_t>();
}

MediaProcessorConfig media_config(const Json& json, std::uint32_t max_tokens) {
    if (!json.is_object()) throw artifact::ArtifactError("GLM media processor must be an object");
    MediaProcessorConfig out;
    out.patch_size          = positive(json, "patch_size", out.patch_size);
    out.temporal_patch_size = positive(json, "temporal_patch_size", out.temporal_patch_size);
    out.merge_size          = positive(json, "merge_size", out.merge_size);
    out.patch_expand_factor = positive(json, "patch_expand_factor", out.patch_expand_factor);
    out.min_tokens          = positive(json, "min_image_tokens", out.min_tokens);
    out.max_tokens          = positive(json, "max_image_tokens", max_tokens);
    out.mean                = json.value("image_mean", out.mean);
    out.std                 = json.value("image_std", out.std);
    if (out.min_tokens > out.max_tokens || out.temporal_patch_size != 2)
        throw artifact::ArtifactError(
            "GLM processor requires ordered token bounds and temporal patch size 2");
    const auto area    = std::uint64_t(out.patch_size) * out.patch_size;
    const auto spatial = std::uint64_t(out.patch_size) * out.merge_size;
    if (area > std::numeric_limits<std::uint32_t>::max() / (3 * out.temporal_patch_size) ||
        spatial > std::numeric_limits<int>::max() / out.patch_expand_factor ||
        out.max_tokens >
            std::numeric_limits<std::size_t>::max() / (out.temporal_patch_size * spatial * spatial))
        throw artifact::ArtifactError("GLM processor geometry exceeds its numeric representation");
    for (int c = 0; c < 3; ++c)
        if (!std::isfinite(out.mean[c]) || !std::isfinite(out.std[c]) || out.std[c] <= 0)
            throw artifact::ArtifactError("invalid GLM processor normalization");
    for (const char* flag : {"do_resize", "do_rescale", "do_normalize", "do_convert_rgb"})
        if (!json.value(flag, true))
            throw artifact::ArtifactError(std::string("GLM processor requires ") + flag);
    if (json.value("resize_mode", std::string("pad")) != "pad" || json.value("resample", 3) != 3 ||
        json.value("rescale_factor", 1.0 / 255.0) != 1.0 / 255.0)
        throw artifact::ArtifactError("GLM processor requires bicubic pad and 1/255 rescaling");
    return out;
}
} // namespace

ProcessorConfig parse_processor_config(std::string_view source) {
    const auto json = artifact::parse_json(source, "processor_config.json");
    try {
        ProcessorConfig out;
        out.image     = media_config(json.at("image_processor"), 8000);
        out.video     = media_config(json.at("video_processor"), 240000);
        out.video_fps = json.at("video_processor").value("fps", 2.0);
        out.max_video_frames =
            positive(json.at("video_processor"), "max_frame_count_dynamic", 2048);
        if (!std::isfinite(out.video_fps) || out.video_fps <= 0 ||
            out.max_video_frames > std::numeric_limits<int>::max())
            throw artifact::ArtifactError("invalid GLM video sampling configuration");
        if (out.image.patch_size != out.video.patch_size ||
            out.image.temporal_patch_size != out.video.temporal_patch_size ||
            out.image.merge_size != out.video.merge_size)
            throw artifact::ArtifactError("GLM image/video patch geometry must agree");
        return out;
    } catch (const nlohmann::json::exception& error) {
        throw artifact::ArtifactError(std::string("invalid GLM processor config: ") + error.what());
    }
}
} // namespace ninfer::models::glm5next
