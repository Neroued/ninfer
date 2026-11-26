#include "models/inkling2/frontend/processor_config.h"

#include "artifact/schema.h"

#include <cmath>
#include <limits>

namespace ninfer::models::inkling2 {
namespace {
using Json = nlohmann::ordered_json;

std::uint32_t positive(const Json& json, const char* key, std::uint32_t fallback) {
    const auto value = json.find(key);
    if (value == json.end()) return fallback;
    if (!value->is_number_integer() || *value <= 0 ||
        *value > std::numeric_limits<std::uint32_t>::max())
        throw artifact::ArtifactError(std::string("invalid Inkling2 processor ") + key);
    return value->get<std::uint32_t>();
}
} // namespace

ProcessorConfig parse_processor_config(std::string_view source) {
    const auto json = artifact::parse_json(source, "processor_config.json");
    try {
        ProcessorConfig out;
        if (!json.is_object())
            throw artifact::ArtifactError("Inkling2 processor_config.json must be an object");
        if (const auto tok = json.find("image_token"); tok != json.end()) {
            if (!tok->is_string() || tok->get_ref<const std::string&>().empty())
                throw artifact::ArtifactError("Inkling2 processor image_token must be nonempty");
            out.image_token = tok->get<std::string>();
        }
        const auto& ip = json.at("image_processor");
        if (!ip.is_object())
            throw artifact::ArtifactError("Inkling2 image_processor must be an object");
        if (const auto size = ip.find("size"); size != ip.end()) {
            if (!size->is_object())
                throw artifact::ArtifactError("Inkling2 image_processor.size must be an object");
            out.cell_height = positive(*size, "height", out.cell_height);
            out.cell_width  = positive(*size, "width", out.cell_width);
        }
        if (out.cell_height != out.cell_width || out.cell_height % out.merge_size != 0) {
            throw artifact::ArtifactError(
                "Inkling2 image_processor requires square cells divisible by merge_size=2");
        }
        out.native_patch_size = out.cell_height / out.merge_size;
        out.mean              = ip.value("image_mean", out.mean);
        out.std               = ip.value("image_std", out.std);
        out.rescale_factor    = ip.value("rescale_factor", out.rescale_factor);
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(out.mean[c]) || !std::isfinite(out.std[c]) || out.std[c] <= 0.0F)
                throw artifact::ArtifactError("invalid Inkling2 processor normalization");
        }
        for (const char* flag : {"do_convert_rgb", "do_rescale", "do_normalize"}) {
            if (!ip.value(flag, true))
                throw artifact::ArtifactError(std::string("Inkling2 processor requires ") + flag);
        }
        if (!std::isfinite(out.rescale_factor) ||
            std::abs(out.rescale_factor - (1.0 / 255.0)) > 1e-9) {
            throw artifact::ArtifactError("Inkling2 processor requires 1/255 rescaling");
        }
        const auto area = std::uint64_t(out.native_patch_size) * out.native_patch_size;
        if (area > std::numeric_limits<std::uint32_t>::max() / 3U)
            throw artifact::ArtifactError("Inkling2 patch size exceeds numeric representation");
        return out;
    } catch (const nlohmann::json::exception& error) {
        throw artifact::ArtifactError(std::string("invalid Inkling2 processor config: ") +
                                      error.what());
    }
}

} // namespace ninfer::models::inkling2
