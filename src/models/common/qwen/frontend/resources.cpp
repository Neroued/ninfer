#include "models/common/qwen/frontend/resources.h"

#include "artifact/schema.h"

namespace ninfer::models::qwen {
namespace {
using Json                      = nlohmann::json;
namespace fi                    = frontend;
constexpr double kRescaleFactor = 1.0 / 255.0;
constexpr double kVideoFps      = 2.0;
constexpr int kVideoMinFrames   = 4;
constexpr int kVideoMaxFrames   = 768;

Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

std::int64_t require_integer(const Json& object, std::string_view field,
                             std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number_integer()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an integer");
    }
    return object.at(key).get<std::int64_t>();
}

double require_number(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be a number");
    }
    return object.at(key).get<double>();
}

double number_or_default(const Json& object, std::string_view field, std::string_view resource,
                         double default_value) {
    const std::string key(field);
    return object.contains(key) ? require_number(object, field, resource) : default_value;
}

std::int64_t integer_or_default(const Json& object, std::string_view field,
                                std::string_view resource, std::int64_t default_value) {
    const std::string key(field);
    return object.contains(key) ? require_integer(object, field, resource) : default_value;
}

const Json& require_object(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_object()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an object");
    }
    return object.at(key);
}

std::uint64_t positive_u64(std::int64_t value, std::string_view field) {
    if (value <= 0) { throw std::invalid_argument(std::string(field) + " must be positive"); }
    return static_cast<std::uint64_t>(value);
}

void validate_pixel_pipeline(const Json& config, std::string_view resource) {
    for (const char* flag : {"do_resize", "do_rescale", "do_normalize", "do_convert_rgb"}) {
        if (config.contains(flag) && (!config[flag].is_boolean() || !config[flag].get<bool>())) {
            throw std::invalid_argument(std::string(resource) + "." + flag +
                                        " must be enabled for the compiled pixel pipeline");
        }
    }
    if (config.contains("resample") &&
        (!config["resample"].is_number_integer() || config["resample"].get<int>() != 3)) {
        throw std::invalid_argument(std::string(resource) +
                                    ".resample must select bicubic interpolation");
    }
    if (require_integer(config, "patch_size", resource) != 16 ||
        require_integer(config, "temporal_patch_size", resource) != 2 ||
        require_integer(config, "merge_size", resource) != 2) {
        throw std::invalid_argument(std::string(resource) +
                                    " does not match the compiled Vision patch geometry");
    }
    const auto require_half_triplet = [&](std::string_view field) {
        const std::string key(field);
        if (!config.contains(key) || !config.at(key).is_array() || config.at(key).size() != 3) {
            throw std::invalid_argument(std::string(resource) + "." + key +
                                        " must contain three values");
        }
        for (const Json& value : config.at(key)) {
            if (!value.is_number() || value.get<double>() != 0.5) {
                throw std::invalid_argument(std::string(resource) + "." + key +
                                            " does not match the compiled normalization");
            }
        }
    };
    require_half_triplet("image_mean");
    require_half_triplet("image_std");
    if (number_or_default(config, "rescale_factor", resource, kRescaleFactor) != kRescaleFactor) {
        throw std::invalid_argument(std::string(resource) +
                                    ".rescale_factor does not match the compiled normalization");
    }
}

} // namespace

frontend::ProcessorOptions parse_processor_options(std::string_view image_json,
                                                   std::string_view video_json) {
    const Json image = parse_resource_json(image_json, "preprocessor_config.json");
    const Json video = parse_resource_json(video_json, "video_preprocessor_config.json");
    validate_pixel_pipeline(image, "preprocessor_config.json");
    validate_pixel_pipeline(video, "video_preprocessor_config.json");

    const Json& image_size = require_object(image, "size", "preprocessor_config.json");
    const Json& video_size = require_object(video, "size", "video_preprocessor_config.json");

    fi::ProcessorOptions options;
    options.image_min_pixels =
        positive_u64(require_integer(image_size, "shortest_edge", "preprocessor_config.json.size"),
                     "image shortest_edge");
    options.image_max_pixels =
        positive_u64(require_integer(image_size, "longest_edge", "preprocessor_config.json.size"),
                     "image longest_edge");
    options.video_min_pixels = positive_u64(
        require_integer(video_size, "shortest_edge", "video_preprocessor_config.json.size"),
        "video shortest_edge");
    options.video_max_pixels = positive_u64(
        require_integer(video_size, "longest_edge", "video_preprocessor_config.json.size"),
        "video longest_edge");
    options.video_fps =
        number_or_default(video, "fps", "video_preprocessor_config.json", kVideoFps);
    options.video_min_frames = static_cast<int>(
        integer_or_default(video, "min_frames", "video_preprocessor_config.json", kVideoMinFrames));
    options.video_max_frames = static_cast<int>(
        integer_or_default(video, "max_frames", "video_preprocessor_config.json", kVideoMaxFrames));
    if (options.video_fps != kVideoFps || options.video_min_frames != kVideoMinFrames ||
        options.video_max_frames != kVideoMaxFrames) {
        throw std::invalid_argument(
            "video_preprocessor_config.json does not match registered sampling defaults");
    }

    return options;
}

void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows,
                     std::optional<MediaGeometry> vision) {
    models::frontend::parse_text_resources(resources.text, embedding_rows);
    if (vision) {
        for (const auto bytes :
             {resources.preprocessor_config_json, resources.video_preprocessor_config_json}) {
            const auto value = artifact::parse_json(bytes, "Vision processor");
            if (!value.is_object() || !value.contains("patch_size") ||
                !value.contains("temporal_patch_size") || !value.contains("merge_size")) {
                throw artifact::ArtifactError("Vision processor is missing patch geometry");
            }
            if (artifact::require_u64(value.at("patch_size"), "patch_size", true) !=
                    vision->patch_size ||
                artifact::require_u64(value.at("temporal_patch_size"), "temporal_patch_size",
                                      true) != vision->temporal_patch_size ||
                artifact::require_u64(value.at("merge_size"), "merge_size", true) !=
                    vision->spatial_merge_size) {
                throw artifact::ArtifactError(
                    "Vision processor geometry differs from component config");
            }
        }
        resources.processor = parse_processor_options(resources.preprocessor_config_json,
                                                      resources.video_preprocessor_config_json);
    }
}

} // namespace ninfer::models::qwen
