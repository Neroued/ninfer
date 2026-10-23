#include "models/glm5next/frontend/resources.h"

#include "artifact/binder.h"
#include "artifact/schema.h"

#include <cmath>
#include <limits>

namespace ninfer::models::glm5next {

void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows) {
    models::frontend::parse_text_resources(resources.text, embedding_rows);
    const auto generation =
        artifact::parse_json(resources.text.generation_config_json, "generation_config.json");
    SamplingPreset sampling;
    try {
        sampling.temperature = generation.value("temperature", 1.0F);
        sampling.top_p       = generation.value("top_p", 0.95F);
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
                              sampling.presence_penalty, sampling.frequency_penalty})
        if (!std::isfinite(value))
            throw artifact::ArtifactError("generation defaults must fit finite float values");
    resources.sampling = sampling;
    resources.processor =
        resources.processor_config_json.empty()
            ? std::nullopt
            : std::optional(parse_processor_config(resources.processor_config_json));
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

} // namespace ninfer::models::glm5next
