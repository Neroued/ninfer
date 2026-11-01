#include "models/registry.h"
#include "artifact/schema.h"

#include <stdexcept>
#include <string>

namespace ninfer::models {

Architecture resolve_architecture(const artifact::Directory& directory) {
    const auto& config = directory.component("text").config;
    const auto& names  = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw artifact::ArtifactError("architectures must name one implementation");
    }
    return resolve_architecture(artifact::require_id(names[0], "architecture"),
                                artifact::require_id(config.at("model_type"), "model_type"));
}

Architecture resolve_architecture(std::string_view architecture, std::string_view model_type) {
    if (architecture == "Glm5NextForCausalLM" && model_type == "glm5next_text")
        return Architecture::Glm5Next;
    if (architecture == "Qwen4ExpForCausalLM" && model_type == "qwen4_exp_text") {
        return Architecture::Qwen4Exp;
    }
    if (architecture == "Qwen3_5ForCausalLM" && model_type == "qwen3_5_text") {
        return Architecture::Qwen3_5;
    }
    if (architecture == "Qwen3_5MoeForCausalLM" && model_type == "qwen3_5_moe_text") {
        return Architecture::Qwen3_5Moe;
    }
    throw std::invalid_argument("unsupported architecture/config pair " +
                                std::string(architecture) + "/" + std::string(model_type));
}

std::string_view architecture_name(Architecture architecture) noexcept {
    switch (architecture) {
    case Architecture::Glm5Next:
        return "Glm5NextForCausalLM";
    case Architecture::Qwen4Exp:
        return "Qwen4ExpForCausalLM";
    case Architecture::Qwen3_5:
        return "Qwen3_5ForCausalLM";
    case Architecture::Qwen3_5Moe:
        return "Qwen3_5MoeForCausalLM";
    }
    return {};
}

} // namespace ninfer::models
