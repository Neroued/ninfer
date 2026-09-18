#include "models/registry.h"

#include <stdexcept>
#include <string>

namespace ninfer::models {

Architecture resolve_architecture(std::string_view architecture, std::string_view model_type) {
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
    case Architecture::Qwen3_5:
        return "Qwen3_5ForCausalLM";
    case Architecture::Qwen3_5Moe:
        return "Qwen3_5MoeForCausalLM";
    }
    return {};
}

} // namespace ninfer::models
