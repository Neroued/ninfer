#include "models/registry.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::models {
namespace {

// The checkpoint names of one implementation: Text architecture, its model_type, and the
// architecture of its MTP component.
struct ArchitectureNames {
    std::string_view text;
    std::string_view model_type;
    std::string_view mtp;
    Architecture architecture;
};

// One entry per Architecture; `text` is its canonical name (architecture_name()).
constexpr std::array kNames{
    ArchitectureNames{"Qwen3_5ForCausalLM", "qwen3_5_text", "Qwen3_5MTP", Architecture::Qwen3_5},
    ArchitectureNames{"Qwen4ExpForCausalLM", "qwen4_exp_text", "Qwen4ExpMTP",
                      Architecture::Qwen4Exp},
};

} // namespace

Architecture resolve_architecture(std::string_view architecture, std::string_view model_type) {
    for (const auto& names : kNames) {
        if (names.text == architecture && names.model_type == model_type) {
            return names.architecture;
        }
    }
    std::string supported;
    for (const auto& names : kNames) {
        supported += (supported.empty() ? "" : ", ") + std::string(names.text) + "/" +
                     std::string(names.model_type);
    }
    throw std::invalid_argument("unsupported architecture " + std::string(architecture) + "/" +
                                std::string(model_type) + " (supported: " + supported + ")");
}

std::string_view mtp_architecture(std::string_view architecture) noexcept {
    for (const auto& names : kNames) {
        if (names.text == architecture) { return names.mtp; }
    }
    return {};
}

std::string_view architecture_name(Architecture architecture) noexcept {
    for (const auto& names : kNames) {
        if (names.architecture == architecture) { return names.text; }
    }
    return {};
}

} // namespace ninfer::models
