#pragma once

#include <string_view>

namespace ninfer::models {

enum class Architecture { Qwen3_5, Qwen3_5Moe };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
