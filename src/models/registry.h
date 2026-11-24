#pragma once

#include <string_view>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models {

enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp, Glm5Next };

[[nodiscard]] Architecture resolve_architecture(const artifact::Directory&);
[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
