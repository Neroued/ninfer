#pragma once

#include <string_view>

namespace ninfer::models {

enum class Architecture { Qwen3_5, Qwen4Exp };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
// The MTP component architecture that pairs with a registered Text architecture name (empty for
// an unregistered one).
[[nodiscard]] std::string_view mtp_architecture(std::string_view architecture) noexcept;
// The canonical Text architecture name of an implementation.
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
