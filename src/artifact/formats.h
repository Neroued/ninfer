#pragma once

#include "core/weight.h"

#include <string_view>

namespace ninfer::artifact {

[[nodiscard]] QType parse_format(std::string_view name);
[[nodiscard]] QuantLayout parse_layout(std::string_view name);
[[nodiscard]] std::string_view format_name(QType format) noexcept;
[[nodiscard]] std::string_view layout_name(QuantLayout layout) noexcept;

inline constexpr std::string_view kRawBytesEncoding = "raw_bytes_v1";

} // namespace ninfer::artifact
