#pragma once

#include "product/config/options.h"

namespace ninfer::product {

[[nodiscard]] std::string format_configuration(const StartupConfigurationView& view);
void log_configuration(spdlog::logger& logger, const StartupConfigurationView& view);
[[nodiscard]] std::string format_context_capacity(const EngineDescription& description);
void log_context_capacity(spdlog::logger& logger, const EngineDescription& description);
void log_resolved_configuration(spdlog::logger& logger, const EngineDescription& description);

} // namespace ninfer::product
