#include "product/logging/configuration.h"

#include "product/logging/pretty_format.h"

#include <spdlog/logger.h>

#include <iomanip>
#include <algorithm>
#include <sstream>
#include <type_traits>

namespace ninfer::product {
namespace {

std::string value_text(const ConfigurationItem& item) {
    return std::visit(
        [&](const auto& value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, bool>)
                return value ? "true" : "false";
            else if constexpr (std::is_same_v<T, std::string>)
                return value.empty() ? "\"\"" : format_pretty_text(value);
            else if constexpr (std::is_arithmetic_v<T>) {
                if (item.unit == ConfigurationUnit::Bytes) return format_pretty_bytes(value);
                if (item.unit == ConfigurationUnit::Milliseconds)
                    return format_pretty_duration(static_cast<double>(value) / 1000.0);
                std::ostringstream text;
                text << std::setprecision(7) << value;
                return text.str();
            } else {
                std::ostringstream text;
                text << '[';
                for (std::size_t i = 0; i < value.size(); ++i) {
                    if (i) text << ", ";
                    if constexpr (std::is_same_v<typename T::value_type, std::string>)
                        text << std::quoted(format_pretty_text(value[i]));
                    else
                        text << value[i];
                }
                text << ']';
                return text.str();
            }
        },
        item.value);
}

} // namespace

std::string format_configuration(const StartupConfigurationView& view) {
    std::ostringstream out;
    out << "configuration  "
        << (view.entry == EntryPoint::Cli     ? "cli"
            : view.entry == EntryPoint::Serve ? "serve"
                                              : "bench")
        << '\n';
    if (!view.config_path.empty())
        out << "  file         " << format_pretty_text(view.config_path.string()) << '\n';
    std::string previous;
    std::size_t label_width = 28;
    for (const auto& item : view.items)
        label_width = std::max(label_width, item.key.size() - item.key.find('.') + 1);
    for (const auto& item : view.items) {
        const auto dot   = item.key.find('.');
        const auto group = item.key.substr(0, dot);
        out << std::left << std::setw(15) << (group == previous ? "" : group)
            << std::setw(label_width) << item.key.substr(dot + 1) << value_text(item);
        if (item.source.kind == ConfigSourceKind::File)
            out << "  [file:" << item.source.line << ']';
        else if (item.source.kind == ConfigSourceKind::CommandLine)
            out << "  [argv:" << item.source.argument << ']';
        else
            out << "  [default]";
        out << '\n';
        previous = group;
    }
    for (const auto& [name, value] : view.environment)
        out << "environment    " << name << " = " << format_pretty_text(value) << '\n';
    return out.str();
}

void log_configuration(spdlog::logger& logger, const StartupConfigurationView& view) {
    if (logger.should_log(spdlog::level::info)) logger.info("\n{}", format_configuration(view));
}

std::string format_context_capacity(const EngineDescription& description) {
    std::uint64_t device_bytes = 0, host_bytes = 0;
    for (const auto& pool : description.context_pools) {
        if (pool.domain == ContextMemoryDomain::Device)
            device_bytes += pool.capacity_bytes;
        else
            host_bytes += pool.capacity_bytes;
    }
    std::ostringstream out;
    out << "context | device " << format_pretty_bytes(device_bytes) << " across "
        << description.ranks.size() << (description.ranks.size() == 1 ? " rank" : " ranks");
    if (host_bytes) out << " | host " << format_pretty_bytes(host_bytes);
    out << " | KV ~" << format_pretty_count(description.context_token_capacity_estimate)
        << " tokens | ~" << std::fixed << std::setprecision(1)
        << static_cast<double>(description.context_token_capacity_estimate) /
               description.max_context
        << " x " << description.max_context << "-token contexts";
    return out.str();
}

void log_context_capacity(spdlog::logger& logger, const EngineDescription& description) {
    logger.info("{}", format_context_capacity(description));
    for (const auto& pool : description.context_pools)
        logger.debug("context rank {} | {} {} | block {}", pool.rank,
                     pool.domain == ContextMemoryDomain::Device ? "device" : "host",
                     format_pretty_bytes(pool.capacity_bytes),
                     format_pretty_bytes(pool.block_bytes));
}

void log_resolved_configuration(spdlog::logger& logger, const EngineDescription& description) {
    logger.info("resolved       prefill chunk={}", description.prefill_chunk);
    log_context_capacity(logger, description);
}

} // namespace ninfer::product
