#include "product/logging/configuration.h"

#include "product/logging/pretty_format.h"

#include <spdlog/logger.h>

#include <iomanip>
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
    out << "configuration  " << (view.entry == EntryPoint::Cli ? "cli" : "serve") << '\n';
    if (!view.config_path.empty())
        out << "  file         " << format_pretty_text(view.config_path.string()) << '\n';
    std::string previous;
    for (const auto& item : view.items) {
        const auto dot   = item.key.find('.');
        const auto group = item.key.substr(0, dot);
        out << std::left << std::setw(15) << (group == previous ? "" : group) << std::setw(28)
            << item.key.substr(dot + 1) << value_text(item);
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

void log_resolved_configuration(spdlog::logger& logger, const EngineOptions& options,
                                const MemorySummary& memory) {
    logger.info("resolved       prefill chunk={}  KV={} tokens  StateImages={}+{}  Host={}",
                options.prefill_chunk, memory.kv_capacity, options.max_concurrency,
                options.context_cache.extra_state_slots.value(),
                format_pretty_bytes(options.context_cache.host_capacity_bytes.value()));
}

} // namespace ninfer::product
