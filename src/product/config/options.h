#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::product {

enum class EntryPoint { Cli, Serve, Bench };
enum class ConfigSourceKind { Default, File, CommandLine };

struct ConfigSource {
    ConfigSourceKind kind = ConfigSourceKind::Default;
    std::uint32_t line    = 0;
    std::uint32_t column  = 0;
    int argument          = 0;
};

struct StartupProvenance {
    std::filesystem::path config_path;
    std::vector<std::string> argv;
    std::map<std::string, ConfigSource, std::less<>> fields;
};

struct GenerationDefaults {
    std::uint32_t max_tokens = 128;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    std::optional<std::uint32_t> thinking_budget;
    // A missing seed requests a fresh seed when each request is formed.
    SamplingOverrides sampling;
};

struct StartupLoggingOptions {
    LogLevel level                  = LogLevel::Info;
    std::uint32_t stats_interval_ms = 5000;
    std::filesystem::path request_jsonl;
};

struct LaunchOptions {
    EngineOptions engine;
    GenerationDefaults generation;
    StartupLoggingOptions logging;
    StartupProvenance provenance;
};

[[nodiscard]] LaunchOptions default_launch_options(EntryPoint entry);
[[nodiscard]] std::uint64_t random_sampling_seed();
[[nodiscard]] DeploymentOptions read_deployment_file(const std::filesystem::path&);

enum class ConstraintKind { None, Grammar, JsonObject, JsonSchema, Regex, Choice };

struct ConstraintOptions {
    ConstraintKind kind = ConstraintKind::None;
    std::filesystem::path path;
    std::string regex;
    std::vector<std::string> choices;
};

struct CliOptions {
    bool help_requested  = false;
    LaunchOptions launch = default_launch_options(EntryPoint::Cli);
    std::string prompt;
    std::filesystem::path messages_path;
    std::optional<ReasoningEffort> reasoning_effort;
    bool raw_output      = false;
    bool print_token_ids = false;
    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;
    ConstraintOptions constraint;
};

inline constexpr std::uint32_t kDefaultServeMaxTokens     = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;

struct ServeOptions {
    bool help_requested  = false;
    LaunchOptions launch = default_launch_options(EntryPoint::Serve);
    std::string host     = "127.0.0.1";
    int port             = 8080;
    std::string api_key;
    std::optional<std::string> model_id_override;
    bool enable_cors                       = false;
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
};

[[nodiscard]] CliOptions parse_cli_options(int argc, char** argv);
[[nodiscard]] ServeOptions parse_serve_options(int argc, char** argv);

struct CommandLineOption {
    std::string_view flag;
    bool takes_value = true;
};

struct CommandLineArgument {
    std::string flag;
    std::string value;
};

struct ParsedLaunchOptions {
    bool help_requested = false;
    LaunchOptions launch;
    std::vector<CommandLineArgument> extra_arguments;
};

// Parse shared startup settings and preserve application-owned arguments in input order.
[[nodiscard]] ParsedLaunchOptions
parse_launch_options(int argc, char** argv, EntryPoint entry,
                     std::span<const CommandLineOption> extra_options = {});
[[nodiscard]] std::string startup_usage_text(std::string_view executable, EntryPoint entry);

using ConfigurationValue = std::variant<bool, std::uint64_t, double, std::string,
                                        std::vector<std::uint32_t>, std::vector<std::string>>;
enum class ConfigurationUnit { None, Bytes, Milliseconds };

struct ConfigurationItem {
    std::string key;
    ConfigurationValue value;
    ConfigSource source;
    ConfigurationUnit unit = ConfigurationUnit::None;
};

struct StartupConfigurationView {
    EntryPoint entry;
    std::filesystem::path config_path;
    std::vector<std::string> argv;
    std::vector<ConfigurationItem> items;
    std::map<std::string, std::string> environment;
};

[[nodiscard]] StartupConfigurationView configuration_view(const CliOptions& options);
[[nodiscard]] StartupConfigurationView configuration_view(const ServeOptions& options);
[[nodiscard]] StartupConfigurationView configuration_view(const LaunchOptions& options,
                                                          EntryPoint entry);

} // namespace ninfer::product
