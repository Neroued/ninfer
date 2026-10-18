#pragma once

#include "product/config/options.h"

#include <array>
#include <span>

namespace ninfer::product::config_detail {

enum class Field {
    ModelPath,
    ChatTemplate,
    Vision,
    Device,
    MaxContext,
    KvDtype,
    KvCapacity,
    PrefillChunk,
    CudaGraph,
    MaxConcurrency,
    MaxPendingRequests,
    PendingTimeout,
    PrefixReuse,
    ExtraStateSlots,
    HostCapacity,
    CostPresets,
    SpecBackend,
    DraftTokens,
    ProposalHead,
    MediaCache,
    MediaLive,
    MediaThreads,
    MaxTokens,
    Thinking,
    PreserveThinking,
    ThinkingBudget,
    Temperature,
    TopK,
    TopP,
    MinP,
    PresencePenalty,
    FrequencyPenalty,
    Seed,
    ServerHost,
    ServerPort,
    ApiKey,
    ModelId,
    Cors,
    MaxRequestSize,
    ResponseRecords,
    ResponseSize,
    LogLevel,
    StatsInterval,
    RequestJsonl,
    Prompt,
    Messages,
    ReasoningEffort,
    RawOutput,
    PrintTokenIds,
    StopIds,
    Stop,
    ReasoningStop,
    ConstraintKind,
    GrammarFile,
    JsonSchemaFile,
    Regex,
    Choices,
    Count,
};

enum class Scope { Common, Cli, Serve };

struct FieldSpec {
    Field field;
    std::string_view key;
    std::string_view flag;
    std::string_view negative_flag;
    std::string_view argument;
    std::string_view description;
    Scope scope            = Scope::Common;
    ConfigurationUnit unit = ConfigurationUnit::None;
};

struct DefaultValue {};

using FieldValue = std::variant<DefaultValue, bool, std::uint64_t, double, std::string,
                                std::vector<std::uint32_t>, std::vector<std::string>>;
using RawValue   = std::variant<bool, std::int64_t, double, std::string, std::vector<std::int64_t>,
                                std::vector<std::string>>;

struct Assignment {
    FieldValue value;
    ConfigSource source;
};

using StartupPatch = std::array<std::optional<Assignment>, static_cast<std::size_t>(Field::Count)>;

[[nodiscard]] std::span<const FieldSpec> fields();
[[nodiscard]] const FieldSpec& field_spec(Field field);
[[nodiscard]] bool applies_to(const FieldSpec& spec, EntryPoint entry);
[[nodiscard]] std::uint32_t parse_duration_ms(const RawValue& value, bool positive);
[[nodiscard]] FieldValue parse_field_value(Field field, const RawValue& value, bool command_line);

struct FileConfiguration {
    StartupPatch fields;
    std::optional<DeploymentOptions> deployment;
    std::map<std::string, ConfigSource, std::less<>> deployment_sources;
};

[[nodiscard]] FileConfiguration read_config_file(const std::filesystem::path& path);
[[noreturn]] void field_error(const FieldSpec& spec, const ConfigSource& source,
                              const std::filesystem::path& config_path, std::string_view message);

inline auto& assignment(StartupPatch& patch, Field field) {
    return patch[static_cast<std::size_t>(field)];
}

inline const auto& assignment(const StartupPatch& patch, Field field) {
    return patch[static_cast<std::size_t>(field)];
}

} // namespace ninfer::product::config_detail
