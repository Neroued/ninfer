#include "product/config/schema.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::product::config_detail {
namespace {

constexpr FieldSpec kFields[] = {
    {Field::ModelPath, "model.path", "", "", "FILE", "Model artifact (.ninfer)."},
    {Field::ChatTemplate, "model.chat_template", "--chat-template", "", "FILE",
     "Chat template override; empty uses the artifact."},
    {Field::Vision, "model.vision", "--vision", "--no-vision", "", "Load Vision resources."},
    {Field::Device, "engine.device", "--device", "", "N", "Local CUDA device ordinal."},
    {Field::MaxContext, "engine.max_context", "--max-context", "", "N",
     "Per-request context limit (8192)."},
    {Field::KvDtype, "engine.kv_dtype", "--kv-dtype", "", "bf16|int8|fp8|nvfp4|k8v4",
     "KV representation."},
    {Field::KvCapacity, "engine.kv_capacity", "--kv-capacity", "", "N|context|auto",
     "Shared Main KV capacity; context follows --max-context."},
    {Field::PrefillChunk, "engine.prefill_chunk", "--prefill-chunk", "", "N|default",
     "Positive multiple of 128; default is resolved by the model."},
    {Field::CudaGraph, "engine.cuda_graph", "--cuda-graph", "--no-cuda-graph", "",
     "Enable CUDA Graph execution."},
    {Field::MaxConcurrency, "engine.max_concurrency", "--max-concurrency", "", "N",
     "Resident execution lanes (1..8)."},
    {Field::MaxPendingRequests, "engine.max_pending_requests", "--max-pending-requests", "", "N",
     "Pending request capacity (16)."},
    {Field::PendingTimeout, "engine.pending_timeout", "--pending-timeout", "", "DURATION",
     "Queue timeout before execution (30s).", Scope::Common, ConfigurationUnit::Milliseconds},
    {Field::PrefixReuse, "context_cache.prefix_reuse", "--prefix-reuse", "--no-prefix-reuse", "",
     "Cross-request history reuse; also accepts =default."},
    {Field::ExtraStateSlots, "context_cache.extra_state_slots", "--extra-state-slots", "",
     "N|default", "Extra complete StateImages beyond active lanes."},
    {Field::HostCapacity, "context_cache.host_capacity", "--host-cache-capacity", "",
     "SIZE|default", "Shared Host State/KV and in-flight capacity; 0B disables backing.",
     Scope::Common, ConfigurationUnit::Bytes},
    {Field::CostPresets, "context_cache.cost_presets", "--context-cost-presets", "", "FILE",
     "Context-cost JSON overrides; empty uses built-in defaults."},
    {Field::SpecBackend, "speculative.backend", "--spec", "", "none|mtp|dflash|dflash2",
     "Speculative backend; none clears drafts and proposal head."},
    {Field::DraftTokens, "speculative.draft_tokens", "--draft-tokens", "", "N",
     "MTP: 1..5; DFlash/DFlash2: 1..15; none: 0."},
    {Field::ProposalHead, "speculative.proposal_head", "--proposal-head", "", "full|optimized",
     "Proposal head representation."},
    {Field::MediaCache, "media.cache_capacity", "--media-cache-capacity", "", "SIZE",
     "Retained media cache (1GiB); 0B disables reuse.", Scope::Common, ConfigurationUnit::Bytes},
    {Field::MediaLive, "media.live_capacity", "--media-live-capacity", "", "SIZE",
     "All live BF16 patch payloads, including retained cache (2GiB).", Scope::Common,
     ConfigurationUnit::Bytes},
    {Field::MediaThreads, "media.preprocess_threads", "--media-preprocess-threads", "", "N|auto",
     "Explicit workers: 1..64; auto: at most 16."},
    {Field::MaxTokens, "generation.max_tokens", "--max-tokens", "", "N|default",
     "Output budget; CLI 128, serve request default 8192."},
    {Field::Thinking, "generation.thinking", "--thinking", "--no-thinking", "",
     "Thinking override; also accepts =model."},
    {Field::PreserveThinking, "generation.preserve_thinking", "--preserve-thinking",
     "--no-preserve-thinking", "", "Retain closed-turn reasoning; also accepts =model."},
    {Field::ThinkingBudget, "generation.thinking_budget", "--thinking-budget", "", "N|none",
     "Model-origin thinking token budget."},
    {Field::Temperature, "generation.sampling.temperature", "--temperature", "", "F|model",
     "Sampling temperature [0,2]; 0 selects greedy."},
    {Field::TopK, "generation.sampling.top_k", "--top-k", "", "N|model",
     "Candidate limit [0,20]; 0 uses the current limit of 20."},
    {Field::TopP, "generation.sampling.top_p", "--top-p", "", "F|model",
     "Cumulative probability cutoff [0,1]."},
    {Field::MinP, "generation.sampling.min_p", "--min-p", "", "F|model",
     "Relative probability cutoff [0,1]."},
    {Field::PresencePenalty, "generation.sampling.presence_penalty", "--presence-penalty", "",
     "F|model", "Presence penalty [-2,2]."},
    {Field::FrequencyPenalty, "generation.sampling.frequency_penalty", "--frequency-penalty", "",
     "F|model", "Frequency penalty [-2,2]."},
    {Field::Seed, "generation.sampling.seed", "--seed", "", "N|default|random",
     "uint64 seed; CLI default 0, serve default random per request."},
    {Field::ServerHost, "server.host", "--host", "", "ADDRESS", "Listen address (127.0.0.1).",
     Scope::Serve},
    {Field::ServerPort, "server.port", "--port", "", "N", "Listen port (8080).", Scope::Serve},
    {Field::ApiKey, "server.api_key", "--api-key", "", "KEY",
     "Authentication key; empty disables authentication.", Scope::Serve},
    {Field::ModelId, "server.model_id", "--model-id", "", "ID",
     "Public model alias; empty uses the artifact name.", Scope::Serve},
    {Field::Cors, "server.cors", "--cors", "--no-cors", "", "Permissive CORS headers.",
     Scope::Serve},
    {Field::MaxRequestSize, "server.max_request_size", "--max-request-size", "", "SIZE",
     "HTTP body limit before JSON parsing (384MiB).", Scope::Serve, ConfigurationUnit::Bytes},
    {Field::ResponseRecords, "server.responses.max_records", "--response-store-max-records", "",
     "N", "Responses store record limit (1024).", Scope::Serve},
    {Field::ResponseSize, "server.responses.max_size", "--response-store-max-size", "", "SIZE",
     "Responses store capacity (256MiB).", Scope::Serve, ConfigurationUnit::Bytes},
    {Field::LogLevel, "logging.level", "--log-level", "", "LEVEL",
     "trace/debug/info/warning/error/critical/off."},
    {Field::StatsInterval, "logging.stats_interval", "--stats-interval", "", "DURATION",
     "Throughput log interval (5s); 0ms disables it.", Scope::Serve,
     ConfigurationUnit::Milliseconds},
    {Field::RequestJsonl, "logging.request_jsonl", "--request-log-jsonl", "", "FILE",
     "Structured startup/request log; empty disables it.", Scope::Serve},
    {Field::Prompt, "cli.prompt", "--prompt", "", "TEXT",
     "Prompt text, mutually exclusive with --messages.", Scope::Cli},
    {Field::Messages, "cli.messages", "--messages", "", "FILE",
     "Messages/tools JSON; local media paths are relative to that file.", Scope::Cli},
    {Field::ReasoningEffort, "cli.reasoning_effort", "--reasoning-effort", "", "EFFORT|model",
     "none/minimal/low/medium/high/xhigh/max; none disables thinking.", Scope::Cli},
    {Field::RawOutput, "cli.raw_output", "--raw-output", "--no-raw-output", "",
     "Emit the raw output representation.", Scope::Cli},
    {Field::PrintTokenIds, "cli.print_token_ids", "--print-token-ids", "--no-print-token-ids", "",
     "Print generated token IDs to stderr.", Scope::Cli},
    {Field::StopIds, "cli.stop_token_ids", "--stop-token-id", "", "N",
     "Token stop condition; repeat to append.", Scope::Cli},
    {Field::Stop, "cli.stop", "--stop", "", "TEXT", "Content stop string; repeat to append.",
     Scope::Cli},
    {Field::ReasoningStop, "cli.reasoning_stop", "--reasoning-stop", "", "TEXT",
     "Reasoning stop string; repeat to append.", Scope::Cli},
    {Field::ConstraintKind, "cli.constraint.kind", "--constraint", "", "none",
     "Clear the inherited output constraint.", Scope::Cli},
    {Field::GrammarFile, "cli.constraint.grammar_file", "--grammar-file", "", "FILE",
     "GBNF output grammar.", Scope::Cli},
    {Field::JsonSchemaFile, "cli.constraint.json_schema_file", "--json-schema-file", "", "FILE",
     "JSON Schema output constraint.", Scope::Cli},
    {Field::Regex, "cli.constraint.regex", "--regex", "", "PATTERN",
     "Regex output constraint; empty is a valid regex.", Scope::Cli},
    {Field::Choices, "cli.constraint.choices", "--choice", "", "TEXT",
     "Literal output choice; repeat to append, empty entries allowed.", Scope::Cli},
};
static_assert(std::size(kFields) == static_cast<std::size_t>(Field::Count));

constexpr auto kU32Max = std::numeric_limits<std::uint32_t>::max();
constexpr auto kU64Max = std::numeric_limits<std::uint64_t>::max();

[[noreturn]] void invalid(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

std::uint64_t decimal(std::string_view text) {
    std::uint64_t result = 0;
    const auto parsed    = std::from_chars(text.data(), text.data() + text.size(), result);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        invalid("expected an unsigned decimal integer");
    return result;
}

std::uint64_t integer(const RawValue& value, bool allow_string, std::uint64_t min,
                      std::uint64_t max) {
    std::uint64_t result = 0;
    if (const auto* number = std::get_if<std::int64_t>(&value)) {
        if (*number < 0) invalid("expected a nonnegative integer");
        result = static_cast<std::uint64_t>(*number);
    } else if (const auto* text = std::get_if<std::string>(&value); text && allow_string) {
        result = decimal(*text);
    } else
        invalid("expected an integer");
    if (result < min || result > max)
        invalid("expected an integer in [" + std::to_string(min) + "," + std::to_string(max) + "]");
    return result;
}

double real(const RawValue& value, bool command_line, double min, double max) {
    double result = 0;
    if (const auto* number = std::get_if<double>(&value))
        result = *number;
    else if (const auto* number = std::get_if<std::int64_t>(&value))
        result = static_cast<double>(*number);
    else if (const auto* text = std::get_if<std::string>(&value); text && command_line) {
        const auto parsed = std::from_chars(text->data(), text->data() + text->size(), result);
        if (text->empty() || parsed.ec != std::errc{} || parsed.ptr != text->data() + text->size())
            invalid("expected a finite number");
    } else
        invalid("expected a number");
    if (!std::isfinite(result) || result < min || result > max)
        invalid("expected a finite number in [" + std::to_string(min) + "," + std::to_string(max) +
                "]");
    return result;
}

std::string text(const RawValue& value, bool nonempty = false) {
    const auto* result = std::get_if<std::string>(&value);
    if (!result || (nonempty && result->empty()))
        invalid(nonempty ? "expected a nonempty string" : "expected a string");
    return *result;
}

std::string enumeration(const RawValue& value, std::initializer_list<std::string_view> choices) {
    const auto result = text(value);
    if (std::find(choices.begin(), choices.end(), result) == choices.end())
        invalid("invalid enum value");
    return result;
}

std::uint64_t quantity(const RawValue& value, bool bytes, bool positive) {
    if (std::holds_alternative<std::int64_t>(value))
        return integer(value, false, positive ? 1 : 0, bytes ? kU64Max : kU32Max);
    const auto* string = std::get_if<std::string>(&value);
    if (!string) invalid("expected an integer or a quantity string");
    const std::string_view input(*string);
    const auto split  = input.find_first_not_of("0123456789.");
    const auto number = input.substr(0, split);
    auto unit         = split == input.npos ? std::string_view{} : input.substr(split);
    while (!unit.empty() && unit.front() == ' ') unit.remove_prefix(1);
    std::uint64_t multiplier = 1;
    if (bytes) {
        if (unit == "KiB")
            multiplier = 1ULL << 10;
        else if (unit == "MiB")
            multiplier = 1ULL << 20;
        else if (unit == "GiB")
            multiplier = 1ULL << 30;
        else if (!unit.empty() && unit != "B")
            invalid("expected bytes or a B/KiB/MiB/GiB capacity");
    } else {
        if (unit == "s")
            multiplier = 1000;
        else if (!unit.empty() && unit != "ms")
            invalid("expected milliseconds or an ms/s duration");
    }
    const auto point = number.find('.');
    const auto whole = decimal(number.substr(0, point));
    auto fraction    = point == number.npos ? std::string_view{} : number.substr(point + 1);
    if (point != number.npos &&
        (fraction.empty() || fraction.find_first_not_of("0123456789") != fraction.npos))
        invalid("expected a nonnegative decimal quantity");
    while (!fraction.empty() && fraction.back() == '0') fraction.remove_suffix(1);
    if (fraction.size() > 30) invalid("quantity must resolve to whole bytes or milliseconds");
    unsigned __int128 numerator = 0;
    unsigned __int128 divisor   = 1;
    for (char digit : fraction) {
        numerator = numerator * 10 + static_cast<unsigned>(digit - '0');
        divisor *= 10;
    }
    // Cancel before multiplication, preserving exact budgets across the entire uint64 range.
    auto denominator = divisor;
    auto scale       = static_cast<unsigned __int128>(multiplier);
    while (scale != 0) {
        const auto remainder = denominator % scale;
        denominator          = scale;
        scale                = remainder;
    }
    const auto common = denominator;
    divisor /= common;
    if (numerator % divisor != 0) invalid("quantity must resolve to whole bytes or milliseconds");
    const auto fractional =
        static_cast<std::uint64_t>((numerator / divisor) * (multiplier / common));
    const std::uint64_t maximum = bytes ? kU64Max : kU32Max;
    if (whole > maximum / multiplier || fractional > maximum - whole * multiplier)
        invalid("quantity is out of range");
    const auto result = whole * multiplier + fractional;
    if (positive && result == 0) invalid("quantity must be positive");
    return result;
}

} // namespace

std::span<const FieldSpec> fields() { return kFields; }

const FieldSpec& field_spec(Field field) { return kFields[static_cast<std::size_t>(field)]; }

bool applies_to(const FieldSpec& spec, EntryPoint entry) {
    return spec.scope == Scope::Common || (spec.scope == Scope::Cli && entry == EntryPoint::Cli) ||
           (spec.scope == Scope::Serve && entry == EntryPoint::Serve);
}

std::uint32_t parse_duration_ms(const RawValue& value, bool positive) {
    return static_cast<std::uint32_t>(quantity(value, false, positive));
}

FieldValue parse_field_value(Field field, const RawValue& value, bool command_line) {
    const auto* string = std::get_if<std::string>(&value);
    const auto is      = [&](std::string_view word) { return string && *string == word; };
    switch (field) {
    case Field::PrefixReuse:
    case Field::ExtraStateSlots:
    case Field::HostCapacity:
    case Field::PrefillChunk:
    case Field::MaxTokens:
    case Field::Seed:
        if (is("default")) return DefaultValue{};
        break;
    case Field::Thinking:
    case Field::PreserveThinking:
    case Field::Temperature:
    case Field::TopK:
    case Field::TopP:
    case Field::MinP:
    case Field::PresencePenalty:
    case Field::FrequencyPenalty:
    case Field::ReasoningEffort:
        if (is("model")) return DefaultValue{};
        break;
    case Field::ThinkingBudget:
        if (is("none")) return DefaultValue{};
        break;
    default:
        break;
    }
    switch (field) {
    case Field::Vision:
    case Field::CudaGraph:
    case Field::PrefixReuse:
    case Field::Thinking:
    case Field::PreserveThinking:
    case Field::Cors:
    case Field::RawOutput:
    case Field::PrintTokenIds:
        if (const auto* boolean = std::get_if<bool>(&value)) return *boolean;
        if (command_line && (is("true") || is("false"))) return is("true");
        invalid("expected a boolean");
    case Field::Device:
        return integer(value, command_line, 0, std::numeric_limits<int>::max());
    case Field::MaxContext:
    case Field::MaxPendingRequests:
    case Field::MaxTokens:
    case Field::ThinkingBudget:
        return integer(value, command_line, 1, kU32Max);
    case Field::ExtraStateSlots:
    case Field::DraftTokens:
        return integer(value, command_line, 0, kU32Max);
    case Field::MaxConcurrency:
        return integer(value, command_line, 1, kMaximumConcurrency);
    case Field::ServerPort:
        return integer(value, command_line, 1, 65535);
    case Field::ResponseRecords:
        return integer(value, command_line, 1, kU64Max);
    case Field::TopK:
        return integer(value, command_line, 0, 20);
    case Field::Seed:
        if (is("random")) return *string;
        return integer(value, true, 0, kU64Max);
    case Field::KvCapacity:
        if (is("auto") || is("context")) return *string;
        return integer(value, command_line, 1, kU32Max);
    case Field::PrefillChunk: {
        const auto count = integer(value, command_line, 1, kU32Max);
        if (count % 128 != 0) invalid("expected a positive multiple of 128");
        return count;
    }
    case Field::MediaThreads:
        if (is("auto")) return DefaultValue{};
        return integer(value, command_line, 1, 64);
    case Field::HostCapacity:
    case Field::MediaCache:
        return quantity(value, true, false);
    case Field::MediaLive:
    case Field::MaxRequestSize:
    case Field::ResponseSize:
        return quantity(value, true, true);
    case Field::PendingTimeout:
        return std::uint64_t{parse_duration_ms(value, true)};
    case Field::StatsInterval:
        return std::uint64_t{parse_duration_ms(value, false)};
    case Field::Temperature:
        return real(value, command_line, 0, 2);
    case Field::TopP:
    case Field::MinP:
        return real(value, command_line, 0, 1);
    case Field::PresencePenalty:
    case Field::FrequencyPenalty:
        return real(value, command_line, -2, 2);
    case Field::KvDtype:
        return enumeration(value, {"bf16", "int8", "fp8", "nvfp4", "k8v4"});
    case Field::SpecBackend:
        return enumeration(value, {"none", "mtp", "dflash", "dflash2"});
    case Field::ProposalHead:
        return enumeration(value, {"full", "optimized"});
    case Field::ReasoningEffort:
        return enumeration(value, {"none", "minimal", "low", "medium", "high", "xhigh", "max"});
    case Field::ConstraintKind:
        if (command_line) return enumeration(value, {"none"});
        return enumeration(value,
                           {"none", "grammar", "json_object", "json_schema", "regex", "choice"});
    case Field::LogLevel:
        return enumeration(value,
                           {"trace", "debug", "info", "warning", "error", "critical", "off"});
    case Field::ModelPath:
    case Field::ServerHost:
    case Field::Prompt:
    case Field::Messages:
    case Field::GrammarFile:
    case Field::JsonSchemaFile:
        return text(value, true);
    case Field::ChatTemplate:
    case Field::CostPresets:
    case Field::ApiKey:
    case Field::ModelId:
    case Field::RequestJsonl:
    case Field::Regex:
        return text(value);
    case Field::StopIds: {
        std::vector<std::uint32_t> result;
        if (command_line) {
            result.push_back(static_cast<std::uint32_t>(
                integer(value, true, 0, std::numeric_limits<TokenId>::max())));
        } else {
            const auto* values = std::get_if<std::vector<std::int64_t>>(&value);
            if (!values) invalid("expected an array of integer token IDs");
            for (auto token : *values) {
                if (token < 0 || token > std::numeric_limits<TokenId>::max())
                    invalid("token ID is out of range");
                result.push_back(static_cast<std::uint32_t>(token));
            }
        }
        return result;
    }
    case Field::Stop:
    case Field::ReasoningStop:
    case Field::Choices: {
        std::vector<std::string> result;
        if (command_line)
            result.push_back(text(value));
        else {
            const auto* values = std::get_if<std::vector<std::string>>(&value);
            if (!values) invalid("expected an array of strings");
            result = *values;
        }
        if (field == Field::Choices) {
            if (result.empty()) invalid("choices must contain at least one string");
        } else if (std::any_of(result.begin(), result.end(),
                               [](const auto& v) { return v.empty(); }))
            invalid("stop strings must be nonempty");
        return result;
    }
    case Field::Count:
        break;
    }
    throw std::logic_error("unhandled startup field");
}

void field_error(const FieldSpec& spec, const ConfigSource& source,
                 const std::filesystem::path& config_path, std::string_view message) {
    std::string location;
    if (source.kind == ConfigSourceKind::File)
        location = config_path.string() + ":" + std::to_string(source.line) + ":" +
                   std::to_string(source.column) + ": ";
    else if (source.kind == ConfigSourceKind::CommandLine)
        location = "argument " + std::to_string(source.argument) + " (" +
                   std::string(spec.flag.empty() ? spec.key : spec.flag) + "): ";
    throw std::invalid_argument(location + std::string(spec.key) + ": " + std::string(message));
}

} // namespace ninfer::product::config_detail
