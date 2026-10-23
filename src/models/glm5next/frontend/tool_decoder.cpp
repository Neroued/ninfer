#include "models/glm5next/frontend/tool_decoder.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>

namespace ninfer::models::glm5next {
namespace {
using Contract = ToolCallOutputContract;
using Codec    = Contract::Encoding;
using Json     = nlohmann::json;
using Events   = models::frontend::PublishedOutput;

bool whitespace(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

std::string_view trim(std::string_view text) {
    while (!text.empty() && whitespace(text.front())) text.remove_prefix(1);
    while (!text.empty() && whitespace(text.back())) text.remove_suffix(1);
    return text;
}

std::size_t safe_bytes(std::string_view text) {
    std::size_t n = 0;
    while (n < text.size()) {
        const auto c            = static_cast<unsigned char>(text[n]);
        const std::size_t width = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
        if (n + width > text.size()) break;
        n += width;
    }
    return n;
}

std::size_t suffix(std::string_view text, std::string_view marker) {
    for (auto n = std::min(text.size(), marker.size() - 1); n; --n)
        if (text.ends_with(marker.substr(0, n))) return n;
    return 0;
}

std::string json_string(std::string_view text) { return Json(std::string(text)).dump(); }

bool admits(const Contract::Parameter& parameter, const Json& value) {
    using Type         = text::SchemaType;
    const auto bit     = [](Type type) { return static_cast<std::uint8_t>(type); };
    std::uint8_t types = 0;
    if (value.is_string())
        types = bit(Type::String);
    else if (value.is_boolean())
        types = bit(Type::Boolean);
    else if (value.is_null())
        types = bit(Type::Null);
    else if (value.is_array())
        types = bit(Type::Array);
    else if (value.is_object())
        types = bit(Type::Object);
    else {
        types = bit(Type::Number);
        if (value.is_number_integer() || std::floor(value.get<double>()) == value.get<double>())
            types |= bit(Type::Integer);
    }
    return parameter.policy != Contract::NormalizationPolicy::DeclaredTypes ||
           (parameter.types.bits & types) != 0;
}
} // namespace

class ToolCallOutputDecoder::Impl {
public:
    enum class State { Text, Header, Parameters, ValueStart, Value, Between, Plain, Failed };

    Impl(std::shared_ptr<const Contract> contract_, std::size_t maximum, bool combined_)
        : contract(std::move(contract_)), max_name(maximum), combined(combined_) {}

    void content(std::string_view bytes) {
        const auto skip = std::min(prefix_bytes, bytes.size());
        bytes.remove_prefix(skip);
        prefix_bytes -= skip;
        if (bytes.empty()) return;
        if (!events.empty())
            if (auto* delta = std::get_if<OutputDelta>(&events.back())) {
                delta->text.append(bytes);
                return;
            }
        events.emplace_back(OutputDelta{OutputChannel::Content, std::string(bytes)});
    }

    std::uint32_t index() const { return static_cast<std::uint32_t>(calls.size() - 1); }

    void arguments(std::string bytes, bool complete = false) {
        if (bytes.empty() && !complete) return;
        calls.back().arguments_json += bytes;
        if (!events.empty())
            if (auto* delta = std::get_if<ToolArgumentsDelta>(&events.back());
                delta && delta->index == index() && !delta->parameter_complete) {
                delta->text += bytes;
                delta->parameter_complete = complete;
                return;
            }
        events.emplace_back(ToolArgumentsDelta{index(), std::move(bytes), complete});
    }

    void end_call(ToolCallStatus status) {
        if (calls.empty() || calls.back().status != ToolCallStatus::InProgress) return;
        calls.back().status = status;
        if (status == ToolCallStatus::Complete) ++diagnostic.structured_call_count;
        events.emplace_back(ToolCallEnd{index(), status});
    }

    bool fail(std::string message) {
        failure = models::frontend::ToolOutputError{
            contract && contract->constrained
                ? models::frontend::ToolOutputErrorKind::ContractViolation
                : models::frontend::ToolOutputErrorKind::InvalidModelOutput,
            std::move(message)};
        end_call(ToolCallStatus::Incomplete);
        state = State::Failed;
        return false;
    }

    bool fallback(ToolCallParseFallbackReason reason) {
        if (contract->constrained || !calls.empty()) return fail("invalid GLM tool header");
        diagnostic.fallback_reason = reason;
        state                      = State::Plain;
        return true;
    }

    void begin_value() {
        codec = parameter ? parameter->encoding : Codec::BufferedNormalize;
        buffer.clear();
        quoted = escaped = value_seen = buffered_json = false;
        if (codec == Codec::BufferedNormalize) return;
        arguments((members++ ? "," : "") + json_string(name) + ":");
        value_offset = calls.back().arguments_json.size();
        if (codec == Codec::RawString) arguments("\"");
    }

    void value_bytes(std::string_view bytes) {
        if (codec == Codec::RawString) {
            const auto encoded = json_string(bytes);
            arguments(encoded.substr(1, encoded.size() - 2));
            return;
        }
        if (codec == Codec::BufferedNormalize)
            buffer.append(bytes);
        else
            arguments(std::string(bytes));
        for (const char c : bytes) {
            if (!value_seen && !whitespace(c)) {
                value_seen    = true;
                buffered_json = c == '{' || c == '[' || c == '"';
            }
            if (codec == Codec::BufferedNormalize && !buffered_json) continue;
            if (codec == Codec::BufferedNormalize && quoted && (c == '\n' || c == '\r')) {
                buffered_json = quoted = escaped = false;
            } else if (escaped)
                escaped = false;
            else if (quoted && c == '\\')
                escaped = true;
            else if (c == '"')
                quoted = !quoted;
        }
    }

    bool finish_value() {
        if (codec == Codec::RawString)
            arguments("\"", true);
        else if (codec == Codec::BufferedNormalize) {
            const auto value  = trim(buffer);
            const auto parsed = Json::parse(value, nullptr, false);
            // Unknown encodings retain the original string, including its whitespace.
            arguments((members++ ? "," : "") + json_string(name) + ":" +
                          (parsed.is_discarded() ? json_string(buffer) : std::string(value)),
                      true);
        } else {
            const auto value  = std::string_view(calls.back().arguments_json).substr(value_offset);
            const auto parsed = Json::parse(value, nullptr, false);
            if (parsed.is_discarded()) return fail("invalid JSON GLM tool parameter");
            if (parameter && !admits(*parameter, parsed)) ++diagnostic.schema_mismatch_arguments;
            arguments({}, true);
        }
        if (!parameter) ++diagnostic.schema_mismatch_arguments;
        state = State::Parameters;
        return true;
    }

    bool step() {
        const auto input = std::string_view(pending).substr(cursor);
        if (input.empty()) return false;
        switch (state) {
        case State::Failed:
            return false;
        case State::Plain:
            content(input);
            cursor += input.size();
            return false;
        case State::Text: {
            const auto marker = input.find(kToolOpen);
            if (marker != std::string_view::npos) {
                content(input.substr(0, marker));
                cursor += marker;
                diagnostic.marker_seen = true;
                state                  = State::Header;
                return true;
            }
            const auto count = safe_bytes(input.substr(0, input.size() - suffix(input, kToolOpen)));
            content(input.substr(0, count));
            cursor += count;
            return false;
        }
        case State::Header: {
            const auto end = input.find('<', kToolOpen.size());
            if (end == std::string_view::npos) {
                if (trim(input.substr(kToolOpen.size())).size() > max_name)
                    return fallback(ToolCallParseFallbackReason::InvalidToolName);
                return false;
            }
            const auto tail = input.substr(end);
            if (tail.size() < kKeyOpen.size() && kKeyOpen.starts_with(tail)) return false;
            if (tail.size() < kToolClose.size() && kToolClose.starts_with(tail)) return false;
            if (!tail.starts_with(kKeyOpen) && !tail.starts_with(kToolClose))
                return fallback(ToolCallParseFallbackReason::MalformedStructure);
            const auto function = trim(input.substr(kToolOpen.size(), end - kToolOpen.size()));
            if (!valid_tool_name(function, max_name))
                return fallback(ToolCallParseFallbackReason::InvalidToolName);
            const auto found =
                std::find_if(contract->tools.begin(), contract->tools.end(),
                             [&](const auto& item) { return item.name == function; });
            if (found == contract->tools.end())
                return fallback(ToolCallParseFallbackReason::UndeclaredTool);
            if (!contract->parallel && !calls.empty())
                return fail("multiple GLM calls are not allowed");
            tool = &*found;
            calls.push_back({std::string(function), {}, ToolCallStatus::InProgress});
            events.emplace_back(ToolCallStart{index(), std::string(function)});
            arguments("{");
            cursor += end;
            prefix_bytes = members = 0;
            previous_parameter     = -1;
            names.clear();
            state = State::Parameters;
            return true;
        }
        case State::Between:
        case State::Parameters: {
            std::size_t skip = 0;
            while (skip < input.size() && whitespace(input[skip])) ++skip;
            if (skip) {
                cursor += skip;
                return true;
            }
            if (state == State::Between) {
                if (kToolOpen.starts_with(input) && input != kToolOpen) return false;
                if (input.starts_with(kToolOpen)) {
                    state = State::Header;
                    return true;
                }
                return fail("unexpected text after GLM tool call");
            }
            if (input.starts_with(kToolClose)) {
                arguments("}");
                cursor += kToolClose.size();
                end_call(ToolCallStatus::Complete);
                state = State::Between;
                return true;
            }
            if (kToolClose.starts_with(input) || kKeyOpen.starts_with(input)) return false;
            if (!input.starts_with(kKeyOpen)) return fail("malformed GLM parameter key");
            const auto end = input.find(kKeyClose, kKeyOpen.size());
            if (end == std::string_view::npos) return false;
            name = input.substr(kKeyOpen.size(), end - kKeyOpen.size());
            if (!valid_parameter_name(name) || !names.insert(name).second)
                return fail("invalid or duplicate GLM parameter key");
            const auto found = std::find_if(tool->parameters.begin(), tool->parameters.end(),
                                            [&](const auto& item) { return item.name == name; });
            parameter        = found == tool->parameters.end() ? nullptr : &*found;
            if (tool->strict) {
                if (!parameter) return fail("undeclared strict GLM parameter");
                const auto ordinal = parameter - tool->parameters.data();
                if (ordinal <= previous_parameter)
                    return fail("strict GLM parameter order mismatch");
                previous_parameter = ordinal;
            }
            cursor += end + kKeyClose.size();
            state = State::ValueStart;
            return true;
        }
        case State::ValueStart: {
            std::size_t skip = 0;
            while (skip < input.size() && whitespace(input[skip])) ++skip;
            if (skip) {
                cursor += skip;
                return true;
            }
            if (kValueOpen.starts_with(input) && input != kValueOpen) return false;
            if (!input.starts_with(kValueOpen)) return fail("missing GLM parameter value tag");
            cursor += kValueOpen.size();
            begin_value();
            state = State::Value;
            return true;
        }
        case State::Value: {
            if (!(quoted &&
                  (codec == Codec::JsonValue || (buffered_json && !contract->constrained)))) {
                if (input.size() < kValueClose.size() && kValueClose.starts_with(input))
                    return false;
                if (input.starts_with(kValueClose)) {
                    cursor += kValueClose.size();
                    return finish_value();
                }
            }
            if (codec == Codec::RawString) {
                const auto marker = input.find('<');
                const auto count  = safe_bytes(input.substr(0, marker == 0 ? 1 : marker));
                if (!count) return false;
                value_bytes(input.substr(0, count));
                cursor += count;
                return true;
            }
            const auto c            = static_cast<unsigned char>(input.front());
            const std::size_t width = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
            if (input.size() < width) return false;
            value_bytes(input.substr(0, width));
            cursor += width;
            return true;
        }
        }
        return false;
    }

    Events feed(std::string_view bytes) {
        if (finished) throw std::logic_error("GLM output decoder is finished");
        events = std::exchange(seed, {});
        pending.append(bytes);
        if (combined && output_branch == ConstraintOutputBranch::Undecided) {
            const auto first = pending.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return std::exchange(events, {});
            output_branch = pending[first] == '<' ? ConstraintOutputBranch::Tools
                                                  : ConstraintOutputBranch::Content;
        }
        if (!contract || output_branch == ConstraintOutputBranch::Content) {
            const auto count = safe_bytes(pending);
            content(std::string_view(pending).substr(0, count));
            cursor = count;
        } else if (!failure) {
            while (step()) {}
        }
        pending.erase(0, cursor);
        cursor = 0;
        if (failure) pending.clear();
        return std::exchange(events, {});
    }

    Events finish(FinishReason reason) {
        events   = feed({});
        finished = true;
        if (failure) return std::exchange(events, {});
        const bool interrupted =
            reason == FinishReason::OutputLimit || reason == FinishReason::ContextCapacity ||
            reason == FinishReason::Cancelled || reason == FinishReason::StopString;
        if (!contract || output_branch == ConstraintOutputBranch::Content || state == State::Text ||
            state == State::Plain)
            content(pending);
        else if (state == State::Header && calls.empty() && !contract->constrained) {
            diagnostic.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
            content(pending);
        } else if (state != State::Between || !pending.empty()) {
            if (interrupted)
                end_call(ToolCallStatus::Incomplete);
            else
                (void)fail("incomplete GLM tool call at model stop");
        }
        pending.clear();
        return std::exchange(events, {});
    }

    std::shared_ptr<const Contract> contract;
    std::size_t max_name;
    bool combined;
    State state                          = State::Text;
    Codec codec                          = Codec::BufferedNormalize;
    ConstraintOutputBranch output_branch = ConstraintOutputBranch::Undecided;
    const Contract::Tool* tool           = nullptr;
    const Contract::Parameter* parameter = nullptr;
    std::unordered_set<std::string> names;
    std::string pending, name, buffer;
    std::size_t cursor = 0, members = 0, value_offset = 0, prefix_bytes = 0;
    std::ptrdiff_t previous_parameter = -1;
    bool quoted = false, escaped = false, value_seen = false, buffered_json = false,
         finished = false;
    std::vector<GeneratedToolCall> calls;
    Events events, seed;
    ToolCallParseDiagnostics diagnostic;
    std::optional<models::frontend::ToolOutputError> failure;
};

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const Contract> contract,
                                             std::size_t max_name, bool combined)
    : impl_(std::make_unique<Impl>(std::move(contract), max_name, combined)) {}

ToolCallOutputDecoder::~ToolCallOutputDecoder() = default;

Events ToolCallOutputDecoder::feed(std::string_view bytes) { return impl_->feed(bytes); }

Events ToolCallOutputDecoder::finish(FinishReason reason) { return impl_->finish(reason); }

std::vector<GeneratedToolCall> ToolCallOutputDecoder::take_tool_calls() noexcept {
    return std::move(impl_->calls);
}

ToolCallParseDiagnostics ToolCallOutputDecoder::diagnostics() const noexcept {
    return impl_->diagnostic;
}

std::optional<models::frontend::ToolOutputError> ToolCallOutputDecoder::error() const {
    return impl_->failure;
}

ConstraintOutputBranch ToolCallOutputDecoder::branch() const noexcept {
    if (impl_->combined) return impl_->output_branch;
    return impl_->calls.empty() ? ConstraintOutputBranch::Undecided : ConstraintOutputBranch::Tools;
}

void ToolCallOutputDecoder::initialize_continuation(std::string_view prefix) {
    const auto events = impl_->feed(prefix);
    if (impl_->failure ||
        std::any_of(impl_->calls.begin(), impl_->calls.end(),
                    [](const auto& call) { return call.status == ToolCallStatus::Complete; }))
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "GLM tool continuation must contain no completed or malformed calls");
    for (const auto& event : events)
        if (!std::holds_alternative<OutputDelta>(event)) impl_->seed.push_back(event);
    if (impl_->calls.empty()) impl_->prefix_bytes = impl_->pending.size();
}
} // namespace ninfer::models::glm5next
