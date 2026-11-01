#include "models/common/qwen/frontend/tool_call_parser.h"
#include "models/common/qwen/frontend/tool_format.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace ninfer::models::qwen::frontend {
namespace {

using Json                = nlohmann::json;
using Contract            = ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = text::SchemaType;
using TypeSet             = text::SchemaTypes;

enum class JsonValueKind : std::uint8_t {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
};

constexpr bool is_format_whitespace(char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

constexpr bool is_ascii_digit(char byte) { return byte >= '0' && byte <= '9'; }

constexpr bool is_ascii_alphanumeric(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') || is_ascii_digit(byte);
}

std::string_view trim_format_whitespace(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && is_format_whitespace(text[begin])) { ++begin; }
    std::size_t end = text.size();
    while (end > begin && is_format_whitespace(text[end - 1])) { --end; }
    return text.substr(begin, end - begin);
}

std::string rtrim_format_whitespace(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && is_format_whitespace(text[end - 1])) { --end; }
    return std::string(text.substr(0, end));
}

void skip_format_whitespace(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && is_format_whitespace(text[pos])) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    return std::all_of(name.begin(), name.end(), [](char byte) {
        return is_ascii_alphanumeric(byte) || byte == '_' || byte == '-';
    });
}

constexpr std::uint8_t type_bit(SchemaType type) { return static_cast<std::uint8_t>(type); }

constexpr bool admits_type(TypeSet types, SchemaType type) {
    return (types.bits & type_bit(type)) != 0;
}

const Contract::Tool* find_tool_contract(const Contract& contract, std::string_view tool_name) {
    const auto tool =
        std::find_if(contract.tools.begin(), contract.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    return tool == contract.tools.end() ? nullptr : &*tool;
}

const Contract::Parameter* find_parameter_contract(const Contract::Tool& tool,
                                                   std::string_view parameter_name) {
    const auto parameter =
        std::find_if(tool.parameters.begin(), tool.parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool.parameters.end() ? nullptr : &*parameter;
}

bool json_number_is_integer(std::string_view number) {
    std::size_t pos = number.starts_with('-') ? 1 : 0;
    if (pos >= number.size()) { return false; }

    const std::size_t integer_begin = pos;
    while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
    const std::size_t integer_end = pos;

    std::size_t fraction_begin = pos;
    std::size_t fraction_end   = pos;
    if (pos < number.size() && number[pos] == '.') {
        fraction_begin = ++pos;
        while (pos < number.size() && is_ascii_digit(number[pos])) { ++pos; }
        fraction_end = pos;
    }

    bool exponent_negative     = false;
    std::size_t exponent_value = 0;
    if (pos < number.size() && (number[pos] == 'e' || number[pos] == 'E')) {
        ++pos;
        if (pos < number.size() && (number[pos] == '+' || number[pos] == '-')) {
            exponent_negative = number[pos] == '-';
            ++pos;
        }
        const std::size_t cap = number.size();
        while (pos < number.size() && is_ascii_digit(number[pos])) {
            const std::size_t digit = static_cast<std::size_t>(number[pos] - '0');
            if (exponent_value != cap) {
                if (exponent_value > cap / 10 || (exponent_value == cap / 10 && digit > cap % 10)) {
                    exponent_value = cap;
                } else {
                    exponent_value = exponent_value * 10 + digit;
                }
            }
            ++pos;
        }
    }
    if (integer_begin == integer_end || pos != number.size()) { return false; }

    bool coefficient_is_zero   = true;
    std::size_t trailing_zeros = 0;
    const auto observe_digit   = [&](char digit) {
        if (digit == '0') {
            ++trailing_zeros;
        } else {
            coefficient_is_zero = false;
            trailing_zeros      = 0;
        }
    };
    for (std::size_t i = integer_begin; i < integer_end; ++i) { observe_digit(number[i]); }
    for (std::size_t i = fraction_begin; i < fraction_end; ++i) { observe_digit(number[i]); }
    if (coefficient_is_zero) { return true; }

    const std::size_t fraction_digits = fraction_end - fraction_begin;
    if (!exponent_negative) {
        if (exponent_value >= fraction_digits) { return true; }
        return fraction_digits - exponent_value <= trailing_zeros;
    }
    if (exponent_value > trailing_zeros) { return false; }
    return fraction_digits <= trailing_zeros - exponent_value;
}

bool classify_json_value(std::string_view value, JsonValueKind& kind) {
    if (value.empty() || !Json::accept(value.begin(), value.end())) { return false; }
    switch (value.front()) {
    case 'n':
        kind = JsonValueKind::Null;
        return true;
    case 't':
    case 'f':
        kind = JsonValueKind::Boolean;
        return true;
    case '"':
        kind = JsonValueKind::String;
        return true;
    case '{':
        kind = JsonValueKind::Object;
        return true;
    case '[':
        kind = JsonValueKind::Array;
        return true;
    default:
        if (value.front() == '-' || is_ascii_digit(value.front())) {
            kind = json_number_is_integer(value) ? JsonValueKind::Integer : JsonValueKind::Number;
            return true;
        }
        return false;
    }
}

bool admits_value(TypeSet types, JsonValueKind kind) {
    switch (kind) {
    case JsonValueKind::Null:
        return admits_type(types, SchemaType::Null);
    case JsonValueKind::Boolean:
        return admits_type(types, SchemaType::Boolean);
    case JsonValueKind::Integer:
        return admits_type(types, SchemaType::Integer) || admits_type(types, SchemaType::Number);
    case JsonValueKind::Number:
        return admits_type(types, SchemaType::Number);
    case JsonValueKind::String:
        return admits_type(types, SchemaType::String);
    case JsonValueKind::Object:
        return admits_type(types, SchemaType::Object);
    case JsonValueKind::Array:
        return admits_type(types, SchemaType::Array);
    }
    return false;
}

std::string encode_json_string(std::string_view value) { return Json(std::string(value)).dump(); }

// A split UTF-8 codepoint stays in the scanner until its bytes can be published together.
std::size_t codepoint_bytes(std::string_view input) {
    if (input.empty()) return 0;
    const auto lead         = static_cast<unsigned char>(input.front());
    const std::size_t count = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
    return input.size() >= count ? count : 0;
}

std::size_t suffix_prefix(std::string_view value, std::string_view marker) {
    for (auto n = std::min(value.size(), marker.size() - 1); n != 0; --n)
        if (value.substr(value.size() - n) == marker.substr(0, n)) return n;
    return 0;
}

} // namespace

class ToolCallOutputDecoder::Impl {
public:
    enum class State {
        Text,
        Header,
        Parameters,
        ValueStart,
        Value,
        ToolClose,
        Between,
        Plain,
        Failed
    };
    using Codec = Contract::Encoding;

    Impl(std::shared_ptr<const Contract> bound, std::size_t max_name)
        : contract(std::move(bound)), max_name_length(max_name) {}

    void content(std::string value) {
        const auto suppressed = std::min(prefix_bytes, value.size());
        value.erase(0, suppressed);
        prefix_bytes -= suppressed;
        if (value.empty()) return;
        if (!events.empty()) {
            if (auto* last = std::get_if<OutputDelta>(&events.back())) {
                last->text += value;
                return;
            }
        }
        events.emplace_back(OutputDelta{.text = std::move(value)});
    }

    std::uint32_t index() const { return static_cast<std::uint32_t>(calls.size() - 1); }

    void arguments(std::string value, bool parameter_complete = false) {
        if (value.empty() && !parameter_complete) return;
        calls.back().arguments_json += value;
        if (!events.empty()) {
            if (auto* last = std::get_if<ToolArgumentsDelta>(&events.back());
                last && last->index == index() && !last->parameter_complete) {
                last->text += value;
                last->parameter_complete = parameter_complete;
                return;
            }
        }
        events.emplace_back(ToolArgumentsDelta{index(), std::move(value), parameter_complete});
    }

    void end_call(ToolCallStatus status) {
        if (calls.empty() || calls.back().status != ToolCallStatus::InProgress) return;
        calls.back().status = status;
        if (status == ToolCallStatus::Complete) ++diagnostic.structured_call_count;
        events.emplace_back(ToolCallEnd{index(), status});
    }

    bool fail(std::string message) {
        failure = models::frontend::ToolOutputError{
            contract->constrained ? models::frontend::ToolOutputErrorKind::ContractViolation
                                  : models::frontend::ToolOutputErrorKind::InvalidModelOutput,
            std::move(message)};
        end_call(ToolCallStatus::Incomplete);
        state = State::Failed;
        pending.clear();
        cursor = 0;
        return false;
    }

    bool fallback_header(FallbackReason reason) {
        if (contract->constrained) return fail("invalid constrained tool header");
        if (!calls.empty()) return fail("invalid tool call header");
        diagnostic.fallback_reason = reason;
        content(std::move(preamble));
        flush_pending();
        state = State::Plain;
        return false;
    }

    // The prepared contract determines encoding before any value bytes are published.
    void begin_value(Codec selected) {
        codec = selected;
        if (selected == Codec::BufferedNormalize) return;
        arguments((members++ ? "," : "") + encode_json_string(parameter_name) + ":");
        value_offset = calls.back().arguments_json.size();
        if (selected == Codec::RawString) arguments("\"");
    }

    void raw_bytes(std::string_view bytes) {
        value_tail.append(bytes);
        std::size_t keep = 0;
        if (!contract->constrained) {
            // Free generation may put an LF/CRLF immediately before its loose closing tag.
            // Only those final framing bytes are undecided; spaces and tabs stream immediately.
            if (value_tail.ends_with("\r\n"))
                keep = 2;
            else if (value_tail.ends_with('\r') || value_tail.ends_with('\n'))
                keep = 1;
        }
        const auto count   = value_tail.size() - keep;
        const auto encoded = encode_json_string(std::string_view(value_tail).substr(0, count));
        arguments(encoded.substr(1, encoded.size() - 2));
        value_tail.erase(0, count);
    }

    void value_bytes(std::string_view bytes) {
        if (codec == Codec::BufferedNormalize) {
            buffered_value.append(bytes);
            // This is only a framing hint: no JSON bytes are published. A valid JSON string
            // can contain a bare closing tag, while a literal newline ends that interpretation.
            for (const char byte : bytes) {
                if (!value_seen && !is_format_whitespace(byte)) {
                    value_seen    = true;
                    buffered_json = byte == '{' || byte == '[' || byte == '"';
                }
                if (!buffered_json) continue;
                if (quoted && (byte == '\n' || byte == '\r')) {
                    buffered_json = false;
                    quoted = escaped = false;
                } else if (escaped) {
                    escaped = false;
                } else if (quoted && byte == '\\') {
                    escaped = true;
                } else if (byte == '"') {
                    quoted = !quoted;
                }
            }
            return;
        }
        if (codec == Codec::RawString) {
            raw_bytes(bytes);
            return;
        }
        if (!tool->strict && !value_seen &&
            std::all_of(bytes.begin(), bytes.end(), is_format_whitespace))
            return;
        value_seen = true;
        for (char c : bytes) {
            if (quoted) {
                if (escaped)
                    escaped = false;
                else if (c == '\\')
                    escaped = true;
                else if (c == '"')
                    quoted = false;
            } else if (c == '"')
                quoted = true;
        }
        // Non-strict JSON normalization trims outer whitespace. Keep only its undecided tail.
        if (!tool->strict && !quoted &&
            std::all_of(bytes.begin(), bytes.end(), is_format_whitespace)) {
            value_tail.append(bytes);
            return;
        }
        std::string value = std::move(value_tail);
        value_tail.clear();
        value.append(bytes);
        arguments(std::move(value));
    }

    bool finish_value() {
        if (codec == Codec::BufferedNormalize) {
            const auto value   = trim_format_whitespace(buffered_value);
            const auto encoded = Json::accept(value.begin(), value.end())
                                     ? std::string(value)
                                     : encode_json_string(value);
            arguments((members++ ? "," : "") + encode_json_string(parameter_name) + ":" + encoded,
                      true);
        } else if (codec == Codec::RawString) {
            if (!contract->constrained) {
                if (value_tail.ends_with("\r\n"))
                    value_tail.resize(value_tail.size() - 2);
                else if (value_tail.ends_with('\n'))
                    value_tail.pop_back();
            }
            const auto encoded = encode_json_string(value_tail);
            arguments(encoded.substr(1, encoded.size() - 2));
            arguments("\"", true);
        } else {
            if (tool->strict) arguments(std::move(value_tail));
            const auto value = std::string_view(calls.back().arguments_json).substr(value_offset);
            JsonValueKind kind;
            if (!classify_json_value(value, kind)) return fail("invalid JSON tool parameter");
            if (parameter && parameter->policy == NormalizationPolicy::DeclaredTypes &&
                !admits_value(parameter->types, kind))
                ++diagnostic.schema_mismatch_arguments;
            arguments({}, true);
        }
        if (!parameter) ++diagnostic.schema_mismatch_arguments;
        buffered_value.clear();
        value_tail.clear();
        state = State::Parameters;
        return true;
    }

    void consume(std::size_t count) { cursor += count; }

    void flush_pending() {
        content(std::string(std::string_view(pending).substr(cursor)));
        pending.clear();
        cursor = 0;
    }

    bool step() {
        if (cursor == pending.size()) return false;
        const auto input = std::string_view(pending).substr(cursor);
        switch (state) {
        case State::Plain:
            flush_pending();
            return false;
        case State::Failed:
            return false;
        case State::Text: {
            const auto marker = input.find(kToolOpen);
            if (marker != std::string::npos) {
                const auto before  = input.substr(0, marker);
                const auto trimmed = rtrim_format_whitespace(before);
                content(trimmed);
                preamble = before.substr(trimmed.size());
                consume(marker);
                diagnostic.marker_seen = true;
                state                  = State::Header;
                return true;
            }
            auto count = input.size() - suffix_prefix(input, kToolOpen);
            while (count && is_format_whitespace(input[count - 1])) --count;
            // Preserve split UTF-8 without blocking preceding complete codepoints.
            std::size_t complete = 0;
            while (complete < count) {
                const auto n = codepoint_bytes(input.substr(complete, count - complete));
                if (!n) break;
                complete += n;
            }
            if (!complete) return false;
            content(std::string(input.substr(0, complete)));
            consume(complete);
            return true;
        }
        case State::Header: {
            std::size_t pos = kToolOpen.size();
            skip_format_whitespace(input, pos);
            if (kFunctionOpen.starts_with(input.substr(pos))) return false;
            if (!starts_with_at(input, pos, kFunctionOpen))
                return fallback_header(FallbackReason::MalformedStructure);
            pos += kFunctionOpen.size();
            const auto end = input.find('>', pos);
            if (end == std::string::npos) {
                if (input.size() - pos > max_name_length)
                    return fallback_header(FallbackReason::InvalidToolName);
                return false;
            }
            const auto name = input.substr(pos, end - pos);
            if (!valid_function_name(name, max_name_length))
                return fallback_header(FallbackReason::InvalidToolName);
            tool = find_tool_contract(*contract, name);
            if (!tool) return fallback_header(FallbackReason::UndeclaredTool);
            if (!contract->parallel && !calls.empty())
                return fail("multiple tool calls are not allowed");
            calls.push_back({std::string(name), {}, ToolCallStatus::InProgress});
            events.emplace_back(ToolCallStart{index(), std::string(name)});
            arguments("{");
            consume(end + 1);
            preamble.clear();
            prefix_bytes       = 0;
            members            = 0;
            previous_parameter = -1;
            names.clear();
            state = State::Parameters;
            return true;
        }
        case State::Parameters: {
            std::size_t pos = 0;
            skip_format_whitespace(input, pos);
            if (pos) {
                consume(pos);
                return true;
            }
            if (input.starts_with(kFunctionClose)) {
                arguments("}");
                consume(kFunctionClose.size());
                state = State::ToolClose;
                return true;
            }
            if (kFunctionClose.starts_with(input) || kParamOpen.starts_with(input)) return false;
            if (!input.starts_with(kParamOpen)) return fail("malformed tool parameter header");
            const auto end = input.find('>', kParamOpen.size());
            if (end == std::string::npos) return false;
            parameter_name = input.substr(kParamOpen.size(), end - kParamOpen.size());
            if (!valid_parameter_name(parameter_name)) return fail("invalid tool parameter name");
            if (!names.insert(parameter_name).second) return fail("duplicate tool parameter");
            parameter = find_parameter_contract(*tool, parameter_name);
            if (tool->strict) {
                if (!parameter) return fail("undeclared strict tool parameter");
                const auto ordinal = parameter - tool->parameters.data();
                if (ordinal <= previous_parameter)
                    return fail("strict tool parameter order mismatch");
                previous_parameter = ordinal;
            }
            consume(end + 1);
            state = State::ValueStart;
            return true;
        }
        case State::ValueStart: {
            if (input == "\r") return false;
            std::size_t frame = input.starts_with("\r\n") ? 2 : input.starts_with('\n') ? 1 : 0;
            if (contract->constrained && frame != 1)
                return fail("missing tool parameter framing newline");
            consume(frame);
            buffered_value.clear();
            value_tail.clear();
            quoted        = false;
            escaped       = false;
            value_seen    = false;
            buffered_json = false;
            nested        = 0;
            state         = State::Value;
            begin_value(parameter ? parameter->encoding : Codec::BufferedNormalize);
            return true;
        }
        case State::Value: {
            const auto delimiter = contract->constrained ? kParameterEnd : kParamClose;
            if (!(quoted &&
                  (codec == Codec::JsonValue || (buffered_json && !contract->constrained)))) {
                if (input.size() < delimiter.size() && delimiter.starts_with(input)) return false;
                if (input.starts_with(delimiter)) {
                    if (!contract->constrained && nested) {
                        --nested;
                        value_bytes(delimiter);
                        consume(delimiter.size());
                        return true;
                    }
                    consume(delimiter.size());
                    return finish_value();
                }
                if (!contract->constrained && codec != Codec::JsonValue) {
                    if (kParamOpen.starts_with(input)) return false;
                    if (input.starts_with(kParamOpen)) {
                        const auto end = input.find('>', kParamOpen.size());
                        if (end == std::string::npos) return false;
                        if (end > kParamOpen.size()) ++nested;
                        value_bytes(input.substr(0, end + 1));
                        consume(end + 1);
                        return true;
                    }
                }
            }
            const auto count = codepoint_bytes(input);
            if (!count) return false;
            value_bytes(input.substr(0, count));
            consume(count);
            return true;
        }
        case State::ToolClose:
        case State::Between: {
            std::size_t pos = 0;
            skip_format_whitespace(input, pos);
            if (pos) {
                consume(pos);
                return true;
            }
            const auto marker = state == State::ToolClose ? kToolClose : kToolOpen;
            if (marker.starts_with(input) && marker != input) return false;
            if (!input.starts_with(marker)) return fail("unexpected text in tool call sequence");
            if (state == State::ToolClose) {
                consume(marker.size());
                end_call(ToolCallStatus::Complete);
                state = State::Between;
            } else
                state = State::Header;
            return true;
        }
        }
        return false;
    }

    std::vector<OutputEvent> feed(std::string_view bytes) {
        if (finished) throw std::logic_error("tool output decoder is finished");
        events = std::exchange(seed, {});
        if (!failure) {
            pending.append(bytes);
            if (!contract)
                flush_pending();
            else
                while (step()) {}
            pending.erase(0, cursor);
            cursor = 0;
        }
        return std::exchange(events, {});
    }

    std::vector<OutputEvent> finish(FinishReason reason) {
        if (finished) throw std::logic_error("tool output decoder is finished");
        auto initial = feed({});
        events       = std::move(initial);
        finished     = true;
        if (failure) return std::exchange(events, {});
        const bool interrupted =
            reason == FinishReason::OutputLimit || reason == FinishReason::ContextCapacity ||
            reason == FinishReason::Cancelled || reason == FinishReason::StopString;
        if (state == State::Text || state == State::Plain)
            content(std::exchange(pending, {}));
        else if (state == State::Header && calls.empty() && !contract->constrained)
            (void)fallback_header(FallbackReason::MalformedStructure);
        else if (state != State::Between || !pending.empty()) {
            if (!interrupted)
                (void)fail("incomplete tool call at model stop");
            else
                end_call(ToolCallStatus::Incomplete);
        }
        return std::exchange(events, {});
    }

    std::shared_ptr<const Contract> contract;
    std::size_t max_name_length;
    State state = State::Text;
    Codec codec = Codec::BufferedNormalize;
    std::string pending, preamble, parameter_name, buffered_value, value_tail;
    bool combined                        = false;
    ConstraintOutputBranch output_branch = ConstraintOutputBranch::Undecided;
    const Contract::Tool* tool           = nullptr;
    const Contract::Parameter* parameter = nullptr;
    std::unordered_set<std::string> names;
    std::ptrdiff_t previous_parameter = -1;
    std::size_t members = 0, nested = 0, value_offset = 0, prefix_bytes = 0, cursor = 0;
    bool quoted = false, escaped = false, value_seen = false, buffered_json = false,
         finished = false;
    std::vector<GeneratedToolCall> calls;
    std::vector<OutputEvent> events, seed;
    ToolCallParseDiagnostics diagnostic;
    std::optional<models::frontend::ToolOutputError> failure;
};

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const Contract> contract,
                                             std::size_t max_tool_name_length, bool combined)
    : impl_(std::make_unique<Impl>(std::move(contract), max_tool_name_length)) {
    impl_->combined = combined;
}

ToolCallOutputDecoder::~ToolCallOutputDecoder()                                           = default;
ToolCallOutputDecoder::ToolCallOutputDecoder(ToolCallOutputDecoder&&) noexcept            = default;
ToolCallOutputDecoder& ToolCallOutputDecoder::operator=(ToolCallOutputDecoder&&) noexcept = default;

bool ToolCallOutputDecoder::in_tool_region() const noexcept {
    return impl_->diagnostic.marker_seen ||
           (impl_->state == Impl::State::Text && impl_->pending.find('<') != std::string::npos);
}

ConstraintOutputBranch ToolCallOutputDecoder::branch() const noexcept {
    if (impl_->combined) return impl_->output_branch;
    return in_tool_region() ? ConstraintOutputBranch::Tools : ConstraintOutputBranch::Undecided;
}

std::vector<OutputEvent> ToolCallOutputDecoder::feed(std::string_view text) {
    if (impl_->combined && impl_->output_branch == ConstraintOutputBranch::Undecided &&
        !text.empty())
        impl_->output_branch =
            text.front() == '<' ? ConstraintOutputBranch::Tools : ConstraintOutputBranch::Content;
    if (impl_->combined && impl_->output_branch == ConstraintOutputBranch::Content)
        return text.empty() ? std::vector<OutputEvent>{}
                            : std::vector<OutputEvent>{
                                  OutputDelta{OutputChannel::Content, std::string(text)}};
    return impl_->feed(text);
}

std::vector<OutputEvent> ToolCallOutputDecoder::finish(FinishReason reason) {
    if (impl_->combined && impl_->output_branch == ConstraintOutputBranch::Content) return {};
    return impl_->finish(reason);
}

std::vector<GeneratedToolCall> ToolCallOutputDecoder::take_tool_calls() noexcept {
    return std::move(impl_->calls);
}

ToolCallParseDiagnostics ToolCallOutputDecoder::diagnostics() const noexcept {
    return impl_->diagnostic;
}

std::optional<models::frontend::ToolOutputError> ToolCallOutputDecoder::error() const {
    return impl_->failure;
}

void ToolCallOutputDecoder::initialize_continuation(std::string_view prefix) {
    if (impl_->combined && !prefix.empty()) {
        impl_->output_branch =
            prefix.front() == '<' ? ConstraintOutputBranch::Tools : ConstraintOutputBranch::Content;
        if (impl_->output_branch == ConstraintOutputBranch::Content) return;
    }
    auto events = impl_->feed(prefix);
    if (impl_->failure ||
        std::any_of(impl_->calls.begin(), impl_->calls.end(),
                    [](const auto& call) { return call.status == ToolCallStatus::Complete; }))
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool continuation must contain no completed or malformed calls");
    for (auto& event : events)
        if (!std::holds_alternative<OutputDelta>(event)) impl_->seed.push_back(std::move(event));
    if (impl_->calls.empty()) impl_->prefix_bytes = impl_->pending.size() + impl_->preamble.size();
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text, std::size_t max_name,
                                                 const Contract& contract) {
    ToolCallOutputDecoder decoder(std::make_shared<Contract>(contract), max_name);
    ParsedToolCallOutput result;
    const auto append = [&](const auto& events) {
        for (const auto& event : events)
            if (const auto* delta = std::get_if<OutputDelta>(&event)) result.content += delta->text;
    };
    append(decoder.feed(text));
    append(decoder.finish());
    if (const auto error = decoder.error()) std::rethrow_exception(error->exception());
    result.tool_calls            = decoder.take_tool_calls();
    result.diagnostics           = decoder.diagnostics();
    result.is_tool_call_response = !result.tool_calls.empty();
    return result;
}

} // namespace ninfer::models::qwen::frontend
