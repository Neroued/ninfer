#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using Json                = nlohmann::json;
using Contract            = ToolCallOutputContract;
using FallbackReason      = ToolCallParseFallbackReason;
using NormalizationPolicy = Contract::NormalizationPolicy;
using SchemaType          = Contract::SchemaType;
using TypeSet             = Contract::TypeSet;

constexpr std::string_view kToolOpen      = "<tool_call>";
constexpr std::string_view kToolClose     = "</tool_call>";
constexpr std::string_view kFunctionOpen  = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParamOpen     = "<parameter=";
constexpr std::string_view kParamClose    = "</parameter>";

struct RawParameter {
    std::string_view name;
    std::string_view value;
};

struct RawToolCall {
    std::string_view name;
    std::vector<RawParameter> parameters;
};

enum class JsonValueKind : std::uint8_t {
    Null,
    Boolean,
    Integer,
    Number,
    String,
    Object,
    Array,
};

enum class ParameterNormalization : std::uint8_t {
    Emitted,
    Omitted,
    SchemaMismatch,
};

struct NormalizedParameter {
    ParameterNormalization disposition = ParameterNormalization::Emitted;
    std::string json_value;
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

bool schema_type(std::string_view name, SchemaType& type) {
    if (name == "null") {
        type = SchemaType::Null;
    } else if (name == "boolean") {
        type = SchemaType::Boolean;
    } else if (name == "integer") {
        type = SchemaType::Integer;
    } else if (name == "number") {
        type = SchemaType::Number;
    } else if (name == "string") {
        type = SchemaType::String;
    } else if (name == "object") {
        type = SchemaType::Object;
    } else if (name == "array") {
        type = SchemaType::Array;
    } else {
        return false;
    }
    return true;
}

bool compile_direct_types(const Json& type_definition, TypeSet& types) {
    types = {};
    if (type_definition.is_string()) {
        SchemaType type;
        if (!schema_type(type_definition.get_ref<const std::string&>(), type)) { return false; }
        types.bits = type_bit(type);
        return true;
    }
    if (!type_definition.is_array() || type_definition.empty()) { return false; }
    for (const Json& member : type_definition) {
        if (!member.is_string()) { return false; }
        SchemaType type;
        if (!schema_type(member.get_ref<const std::string&>(), type)) { return false; }
        types.bits |= type_bit(type);
    }
    return types.bits != 0;
}

bool compile_schema_types(const Json& schema, TypeSet& types) {
    if (!schema.is_object()) { return false; }
    const auto direct = schema.find("type");
    if (direct != schema.end()) { return compile_direct_types(*direct, types); }

    const auto any_of     = schema.find("anyOf");
    const auto one_of     = schema.find("oneOf");
    const bool has_any_of = any_of != schema.end();
    const bool has_one_of = one_of != schema.end();
    if (has_any_of == has_one_of) { return false; }

    const Json& alternatives = has_any_of ? *any_of : *one_of;
    if (!alternatives.is_array() || alternatives.empty()) { return false; }

    TypeSet combined;
    for (const Json& alternative : alternatives) {
        TypeSet branch;
        if (!compile_schema_types(alternative, branch)) { return false; }
        combined.bits |= branch.bits;
    }
    if (combined.bits == 0) { return false; }
    types = combined;
    return true;
}

Contract::Tool compile_tool_contract(const Json& definition) {
    Contract::Tool contract;
    if (!definition.is_object()) { return contract; }
    const auto function = definition.find("function");
    if (function == definition.end() || !function->is_object()) { return contract; }
    const auto name = function->find("name");
    if (name == function->end() || !name->is_string()) { return contract; }
    contract.name = name->get<std::string>();

    const auto schema = function->find("parameters");
    if (schema == function->end() || !schema->is_object()) { return contract; }
    const auto properties = schema->find("properties");
    if (properties == schema->end() || !properties->is_object()) { return contract; }

    contract.parameters.reserve(properties->size());
    for (const auto& [parameter_name, property] : properties->items()) {
        Contract::Parameter parameter;
        parameter.name = parameter_name;
        if (compile_schema_types(property, parameter.types)) {
            parameter.policy = NormalizationPolicy::DeclaredTypes;
        }
        contract.parameters.push_back(std::move(parameter));
    }
    return contract;
}

bool same_contract(const Contract::Tool& lhs, const Contract::Tool& rhs) {
    if (lhs.parameters.size() != rhs.parameters.size()) { return false; }
    for (std::size_t i = 0; i < lhs.parameters.size(); ++i) {
        const Contract::Parameter& left  = lhs.parameters[i];
        const Contract::Parameter& right = rhs.parameters[i];
        if (left.name != right.name || left.policy != right.policy ||
            left.types.bits != right.types.bits) {
            return false;
        }
    }
    return true;
}

void append_tool_contract(Contract& contracts, const Json& definition) {
    Contract::Tool compiled = compile_tool_contract(definition);
    if (compiled.name.empty()) { return; }
    const auto existing =
        std::find_if(contracts.tools.begin(), contracts.tools.end(),
                     [&](const auto& tool) { return tool.name == compiled.name; });
    if (existing == contracts.tools.end()) {
        contracts.tools.push_back(std::move(compiled));
        return;
    }
    if (existing->unambiguous && !same_contract(*existing, compiled)) {
        existing->parameters.clear();
        existing->unambiguous = false;
    }
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

std::string_view remove_parameter_framing_newlines(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    if (text.starts_with("\r\n")) {
        begin = 2;
    } else if (text.starts_with('\n')) {
        begin = 1;
    }
    if (end >= begin + 2 && text.substr(end - 2, 2) == "\r\n") {
        end -= 2;
    } else if (end > begin && text[end - 1] == '\n') {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool ascii_case_equal(std::string_view text, std::string_view lowercase) {
    if (text.size() != lowercase.size()) { return false; }
    for (std::size_t i = 0; i < text.size(); ++i) {
        char byte = text[i];
        if (byte >= 'A' && byte <= 'Z') { byte = static_cast<char>(byte + ('a' - 'A')); }
        if (byte != lowercase[i]) { return false; }
    }
    return true;
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

NormalizedParameter normalize_declared_parameter(std::string_view encoded_value, TypeSet types) {
    const std::string_view framed = remove_parameter_framing_newlines(encoded_value);
    if (admits_type(types, SchemaType::String)) {
        return {.json_value = encode_json_string(framed)};
    }

    const std::string_view value = trim_format_whitespace(framed);
    if (value.empty()) { return {.disposition = ParameterNormalization::Omitted}; }

    JsonValueKind kind;
    if (classify_json_value(value, kind)) {
        return {.disposition = admits_value(types, kind) ? ParameterNormalization::Emitted
                                                         : ParameterNormalization::SchemaMismatch,
                .json_value  = std::string(value)};
    }

    if (admits_type(types, SchemaType::Boolean)) {
        if (ascii_case_equal(value, "true")) { return {.json_value = "true"}; }
        if (ascii_case_equal(value, "false")) { return {.json_value = "false"}; }
    }
    return {.disposition = ParameterNormalization::SchemaMismatch,
            .json_value  = encode_json_string(framed)};
}

NormalizedParameter normalize_parameter(std::string_view encoded_value,
                                        const Contract::Parameter* parameter) {
    if (parameter != nullptr && parameter->policy == NormalizationPolicy::DeclaredTypes) {
        return normalize_declared_parameter(encoded_value, parameter->types);
    }

    const std::string_view value = trim_format_whitespace(encoded_value);
    if (Json::accept(value.begin(), value.end())) { return {.json_value = std::string(value)}; }
    return {.json_value = encode_json_string(value)};
}

class QwenToolRegionParser {
public:
    QwenToolRegionParser(std::string_view text, std::size_t max_name_length,
                         const Contract& contract)
        : text_(text), max_name_length_(max_name_length), contract_(contract) {}

    FallbackReason parse(std::vector<RawToolCall>& calls) const {
        std::size_t pos = 0;
        for (;;) {
            skip_format_whitespace(text_, pos);
            if (pos == text_.size()) {
                return calls.empty() ? FallbackReason::MalformedStructure : FallbackReason::None;
            }
            if (!starts_with_at(text_, pos, kToolOpen)) {
                return calls.empty() ? FallbackReason::MalformedStructure
                                     : FallbackReason::TrailingContent;
            }

            RawToolCall call;
            const FallbackReason failure = parse_tool_call(pos, call);
            if (failure != FallbackReason::None) { return failure; }
            calls.push_back(std::move(call));
        }
    }

private:
    bool consume(std::size_t& pos, std::string_view token) const {
        if (!starts_with_at(text_, pos, token)) { return false; }
        pos += token.size();
        return true;
    }

    FallbackReason parse_tool_call(std::size_t& pos, RawToolCall& call) const {
        if (!consume(pos, kToolOpen)) { return FallbackReason::MalformedStructure; }
        skip_format_whitespace(text_, pos);
        const FallbackReason failure = parse_function(pos, call);
        if (failure != FallbackReason::None) { return failure; }
        skip_format_whitespace(text_, pos);
        return consume(pos, kToolClose) ? FallbackReason::None : FallbackReason::MalformedStructure;
    }

    FallbackReason parse_function(std::size_t& pos, RawToolCall& call) const {
        if (!consume(pos, kFunctionOpen)) { return FallbackReason::MalformedStructure; }
        const std::size_t name_begin = pos;
        const std::size_t name_end   = text_.find('>', name_begin);
        if (name_end == std::string_view::npos || name_end == name_begin) {
            return FallbackReason::InvalidToolName;
        }
        call.name = text_.substr(name_begin, name_end - name_begin);
        if (!valid_function_name(call.name, max_name_length_)) {
            return FallbackReason::InvalidToolName;
        }
        if (contract_.enforce_declared_names &&
            find_tool_contract(contract_, call.name) == nullptr) {
            return FallbackReason::UndeclaredTool;
        }
        pos = name_end + 1;

        for (;;) {
            skip_format_whitespace(text_, pos);
            if (consume(pos, kFunctionClose)) { return FallbackReason::None; }
            const FallbackReason failure = parse_parameter(pos, call);
            if (failure != FallbackReason::None) { return failure; }
        }
    }

    FallbackReason parse_parameter(std::size_t& pos, RawToolCall& call) const {
        if (!consume(pos, kParamOpen)) { return FallbackReason::MalformedStructure; }
        const std::size_t name_begin = pos;
        const std::size_t name_end   = text_.find('>', name_begin);
        if (name_end == std::string_view::npos || name_end == name_begin) {
            return FallbackReason::MalformedStructure;
        }
        const std::string_view name = text_.substr(name_begin, name_end - name_begin);
        if (std::any_of(call.parameters.begin(), call.parameters.end(),
                        [&](const RawParameter& existing) { return existing.name == name; })) {
            return FallbackReason::DuplicateParameter;
        }

        const std::size_t value_begin = name_end + 1;
        std::size_t value_end         = 0;
        if (!find_parameter_close(value_begin, value_end)) {
            return FallbackReason::MalformedStructure;
        }
        call.parameters.push_back(RawParameter{
            .name = name, .value = text_.substr(value_begin, value_end - value_begin)});
        pos = value_end + kParamClose.size();
        return FallbackReason::None;
    }

    bool find_parameter_open_before(std::size_t scan, std::size_t limit,
                                    std::size_t& open_end) const {
        std::size_t candidate = text_.find(kParamOpen, scan);
        while (candidate != std::string_view::npos && candidate < limit) {
            const std::size_t name_begin = candidate + kParamOpen.size();
            const std::size_t name_end   = text_.find('>', name_begin);
            if (name_end != std::string_view::npos && name_end < limit && name_end != name_begin) {
                open_end = name_end + 1;
                return true;
            }
            candidate = text_.find(kParamOpen, candidate + 1);
        }
        return false;
    }

    bool find_parameter_close(std::size_t value_begin, std::size_t& value_end) const {
        std::size_t depth = 1;
        std::size_t scan  = value_begin;
        for (;;) {
            const std::size_t close = text_.find(kParamClose, scan);
            if (close == std::string_view::npos) { return false; }

            std::size_t nested_open_end = 0;
            if (find_parameter_open_before(scan, close, nested_open_end)) {
                ++depth;
                scan = nested_open_end;
                continue;
            }

            --depth;
            if (depth == 0) {
                value_end = close;
                return true;
            }
            scan = close + kParamClose.size();
        }
    }

    std::string_view text_;
    std::size_t max_name_length_;
    const Contract& contract_;
};

GeneratedToolCall normalize_raw_tool_call(const RawToolCall& raw, const Contract& contract,
                                          ToolCallParseDiagnostics& diagnostics) {
    const Contract::Tool* tool = find_tool_contract(contract, raw.name);
    if (tool != nullptr && !tool->unambiguous) { tool = nullptr; }

    std::string arguments = "{";
    bool first            = true;
    for (const RawParameter& raw_parameter : raw.parameters) {
        const Contract::Parameter* parameter =
            tool == nullptr ? nullptr : find_parameter_contract(*tool, raw_parameter.name);
        NormalizedParameter normalized = normalize_parameter(raw_parameter.value, parameter);
        if (tool != nullptr && parameter == nullptr) {
            normalized.disposition = ParameterNormalization::SchemaMismatch;
        }
        if (normalized.disposition == ParameterNormalization::Omitted) {
            ++diagnostics.empty_arguments_omitted;
            continue;
        }
        if (normalized.disposition == ParameterNormalization::SchemaMismatch) {
            ++diagnostics.schema_mismatch_arguments;
        }

        if (!first) { arguments.push_back(','); }
        first = false;
        arguments += encode_json_string(raw_parameter.name);
        arguments.push_back(':');
        arguments += normalized.json_value;
    }
    arguments.push_back('}');

    return GeneratedToolCall{.name = std::string(raw.name), .arguments_json = std::move(arguments)};
}

ParsedToolCallOutput fallback(const std::string& text, ToolCallParseDiagnostics diagnostics = {}) {
    ParsedToolCallOutput out;
    out.content     = text;
    out.diagnostics = diagnostics;
    return out;
}

} // namespace

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled) {
    if (!enabled) { return {}; }
    auto contract                    = std::make_shared<ToolCallOutputContract>();
    contract->enforce_declared_names = true;
    contract->tools.reserve(tool_jsons.size());
    for (const std::string& tool_json : tool_jsons) {
        const Json definition = Json::parse(tool_json, nullptr, false);
        if (!definition.is_discarded()) { append_tool_contract(*contract, definition); }
    }
    return contract;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolCallOutputContract& contract) {
    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content                 = rtrim_format_whitespace(std::string_view(text).substr(0, first));
    out.diagnostics.marker_seen = true;

    std::vector<RawToolCall> raw_calls;
    const std::string_view tool_region = std::string_view(text).substr(first);
    const QwenToolRegionParser parser(tool_region, max_tool_name_length, contract);
    const FallbackReason failure = parser.parse(raw_calls);
    if (failure != FallbackReason::None) {
        out.diagnostics.fallback_reason = failure;
        return fallback(text, out.diagnostics);
    }

    out.tool_calls.reserve(raw_calls.size());
    for (const RawToolCall& raw : raw_calls) {
        out.tool_calls.push_back(normalize_raw_tool_call(raw, contract, out.diagnostics));
    }

    out.diagnostics.structured_call_count = static_cast<std::uint32_t>(out.tool_calls.size());
    out.is_tool_call_response             = true;
    return out;
}

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                                             std::size_t max_tool_name_length)
    : contract_(std::move(contract)), max_tool_name_length_(max_tool_name_length) {}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    std::string visible;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char byte = text[index];
        if (marker_prefix_bytes_ != 0) {
            if (byte == kToolOpen[marker_prefix_bytes_]) {
                ++marker_prefix_bytes_;
                if (marker_prefix_bytes_ == kToolOpen.size()) {
                    tool_region_ = std::move(trailing_whitespace_);
                    trailing_whitespace_.clear();
                    tool_region_.append(kToolOpen);
                    tool_region_.append(text.substr(index + 1));
                    marker_prefix_bytes_ = 0;
                    saw_tool_marker_     = true;
                    break;
                }
                continue;
            }
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.append(kToolOpen.substr(0, marker_prefix_bytes_));
            marker_prefix_bytes_ = 0;
        }

        if (byte == kToolOpen.front()) {
            marker_prefix_bytes_ = 1;
        } else if (is_format_whitespace(byte)) {
            trailing_whitespace_.push_back(byte);
        } else {
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.push_back(byte);
        }
    }
    return visible;
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish() {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    ParsedToolCallOutput parsed =
        parse_qwen_tool_call_output(tool_region_, max_tool_name_length_, *contract_);
    if (saw_tool_marker_ && parsed.is_tool_call_response) {
        trailing_whitespace_.clear();
        tool_region_.clear();
        marker_prefix_bytes_ = 0;
        return Terminal{.content     = {},
                        .tool_calls  = std::move(parsed.tool_calls),
                        .diagnostics = parsed.diagnostics};
    }

    std::string tail = std::move(trailing_whitespace_);
    tail.append(kToolOpen.substr(0, marker_prefix_bytes_));
    marker_prefix_bytes_ = 0;
    tail += tool_region_;
    tool_region_.clear();
    return Terminal{
        .content = std::move(tail), .tool_calls = {}, .diagnostics = parsed.diagnostics};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
