#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// Qwen's tool syntax carries each argument as untyped text. This terminal contract records only
// the supported top-level JSON Schema types needed to normalize that text. A type mismatch remains
// a structured call for consumer validation; recursive validation is outside this non-strict
// contract.
struct ToolCallOutputContract {
    enum class SchemaType : std::uint8_t {
        Null    = 1U << 0U,
        Boolean = 1U << 1U,
        Integer = 1U << 2U,
        Number  = 1U << 3U,
        String  = 1U << 4U,
        Object  = 1U << 5U,
        Array   = 1U << 6U,
    };

    struct TypeSet {
        std::uint8_t bits = 0;
    };

    enum class NormalizationPolicy : std::uint8_t {
        Legacy,
        DeclaredTypes,
    };

    struct Parameter {
        std::string name;
        NormalizationPolicy policy = NormalizationPolicy::Legacy;
        TypeSet types;
    };

    struct Tool {
        std::string name;
        std::vector<Parameter> parameters;
        bool unambiguous = true;
    };

    std::vector<Tool> tools;
    // Declared in the request but removed for this turn by tool_choice.allowed_tools: calls naming
    // them keep falling back to text so the filter stays a restriction.
    std::vector<std::string> filtered_out_names;
};

// How the text of a tool-call region ended. The structural repairs that read the end of the text
// as the end of a call (missing function/tool closers, an unterminated last parameter) apply only
// when the model ended its own response with a stop token. A token budget, a context limit, a stop
// string or a cancellation cuts the call short instead: its last value may be incomplete, so the
// region stays ordinary text, as it would without those repairs.
enum class ToolCallRegionEnd : std::uint8_t {
    ModelStop,
    Cut,
};

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    // Prose after the last complete call, right-trimmed; the decoder appends it to the content.
    std::string trailing_content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

[[nodiscard]] std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons,
                                std::span<const std::string> filtered_out_names, bool enabled);

[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract,
                            ToolCallRegionEnd end = ToolCallRegionEnd::ModelStop);

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix. At terminal time the region becomes structured calls when it parses, possibly after the
// recoveries of docs/serving.md (undeclared or catalog-spelled names, closers at the model's stop,
// stray closers, trailing prose); otherwise it is restored verbatim as content.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length);

    [[nodiscard]] std::string feed(std::string_view text);
    // Trailing prose kept beside salvaged calls is preceded by the whitespace the model wrote
    // before its first call (a newline when there was none) when visible content was already
    // published.
    [[nodiscard]] Terminal finish(ToolCallRegionEnd end);

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string trailing_whitespace_;
    std::string tool_region_;
    std::size_t marker_prefix_bytes_  = 0;
    std::size_t max_tool_name_length_ = 0;
    bool saw_tool_marker_             = false;
    bool published_content_           = false;
    bool finished_                    = false;
};

} // namespace ninfer::models::qwen3_5::frontend
