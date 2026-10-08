#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract);

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix. At terminal time, constrained output retains completed calls across interruptions;
// free output restores malformed regions verbatim.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length);

    [[nodiscard]] bool in_tool_region() const noexcept {
        return saw_tool_marker_ || marker_prefix_bytes_ != 0;
    }
    [[nodiscard]] std::string feed(std::string_view text);
    void initialize_continuation(std::string_view prefix);
    [[nodiscard]] Terminal finish(FinishReason reason = FinishReason::StopToken);

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string trailing_whitespace_;
    std::string tool_region_;
    std::size_t marker_prefix_bytes_         = 0;
    std::size_t max_tool_name_length_        = 0;
    bool saw_tool_marker_                    = false;
    bool finished_                           = false;
    std::size_t continuation_withheld_bytes_ = 0;
};

} // namespace ninfer::models::qwen3_5::frontend
