#pragma once

#include "serve/request.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

// Parse Qwen's XML-like tool-call format. In tolerant mode, a complete function
// call is recovered even when the model adds wrapper garbage or suffix text.
ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 bool tolerant = false);

// Incrementally publishes text that is provably outside a possible Qwen
// <tool_call> suffix. At terminal time, a valid tool response discards the
// buffered tool region; malformed/non-tool output flushes it verbatim.
class ToolCallStreamFilter {
public:
    std::string feed(std::string_view text);
    std::string finish(bool is_tool_call_response);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    std::string pending_;
    std::string tool_region_;
    std::size_t emitted_bytes_ = 0;
    bool saw_tool_marker_      = false;
    bool finished_             = false;
};

} // namespace ninfer::serve
