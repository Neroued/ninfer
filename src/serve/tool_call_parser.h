#pragma once

#include "serve/request.h"

#include <cstddef>
#include <string>
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

} // namespace ninfer::serve
