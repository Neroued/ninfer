#pragma once

#include "qus/serve/request.h"

#include <string>
#include <vector>

namespace qus::serve {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text);

} // namespace qus::serve
