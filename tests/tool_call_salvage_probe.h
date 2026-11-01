#pragma once

// Shared model output for the protocol tests' salvaged tool-call round trips: one Qwen call per
// probe name, parsed against the output contract the server builds from a request's PromptInput.
// A name the parser keeps as a call must come back in the next request's history; one the request
// parsers reject would turn the kept call into an HTTP 400 one turn later.

#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "ninfer/types.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::test {

// Declared, undeclared, catalog-spelled, punctuated, non-ASCII and boundary-length names.
inline std::vector<std::string> salvage_probe_names() {
    return {"Read",
            "other_tool",
            "a-b_c9",
            "srv.Read",
            "srv/Read",
            "tools[\"srv\"][\"Read\"]",
            "tools[\"context-mode\"].ctx_batch_execute",
            "bad.name",
            ".Read",
            "Read.",
            "bad name",
            "<b>",
            "\xc3\xbcnicode",
            std::string(64, 'x'),
            std::string(65, 'x'),
            std::string(128, 'y'),
            std::string(129, 'y')};
}

inline std::string qwen_tool_call(std::string_view name) {
    std::string text = "<tool_call>\n<function=";
    text.append(name);
    text += ">\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>";
    return text;
}

// The model calling `name` once, parsed as the server parses it for the request behind `input`.
inline models::qwen3_5::frontend::ParsedToolCallOutput
parse_model_call(const PromptInput& input, std::size_t max_tool_name_length,
                 std::string_view name) {
    namespace fi        = models::qwen3_5::frontend;
    const auto contract = fi::build_tool_call_output_contract(
        input.options.tool_jsons, input.options.filtered_out_tool_names, true);
    return fi::parse_qwen_tool_call_output(qwen_tool_call(name), max_tool_name_length, *contract);
}

} // namespace ninfer::test
