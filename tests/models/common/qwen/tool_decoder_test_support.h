#pragma once

#include "models/common/qwen/frontend/tool_call_parser.h"

#include <algorithm>

namespace tool_decoder_test {
using Decoder = ninfer::models::qwen::frontend::ToolCallOutputDecoder;

inline std::string text(const std::vector<ninfer::OutputEvent>& events) {
    std::string result;
    for (const auto& event : events)
        if (const auto* delta = std::get_if<ninfer::OutputDelta>(&event)) result += delta->text;
    return result;
}

inline std::string feed(Decoder& decoder, std::string_view input) {
    return text(decoder.feed(input));
}

inline ninfer::models::qwen::frontend::ParsedToolCallOutput
finish(Decoder& decoder, ninfer::FinishReason reason = ninfer::FinishReason::StopToken) {
    auto content = text(decoder.finish(reason));
    if (const auto error = decoder.error()) std::rethrow_exception(error->exception());
    auto calls           = decoder.take_tool_calls();
    const bool has_calls = !calls.empty();
    return {has_calls, std::move(content), std::move(calls), decoder.diagnostics()};
}

inline std::size_t completed(const std::vector<ninfer::GeneratedToolCall>& calls) {
    return std::count_if(calls.begin(), calls.end(), [](const auto& call) {
        return call.status == ninfer::ToolCallStatus::Complete;
    });
}
} // namespace tool_decoder_test
