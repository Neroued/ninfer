#pragma once

#include "ninfer/types.h"
#include <iterator>
#include <string>
#include <vector>

// Drive the public protocol encoders with the same append-only events as Engine consumers.
template <class Stream>
std::vector<std::string> emit_tools(Stream& stream,
                                    const std::vector<ninfer::GeneratedToolCall>& calls) {
    std::vector<std::string> wire;
    const auto append = [&](std::vector<std::string> events) {
        wire.insert(wire.end(), std::make_move_iterator(events.begin()),
                    std::make_move_iterator(events.end()));
    };
    for (std::uint32_t i = 0; i < calls.size(); ++i) {
        const auto& call = calls[i];
        append(stream.tool_start({i, call.name}));
        const auto split = call.arguments_json.size() / 2;
        append(stream.tool_arguments({i, call.arguments_json.substr(0, split)}));
        append(stream.tool_arguments({i, call.arguments_json.substr(split),
                                      call.status == ninfer::ToolCallStatus::Complete}));
        append(stream.tool_end({i, call.status}));
    }
    return wire;
}
