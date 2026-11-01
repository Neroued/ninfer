#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::serve {

// Protocol-local accumulation of the Engine's ordered call events. Wire IDs remain adapter-owned.
class ToolStreamState {
public:
    void start(const ToolCallStart& event) {
        if (event.index != calls_.size() ||
            (!calls_.empty() && calls_.back().status == ToolCallStatus::InProgress))
            throw std::logic_error("invalid tool start order");
        calls_.push_back({event.name, {}, ToolCallStatus::InProgress});
    }

    void append(const ToolArgumentsDelta& event) {
        active(event.index).arguments_json += event.text;
    }

    void end(const ToolCallEnd& event) {
        if (event.status == ToolCallStatus::InProgress)
            throw std::logic_error("tool end has no terminal status");
        active(event.index).status = event.status;
    }

    const GeneratedToolCall& at(std::size_t index) const { return calls_.at(index); }

    std::span<const GeneratedToolCall> calls() const noexcept { return calls_; }

    void verify(std::span<const GeneratedToolCall> final) const {
        if (final.size() != calls_.size() ||
            !std::equal(final.begin(), final.end(), calls_.begin()))
            throw std::logic_error("streamed tools disagree with terminal calls");
    }

private:
    GeneratedToolCall& active(std::uint32_t index) {
        if (index >= calls_.size() || calls_[index].status != ToolCallStatus::InProgress)
            throw std::logic_error("tool delta has no active call");
        return calls_[index];
    }

    std::vector<GeneratedToolCall> calls_;
};

} // namespace ninfer::serve
