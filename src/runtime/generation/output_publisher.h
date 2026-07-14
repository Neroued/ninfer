#pragma once

#include "ninfer/types.h"

#include <utility>

namespace ninfer::runtime {

class OutputPublisher {
public:
    explicit OutputPublisher(OutputSink* sink) noexcept : sink_(sink) {}

    template <class PublishedOutput>
    void publish(PublishedOutput&& output) {
        for (OutputDelta& delta : output) {
            std::string& full = delta.channel == OutputChannel::Reasoning ? reasoning_ : content_;
            full += delta.text;
            if (sink_ != nullptr) { sink_->publish(std::move(delta)); }
        }
    }

    [[nodiscard]] std::string take_content() noexcept { return std::move(content_); }

    [[nodiscard]] std::string take_reasoning() noexcept { return std::move(reasoning_); }

private:
    OutputSink* sink_ = nullptr;
    std::string content_;
    std::string reasoning_;
};

} // namespace ninfer::runtime
