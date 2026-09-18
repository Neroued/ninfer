#pragma once
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"

#include "core/nvtx.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        executable->launch(state.execution.device.stream);
    } else {
        nvtx::ScopedRange eager_range(nvtx::Name::DecodeEager, nvtx::Category::Decode);
        body();
    }
}

template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    definition.capture(state.execution.device.stream, body);
}

} // namespace ninfer::models::qwen3_5::execution
