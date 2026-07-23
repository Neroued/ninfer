#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

template <class Body>
void run_prepared(State& state, DecodeGraph* graph, Body&& body) {
    if (graph != nullptr) {
        if (!graph->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        graph->launch(state.device.stream);
    } else {
        body();
    }
}

template <class Body>
void warm_capture(State& state, DecodeGraph& graph, const GraphPrepare& prepare, Body&& body) {
    prepare();
    state.device.synchronize();
    body();
    state.device.synchronize();
    prepare();
    state.device.synchronize();
    graph.capture(state.device.stream, body);
    prepare();
    state.device.synchronize();
    graph.launch(state.device.stream);
    state.device.synchronize();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
