#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/text.h"

#include <functional>
#include <memory>
#include <span>

namespace ninfer::models::qwen3_5 {
struct GraphProfile;
}

namespace ninfer::models::qwen3_5::execution {

// One exact width/batch traversal. Resident execution submits one complete graph;
// offloaded execution retains local CPU/cache readiness between captured layer bodies.
class TextGraph {
public:
    TextGraph();
    ~TextGraph();
    TextGraph(TextGraph&&) noexcept;
    TextGraph& operator=(TextGraph&&) noexcept;
    TextGraph(const TextGraph&)            = delete;
    TextGraph& operator=(const TextGraph&) = delete;

    void prepare(TextRun&, const Parameters&, moe::Schedule*, std::span<const GraphProfile>,
                 std::uint32_t width, const std::function<void()>& prologue,
                 const std::function<void()>& epilogue, MemoryArena&, DeviceContext&,
                 const TextGraph* shared = nullptr);
    void select(std::uint32_t frontier);
    [[nodiscard]] bool start(CudaExecutionView);
    [[nodiscard]] bool poll();
    void cancel();
    [[nodiscard]] std::size_t workspace_peak() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen3_5::execution
