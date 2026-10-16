#pragma once

#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/moe.h"
#include "models/common/moe/schedule.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// The prefix/suffix contain the surrounding ordinary or speculative forward body.
// MoE definitions share exact-T bindings across the attention context profiles.
struct TextGraphDefinition {
    DecodeGraphDefinition prefix, suffix;
    std::vector<std::shared_ptr<const moe::LayerGraphDefinition>> layers;

    explicit TextGraphDefinition(std::size_t layers, const TextGraphDefinition* shared = nullptr);
};

class TextGraphExecutable {
public:
    explicit TextGraphExecutable(const TextGraphDefinition&,
                                 const TextGraphExecutable* shared = nullptr);
    void update(const TextGraphDefinition&);
    // The target is shared across draft topology classes of the same exact batch.
    // Activate it even when this prefix already has the requested profile installed.
    void activate(const TextGraphDefinition&);
    void upload(cudaStream_t);

private:
    friend class TextGraphRun;
    struct Target;
    DecodeGraphExecutable prefix_, suffix_;
    std::shared_ptr<Target> target_;
};

// One synchronous Host traversal; GPU work remains asynchronous on the compute stream.
// Capture records the same GPU bodies that eager execution emits. Replay leaves the
// layer cache/CPU handoffs on the caller and skips the captured prefix/suffix bodies.
class TextGraphRun {
public:
    TextGraphRun(TextGraphDefinition&, CudaExecutionView);
    TextGraphRun(TextGraphExecutable&, CudaExecutionView,
                 const std::function<void()>* after_first_submit = nullptr);

    [[nodiscard]] bool recording() const noexcept { return definition_ != nullptr; }

    void begin();
    void begin_layers();
    void end_layers();
    void finish();
    void capture_layer(std::size_t layer, moe::Schedule&, ExpertBankId, std::int32_t tokens,
                       const std::function<void()>& mixer,
                       const std::function<void(const moe::GpuCall&)>& gpu_moe,
                       const std::function<void()>& feature = {});
    void replay_layer(std::size_t layer, Tensor& output);

private:
    void first_submitted();

    TextGraphDefinition* definition_ = nullptr;
    TextGraphExecutable* executable_ = nullptr;
    CudaExecutionView cuda_;
    std::optional<DecodeGraphCapture> capture_;
    const std::function<void()>* after_first_submit_ = nullptr;
    bool traversed_                                  = false;
    bool notified_                                   = false;
};

[[nodiscard]] inline bool emit_forward_gpu(const TextGraphRun* run) {
    return !run || run->recording();
}

} // namespace ninfer::models::qwen3_5::execution
