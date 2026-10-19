#pragma once

#include "core/buffer.h"
#include "models/common/context/storage.h"

#include <span>
#include <vector>

namespace ninfer::models::context {

// One rank-local action slot. Its event and pinned mapping storage are fixed at startup. Compute
// and independent context migration use separate slots; neither owns directory reservations.
class ContextMigration {
public:
    ContextMigration(DeviceContext& device, ContextStorage& storage,
                     std::size_t mapping_capacity_elements = 0,
                     std::size_t copy_segment_capacity     = 0);
    ContextMigration(DeviceContext& device, ContextStorage& storage, CudaExecutionView execution,
                     std::size_t mapping_capacity_elements = 0,
                     std::size_t copy_segment_capacity     = 0);
    ~ContextMigration();
    ContextMigration(const ContextMigration&)            = delete;
    ContextMigration& operator=(const ContextMigration&) = delete;

    void validate(const ContextActions& actions) const;

    // Transfer-stream work inherits all compute already submitted at this boundary.
    void submit(const ContextActions& actions);
    [[nodiscard]] bool ready() const;

    [[nodiscard]] bool occupied() const noexcept { return occupied_; }

    [[nodiscard]] std::size_t pinned_bytes() const noexcept { return mapping_.size(); }

    [[nodiscard]] std::span<const runtime::ContextTransferObservation> observations();
    void finish();
    void drain();

private:
    struct Segment {
        runtime::ContextTransferObservation observed;
        std::size_t begin = 0;
        std::size_t end   = 0;
    };

    void prepare(const ContextActions& actions);
    void enqueue(const ContextActions& actions, CudaExecutionView execution);

    DeviceContext* device_;
    ContextStorage* storage_;
    CudaExecutionView execution_;
    PinnedHostBuffer mapping_;
    CudaCompletionEvent source_ready_;
    CudaCompletionEvent completed_;
    std::vector<CudaEventTimer> timers_;
    std::vector<Segment> segments_;
    std::vector<runtime::ContextTransferObservation> observations_;
    cudaStream_t submitted_stream_ = nullptr;
    bool occupied_                 = false;
    bool fenced_                   = false;
};

} // namespace ninfer::models::context
