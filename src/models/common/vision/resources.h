#pragma once

#include "core/buffer.h"
#include "core/tensor.h"
#include "core/device.h"
#include "models/common/vision/window.h"

#include <cstdint>
#include <functional>

namespace ninfer::models::vision {

struct EncoderMemory {
    std::size_t workspace_bytes    = 0;
    std::size_t host_staging_bytes = 0;
};

struct Capacity {
    std::size_t budget_bytes   = 0;
    std::uint32_t max_features = 0;
    std::int32_t output_width  = 0;
    EncoderMemory encoder;
    std::size_t output_bytes = 0;

    [[nodiscard]] std::size_t device_bytes() const;
};

// The model query covers all legal batches with at most the supplied feature count.
[[nodiscard]] Capacity plan_capacity(std::size_t budget, std::uint32_t maximum_features,
                                     std::int32_t output_width,
                                     const std::function<EncoderMemory(std::uint32_t)>&);

// The Program owns this storage and retires its readers before destruction. Encoder scratch
// is borrowed from the Program's shared compute workspace.
class Storage {
public:
    Storage(DeviceContext&, Capacity);

    [[nodiscard]] Tensor output(std::uint32_t first, std::uint32_t count);

    [[nodiscard]] MemorySpan staging() { return staging_.span(); }

    [[nodiscard]] const Capacity& capacity() const noexcept { return capacity_; }

    [[nodiscard]] bool begin(const WindowUse&);
    void produced(std::size_t workspace_bytes);
    void release(std::uint64_t id) noexcept;
    [[nodiscard]] std::optional<std::uint64_t> take_timing();
    [[nodiscard]] std::size_t active_bytes() const noexcept;

    [[nodiscard]] const WindowUse* current() const noexcept {
        return active_ ? &*active_ : nullptr;
    }

    [[nodiscard]] std::size_t workspace_peak_bytes() const noexcept { return workspace_peak_; }

    void reset_peaks() noexcept;
    [[nodiscard]] VisionMemorySummary observe() const noexcept;

private:
    Capacity capacity_;
    DeviceBuffer output_;
    PinnedHostBuffer staging_;
    CudaEventTimer timer_;
    std::optional<WindowUse> active_;
    std::uint64_t last_id_   = 0;
    std::size_t output_peak_ = 0, workspace_peak_ = 0;
    bool timing_pending_ = false;
};

} // namespace ninfer::models::vision
