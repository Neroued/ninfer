#pragma once

#include "runtime/contract/transient_region.h"

#include <cstddef>
#include <memory>

namespace ninfer {

struct DeviceContext;

namespace runtime {

// Owns the request-lifetime device buffer. ensure() may grow the allocation and
// therefore must be called before target execution begins for the request.
class RequestMemory {
public:
    static constexpr std::size_t kDeviceAllocationAlignment = 256;

    explicit RequestMemory(DeviceContext& device);
    ~RequestMemory();

    RequestMemory(const RequestMemory&)            = delete;
    RequestMemory& operator=(const RequestMemory&) = delete;
    RequestMemory(RequestMemory&&)                 = delete;
    RequestMemory& operator=(RequestMemory&&)      = delete;

    void ensure(std::size_t bytes, std::size_t alignment);
    void reset() noexcept;

    [[nodiscard]] TransientRegion region() const noexcept;
    [[nodiscard]] std::size_t capacity_bytes() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace ninfer
