#include "runtime/engine/request_memory.h"

#include "core/arena.h"
#include "core/device.h"

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

class RequestMemory::Impl {
public:
    explicit Impl(DeviceContext& context) : device(context.device) {}

    ~Impl() {
        if (arena != nullptr) {
            (void)cudaSetDevice(device);
            arena.reset();
        }
    }

    int device = 0;
    std::unique_ptr<DeviceArena> arena;
    std::size_t active_bytes     = 0;
    std::size_t active_alignment = 1;
};

RequestMemory::RequestMemory(DeviceContext& device) : impl_(std::make_unique<Impl>(device)) {}

RequestMemory::~RequestMemory() = default;

void RequestMemory::ensure(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        if (alignment != 1) {
            throw std::invalid_argument("an empty transient region must use alignment one");
        }
        reset();
        return;
    }

    if (!is_power_of_two(alignment) || alignment > kDeviceAllocationAlignment) {
        throw std::invalid_argument("unsupported transient region alignment");
    }

    if (impl_->arena == nullptr || impl_->arena->capacity() < bytes) {
        CUDA_CHECK(cudaSetDevice(impl_->device));
        auto replacement = std::make_unique<DeviceArena>(bytes);
        impl_->arena     = std::move(replacement);
    }

    impl_->active_bytes     = bytes;
    impl_->active_alignment = alignment;
}

void RequestMemory::reset() noexcept {
    impl_->active_bytes     = 0;
    impl_->active_alignment = 1;
}

TransientRegion RequestMemory::region() const noexcept {
    if (impl_->active_bytes == 0) { return {}; }
    return {static_cast<std::byte*>(impl_->arena->base()), impl_->active_bytes,
            impl_->active_alignment};
}

std::size_t RequestMemory::capacity_bytes() const noexcept {
    return impl_->arena != nullptr ? impl_->arena->capacity() : 0;
}

} // namespace ninfer::runtime
