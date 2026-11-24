#pragma once

#include "core/dtype.h"
#include "core/memory.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct Tensor {
    void* data  = nullptr;
    DType dtype = DType::BF16;
    Device device;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    // A null, unlocated span describes planned capacity; a bound span supplies actual residency.
    Tensor(MemorySpan memory, DType dtype, std::initializer_list<std::int32_t> shape);
    Tensor(void* data, Device device, DType dtype, std::initializer_list<std::int32_t> shape);
    [[nodiscard]] static Tensor metadata(DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

} // namespace ninfer
