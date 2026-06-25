#pragma once

#include "qus/core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace qus {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

enum class QuantLayout : std::uint8_t {
    W4A16KernelPackedV1 = 0,
};

struct QuantWeight {
    const void* qdata        = nullptr;
    const void* scales       = nullptr;
    std::int32_t n           = 0;
    std::int32_t k           = 0;
    std::int32_t group       = 0;
    QuantLayout layout       = QuantLayout::W4A16KernelPackedV1;
    DType scale_dtype        = DType::FP32;
    std::int32_t scale_ne[4] = {1, 1, 1, 1};
    std::int64_t scale_nb[4] = {0, 0, 0, 0};
};

} // namespace qus
