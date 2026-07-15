#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

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

enum class QType : std::uint16_t {
    Q4G64_F16S = 0,
    Q5G64_F16S = 1,
    Q6G64_F16S = 2,
    W8G32_F16S = 3,
    BF16_CTRL  = 4,
    FP32_CTRL  = 5,
    I32_CTRL   = 6,
};

enum class QuantLayout : std::uint16_t {
    RowSplit   = 0,
    Contiguous = 1,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4G64_F16S;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata        = nullptr;
    const void* qhigh        = nullptr;
    const void* scales       = nullptr;
    std::int32_t n           = 0;
    std::int32_t k           = 0;
    std::int32_t group       = 0;
    QuantLayout layout       = QuantLayout::RowSplit;
    DType scale_dtype        = DType::FP32;
    std::int32_t scale_ne[4] = {1, 1, 1, 1};
    std::int64_t scale_nb[4] = {0, 0, 0, 0};
};

} // namespace ninfer
