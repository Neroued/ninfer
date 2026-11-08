#pragma once

#include "core/dtype.h"
#include "core/memory.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16                  = 0,
    Q5_G64_FP16                  = 1,
    Q6_G64_FP16                  = 2,
    Q8_G32_FP16                  = 3,
    BF16                         = 4,
    FP32                         = 5,
    INT32                        = 6,
    NVFP4                        = 7,
    FP8_E4M3FN_ROW_BF16          = 8,
    NF4_G16_E4M3FN               = 9,
    FP8_E4M3FN_BLOCK128X128_BF16 = 10,
    FP8_E4M3FN_BLOCK64X64_BF16   = 11,
    NVFP4_MUL                    = 12,
    MXFP8                        = 13,
    FP8_E4M3FN_BLOCK128X128_FP32 = 14,
};

enum class QuantLayout : std::uint16_t {
    RowSplit                   = 0,
    Contiguous                 = 1,
    BlockScaleK16M128x4        = 2,
    RowScale                   = 3,
    RowPagedG16_4K             = 4,
    BatchedBlockScaleK16M128x4 = 5,
    BlockScaleM128K128         = 6,
    BlockScaleM64K64           = 7,
    Csf32                      = 8,
    RowGroupScale              = 9,
};

enum class MatrixScaleOperation : std::uint8_t { None, Divide, Multiply };

[[nodiscard]] constexpr MatrixScaleOperation matrix_scale_operation(QType format) noexcept {
    switch (format) {
    case QType::NVFP4:
    case QType::NF4_G16_E4M3FN:
        return MatrixScaleOperation::Divide;
    case QType::NVFP4_MUL:
        return MatrixScaleOperation::Multiply;
    default:
        return MatrixScaleOperation::None;
    }
}

// Borrowed compressed scale planes. Bases and indices begin at one matrix; exceptions
// remain bank-global. A submatrix retains the original geometry and a logical row origin.
struct Csf32ScaleView {
    const std::uint8_t* bases       = nullptr;
    const std::uint8_t* nibbles     = nullptr;
    const std::uint16_t* offsets    = nullptr;
    const std::uint32_t* page_bases = nullptr;
    const std::uint16_t* exceptions = nullptr;
    std::uint64_t rows = 0, columns = 0, row_begin = 0;
    std::uint64_t exception_count = 0;
};

struct Weight {
    Device device;
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
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
    float weight_scale       = 0.0F;
    float input_scale        = 0.0F;
    Csf32ScaleView csf;
};

} // namespace ninfer
