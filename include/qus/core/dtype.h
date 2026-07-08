#pragma once

#include <cstddef>
#include <cstdint>

namespace qus {

enum class DType : std::uint8_t {
    BF16 = 0,
    FP32 = 1,
    I32  = 2,
    U8   = 3,
    I64  = 4,
    I8   = 5,
    FP16 = 6,
};

std::size_t dtype_size(DType dtype);

} // namespace qus
