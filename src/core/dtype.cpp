#include "ninfer/core/dtype.h"

#include <stdexcept>

namespace ninfer {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return 2;
    case DType::FP32:
        return 4;
    case DType::I32:
        return 4;
    case DType::U8:
        return 1;
    case DType::I64:
        return 8;
    case DType::I8:
        return 1;
    case DType::FP16:
        return 2;
    }
    throw std::invalid_argument("invalid DType");
}

} // namespace ninfer
