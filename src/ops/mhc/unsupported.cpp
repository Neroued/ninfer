#include "ops/mhc/implementation.h"
#include <stdexcept>

namespace ninfer::ops::detail::mhc {
Plan select(Form, int, int) {
    throw std::invalid_argument("mhc: no implementation for this target");
}

void launch(Form, Operands, Plan, void*, CudaExecutionView) {
    throw std::invalid_argument("mhc: no implementation for this target");
}
} // namespace ninfer::ops::detail::mhc
