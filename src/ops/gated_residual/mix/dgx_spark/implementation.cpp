#include "ops/gated_residual/mix/implementation.h"

#include <stdexcept>

namespace ninfer::ops::detail::hc::mix {

void execute(Weights, const Tensor&, MemoryArena&, Tensor&, Tensor*, CudaExecutionView) {
    throw std::invalid_argument("hc_mix: no implementation for dgx_spark");
}

std::size_t capacity(Weights, std::int32_t, std::int32_t) {
    throw std::invalid_argument("hc_mix: no implementation for dgx_spark");
}

} // namespace ninfer::ops::detail::hc::mix
