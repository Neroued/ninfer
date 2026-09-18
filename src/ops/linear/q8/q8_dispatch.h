#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/q8/q8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace ninfer::ops::detail
