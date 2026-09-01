#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/q4/q4_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
