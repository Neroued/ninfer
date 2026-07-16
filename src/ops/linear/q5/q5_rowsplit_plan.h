#pragma once

#include "ops/linear/q5/q5_rowsplit_launch.h"

namespace ninfer::ops::detail {

Q5Problem q5_rowsplit_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept;

bool q5_rowsplit_admits(const Q5Problem& problem) noexcept;
Q5Plan q5_rowsplit_resolve_plan(const Q5Problem& problem);

void q5_rowsplit_execute_plan(Q5Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void q5_rowsplit_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
