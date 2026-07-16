#pragma once

#include "ops/linear/w8/w8_rowsplit_launch.h"

namespace ninfer::ops::detail {

W8Problem w8_rowsplit_problem(const Tensor& x, const Weight& w, const Tensor& out) noexcept;

bool w8_rowsplit_admits(const W8Problem& problem) noexcept;
W8Plan w8_rowsplit_resolve_plan(const W8Problem& problem);

void w8_rowsplit_execute_plan(W8Plan plan, const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);
void w8_rowsplit_dispatch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
