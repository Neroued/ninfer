#pragma once

// The NVFP4 gate|up problems of a dense FFN of intermediate width 14336 over hidden 5120: one
// device's [28672,5120] and its two-device output-row half [14336,5120], out [14336,T] and
// [7168,T].
//
// Same scheme as the [17408,5120] half (tp2/linear_swiglu_half.h): upstream's NVFP4 plan and A4
// launcher compiled again with the problem geometry and the entry points renamed
// (*_names.h); the decode, small-T and TMA launchers read the rows from the weight and are shared.
// Every route, schedule and token cutoff is the [34816,5120] parent's, not re-measured at these
// shapes.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kLinearSwiGluFfn14336Rows     = 28672;
inline constexpr std::int32_t kLinearSwiGluFfn14336HalfRows = 14336;

void nvfp4_linear_swiglu_n28672_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t
nvfp4_linear_swiglu_n28672_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens);
void nvfp4_linear_swiglu_n28672_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                         LinearPolicy policy, WorkspaceArena* workspace,
                                         cudaStream_t stream);

void nvfp4_linear_swiglu_n14336_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t
nvfp4_linear_swiglu_n14336_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens);
void nvfp4_linear_swiglu_n14336_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                         LinearPolicy policy, WorkspaceArena* workspace,
                                         cudaStream_t stream);

} // namespace ninfer::ops::detail
