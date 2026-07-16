#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kBf16GdnGatingMaxCols       = 128 * 65535;
inline constexpr std::int32_t kBf16GdnGatingQualifiedCols = 1024;

enum class Bf16GdnGatingScheduleId {
    GemvPairedRows,
    SmallTSplit10,
    SimtWarpRowC4,
    SimtWarpRowC8,
    MmaCooperativeSplit32,
    MmaCooperativeSplit16,
    MmaCooperativeSplit8,
    MmaCooperativeSplit4,
    MmaCooperativeSplit2,
    MmaUnsplit,
};

struct Bf16GdnGatingProblem {
    std::int32_t heads;
    std::int32_t input_rows;
    std::int32_t cols;
};

struct Bf16GdnGatingPlan {
    Bf16GdnGatingScheduleId schedule;
    Bf16GdnGatingTokenVariant token_variant;
    std::size_t workspace_bytes;
    bool performance_qualified;
};

const char* bf16_gdn_gating_schedule_name(Bf16GdnGatingScheduleId schedule) noexcept;

bool bf16_gdn_gating_admits(const Bf16GdnGatingProblem& problem) noexcept;
Bf16GdnGatingPlan bf16_gdn_gating_resolve_plan(const Bf16GdnGatingProblem& problem);
Bf16GdnGatingPlan bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId schedule,
                                                    const Bf16GdnGatingProblem& problem);

std::size_t bf16_gdn_gating_capacity_workspace_bytes(std::int32_t max_cols);

void bf16_gdn_gating_execute_plan(const Bf16GdnGatingPlan& plan, const Tensor& x,
                                  const Weight& a_weight, const Weight& b_weight,
                                  const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                  Tensor& g, Tensor& beta, cudaStream_t stream);
void bf16_gdn_gating_execute_candidate(Bf16GdnGatingScheduleId schedule, const Tensor& x,
                                       const Weight& a_weight, const Weight& b_weight,
                                       const Tensor& A_log, const Tensor& dt_bias,
                                       WorkspaceArena& ws, Tensor& g, Tensor& beta,
                                       cudaStream_t stream);
void bf16_gdn_gating_dispatch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                              Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops::detail
