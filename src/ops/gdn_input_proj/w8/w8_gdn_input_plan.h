#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class W8GdnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    MmaR64C128,
    SimtRowViewSplit,
    CutlassSm70,
};

enum class W8GdnInputConvScheduleId {
    DecodeFused,
    SplitKMmaFused,
    Materialized,
};

struct W8GdnInputProblem {
    std::int32_t input_rows;
    std::int32_t qkv_rows;
    std::int32_t z_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct W8GdnInputPlan {
    W8GdnInputScheduleId schedule;
};

struct W8GdnInputConvPlan {
    W8GdnInputConvScheduleId schedule;
};

const char* w8_gdn_input_schedule_name(W8GdnInputScheduleId schedule) noexcept;
const char* w8_gdn_input_conv_schedule_name(W8GdnInputConvScheduleId schedule) noexcept;
bool w8_gdn_input_admits(const W8GdnInputProblem& problem) noexcept;
W8GdnInputPlan w8_gdn_input_resolve_plan(const W8GdnInputProblem& problem);
W8GdnInputConvPlan w8_gdn_input_conv_resolve_plan(const W8GdnInputProblem& problem,
                                                  std::int32_t batch_size);

// `workspace` may be null; it is required only by schedules that stage a
// dequantized weight (CutlassSm70). Capacity is reported by
// gdn_input_proj_workspace_capacity_bytes().
void w8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           WorkspaceArena* workspace, cudaStream_t stream);

// Transient bytes the W8 GDN input projection needs across [min_cols, max_cols].
[[nodiscard]] std::size_t w8_gdn_input_workspace_bytes(std::int32_t min_cols,
                                                       std::int32_t max_cols);

} // namespace ninfer::ops::detail
