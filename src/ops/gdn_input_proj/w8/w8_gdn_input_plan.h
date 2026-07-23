#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class W8GdnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    MmaR64C128,
};

enum class W8GdnInputSnapshotScheduleId {
    DecodeFused,
    SplitKMmaFused,
    Composed,
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
    W8KernelVariant variant;
    std::size_t workspace_bytes;
};

struct W8GdnInputSnapshotPlan {
    W8GdnInputSnapshotScheduleId schedule;
};

const char* w8_gdn_input_schedule_name(W8GdnInputScheduleId schedule) noexcept;
const char* w8_gdn_input_snapshot_schedule_name(W8GdnInputSnapshotScheduleId schedule) noexcept;
bool w8_gdn_input_admits(const W8GdnInputProblem& problem) noexcept;
W8GdnInputPlan w8_gdn_input_resolve_plan(const W8GdnInputProblem& problem);
W8GdnInputSnapshotPlan w8_gdn_input_snapshot_resolve_plan(const W8GdnInputProblem& problem);
std::size_t w8_gdn_input_capacity_workspace_bytes(std::int32_t qkv_rows, std::int32_t z_rows,
                                                  std::int32_t max_cols);

void w8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream);

} // namespace ninfer::ops::detail
