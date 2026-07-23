#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class W8AttnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaExactT,
    SimtR8C4,
    MmaR32C128,
    MmaR64C128,
};

struct W8AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct W8AttnInputPlan {
    W8AttnInputScheduleId schedule;
    W8KernelVariant variant;
    std::size_t workspace_bytes;
};

const char* w8_attn_input_schedule_name(W8AttnInputScheduleId schedule) noexcept;
bool w8_attn_input_admits(const W8AttnInputProblem& problem) noexcept;
W8AttnInputPlan w8_attn_input_resolve_plan(const W8AttnInputProblem& problem);

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
