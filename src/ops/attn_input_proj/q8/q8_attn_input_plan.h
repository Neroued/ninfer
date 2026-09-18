#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8AttnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    SimtR8C4,
    MmaR32C64,
    MmaR32C128,
    MmaR64C64,
    MmaR64C96,
    MmaR64C128,
    MmaR128C64,
    MmaR128C80,
    DFlash2SmallT,
    DFlash2MmaR16C64K128,
    DFlash2MmaR32C32K128,
    DFlash2MmaR32C64K128,
    DFlash2MmaR32C64,
    DFlash2MmaR64C128,
};

struct Q8AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q8AttnInputPlan {
    Q8AttnInputScheduleId schedule;
};

const char* q8_attn_input_schedule_name(Q8AttnInputScheduleId schedule) noexcept;
bool q8_attn_input_admits(const Q8AttnInputProblem& problem) noexcept;
Q8AttnInputPlan q8_attn_input_resolve_plan(const Q8AttnInputProblem& problem);

void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_execute_plan(const Q8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, cudaStream_t stream);
void q8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
