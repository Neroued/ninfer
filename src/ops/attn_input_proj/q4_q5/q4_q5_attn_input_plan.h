#pragma once

#include "ops/linear/q4/q4_rowsplit_plan.h"
#include "ops/linear/q5/q5_rowsplit_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kQ4Q5AttnInputMaxCols       = 128 * 65535;
inline constexpr std::int32_t kQ4Q5AttnInputQualifiedCols = 1024;

enum class Q4Q5AttnInputScheduleId {
    IndependentFixed,
    GroupedHomogeneousPairMmaR64C128,
};

struct Q4Q5AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct Q4Q5AttnInputSubplans {
    Q4Plan query;
    Q5Plan gate;
    Q4Plan key;
    Q5Plan value;
};

struct Q4Q5AttnInputPlan {
    Q4Q5AttnInputScheduleId schedule;
    Q4KernelVariant grouped_variant;
    std::optional<Q4Q5AttnInputSubplans> independent;
    std::size_t workspace_bytes;
    bool performance_qualified;
};

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept;

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept;
Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem);

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& q_weight, const Weight& gate_weight,
                                   const Weight& k_weight, const Weight& v_weight, Tensor& q,
                                   Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& q_weight, const Weight& gate_weight,
                               const Weight& k_weight, const Weight& v_weight, Tensor& q,
                               Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
