#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        return "attn_input_proj.q4_q5.mixed.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        return "attn_input_proj.q4_q5.pair.r32.c64.s3";
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        return "attn_input_proj.q4_q5.mixed.r64.c128.s2";
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        return "attn_input_proj.q4_q5.pair.r32.c64.s4";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    if (problem.cols <= 12) return {Q4Q5AttnInputScheduleId::ParentSplitFixed};
    if (problem.cols <= 64) return {Q4Q5AttnInputScheduleId::MixedR32C64S3};
    if (problem.cols <= 104) return {Q4Q5AttnInputScheduleId::PairR32C64S3};
    if (problem.cols <= 128 || problem.cols >= 193)
        return {Q4Q5AttnInputScheduleId::MixedR64C128S2};
    return {Q4Q5AttnInputScheduleId::PairR32C64S4};
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR32C64S3:
        q4_q5_attn_input_mixed_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                 v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S3:
        q4_q5_attn_input_pair_r32_c64_s3_launch(x, query_key_weight, gate_value_weight, q, gate, k,
                                                v, stream);
        return;
    case Q4Q5AttnInputScheduleId::MixedR64C128S2:
        q4_q5_attn_input_mixed_r64_c128_s2_launch(x, query_key_weight, gate_value_weight, q, gate,
                                                  k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::PairR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  stream);
}

} // namespace ninfer::ops::detail
