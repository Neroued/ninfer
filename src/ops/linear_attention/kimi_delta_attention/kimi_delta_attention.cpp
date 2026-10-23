#include "ninfer/ops/kimi_delta_attention.h"
#include "ops/common/host/operands.h"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "ops/linear_attention/kimi_delta_attention/launch.h"

namespace ninfer::ops {
namespace {
namespace kda            = detail::kimi_delta_attention;
constexpr const char* op = "kimi_delta_attention";

kda::Arguments arguments(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                         const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                         float lower_bound, float scale, const float* state_in, float* state_out,
                         Tensor& out) {
    return {static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const __nv_bfloat16*>(g.data),
            static_cast<const __nv_bfloat16*>(beta.data),
            static_cast<const float*>(a_log.data),
            static_cast<const float*>(dt_bias.data),
            state_in,
            state_out,
            static_cast<__nv_bfloat16*>(out.data),
            q.ne[1],
            v.ne[1],
            q.ne[2],
            lower_bound,
            scale};
}
} // namespace

std::size_t kimi_delta_attention_workspace_capacity_bytes(int qk, int value, int first, int last) {
    if (!kda::valid_heads(qk, value) || first < 1 || last < first)
        throw std::invalid_argument("kimi_delta_attention workspace: invalid heads or interval");
    return kda::recurrence_workspace_bytes(value, last);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace, Tensor& state,
                          Tensor& out, CudaExecutionView execution) {
    kimi_delta_attention(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, workspace, state,
                         state, out, execution);
}

void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          CudaExecutionView execution) {
    common::operand::device(workspace.device(), execution.device(), op, "workspace");
    const auto p = kda::validate_inputs(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, out,
                                        execution.device());
    if (p.batch != 1) throw std::invalid_argument("kimi_delta_attention: full window requires B=1");
    for (const auto* state : {&state_in, static_cast<const Tensor*>(&state_out)}) {
        common::operand::tensor(*state, DType::FP32, {128, 128, p.value_heads}, 16, op, "State",
                                execution.device());
        for (const auto* t :
             {&q, &k, &v, &g, &beta, &a_log, &dt_bias, static_cast<const Tensor*>(&out)})
            common::operand::disjoint(common::operand::range(*state, op),
                                      common::operand::range(*t, op), op);
    }
    if (state_in.data != state_out.data)
        common::operand::disjoint(common::operand::range(state_in, op),
                                  common::operand::range(state_out, op), op);
    const auto args = arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                                static_cast<const float*>(state_in.data),
                                static_cast<float*>(state_out.data), out);
    const auto plan =
        kda::select_recurrence_plan(p.value_heads, p.tokens, execution.multiprocessor_count());
    if (plan.route == kda::RecurrenceRoute::Recurrent) {
        kda::launch_recurrent(args, execution.stream());
        return;
    }
    auto scope    = workspace.scope();
    auto* packets = static_cast<kda::Chunk*>(workspace.alloc_bytes(plan.workspace_bytes).data);
    for (const auto* tensor :
         {&q, &k, &v, &g, &beta, &a_log, &dt_bias, &state_in,
          static_cast<const Tensor*>(&state_out), static_cast<const Tensor*>(&out)})
        common::operand::disjoint(common::operand::range(packets, plan.workspace_bytes, op),
                                  common::operand::range(*tensor, op), op);
    kda::launch_prepare(args, packets, execution.stream());
    kda::launch_chunk_recurrence(args, packets, plan.value_tile, execution.stream());
}

void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       ConstStateFieldView source, StateFieldView destination,
                                       Tensor& out, CudaExecutionView execution) {
    const auto p = kda::validate_inputs(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, out,
                                        execution.device());
    if (p.tokens != 1)
        throw std::invalid_argument("kimi_delta_attention: batch update requires T=1");
    kda::validate_state(source, p, execution.device());
    kda::validate_state(destination, p, execution.device());
    for (const auto* metadata : {&source.records.byte_offsets, &destination.records.byte_offsets})
        common::operand::disjoint(common::operand::range(out, op),
                                  common::operand::range(*metadata, op), op);
    kda::launch_batch_update(
        arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                  common::operand::state_field_base<const float>(source),
                  common::operand::state_field_base<float>(destination), out),
        static_cast<const std::int64_t*>(source.records.byte_offsets.data),
        static_cast<const std::int64_t*>(destination.records.byte_offsets.data), p.batch,
        execution.stream());
}

void kimi_delta_attention_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& g, const Tensor& beta, const Tensor& a_log,
                                        const Tensor& dt_bias, float lower_bound, float scale,
                                        ConstStateFieldView source, const Tensor& valid,
                                        KdaRecurrenceRecords records, Tensor& out,
                                        CudaExecutionView execution) {
    const auto p = kda::validate_inputs(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale, out,
                                        execution.device());
    kda::validate_state(source, p, execution.device());
    if (valid.data)
        common::operand::tensor(valid, DType::I32, {p.batch}, 4, op, "valid columns",
                                execution.device());
    common::operand::tensor(records.key, DType::BF16, {128, p.qk_heads, p.tokens, p.batch}, 16, op,
                            "key record", execution.device());
    common::operand::tensor(records.value, DType::BF16, {128, p.value_heads, p.tokens, p.batch}, 16,
                            op, "value record", execution.device());
    common::operand::tensor(records.log_decay, DType::FP32, {128, p.value_heads, p.tokens, p.batch},
                            16, op, "decay record", execution.device());
    common::operand::tensor(records.beta, DType::FP32, {p.value_heads, p.tokens, p.batch}, 4, op,
                            "beta record", execution.device());
    const Tensor* outputs[] = {&records.key, &records.value, &records.log_decay, &records.beta,
                               &out};
    for (std::size_t i = 0; i < std::size(outputs); ++i) {
        for (std::size_t j = 0; j < i; ++j)
            common::operand::disjoint(common::operand::range(*outputs[i], op),
                                      common::operand::range(*outputs[j], op), op);
        for (const auto* input :
             {&q, &k, &v, &g, &beta, &a_log, &dt_bias,
              static_cast<const Tensor*>(&source.records.byte_offsets), &valid}) {
            if (!input->data ||
                (outputs[i] == &records.key && input == &k && records.key.data == k.data) ||
                (outputs[i] == &records.value && input == &v && records.value.data == v.data))
                continue;
            common::operand::disjoint(common::operand::range(*outputs[i], op),
                                      common::operand::range(*input, op), op);
        }
    }
    kda::launch_record(
        arguments(q, k, v, g, beta, a_log, dt_bias, lower_bound, scale,
                  common::operand::state_field_base<const float>(source), nullptr, out),
        static_cast<const std::int64_t*>(source.records.byte_offsets.data),
        static_cast<const std::int32_t*>(valid.data),
        {static_cast<__nv_bfloat16*>(records.key.data),
         static_cast<__nv_bfloat16*>(records.value.data),
         static_cast<float*>(records.log_decay.data), static_cast<float*>(records.beta.data)},
        p.batch, execution.stream());
}
} // namespace ninfer::ops
