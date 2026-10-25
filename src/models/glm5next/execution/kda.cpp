#include "models/glm5next/execution/kda.h"

#include "models/glm5next/execution/workspace.h"
#include "ninfer/ops/kda_input_proj.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/rmsnorm.h"

#include <cmath>

namespace ninfer::models::glm5next::execution {
void kda(const Tensor& input, const ModelProfile& profile, const KdaParameters& p,
         const BatchView& batch, const LinearAttentionStateView& states, std::uint32_t layer,
         const StateUpdate& update, Tensor& output, MemoryArena& arena, CudaExecutionView cuda) {
    const auto& config = profile.config.text;
    const auto& c      = config.kda;
    const int t = batch.tokens, rows = batch.rows, m = t * rows;
    const int heads = dimension(profile.local.kda_heads.size), d = dimension(c.head_dim),
              width    = heads * d;
    const auto* record = std::get_if<RecordWindow>(&update);
    auto scope         = arena.scope();
    auto b             = workspace::kda(arena, profile, m, record != nullptr);
    auto before        = states.layer_view(layer, batch.source_offsets);
    auto after         = states.layer_view(layer, batch.destination_offsets);
    auto q             = b.query.view({d, heads, t, rows});
    Tensor key, value;
    std::optional<KdaReplayRecordLayer> records;
    if (record) {
        if (record->kda.spec.width != t)
            throw std::logic_error("KDA record width differs from physical forward");
        records = record->kda.layer(layer, rows);
        key     = records->key.view({d, heads, t, rows});
        value   = records->value.view({d, heads, t, rows});
    } else {
        key   = b.key.view({d, heads, t, rows});
        value = b.value.view({d, heads, t, rows});
    }
    const ops::KdaConvOutputs outputs{q.view({width, t, rows}), key.view({width, t, rows}),
                                      value.view({width, t, rows})};
    if (records)
        ops::kda_input_proj_conv_record(input.view({input.ne[0], t, rows}), p.qkv, p.convolution,
                                        {before.conv}, batch.valid_columns, outputs, records->conv,
                                        arena, cuda);
    else
        ops::kda_input_proj_conv_update(input.view({input.ne[0], t, rows}), p.qkv, p.convolution,
                                        {before.conv, after.conv}, batch.valid_columns, outputs,
                                        arena, cuda);
    const std::array controls{b.beta, b.forget_lowrank, b.gate_lowrank};
    ops::linear_split(input, p.control, controls, arena, cuda);
    ops::linear(b.forget_lowrank, p.forget.weight, b.forget, p.forget.policy, arena, cuda);
    ops::linear(b.gate_lowrank, p.gate.weight, b.gate, p.gate.policy, arena, cuda);
    auto forget       = b.forget.view({d, heads, t, rows});
    auto beta         = b.beta.view({heads, t, rows});
    auto recurrent    = b.output.view({d, heads, t, rows});
    const float scale = static_cast<float>(1.0 / std::sqrt(double(c.head_dim)));
    if (const auto* sequence = std::get_if<SequenceUpdate>(&update)) {
        auto source      = states.recurrent_tensor(layer, sequence->source);
        auto destination = states.recurrent_tensor(layer, sequence->destination);
        ops::kimi_delta_attention(q.view({d, heads, t}), key.view({d, heads, t}),
                                  value.view({d, heads, t}), forget.view({d, heads, t}),
                                  beta.view({heads, t}), p.a_log, p.dt_bias, c.gate_lower_bound,
                                  scale, arena, source, destination, recurrent, cuda);
    } else if (records) {
        ops::kimi_delta_attention_replay_record(
            q, key, value, forget, beta, p.a_log, p.dt_bias, c.gate_lower_bound, scale,
            before.recurrent, batch.valid_columns,
            {key, value, records->log_decay.view({d, heads, t, rows}),
             records->beta.view({heads, t, rows})},
            recurrent, cuda);
    } else {
        ops::kimi_delta_attention_batch_update(q, key, value, forget, beta, p.a_log, p.dt_bias,
                                               c.gate_lower_bound, scale, before.recurrent,
                                               after.recurrent, recurrent, cuda);
    }
    auto normalized = b.normalized.view({d, heads, m});
    ops::sigmoid_gated_rmsnorm(recurrent.view({d, heads, m}), p.norm, b.gate.view({d, heads, m}),
                               config.norm_epsilon, normalized, cuda);
    ops::linear(b.normalized, p.output.weight, output, p.output.policy, arena, cuda);
}
} // namespace ninfer::models::glm5next::execution
