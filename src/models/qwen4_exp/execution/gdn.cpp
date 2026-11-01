#include "models/qwen4_exp/execution/gdn.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/rmsnorm.h"

#include <cmath>

namespace ninfer::models::qwen4_exp::execution {

void gdn(const Tensor& input, const GdnConfig& c, const GdnParameters& p, const BatchView& batch,
         const LinearAttentionStateView& states, std::uint32_t layer, const StateUpdate& update,
         float epsilon, Tensor& output, MemoryArena& arena, CudaExecutionView cuda) {
    const auto kd = dimension(c.key_dim), vd = dimension(c.value_dim);
    const auto kh = dimension(c.key_heads), vh = dimension(c.value_heads);
    const auto kw = dimension(c.key_width()), vw = dimension(c.value_width());
    const float scale    = static_cast<float>(1.0 / std::sqrt(double(c.key_dim)));
    auto scope           = arena.scope();
    const auto m         = batch.tokens * batch.batch;
    const auto* sequence = std::get_if<SequenceState>(&update);
    auto b               = workspace::gdn(arena, c, m, sequence != nullptr);
    {
        auto scratch = arena.scope();
        ops::gdn_control_proj(input, p.control, p.a_log, p.dt_bias, {b.g, b.beta}, arena, cuda);
    }
    if (sequence) {
        {
            auto scratch = arena.scope();
            const std::array outputs{b.qkv, b.z};
            ops::linear_split(input, p.input, outputs, arena, cuda);
        }
        auto before = states.conv_tensor(layer, sequence->source);
        auto after  = states.conv_tensor(layer, sequence->destination);
        auto q      = b.query.view({kw, m});
        auto k      = b.key.view({kw, m});
        auto v      = b.value.view({vw, m});
        ops::causal_conv1d_silu_split(b.qkv, p.convolution, before, after, q, k, v, cuda);
        auto initial   = states.recurrent_tensor(layer, sequence->source);
        auto final     = states.recurrent_tensor(layer, sequence->destination);
        auto recurrent = b.recurrent.view({vd, vh, m});
        auto scratch   = arena.scope();
        ops::gated_delta_net(b.query, b.key, b.value, b.g, b.beta, scale, true, arena, initial,
                             final, recurrent, cuda);
    } else if (const auto* record = std::get_if<VerifyRecords>(&update)) {
        auto state         = states.layer_view(layer, batch.source_offsets);
        const auto records = record->gdn.layer(layer, batch.batch);
        const auto t       = batch.tokens;
        auto q             = b.query.view({kd, kh, t, batch.batch});
        auto k             = b.key.view({kd, kh, t, batch.batch});
        auto v             = b.value.view({vd, vh, t, batch.batch});
        auto z             = b.z.view({vw, t, batch.batch});
        {
            auto scratch = arena.scope();
            auto conv    = records.conv;
            ops::gdn_input_proj_conv_record(input.view({input.ne[0], t, batch.batch}), p.input,
                                            p.convolution, state.conv, batch.valid_columns,
                                            {q.view({kw, t, batch.batch}),
                                             k.view({kw, t, batch.batch}),
                                             v.view({vw, t, batch.batch}), z},
                                            conv, arena, cuda);
        }
        auto recurrent = b.recurrent.view({vd, vh, t, batch.batch});
        ops::gated_delta_net_replay_record(
            q, k, v, b.g.view({vh, t, batch.batch}), b.beta.view({vh, t, batch.batch}), scale,
            state.recurrent, batch.valid_columns, {records.key, records.value, records.gate},
            recurrent, cuda);
    } else {
        auto before = states.layer_view(layer, batch.source_offsets);
        auto after  = states.layer_view(layer, batch.destination_offsets);
        auto q      = b.query.view({kd, kh, 1, batch.batch});
        auto k      = b.key.view({kd, kh, 1, batch.batch});
        auto v      = b.value.view({vd, vh, 1, batch.batch});
        auto z      = b.z.view({vw, 1, batch.batch});
        {
            auto scratch = arena.scope();
            ops::gdn_input_proj_conv_update(
                input.view({input.ne[0], 1, batch.batch}), p.input, p.convolution, before.conv,
                after.conv, batch.valid_columns,
                {q.view({kw, 1, batch.batch}), k.view({kw, 1, batch.batch}),
                 v.view({vw, 1, batch.batch}), z},
                arena, cuda);
        }
        auto recurrent = b.recurrent.view({vd, vh, 1, batch.batch});
        ops::gated_delta_net_batch_update(q, k, v, b.g.view({vh, 1, batch.batch}),
                                          b.beta.view({vh, 1, batch.batch}), scale, true,
                                          before.recurrent, after.recurrent, recurrent, cuda);
    }
    auto normalized = b.normalized.view({vd, vh, m});
    ops::sigmoid_gated_rmsnorm(b.recurrent.view({vd, vh, m}), p.norm, b.z.view({vd, vh, m}),
                               epsilon, normalized, cuda);
    ops::linear(b.normalized, p.output.weight, output, p.output.policy, arena, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
