#include "models/qwen4_exp/execution/ple.h"
#include "models/qwen4_exp/execution/workspace.h"

namespace ninfer::models::qwen4_exp::execution {

void ple(const Tensor& embedding, const TextConfig& c, const PleParameters& p,
         const BatchView& batch, const ops::PleConvStateView& before,
         std::optional<ops::PleConvStateView> after, ops::PleConvRecordWindowView records,
         const HcParameters& next, Tensor& residual, Tensor& normalized, MemoryArena& arena,
         CudaExecutionView cuda) {
    const auto h = dimension(c.hidden), streams = dimension(c.hc.streams);
    const auto channels = dimension(c.residual_width());
    auto scope          = arena.scope();
    const auto t = batch.tokens, b = batch.batch, m = t * b;
    auto buffers = workspace::ple(arena, c, m, !records.values.data);
    if (!records.values.data)
        records = {buffers.records.view({channels, t, b}), batch.record_offsets};
    {
        auto scratch = arena.scope();
        ops::linear(embedding.view({dimension(c.ple.embedding), m}), p.projection.weight,
                    buffers.projection, p.projection.policy, arena, cuda);
    }
    const auto parent = buffers.projection.view({dimension(c.residual_width() + c.hidden), t, b});
    Tensor key(parent.data, parent.device, DType::BF16, {h, streams, t, b});
    key.nb[2]            = parent.nb[1];
    key.nb[3]            = parent.nb[2];
    const auto value     = parent.slice(0, channels, h);
    auto residual_view   = residual.view({h, streams, t, b});
    auto normalized_view = normalized.view({h, streams, t, b});
    auto gated           = buffers.gated.view({h, streams, t, b});
    ops::ple_gate({key, value}, residual_view, p.gate, {batch.valid_columns}, records, gated, cuda);
    ops::ple_conv_add_norm(residual_view, gated, p.convolution, {batch.valid_columns}, before,
                           records, next.norm, p.gate.rms_epsilon, residual_view, normalized_view,
                           cuda);
    if (after)
        ops::ple_state_extract(before, {records.values, batch.valid_columns}, batch.valid_columns,
                               *after, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
