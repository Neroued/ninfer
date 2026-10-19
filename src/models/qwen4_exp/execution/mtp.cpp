#include "models/qwen4_exp/execution/mtp.h"
#include "models/qwen4_exp/execution/qsa.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "core/transfer.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/rmsnorm.h"

namespace ninfer::models::qwen4_exp::execution {
namespace {
void stem(const TextConfig& c, const MtpParameters& p, const Tensor& conditioning,
          const BatchView& batch, std::optional<VisualEmbeddingView> visual,
          workspace::Roots& roots, MemoryArena& arena, CudaExecutionView cuda) {
    const auto m = batch.tokens * batch.batch;
    const auto h = dimension(c.hidden), streams = dimension(c.hc.streams);
    auto scope = arena.scope();
    auto b     = workspace::mtp_stem(arena, c, m);
    ops::embedding(batch.ids.view({m}), p.embedding, b.embedding, cuda);
    if (visual) ops::scatter(visual->features, visual->columns, b.embedding, cuda);
    ops::rmsnorm(b.embedding, p.embedding_norm, c.norm_epsilon, true, b.normalized_embedding, cuda);
    ops::rmsnorm(conditioning.view({dimension(c.residual_width()), m}), p.hidden_norm,
                 c.norm_epsilon, true, b.normalized_hidden, cuda);
    {
        auto scratch = arena.scope();
        ops::linear(b.normalized_embedding, p.fc_embedding.weight, b.projected_embedding,
                    p.fc_embedding.policy, arena, cuda);
    }
    {
        auto scratch  = arena.scope();
        auto residual = roots.residual.view({h, streams * m});
        ops::linear(b.normalized_hidden.view({h, streams * m}), p.fc_hidden.weight, residual,
                    p.fc_hidden.policy, arena, cuda);
    }
    ops::hc_combine_norm(roots.residual, b.projected_embedding, p.layer.attention_hc.norm,
                         c.norm_epsilon, roots.residual, roots.normalized, cuda);
}

void mix(const Tensor& normalized, const HcParameters& p, workspace::Roots& roots,
         MemoryArena& arena, CudaExecutionView cuda) {
    auto scratch = arena.scope();
    ops::hc_mix(normalized, p.mix, arena, roots.input, roots.injection, cuda);
}
} // namespace

void mtp_forward(const TextConfig& c, const QsaConfig& qsa_config, const MtpParameters& p,
                 const Tensor& conditioning, const BatchView& batch,
                 const ops::QsaTailStateView& before, ops::QsaTailStateView after,
                 ops::QsaCacheView cache, TextOutputs output, MemoryArena& arena,
                 CudaExecutionView cuda) {
    const auto m = batch.tokens * batch.batch;
    auto scope   = arena.scope();
    auto roots   = workspace::roots(arena, c, m);
    stem(c, p, conditioning, batch, {}, roots, arena, cuda);
    mix(roots.normalized, p.layer.attention_hc, roots, arena, cuda);
    qsa(roots.input, qsa_config, std::get<QsaParameters>(p.layer.mixer), batch, before, after,
        cache, {}, roots.output, arena, cuda);
    ops::hc_combine_norm(roots.residual, roots.output, roots.injection, p.layer.moe_hc.norm,
                         c.norm_epsilon, roots.residual, roots.normalized, cuda);
    mix(roots.normalized, p.layer.moe_hc, roots, arena, cuda);
    {
        auto scratch = arena.scope();
        moe::execute(roots.input, p.layer.moe, p.layer.moe.resident, ops::MoeEpilogue::Write,
                     roots.output, arena, cuda);
    }
    ops::hc_combine_norm(roots.residual, roots.output, roots.injection, p.final_hc.norm,
                         c.norm_epsilon, roots.residual, roots.normalized, cuda);
    copy_bytes_async({output.conditioning.data, roots.residual.bytes(), cuda.device()},
                     {roots.residual.data, roots.residual.bytes(), cuda.device()}, cuda);
    auto sample = output.sample.view({dimension(c.hidden), m});
    ops::hc_mix(roots.normalized, p.final_hc.mix, arena, sample, cuda);
}

void mtp_history(const TextConfig& c, const QsaConfig& qsa_config, const MtpParameters& p,
                 const Tensor& conditioning, const BatchView& batch,
                 const ops::QsaTailStateView& before, ops::QsaTailStateView after,
                 ops::QsaCacheView cache, std::optional<VisualEmbeddingView> visual,
                 MemoryArena& arena, CudaExecutionView cuda) {
    auto scope = arena.scope();
    auto roots = workspace::roots(arena, c, batch.tokens * batch.batch);
    stem(c, p, conditioning, batch, visual, roots, arena, cuda);
    mix(roots.normalized, p.layer.attention_hc, roots, arena, cuda);
    qsa_history(roots.input, qsa_config, std::get<QsaParameters>(p.layer.mixer), batch, before,
                after, cache, arena, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
