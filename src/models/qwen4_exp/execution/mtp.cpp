#include "models/qwen4_exp/execution/mtp.h"
#include "models/qwen4_exp/execution/qsa.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "core/transfer.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/rmsnorm.h"

namespace ninfer::models::qwen4_exp::execution {
namespace {
void mtp_stem(const TextConfig& c, const MtpParameters& p, const Tensor& conditioning,
              const BatchView& batch, std::optional<models::vision::EmbeddingView> visual,
              const TokenPartition& columns, Tensor& residual, Tensor& normalized,
              MemoryArena& arena, CudaExecutionView cuda) {
    const auto m = columns.count;
    const auto h = dimension(c.hidden), streams = dimension(c.hc.streams);
    auto scope = arena.scope();
    auto b     = workspace::mtp_stem(arena, c, m, columns.columns);
    ops::embedding(batch.ids.view({columns.columns}), p.embedding, b.embedding, cuda);
    if (visual)
        for (const auto& slice : visual->slices)
            ops::scatter(slice.features, slice.columns, b.embedding, cuda);
    const auto embedding = columns.local(b.embedding, 1);
    ops::rmsnorm(embedding, p.embedding_norm, c.norm_epsilon, true, b.normalized_embedding, cuda);
    ops::rmsnorm(conditioning.view({dimension(c.residual_width()), m}), p.hidden_norm,
                 c.norm_epsilon, true, b.normalized_hidden, cuda);
    {
        auto scratch = arena.scope();
        ops::linear(b.normalized_embedding, p.fc_embedding.weight, b.projected_embedding,
                    p.fc_embedding.policy, arena, cuda);
    }
    {
        auto scratch   = arena.scope();
        auto projected = residual.view({h, streams * m});
        ops::linear(b.normalized_hidden.view({h, streams * m}), p.fc_hidden.weight, projected,
                    p.fc_hidden.policy, arena, cuda);
    }
    ops::hc_combine_norm(residual, b.projected_embedding, p.layer.attention_hc.norm, c.norm_epsilon,
                         residual, normalized, cuda);
}

void mix(const Tensor& normalized, const HcParameters& p, workspace::Roots& roots,
         MemoryArena& arena, CudaExecutionView cuda) {
    auto scratch = arena.scope();
    ops::hc_mix(normalized, p.mix, arena, roots.input, roots.injection, cuda);
}
} // namespace

MtpRun::MtpRun(const TextConfig& config, const QsaConfig& qsa, const MtpParameters& parameters,
               const Tensor& conditioning, BatchView batch, ops::QsaTailStateView before,
               ops::QsaTailStateView after, ops::QsaCacheView cache, MemoryArena& arena,
               CudaExecutionView cuda, ParallelExecution* parallel)
    : config_(config), qsa_(qsa), parameters_(parameters), conditioning_(conditioning),
      batch_(batch), before_(before), after_(after), cache_(cache), arena_(arena), cuda_(cuda),
      parallel_(parallel), scope_(arena.scope()),
      roots_(workspace::roots(arena, config, batch.tokens * batch.batch)) {}

void MtpRun::stem() {
    mtp_stem(config_, parameters_, conditioning_, batch_, {},
             TokenPartition(batch_.tokens * batch_.batch), roots_.residual, roots_.normalized,
             arena_, cuda_);
}

QsaRun MtpRun::local_qsa() {
    return QsaRun(roots_.input, qsa_, std::get<QsaParameters>(parameters_.layer.mixer), batch_,
                  before_, after_, cache_, {}, roots_.output, arena_, cuda_);
}

void MtpRun::mixer(const QsaSubmit& submit) {
    mix(roots_.normalized, parameters_.layer.attention_hc, roots_, arena_, cuda_);
    local_qsa().execute(submit);
    if (parallel_) parallel_->sum(roots_.output, cuda_);
    ops::hc_combine_norm(roots_.residual, roots_.output, roots_.injection,
                         parameters_.layer.moe_hc.norm, config_.norm_epsilon, roots_.residual,
                         roots_.normalized, cuda_);
    mix(roots_.normalized, parameters_.layer.moe_hc, roots_, arena_, cuda_);
}

void MtpRun::feed_forward() {
    {
        auto scratch = arena_.scope();
        moe::execute(roots_.input, parameters_.layer.moe, parameters_.layer.moe.resident,
                     ops::MoeEpilogue::Write, roots_.output, {batch_.valid_columns, batch_.tokens},
                     arena_, cuda_);
    }
    if (parallel_) parallel_->sum(roots_.output, cuda_);
    ops::hc_combine_norm(roots_.residual, roots_.output, roots_.injection,
                         parameters_.final_hc.norm, config_.norm_epsilon, roots_.residual,
                         roots_.normalized, cuda_);
}

void MtpRun::finish(TextOutputs output) {
    copy_bytes_async({output.conditioning.data, roots_.residual.bytes(), cuda_.device()},
                     {roots_.residual.data, roots_.residual.bytes(), cuda_.device()}, cuda_);
    auto sample = output.sample.view({dimension(config_.hidden), batch_.tokens * batch_.batch});
    ops::hc_mix(roots_.normalized, parameters_.final_hc.mix, arena_, sample, cuda_);
}

void mtp_forward(const TextConfig& c, const QsaConfig& qsa_config, const MtpParameters& p,
                 const Tensor& conditioning, const BatchView& batch,
                 const ops::QsaTailStateView& before, ops::QsaTailStateView after,
                 ops::QsaCacheView cache, TextOutputs output, MemoryArena& arena,
                 CudaExecutionView cuda, ParallelExecution* parallel) {
    MtpRun run(c, qsa_config, p, conditioning, batch, before, after, cache, arena, cuda, parallel);
    run.stem();
    run.mixer();
    run.feed_forward();
    run.finish(output);
}

void mtp_history(const TextConfig& c, const QsaConfig& qsa_config, const MtpParameters& p,
                 MtpHistoryInput input, const BatchView& batch, const ops::QsaTailStateView& before,
                 ops::QsaTailStateView after, ops::QsaCacheView cache,
                 std::optional<models::vision::EmbeddingView> visual, MemoryArena& arena,
                 CudaExecutionView cuda, ParallelExecution* parallel) {
    const auto& columns = input.columns;
    if (columns.columns != batch.tokens * batch.batch ||
        (columns.split() && (!parallel || batch.batch != 1)))
        throw std::invalid_argument("Qwen4Exp MTP history partition differs from its sequence");
    auto scope = arena.scope();
    auto roots = workspace::history_roots(arena, c, columns.count, columns.padded);
    columns.clear_padding(roots.input, 1, cuda);
    mtp_stem(c, p, input.conditioning, batch, visual, columns, roots.residual, roots.normalized,
             arena, cuda);
    {
        auto scratch = arena.scope();
        auto local   = columns.local(roots.input, 1);
        ops::hc_mix(roots.normalized, p.history.mix, arena, local, cuda);
    }
    if (columns.split()) parallel->gather(columns.exchange(roots.input, 1), roots.input, cuda);
    const auto gathered = roots.input.slice(1, 0, columns.columns);
    qsa_history(gathered, qsa_config, p.history.attention, batch, before, after, cache, arena, cuda,
                input.ragged);
}

} // namespace ninfer::models::qwen4_exp::execution
