#include "models/glm5next/execution/mtp.h"

#include "ninfer/ops/embedding.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/paired_rmsnorm_concat.h"
#include "ninfer/ops/residual_add_rmsnorm.h"
#include "ninfer/ops/rmsnorm.h"

namespace ninfer::models::glm5next::execution {
namespace {
void prepare_stem(const Parameters& p, const BatchView& batch, const Tensor& hidden,
                  Tensor& residual, Tensor& normalized, MemoryArena& arena,
                  CudaExecutionView cuda) {
    const auto& c = p.profile->config.text;
    const auto& m = *p.mtp;
    auto scope    = arena.scope();
    auto b        = workspace::mtp_stem(arena, c, batch.tokens * batch.rows);
    ops::embedding(batch.ids.view({batch.tokens * batch.rows}), m.embedding, b.embedding, cuda);
    ops::paired_rmsnorm_concat(b.embedding, hidden, m.embedding_norm, m.hidden_norm,
                               batch.positions.view({batch.tokens * batch.rows}), c.norm_epsilon,
                               b.normalized, cuda);
    ops::linear(b.normalized, m.stem.weight, residual, m.stem.policy, arena, cuda);
    ops::rmsnorm(residual, m.input_norm, c.norm_epsilon, false, normalized, cuda);
}

} // namespace

MtpRun::MtpRun(const Parameters& p, MtpExecution execution, BatchView batch, CudaExecutionView cuda)
    : parameters_(p), execution_(execution), batch_(batch), cuda_(cuda),
      scope_(execution.workspace.scope()),
      roots_(workspace::mtp_roots(execution.workspace, p.profile->config.text,
                                  batch.tokens * batch.rows)) {
    if (execution.parallel)
        attention_ =
            workspace::attention(execution.workspace, dimension(p.profile->config.text.hidden),
                                 batch.tokens * batch.attention.rows, p.profile->local.size);
}

void MtpRun::stem(const Tensor& conditioning) {
    prepare_stem(parameters_, batch_, conditioning, roots_.residual, roots_.input,
                 execution_.workspace, cuda_);
}

MlaRun MtpRun::local_mla(MlaSelection selection) {
    const auto& p     = *parameters_.profile;
    const auto& local = batch_.attention;
    return MlaRun(
        attention_.input.data ? attention_.input : roots_.input, p.config.text,
        parameters_.mtp->attention, {local.positions, local.valid_columns, local.table_rows},
        execution_.before, execution_.after, execution_.cache.layer(0), {}, local.selection_rows,
        local.max_visible_tokens, attention_.output.data ? attention_.output : roots_.output,
        execution_.workspace, cuda_, selection);
}

void MtpRun::local_attention(MlaSelection selection, const MlaSubmit& submit) {
    if (attention_.input.data) {
        const int h = parameters_.profile->config.text.hidden, t = batch_.tokens;
        auto input = attention_.input.view({h, t, batch_.attention.rows});
        ops::gather_batch_rows(roots_.input.view({h, t, batch_.rows}), batch_.attention.to_global,
                               input, cuda_);
    }
    auto run = local_mla(selection);
    run.execute(submit);
}

void MtpRun::finish_attention() {
    const auto& p = *parameters_.profile;
    if (execution_.parallel) {
        execution_.parallel->gather(attention_.output, attention_.gathered, cuda_);
        const int h = p.config.text.hidden, t = batch_.tokens;
        auto output = roots_.output.view({h, t, batch_.rows});
        ops::gather_batch_rows(
            attention_.gathered.view({h, t, batch_.attention.rows * int(p.local.size)}),
            batch_.attention.from_gather, output, cuda_);
    }
    ops::residual_add_rmsnorm(roots_.residual, roots_.output, parameters_.mtp->post_attention_norm,
                              p.config.text.norm_epsilon, roots_.residual, roots_.input, cuda_);
}

void MtpRun::ffn() {
    const auto& m = parameters_.mtp->moe;
    ops::topk_sigmoid_swiglu_moe(roots_.input, parameters_.profile->mtp->moe.math, m.weights,
                                 m.subset, roots_.output, execution_.workspace, cuda_);
}

void MtpRun::finish(Tensor& hidden) {
    if (execution_.parallel) execution_.parallel->sum(roots_.output, cuda_);
    ops::residual_add_rmsnorm(roots_.residual, roots_.output, parameters_.mtp->final_norm,
                              parameters_.profile->config.text.norm_epsilon, roots_.residual,
                              hidden, cuda_);
}

void mtp_history(const Parameters& p, MtpExecution e, const BatchView& batch,
                 const Tensor& conditioning, CudaExecutionView cuda) {
    if (batch.sequence_owner && *batch.sequence_owner != p.profile->local.rank) return;
    const auto& local = batch.attention;
    const int h = p.profile->config.text.hidden, t = batch.tokens, rows = local.rows;
    auto scope      = e.workspace.scope();
    auto residual   = e.workspace.alloc(DType::BF16, {h, t * rows});
    auto normalized = e.workspace.alloc(DType::BF16, {h, t * rows});
    prepare_stem(p, batch, conditioning, residual, normalized, e.workspace, cuda);
    mla_history(normalized.view({h, t, rows}), p.mtp->attention,
                {local.positions, local.valid_columns, local.table_rows}, e.before, e.after,
                e.cache.layer(0), e.workspace, cuda);
}
} // namespace ninfer::models::glm5next::execution
