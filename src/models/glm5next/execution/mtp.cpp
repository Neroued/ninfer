#include "models/glm5next/execution/mtp.h"
#include "distributed/communication.h"
#include "ninfer/ops/embedding.h"
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

ops::ConstKpoolTailStateView slice(ops::ConstKpoolTailStateView tail, int begin, int rows) {
    tail.raw_keys.records.byte_offsets  = tail.raw_keys.records.byte_offsets.slice(0, begin, rows);
    tail.raw_gates.records.byte_offsets = tail.raw_gates.records.byte_offsets.slice(0, begin, rows);
    return tail;
}

ops::KpoolTailStateView slice(ops::KpoolTailStateView tail, int begin, int rows) {
    tail.raw_keys.records.byte_offsets  = tail.raw_keys.records.byte_offsets.slice(0, begin, rows);
    tail.raw_gates.records.byte_offsets = tail.raw_gates.records.byte_offsets.slice(0, begin, rows);
    return tail;
}
} // namespace

MtpRun::MtpRun(const Parameters& p, MtpExecution execution, BatchView batch, CudaExecutionView cuda)
    : parameters_(p), execution_(execution), batch_(batch), cuda_(cuda),
      scope_(execution.workspace.scope()),
      roots_(workspace::mtp_roots(execution.workspace, p.profile->config.text,
                                  batch.tokens * batch.rows)) {}

void MtpRun::stem(const Tensor& conditioning) {
    prepare_stem(parameters_, batch_, conditioning, roots_.residual, roots_.input,
                 execution_.workspace, cuda_);
}

void MtpRun::local_attention(MlaSelection selection) {
    const auto& p    = *parameters_.profile;
    const auto local = batch_.owners[p.local.rank];
    if (!local.size) return;
    const int t = batch_.tokens, begin = local.begin, rows = local.size, h = p.config.text.hidden;
    auto output = roots_.output.view({h, t, batch_.rows}).slice(2, begin, rows).view({h, t * rows});
    mla(roots_.input.view({h, t, batch_.rows}).slice(2, begin, rows).view({h, t * rows}),
        p.config.text, parameters_.mtp->attention,
        {batch_.positions.slice(1, begin, rows), batch_.valid_columns.slice(0, begin, rows),
         batch_.table_rows.slice(0, begin, rows)},
        slice(execution_.before, begin, rows), slice(execution_.after, begin, rows),
        execution_.cache.layer(0), {}, batch_.selection_rows.slice(1, begin, rows),
        batch_.max_visible_tokens, output, execution_.workspace, cuda_, selection);
}

void MtpRun::finish_attention() {
    const auto& p = *parameters_.profile;
    if (execution_.parallel) {
        const auto local = batch_.owners[p.local.rank];
        const auto bytes =
            std::size_t(p.config.text.hidden) * batch_.tokens * sizeof(std::uint16_t);
        const auto memory = MemorySpan{roots_.output.data, roots_.output.bytes(), cuda_.device()};
        std::vector<distributed::PeerTransfer> transfers;
        for (std::uint32_t rank = 0; rank < p.local.size; ++rank) {
            if (rank == p.local.rank) continue;
            const auto remote = batch_.owners[rank];
            transfers.push_back({rank, memory.subspan(bytes * local.begin, bytes * local.size),
                                 memory.subspan(bytes * remote.begin, bytes * remote.size)});
        }
        execution_.parallel->exchange(transfers, cuda_);
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
    const auto local = batch.owners[p.profile->local.rank];
    if (!local.size) return;
    const int h = p.profile->config.text.hidden, t = batch.tokens, begin = local.begin,
              rows        = local.size;
    auto scope            = e.workspace.scope();
    auto residual         = e.workspace.alloc(DType::BF16, {h, t * rows});
    auto normalized       = e.workspace.alloc(DType::BF16, {h, t * rows});
    auto local_batch      = batch;
    local_batch.rows      = rows;
    local_batch.ids       = batch.ids.slice(1, begin, rows);
    local_batch.positions = batch.positions.slice(1, begin, rows);
    const auto hidden =
        conditioning.view({h, t, batch.rows}).slice(2, begin, rows).view({h, t * rows});
    prepare_stem(p, local_batch, hidden, residual, normalized, e.workspace, cuda);
    mla_history(normalized.view({h, t, rows}), p.mtp->attention,
                {local_batch.positions, batch.valid_columns.slice(0, begin, rows),
                 batch.table_rows.slice(0, begin, rows)},
                slice(e.before, begin, rows), slice(e.after, begin, rows), e.cache.layer(0),
                e.workspace, cuda);
}
} // namespace ninfer::models::glm5next::execution
