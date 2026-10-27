#include "models/glm5next/execution/mla.h"

#include "ninfer/ops/linear_split.h"

#include <algorithm>
#include <cmath>

namespace ninfer::models::glm5next::execution {
namespace {
ops::KpoolIndexerProfile index_geometry(const MlaReadout& r) {
    return {r.index_query.ne[1], r.index_query.ne[0]};
}

ops::KpoolSparseMlaProfile attention_geometry(const MlaReadout& r) {
    return {r.query.ne[1], r.query.ne[0], r.output.ne[0], r.factors.key_up.ne[0]};
}
} // namespace

ops::KpoolGraphProfile MlaReadout::index_profile(std::uint32_t bound, int sms) const {
    return ops::kpool_indexer_graph_profile(index_geometry(*this), cache.index.format, {bound},
                                            query.ne[3], query.ne[2], sms);
}

ops::KpoolGraphProfile MlaReadout::attention_profile(std::uint32_t bound, int sms) const {
    return ops::kpool_sparse_mla_graph_profile(attention_geometry(*this), cache.latent.storage,
                                               {bound}, query.ne[3], query.ne[2], sms);
}

std::size_t MlaReadout::scratch_bytes(int sms) const {
    const int t = query.ne[2], b = query.ne[3];
    const auto attention = ops::kpool_sparse_mla_workspace_capacity_bytes(
        attention_geometry(*this), cache.latent.storage, {visible}, b, t, t, sms);
    return index ? std::max(attention,
                            ops::kpool_indexer_workspace_capacity_bytes(
                                index_geometry(*this), cache.index.format, {visible}, b, t, t, sms))
                 : attention;
}

void MlaReadout::indexer(std::uint32_t bound, CudaExecutionView cuda) const {
    MemoryArena workspace(scratch);
    auto selected = selection;
    ops::kpool_indexer(index_query, coefficients, cache.index, batch, {bound}, workspace, selected,
                       cuda);
}

void MlaReadout::attention(std::uint32_t bound, CudaExecutionView cuda) const {
    MemoryArena workspace(scratch);
    auto out = output;
    ops::kpool_sparse_mla(query, factors, cache.latent, batch, selection, selection_rows, scale,
                          {bound}, workspace, out, cuda);
}

MlaRun::MlaRun(const Tensor& input, const TextConfig& c, const MlaParameters& p,
               ops::KpoolMlaBatchView batch, ops::ConstKpoolTailStateView before,
               std::optional<ops::KpoolTailStateView> after, ops::KpoolMlaCacheView cache,
               ops::KpoolRawRecordWriteView records, const Tensor& selection_rows,
               std::uint32_t visible, Tensor output, MemoryArena& arena, CudaExecutionView cuda,
               std::optional<MlaSelection> selection)
    : config_(c), parameters_(p), input_(input), output_(output), selection_rows_(selection_rows),
      batch_(batch), before_(before), after_(after), cache_(cache), records_(records),
      visible_(visible), arena_(arena), cuda_(cuda), scope_(arena.scope()),
      buffers_(workspace::mla(arena, c, batch.positions.ne[0] * batch.positions.ne[1],
                              selection && selection->reuse, !selection)),
      selection_(selection ? selection->value : buffers_.selection),
      reuse_(selection && selection->reuse) {}

void MlaRun::prepare() {
    const auto& m = config_.mla;
    const int t = batch_.positions.ne[0], b = batch_.positions.ne[1];
    const auto& p = parameters_.prepare[static_cast<std::size_t>(
        reuse_ ? ops::KpoolMlaPrepareProjection::QueryAndHistory
               : ops::KpoolMlaPrepareProjection::Full)];
    const ops::KpoolMlaPrepareOutputs out{
        buffers_.query_lowrank.view({dimension(m.query_lowrank), t, b}),
        reuse_ ? Tensor{} : buffers_.coefficients.view({dimension(m.index_heads), t, b})};
    const auto x = input_.view({dimension(config_.hidden), t, b});
    if (after_)
        ops::kpool_mla_prepare_update(x, p, batch_, before_, cache_, *after_, out, arena_, cuda_);
    else
        ops::kpool_mla_prepare_record(x, p, batch_, before_, cache_, records_, out, arena_, cuda_);
    if (reuse_)
        ops::linear(buffers_.query_lowrank, parameters_.query.weight, buffers_.query,
                    parameters_.query.policy, arena_, cuda_);
    else {
        const std::array outputs{buffers_.query, buffers_.index_query};
        ops::linear_split(buffers_.query_lowrank, parameters_.query_index, outputs, arena_, cuda_);
    }
}

MlaReadout MlaRun::readout() {
    const auto& m = config_.mla;
    const int t = batch_.positions.ne[0], b = batch_.positions.ne[1];
    MlaReadout r{buffers_.query.view({dimension(m.query_dim), dimension(m.heads), t, b}),
                 reuse_ ? Tensor{}
                        : buffers_.index_query.view(
                              {dimension(m.index_dim), dimension(m.index_heads), t, b}),
                 reuse_ ? Tensor{} : buffers_.coefficients.view({dimension(m.index_heads), t, b}),
                 buffers_.output.view({dimension(m.value_dim), dimension(m.heads), t, b}),
                 selection_rows_,
                 {parameters_.key_up, parameters_.value_up},
                 cache_,
                 batch_,
                 selection_,
                 {},
                 float(1.0 / std::sqrt(double(m.query_dim))),
                 visible_,
                 !reuse_};
    auto scope       = arena_.scope();
    const auto bytes = r.scratch_bytes(cuda_.multiprocessor_count());
    // A captured readout may later select a larger envelope inside the already planned
    // backing. Reserve today's high-water use while retaining the available bounded span.
    if (bytes) {
        const auto allocation = arena_.alloc_bytes(bytes, 256);
        const auto offset     = static_cast<std::byte*>(allocation.data) -
                            static_cast<std::byte*>(arena_.storage().data);
        r.scratch = arena_.storage().subspan(offset);
    } else {
        const auto offset = (arena_.used() + 255) & ~std::size_t(255);
        r.scratch         = arena_.storage().subspan(std::min(offset, arena_.capacity()));
    }
    return r;
}

void MlaRun::execute(const MlaSubmit& submit) {
    prepare();
    const auto call = readout();
    if (submit)
        submit(call);
    else {
        if (call.index) call.indexer(visible_, cuda_);
        call.attention(visible_, cuda_);
    }
    finish();
}

void MlaRun::finish() {
    ops::linear(buffers_.output, parameters_.output.weight, output_, parameters_.output.policy,
                arena_, cuda_);
}

void mla_history(const Tensor& input, const MlaParameters& p, const ops::KpoolMlaBatchView& batch,
                 ops::ConstKpoolTailStateView before, ops::KpoolTailStateView after,
                 ops::KpoolMlaCacheView cache, MemoryArena& arena, CudaExecutionView cuda) {
    ops::kpool_mla_prepare_update(
        input, p.prepare[static_cast<std::size_t>(ops::KpoolMlaPrepareProjection::History)], batch,
        before, cache, after, {}, arena, cuda);
}
} // namespace ninfer::models::glm5next::execution
