#include "models/glm5next/execution/mla.h"
#include "models/glm5next/execution/workspace.h"
#include "ninfer/ops/linear_split.h"
#include <cmath>

namespace ninfer::models::glm5next::execution {
void mla(const Tensor& input, const TextConfig& c, const MlaParameters& p,
         const ops::KpoolMlaBatchView& batch, ops::ConstKpoolTailStateView before,
         std::optional<ops::KpoolTailStateView> after, ops::KpoolMlaCacheView cache,
         ops::KpoolRawRecordWriteView records, const Tensor& selection_rows, std::uint32_t visible,
         Tensor& output, MemoryArena& arena, CudaExecutionView cuda) {
    const int t = batch.positions.ne[0], rows = batch.positions.ne[1], columns = t * rows;
    const auto& m = c.mla;
    auto scope    = arena.scope();
    auto b        = workspace::mla(arena, c, columns, false);
    const ops::KpoolMlaPrepareOutputs prepared{
        b.query_lowrank.view({dimension(m.query_lowrank), t, rows}),
        b.coefficients.view({dimension(m.index_heads), t, rows})};
    const auto x = input.view({dimension(c.hidden), t, rows});
    if (after)
        ops::kpool_mla_prepare_update(x, p.prepare[0], batch, before, cache, *after, prepared,
                                      arena, cuda);
    else
        ops::kpool_mla_prepare_record(x, p.prepare[0], batch, before, cache, records, prepared,
                                      arena, cuda);
    const std::array query_outputs{b.query, b.index_query};
    ops::linear_split(b.query_lowrank, p.query_index, query_outputs, arena, cuda);
    auto index_query =
        b.index_query.view({dimension(m.index_dim), dimension(m.index_heads), t, rows});
    ops::kpool_indexer(index_query, prepared.head_coeff, cache.index, batch, {visible}, arena,
                       b.selection, cuda);
    auto heads = b.output.view({dimension(m.value_dim), dimension(m.heads), t, rows});
    ops::kpool_sparse_mla(b.query.view({dimension(m.query_dim), dimension(m.heads), t, rows}),
                          {p.key_up, p.value_up}, cache.latent, batch, b.selection, selection_rows,
                          static_cast<float>(1.0 / std::sqrt(double(m.query_dim))), {visible},
                          arena, heads, cuda);
    ops::linear(b.output, p.output.weight, output, p.output.policy, arena, cuda);
}
} // namespace ninfer::models::glm5next::execution
