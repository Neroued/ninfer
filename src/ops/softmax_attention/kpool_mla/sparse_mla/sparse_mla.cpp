#include "ops/softmax_attention/kpool_mla/sparse_mla/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

#include <cmath>

namespace ninfer::ops {
using namespace detail::kpool;

KpoolGraphProfile kpool_sparse_mla_graph_profile(const KpoolSparseMlaProfile& profile,
                                                 KpoolMlaStorage storage,
                                                 KpoolExecutionEnvelope envelope, int batch,
                                                 int tokens, int sms) {
    sparse_mla_geometry(profile, storage, envelope, batch, tokens, tokens, sms);
    KpoolGraphProfile key;
    key.words_        = {std::uint64_t(tokens), std::uint64_t(batch), std::uint64_t(sms),
                         std::uint64_t(storage)};
    std::size_t at    = 4;
    const auto append = [&](int columns) {
        if (!columns) return;
        const auto p = sparse_mla_plan(columns, batch, envelope, sms);
        for (const auto word :
             {std::uint64_t(p.factor), std::uint64_t(p.attention), std::uint64_t(p.splits),
              std::uint64_t(p.tile), std::uint64_t(p.direct), std::uint64_t(p.merge_value),
              std::uint64_t(p.bytes)})
            key.words_[at++] = word;
    };
    const auto columns = std::int64_t(tokens) * batch;
    if (columns >= kMlaWindow) append(kMlaWindow);
    append(columns % kMlaWindow);
    return key;
}

std::size_t kpool_sparse_mla_workspace_capacity_bytes(const KpoolSparseMlaProfile& profile,
                                                      KpoolMlaStorage storage,
                                                      KpoolExecutionEnvelope envelope,
                                                      std::int32_t batch, std::int32_t lo,
                                                      std::int32_t hi, std::int32_t sms) {
    sparse_mla_geometry(profile, storage, envelope, batch, lo, hi, sms);
    return sparse_mla_capacity(lo, hi, batch, envelope, sms);
}

void kpool_sparse_mla(const Tensor& query, const KpoolMlaFactors& factors,
                      const ConstKpoolMlaLatentCacheView& cache, const KpoolMlaBatchView& batch,
                      const KpoolBlockSelection& selected, const Tensor& selection_rows,
                      float attention_scale, KpoolExecutionEnvelope envelope,
                      MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    const int T = query.ne[2], B = query.ne[3], sms = execution.multiprocessor_count();
    sparse_mla_geometry({query.ne[1], query.ne[0], out.ne[0], factors.key_up.ne[0]}, cache.storage,
                        envelope, B, T, T, sms);
    require(std::isfinite(attention_scale) && attention_scale > 0, "Sparse MLA scale");
    const auto device = execution.device();
    std::array<operand::Range, 10> inputs;
    int n            = 0;
    const auto input = [&](const Tensor& t, DType type, std::initializer_list<int> shape,
                           int alignment, const char* name) {
        operand::tensor(t, type, shape, alignment, kOperation, name, device);
        inputs[n++] = operand::range(t, kOperation);
    };
    input(query, DType::BF16, {256, 64, T, B}, 16, "MLA query");
    input(factors.key_up, DType::BF16, {512, 256, 64}, 16, "U_K");
    input(factors.value_up, DType::BF16, {512, 256, 64}, 16, "U_V");
    input(batch.positions, DType::I32, {T, B}, 4, "positions");
    input(batch.valid_columns, DType::I32, {B}, 4, "valid columns");
    input(batch.table_rows, DType::I32, {B}, 4, "table rows");
    input(selection_rows, DType::I32, {T, B}, 4, "selection rows");
    const int selected_rows = selected.pool_ids.ne[1];
    input(selected.pool_ids, DType::I32, {512, selected_rows}, 4, "pool IDs");
    input(selected.pool_counts, DType::I32, {selected_rows}, 4, "pool counts");
    input(selected.selection_end, DType::I32, {selected_rows}, 4, "selection end");
    plane(cache.data, 512, 1, device);
    require(!cache.scales.records.storage.data, "BF16 latent has no scale plane");
    const auto& table = cache.data.records.page_offsets;
    require(std::uint64_t(table.ne[0]) * 64 >= envelope.max_visible_tokens, "page table capacity");
    const auto table_blocks = operand::blocks(table, kOperation);
    operand::tensor(out, DType::BF16, {256, 64, T, B}, 16, kOperation, "MLA output", device);
    const auto output = operand::range(out, kOperation);
    for (const auto& src : inputs) operand::disjoint(output, src, kOperation);
    require(!operand::overlaps(table_blocks, output), "output overlaps page table");
    operand::device(workspace.device(), device, kOperation, "workspace");
    SparseMlaArguments a{};
    a.query          = static_cast<const __nv_bfloat16*>(query.data);
    a.key_up         = static_cast<const __nv_bfloat16*>(factors.key_up.data);
    a.value_up       = static_cast<const __nv_bfloat16*>(factors.value_up.data);
    a.latent         = static_cast<const __nv_bfloat16*>(cache.data.data());
    a.latent_bytes   = cache.data.records.storage.bytes - cache.data.plane.tensor.region.offset;
    a.valid          = static_cast<const int*>(batch.valid_columns.data);
    a.table_rows     = static_cast<const int*>(batch.table_rows.data);
    a.selection_rows = static_cast<const int*>(selection_rows.data);
    a.ids            = static_cast<const int*>(selected.pool_ids.data);
    a.counts         = static_cast<const int*>(selected.pool_counts.data);
    a.ends           = static_cast<const int*>(selected.selection_end.data);
    a.tables         = static_cast<const std::int64_t*>(table.data);
    a.table_stride   = table.nb[1] / 8;
    a.out            = static_cast<__nv_bfloat16*>(out.data);
    a.tokens         = T;
    a.batch          = B;
    a.scale          = attention_scale;
    auto scope       = workspace.scope();
    const auto bytes = sparse_mla_capacity(T, T, B, envelope, sms);
    MemorySpan scratch{};
    if (bytes) {
        scratch          = workspace.alloc_bytes(bytes, 256);
        const auto range = operand::range(scratch.data, scratch.bytes, kOperation);
        for (const auto& src : inputs) operand::disjoint(range, src, kOperation);
        operand::disjoint(range, output, kOperation);
        require(!operand::overlaps(table_blocks, range), "workspace overlaps page table");
    }
    launch_sparse_mla(a, scratch, envelope, sms, execution.stream());
}
} // namespace ninfer::ops
