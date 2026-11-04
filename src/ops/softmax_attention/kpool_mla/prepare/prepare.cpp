#include "ops/softmax_attention/kpool_mla/prepare/arguments.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

#include <cmath>

namespace ninfer::ops {
namespace {
using namespace detail::kpool;

template <class Field>
TailField bind_field(const Field& f) {
    return {reinterpret_cast<__nv_bfloat16*>(
                const_cast<std::byte*>(static_cast<const std::byte*>(f.records.storage.data)) +
                f.field.tensor.region.offset),
            static_cast<const std::int64_t*>(f.records.byte_offsets.data),
            f.field.byte_strides[1] / 2};
}

RawField bind_raw(const Tensor& t) {
    return {static_cast<__nv_bfloat16*>(t.data), t.nb[1] / 2, t.nb[2] / 2};
}

void execute(const Tensor& x, const KpoolMlaPrepareParameters& p, const KpoolMlaBatchView& batch,
             ConstKpoolTailStateView before, KpoolMlaCacheView cache,
             KpoolRawRecordWriteView records, KpoolTailStateView after, KpoolMlaPrepareOutputs out,
             bool update, MemoryArena& arena, CudaExecutionView execution) {
    const int T = x.ne[1], B = x.ne[2];
    const auto device   = execution.device();
    const auto profile  = kpool_mla_prepare_profile(p);
    const auto geometry = prepare_geometry(profile, cache.latent.storage, cache.index.format, B, T,
                                           T, execution.multiprocessor_count());
    require(p.rms_epsilon > 0 && std::isfinite(p.rms_epsilon) && p.index_epsilon > 0 &&
                std::isfinite(p.index_epsilon),
            "norm epsilons");
    std::array<operand::Range, 24> inputs{};
    int input_count  = 0;
    const auto input = [&](const Tensor& t, DType type, std::initializer_list<int> dims,
                           std::size_t alignment, const char* name) {
        operand::tensor(t, type, dims, alignment, kOperation, name, device);
        inputs[input_count++] = operand::range(t, kOperation);
    };
    input(x, DType::BF16, {4096, T, B}, 16, "input");
    input(p.latent_norm, DType::BF16, {512}, 2, "latent norm");
    input(p.index_norm, DType::BF16, {128}, 2, "key norm");
    input(p.index_bias, DType::BF16, {128}, 2, "key bias");
    input(p.pool_ape, DType::BF16, {128, 4}, 2, "pool APE");
    if (geometry.query_rank)
        input(p.query_norm, DType::BF16, {1536}, 2, "query norm");
    else
        require(!p.query_norm.data && !out.q_r.data, "History has no query operands");
    input(batch.positions, DType::I32, {T, B}, 4, "positions");
    input(batch.valid_columns, DType::I32, {B}, 4, "valid columns");
    input(batch.table_rows, DType::I32, {B}, 4, "table rows");
    for (const auto* f : {&before.raw_keys, &before.raw_gates}) {
        tail_field(*f, 128, B, 1, device);
        inputs[input_count++] = operand::range(f->records.byte_offsets, kOperation);
    }
    latent_cache(cache.latent, 512, device);
    index_cache(cache.index, 128, device);
    operand::same_paged_records(cache.latent.data, cache.index.keys, kOperation, "cache planes");
    const auto table_blocks = operand::blocks(cache.latent.data.records.page_offsets, kOperation);
    for (const auto* latent : {&cache.latent.data, &cache.latent.scales})
        for (const auto* index : {&cache.index.keys, &cache.index.scales}) {
            if (!latent->records.storage.data || !index->records.storage.data) continue;
            const auto& cr = latent->plane.tensor.region;
            const auto& ir = index->plane.tensor.region;
            require(cr.offset + cr.bytes <= ir.offset || ir.offset + ir.bytes <= cr.offset,
                    "cache planes overlap");
        }
    const auto input_weight = [&](const SingleProjectionWeight& weight) {
        for (const auto plane : weight_operands(weight.weight, weight.weight.n, 4096, device))
            if (plane.begin != plane.end) inputs[input_count++] = plane;
    };
    const auto& first = std::visit(
        [](const auto& w) -> const SingleProjectionWeight& {
            if constexpr (std::is_same_v<std::decay_t<decltype(w)>, SingleProjectionWeight>)
                return w;
            else
                return w.first;
        },
        p.weights);
    input_weight(first);
    const auto* paired = std::get_if<PairedProjectionWeights>(&p.weights);
    if (paired) input_weight(paired->second);
    std::array<operand::Range, 2> outputs{};
    int output_count  = 0;
    const auto output = [&](const Tensor& t, DType type, std::initializer_list<int> dims) {
        operand::tensor(t, type, dims, 16, kOperation, "query output", device);
        const auto range = operand::range(t, kOperation);
        for (int i = 0; i < input_count; ++i) operand::disjoint(range, inputs[i], kOperation);
        for (int i = 0; i < output_count; ++i) operand::disjoint(range, outputs[i], kOperation);
        outputs[output_count++] = range;
    };
    if (geometry.query_rank) output(out.q_r, DType::BF16, {1536, T, B});
    if (geometry.heads)
        output(out.head_coeff, DType::FP32, {32, T, B});
    else
        require(!out.head_coeff.data, "absent head coefficients");
    if (update) {
        tail_field(after.raw_keys, 128, B, 1, device);
        tail_field(after.raw_gates, 128, B, 1, device);
        inputs[input_count++] = operand::range(after.raw_keys.records.byte_offsets, kOperation);
        inputs[input_count++] = operand::range(after.raw_gates.records.byte_offsets, kOperation);
    } else {
        input(records.offsets, DType::I32, {B}, 4, "record offsets");
        const int width = records.raw_keys.ne[1];
        strided(records.raw_keys, DType::BF16, {128, width, B, 1}, device, "raw keys");
        strided(records.raw_gates, DType::BF16, {128, width, B, 1}, device, "raw gates");
    }
    for (int i = 0; i < output_count; ++i)
        require(!operand::overlaps(table_blocks, outputs[i]), "query output overlaps page table");
    for (int i = 0; i < output_count; ++i)
        for (int j = 0; j < input_count; ++j) operand::disjoint(outputs[i], inputs[j], kOperation);
    operand::device(arena.device(), device, kOperation, "workspace");
    const auto layout        = prepare_workspace(geometry, update, T, B);
    auto scope               = arena.scope();
    const auto scratch       = arena.alloc_bytes(layout.bytes, 256);
    const auto scratch_range = operand::range(scratch.data, scratch.bytes, kOperation);
    require(!operand::overlaps(table_blocks, scratch_range), "workspace overlaps page table");
    for (int i = 0; i < input_count; ++i) operand::disjoint(scratch_range, inputs[i], kOperation);
    for (int i = 0; i < output_count; ++i) operand::disjoint(scratch_range, outputs[i], kOperation);
    PrepareArguments a{};
    a.x                    = static_cast<const __nv_bfloat16*>(x.data);
    const auto bind_weight = [](const SingleProjectionWeight& w) {
        return PrepareWeight{w.weight.qdata, w.weight.scales, w.weight.qtype, w.policy, w.weight.n};
    };
    a.weights[0] = bind_weight(first);
    if (paired) a.weights[1] = bind_weight(paired->second);
    a.mapping = p.rows;
    if (layout.activation_codes.region.bytes) {
        a.activation_codes = static_cast<std::uint8_t*>(layout.activation_codes.bind(scratch).data);
        a.activation_scales =
            static_cast<std::uint8_t*>(layout.activation_scales.bind(scratch).data);
    }
    a.latent_norm    = static_cast<const __nv_bfloat16*>(p.latent_norm.data);
    a.query_norm     = static_cast<const __nv_bfloat16*>(p.query_norm.data);
    a.key_norm       = static_cast<const __nv_bfloat16*>(p.index_norm.data);
    a.key_bias       = static_cast<const __nv_bfloat16*>(p.index_bias.data);
    a.ape            = static_cast<const __nv_bfloat16*>(p.pool_ape.data);
    a.positions      = static_cast<const int*>(batch.positions.data);
    a.valid          = static_cast<const int*>(batch.valid_columns.data);
    a.table_rows     = static_cast<const int*>(batch.table_rows.data);
    a.record_offsets = static_cast<const int*>(records.offsets.data);
    a.page_offsets = static_cast<const std::int64_t*>(cache.latent.data.records.page_offsets.data);
    a.table_stride = cache.latent.data.records.page_offsets.nb[1] / 8;
    a.fp8_latent   = cache.latent.storage == KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0;
    if (a.fp8_latent) {
        a.latent_fp8    = static_cast<std::uint8_t*>(cache.latent.data.data());
        a.latent_scales = static_cast<std::uint8_t*>(cache.latent.scales.data());
    } else {
        a.latent = static_cast<__nv_bfloat16*>(cache.latent.data.data());
    }
    a.index        = cache.index.keys.data();
    a.fp8_index    = cache.index.format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    a.index_scales = a.fp8_index ? static_cast<std::uint8_t*>(cache.index.scales.data()) : nullptr;
    a.query        = static_cast<__nv_bfloat16*>(out.q_r.data);
    a.coefficients = static_cast<float*>(out.head_coeff.data);
    a.before[0]    = bind_field(before.raw_keys);
    a.before[1]    = bind_field(before.raw_gates);
    if (update) {
        a.after[0] = bind_field(after.raw_keys);
        a.after[1] = bind_field(after.raw_gates);
        if (layout.gates.region.bytes)
            a.gates = static_cast<__nv_bfloat16*>(layout.gates.bind(scratch).data);
        if (layout.carry.region.bytes)
            a.carry = static_cast<__nv_bfloat16*>(layout.carry.bind(scratch).data);
    } else {
        a.records[0] = bind_raw(records.raw_keys);
        a.records[1] = bind_raw(records.raw_gates);
    }
    a.projected   = static_cast<float*>(layout.projected.bind(scratch).data);
    a.tokens      = T;
    a.batch       = B;
    a.rows        = geometry.rows;
    a.query_rank  = geometry.query_rank;
    a.heads       = geometry.heads;
    a.norm_dim    = geometry.norm_dim();
    a.rms_epsilon = p.rms_epsilon;
    a.key_epsilon = p.index_epsilon;
    a.update      = update;
    launch_prepare(a, geometry, layout.columns, execution.stream());
}
} // namespace

void kpool_mla_prepare_record(const Tensor& x, const KpoolMlaPrepareParameters& p,
                              const KpoolMlaBatchView& batch, ConstKpoolTailStateView before,
                              KpoolMlaCacheView cache, KpoolRawRecordWriteView records,
                              KpoolMlaPrepareOutputs out, MemoryArena& workspace,
                              CudaExecutionView execution) {
    execute(x, p, batch, before, cache, records, {}, out, false, workspace, execution);
}

void kpool_mla_prepare_update(const Tensor& x, const KpoolMlaPrepareParameters& p,
                              const KpoolMlaBatchView& batch, ConstKpoolTailStateView before,
                              KpoolMlaCacheView cache, KpoolTailStateView after,
                              KpoolMlaPrepareOutputs out, MemoryArena& workspace,
                              CudaExecutionView execution) {
    execute(x, p, batch, before, cache, {}, after, out, true, workspace, execution);
}

std::size_t kpool_mla_prepare_record_workspace_capacity_bytes(const KpoolMlaPrepareProfile& p,
                                                              KpoolMlaStorage storage,
                                                              KvCacheVectorFormat index_format,
                                                              int batch, int lo, int hi, int sms) {
    const auto g = detail::kpool::prepare_geometry(p, storage, index_format, batch, lo, hi, sms);
    return detail::kpool::prepare_capacity(g, false, lo, hi, batch);
}

std::size_t kpool_mla_prepare_update_workspace_capacity_bytes(const KpoolMlaPrepareProfile& p,
                                                              KpoolMlaStorage storage,
                                                              KvCacheVectorFormat index_format,
                                                              int batch, int lo, int hi, int sms) {
    const auto g = detail::kpool::prepare_geometry(p, storage, index_format, batch, lo, hi, sms);
    return detail::kpool::prepare_capacity(g, true, lo, hi, batch);
}
} // namespace ninfer::ops
