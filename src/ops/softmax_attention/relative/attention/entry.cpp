#include "ops/softmax_attention/relative/attention/plan.h"

#include "core/paged_kv_storage.h"
#include "core/platform.h"
#include "ops/common/host/operands.h"

#include <cmath>
#include <type_traits>

namespace ninfer::ops {
namespace {

using namespace relative;
namespace operand = common::operand;

struct Call {
    const RelativeAttentionQueryInputs& in;
    Tensor& out;
    Device device;
    operand::Range output;
    operand::Range scratch;

    void read(const Tensor& tensor) const {
        if (tensor.data) read(operand::range(tensor, kOp));
    }

    void read(operand::Range range) const {
        operand::disjoint(output, range, kOp);
        if (scratch.begin != scratch.end) operand::disjoint(scratch, range, kOp);
    }

    void read(ConstMemorySpan mem) const { read(operand::range(mem.data, mem.bytes, kOp)); }
};

MemorySpan borrow(MemoryArena& arena, std::size_t bytes, Device device) {
    if (!bytes) return {};
    operand::device(arena.device(), device, kOp, "workspace");
    return arena.alloc_bytes(bytes, kScratchAlignment);
}

Call validate_queries(const RelativeAttentionQueryInputs& in, Tensor& out, int width, int batch,
                      float scale, MemorySpan scratch, Device device) {
    if (!std::isfinite(scale) || scale <= 0.0f)
        throw std::invalid_argument("relative_softmax_attention: expected finite positive scale");
    operand::tensor(in.q, DType::BF16, {kDim, kHeads, width, batch}, 16, kOp, "Q", device);
    operand::tensor(in.r, DType::BF16, {kRelDim, kHeads, width, batch}, 16, kOp, "R", device);
    operand::tensor(in.rel_logits_proj, DType::BF16, {kRelDim, kExtent}, 16, kOp, "projection",
                    device);
    operand::tensor(in.gate, DType::BF16, {kDim, kHeads, width, batch}, 16, kOp, "gate", device);
    operand::tensor(in.sink, DType::BF16, {kHeads, width, batch}, 16, kOp, "sink", device);
    operand::tensor(in.positions, DType::I32, {width, batch}, 4, kOp, "positions", device);
    operand::tensor(out, DType::BF16, {kDim, kHeads, width, batch}, 16, kOp, "output", device);

    Call call{in, out, device, operand::range(out, kOp),
              operand::range(scratch.data, scratch.bytes, kOp)};
    if (scratch.bytes) operand::disjoint(call.output, call.scratch, kOp);
    for (const auto* tensor :
         {&in.q, &in.r, &in.rel_logits_proj, &in.gate, &in.sink, &in.positions})
        call.read(*tensor);
    return call;
}

#if NINFER_TARGET_RTX5090
Query query(const Call& call, float scale) {
    const auto words = [](const Tensor& t) { return static_cast<const std::uint16_t*>(t.data); };
    const auto& in   = call.in;
    return {words(in.q),
            words(in.r),
            words(in.rel_logits_proj),
            words(in.gate),
            words(in.sink),
            static_cast<const int*>(in.positions.data),
            static_cast<std::uint16_t*>(call.out.data),
            scale};
}
#endif

CacheView bind_cache(const Call& call, const ConstCyclicKVCacheLayerView& cache, const Tensor&,
                     int rows) {
    if (cache.head_dim != kDim || cache.num_kv_heads != kKvHeads || cache.capacity != kWindow ||
        cache.padded_capacity < kWindow || cache.padded_capacity > INT32_MAX)
        throw std::invalid_argument("relative_softmax_attention: invalid cyclic cache geometry");
    const int padded  = static_cast<int>(cache.padded_capacity);
    const auto schema = paged_kv_storage_layout(cache.storage, kDim);
    const auto field  = [&](const ConstStateFieldView& view, DType dtype, int features,
                           const char* name) -> Plane {
        operand::state_field(view, dtype, {features, padded, kKvHeads}, 16, kOp, name, call.device);
        const auto& offsets = view.records.byte_offsets;
        operand::tensor(offsets, DType::I64, {rows}, 8, kOp, name, call.device);
        call.read(view.records.storage);
        call.read(offsets);
        return {operand::state_field_base<const unsigned char>(view),
                static_cast<const std::int64_t*>(offsets.data)};
    };
    CacheView result{field(cache.k, schema.key.data_dtype, kDim, "cyclic K"),
                     field(cache.v, schema.value.data_dtype, kDim, "cyclic V"),
                     {},
                     {},
                     nullptr,
                     padded,
                     rows};
    if (schema.key.has_scale()) {
        result.ks = field(cache.k_scales, schema.key.scale_dtype, 1, "cyclic K scales");
        result.vs = field(cache.v_scales, schema.value.scale_dtype, 1, "cyclic V scales");
    }
    return result;
}

CacheView bind_cache(const Call& call, const ConstPagedKVBatchLayerView& cache,
                     const Tensor& table_rows, int rows) {
    const auto& offsets = cache.k_pages.records.page_offsets;
    if (cache.head_dim != kDim || cache.num_kv_heads != kKvHeads ||
        cache.k_pages.records.page_tokens != kPagedKVPageSize)
        throw std::invalid_argument("relative_softmax_attention: invalid paged cache geometry");
    const auto schema = paged_kv_storage_layout(cache.storage, kDim);
    operand::tensor(table_rows, DType::I32, {rows}, 4, kOp, "table rows", call.device);
    call.read(table_rows);
    const auto plane = [&](const ConstPagedPlaneView& view, DType dtype, int features,
                           const char* name) -> Plane {
        operand::paged_plane(view, dtype, {features, kPagedKVPageSize, kKvHeads}, 16, kOp, name,
                             call.device);
        operand::same_paged_records(cache.k_pages, view, kOp, name);
        call.read(view.records.storage);
        call.read(view.records.page_offsets);
        return {static_cast<const unsigned char*>(view.data()),
                static_cast<const std::int64_t*>(offsets.data)};
    };
    CacheView result{plane(cache.k_pages, schema.key.data_dtype, kDim, "paged K"),
                     plane(cache.v_pages, schema.value.data_dtype, kDim, "paged V"),
                     {},
                     {},
                     static_cast<const int*>(table_rows.data),
                     offsets.ne[0],
                     offsets.ne[1]};
    if (schema.key.has_scale()) {
        result.ks = plane(cache.k_scale_pages, schema.key.scale_dtype, 1, "paged K scales");
        result.vs = plane(cache.v_scale_pages, schema.value.scale_dtype, 1, "paged V scales");
    }
    return result;
}

void validate_chunk_presence(const RelativeAttentionChunkKV& chunk, KvCacheStorage storage,
                             bool required) {
    if (!chunk.k.data) {
        if (required || chunk.v.data || chunk.k_scales.data || chunk.v_scales.data)
            throw std::invalid_argument("relative_softmax_attention: invalid or missing chunk KV");
    } else if (chunk.storage != storage) {
        throw std::invalid_argument("relative_softmax_attention: chunk storage differs from cache");
    }
}

ChunkView bind_chunk(const Call& call, const RelativeAttentionChunkKV& chunk,
                     KvCacheStorage storage, int width, int batch) {
    if (!chunk.k.data) return {};
    const auto layout = paged_kv_storage_layout(storage, kDim);
    const auto bind   = [&](const Tensor& tensor, DType dtype, int dim, int alignment,
                          const char* name) {
        operand::tensor(tensor, dtype, {dim, kKvHeads, width, batch}, alignment, kOp, name,
                          call.device);
        call.read(tensor);
        return static_cast<const unsigned char*>(tensor.data);
    };
    ChunkView view{bind(chunk.k, layout.key.data_dtype, kDim, 16, "chunk K"),
                   bind(chunk.v, layout.value.data_dtype, kDim, 16, "chunk V")};
    if (layout.key.has_scale()) {
        view.ks = bind(chunk.k_scales, layout.key.scale_dtype, 1, 1, "chunk K scales");
        view.vs = bind(chunk.v_scales, layout.value.scale_dtype, 1, 1, "chunk V scales");
    } else if (chunk.k_scales.data || chunk.v_scales.data) {
        throw std::invalid_argument("relative_softmax_attention: BF16 chunk has scale planes");
    }
    return view;
}

template <class Cache>
void batched(const RelativeAttentionQueryInputs& in, const Tensor& valid, const Tensor& table_rows,
             AttentionHeadGeometry geometry, bool swa, float scale, const Cache& cache,
             const RelativeAttentionChunkKV& chunk, ContextBucket bucket, MemoryArena& workspace,
             Tensor& out, CudaExecutionView execution) {
    const int sms = execution.multiprocessor_count();
    validate_profile(geometry, cache.storage, sms);
    (void)context_bucket_tokens(bucket);
    const auto plan = select_decode(swa, cache.storage, bucket, in.q.ne[3], in.q.ne[2], sms);
    if constexpr (std::is_same_v<Cache, ConstCyclicKVCacheLayerView>)
        validate_chunk_presence(chunk, cache.storage, plan.width > 1);
    auto scope         = workspace.scope();
    const auto scratch = borrow(workspace, plan.bytes(), execution.device());
    const auto call =
        validate_queries(in, out, plan.width, plan.batch, scale, scratch, execution.device());
    if (valid.data) {
        operand::tensor(valid, DType::I32, {plan.batch}, 4, kOp, "valid columns", call.device);
        call.read(valid);
    }
    auto view = bind_cache(call, cache, table_rows, plan.batch);
    if constexpr (std::is_same_v<Cache, ConstCyclicKVCacheLayerView>)
        view.chunk = bind_chunk(call, chunk, cache.storage, plan.width, plan.batch);
    auto* values = static_cast<float*>(scratch.data);
    const Partials partials{values, values ? values + plan.rows() * kDim : nullptr};
#if NINFER_TARGET_RTX5090
    launch_decode(query(call, scale), view, swa, cache.storage, static_cast<const int*>(valid.data),
                  plan, partials, execution.stream());
#else
    (void)partials;
    throw std::runtime_error("relative_softmax_attention: requires the RTX 5090 build target");
#endif
}

template <class Cache>
void packed(const RelativeAttentionQueryInputs& in, const RaggedSequenceView& ragged,
            const Tensor& table_rows, AttentionHeadGeometry geometry, bool swa, float scale,
            const Cache& cache, const RelativeAttentionChunkKV& chunk, ContextBucket bucket,
            MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    validate_profile(geometry, cache.storage, execution.multiprocessor_count());
    if constexpr (std::is_same_v<Cache, ConstCyclicKVCacheLayerView>)
        validate_chunk_presence(chunk, cache.storage, true);
    (void)context_bucket_tokens(bucket);
    const auto device = execution.device();
    operand::ragged_sequence(ragged, kOp, device);
    const int n = ragged.total_tokens(), segments = ragged.num_seqs();
    const int tiles    = ragged_tiles(n, segments);
    auto scope         = workspace.scope();
    const auto scratch = borrow(workspace, segments == 1 ? 0 : sizeof(RaggedTile) * tiles, device);
    const auto call    = validate_queries(in, out, n, 1, scale, scratch, device);
    call.read(ragged.cu_seqlens);
    call.read(ragged.seq_lengths);
    call.read(ragged.token_to_seq);
    auto view = bind_cache(call, cache, table_rows, segments);
    if constexpr (std::is_same_v<Cache, ConstCyclicKVCacheLayerView>)
        view.chunk = bind_chunk(call, chunk, cache.storage, n, 1);
#if NINFER_TARGET_RTX5090
    launch_ragged(query(call, scale), view, swa, cache.storage,
                  static_cast<const int*>(ragged.cu_seqlens.data), n, segments,
                  static_cast<RaggedTile*>(scratch.data), tiles, execution.stream());
#else
    (void)swa;
    (void)view;
    throw std::runtime_error("relative_softmax_attention: requires the RTX 5090 build target");
#endif
}

} // namespace

void relative_softmax_attention(const RelativeAttentionQueryInputs& in, const Tensor& valid,
                                AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                                const ConstCyclicKVCacheLayerView& cache,
                                const RelativeAttentionChunkKV& chunk, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    validate_window(window);
    batched(in, valid, {}, geometry, true, scale, cache, chunk, bucket, workspace, out, execution);
}

void relative_softmax_attention(const RelativeAttentionQueryInputs& in, const Tensor& valid,
                                const Tensor& table_rows, AttentionHeadGeometry geometry,
                                float scale, const ConstPagedKVBatchLayerView& cache,
                                ContextBucket bucket, MemoryArena& workspace, Tensor& out,
                                CudaExecutionView execution) {
    batched(in, valid, table_rows, geometry, false, scale, cache, {}, bucket, workspace, out,
            execution);
}

void relative_softmax_attention(const RelativeAttentionQueryInputs& in,
                                const RaggedSequenceView& ragged, AttentionHeadGeometry geometry,
                                std::uint32_t window, float scale,
                                const ConstCyclicKVCacheLayerView& cache,
                                const RelativeAttentionChunkKV& chunk, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    validate_window(window);
    packed(in, ragged, {}, geometry, true, scale, cache, chunk, bucket, workspace, out, execution);
}

void relative_softmax_attention(const RelativeAttentionQueryInputs& in,
                                const RaggedSequenceView& ragged, const Tensor& table_rows,
                                AttentionHeadGeometry geometry, float scale,
                                const ConstPagedKVBatchLayerView& cache, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    packed(in, ragged, table_rows, geometry, false, scale, cache, {}, bucket, workspace, out,
           execution);
}

} // namespace ninfer::ops
