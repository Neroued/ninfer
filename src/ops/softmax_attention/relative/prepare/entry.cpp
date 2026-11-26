#include "kernels.cuh"
#include "ops/common/host/operands.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ninfer::ops {
namespace relative_prepare {
namespace {

namespace operand             = common::operand;
constexpr char kOp[]          = "relative_attention_prepare";
constexpr char kCommitOp[]    = "relative_attention_commit_cyclic_kv";
constexpr KvCacheStorage kFp8 = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;

bool is_fp8(KvCacheStorage storage) { return storage == kFp8; }

[[noreturn]] void invalid(const char* reason, const char* op = kOp) {
    throw std::invalid_argument(std::string(op) + ": " + reason);
}

auto bf16(const Tensor& t) { return static_cast<Bf16*>(t.data); }

auto i32(const Tensor& t) { return static_cast<const int*>(t.data); }

auto i64(const Tensor& t) { return static_cast<const std::int64_t*>(t.data); }

struct RecordRange {
    operand::Range storage;
    operand::Range field;
    bool write;
    const void* selector;
};

// Field offsets are comparable within a single record selection. The caller owns
// disjointness across distinct selections, except for exact per-row history aliases.
struct Validation {
    Device device;
    int sequences;
    const char* op = kOp;
    std::array<operand::Range, 32> reads{};
    std::array<operand::Blocks, 9> writes{};
    std::array<RecordRange, 8> records{};
    std::size_t num_reads = 0, num_writes = 0, num_records = 0;

    void tensor(const Tensor& t, DType dtype, std::initializer_list<int> dims, const char* name,
                bool write = false) {
        operand::tensor(t, dtype, dims,
                        dtype == DType::U8 || dtype == DType::I32 || dtype == DType::I64
                            ? dtype_size(dtype)
                            : 16,
                        op, name, device);
        const auto range = operand::range(t, op);
        if (write)
            writes[num_writes++] = {range, t.bytes(), 1};
        else
            reads[num_reads++] = range;
    }

    void record(ConstMemorySpan storage, const LayoutRegion& region, const void* selector,
                bool write, bool exact_alias = false) {
        const RecordRange current{
            operand::range(storage.data, storage.bytes, op),
            operand::range(storage.subspan(region.offset).data, region.bytes, op), write, selector};
        for (std::size_t j = 0; j < num_records; ++j) {
            const auto& prior = records[j];
            if (!(write || prior.write) || current.storage.begin != prior.storage.begin ||
                current.selector != prior.selector)
                continue;
            if (exact_alias && j + 1 == num_records && !prior.write &&
                current.field.begin == prior.field.begin && current.field.end == prior.field.end)
                continue;
            operand::disjoint(current.field, prior.field, op);
        }
        records[num_records++] = current;
    }

    template <class T, class View>
    Records<T> state(const View& view, DType dtype, std::initializer_list<int> dims,
                     const char* name, bool write, bool exact_alias = false) {
        operand::state_field(view, dtype, dims, dtype == DType::U8 ? 1 : 16, op, name, device);
        if (view.field.layers != 1) invalid("select one state or cache layer", op);
        const auto& offsets = view.records.byte_offsets;
        operand::shape(offsets, {sequences, 1, 1, 1}, op, "state byte offsets");
        reads[num_reads++] = operand::range(offsets, op);
        record(view.records.storage, view.field.tensor.region, offsets.data, write, exact_alias);
        return {operand::state_field_base<T>(view), i64(offsets)};
    }

    void disjoint() const {
        for (std::size_t j = 0; j < num_writes; ++j) {
            for (std::size_t k = 0; k < j; ++k)
                if (operand::overlaps(writes[j], writes[k])) invalid("outputs overlap", op);
            for (std::size_t k = 0; k < num_reads; ++k)
                if (operand::overlaps(writes[j], reads[k])) invalid("output overlaps input", op);
            for (std::size_t k = 0; k < num_records; ++k)
                if (operand::overlaps(writes[j], records[k].storage))
                    invalid("output overlaps state or cache storage", op);
        }
        for (std::size_t j = 0; j < num_records; ++j) {
            const auto& r = records[j];
            if (r.write)
                for (std::size_t k = 0; k < num_reads; ++k)
                    operand::disjoint(r.storage, reads[k], op);
        }
    }
};

struct State {
    ConstStateFieldView k_in, v_in;
    StateFieldView k_out, v_out;
    const Tensor* snapshot_slots = nullptr;
    Tensor k_snapshots, v_snapshots;
};

void bind_state(Arguments& args, Validation& check, const State& state) {
    auto bind = [&](Conv& conv, const ConstStateFieldView& in, const StateFieldView& out,
                    const Tensor& snapshots, std::int64_t& stride) {
        conv.before = check.state<const Bf16>(in, DType::BF16, {kChannels, kHistory},
                                              "source history", false);
        if (state.snapshot_slots) {
            const int n = snapshots.ne[2];
            if (n <= 0) invalid("snapshot tensor must have slots");
            // Slots may be separated by padding, as with other snapshot producers.
            operand::slotted_tensor(snapshots, DType::BF16, {kChannels, kHistory, n}, 2, 16, kOp,
                                    "snapshots", check.device);
            check.writes[check.num_writes++] = operand::blocks(snapshots, kOp);
            conv.after                       = {bf16(snapshots), nullptr};
            stride                           = snapshots.nb[2];
        } else {
            conv.after = check.state<Bf16>(out, DType::BF16, {kChannels, kHistory},
                                           "destination history", true, true);
        }
    };
    if (state.snapshot_slots) {
        check.tensor(*state.snapshot_slots, DType::I32, {check.sequences}, "snapshot base slots");
        args.snapshot_slots = i32(*state.snapshot_slots);
    }
    bind(args.k, state.k_in, state.k_out, state.k_snapshots, args.snapshot_stride_k);
    bind(args.v, state.v_in, state.v_out, state.v_snapshots, args.snapshot_stride_v);
    if (state.snapshot_slots && state.k_snapshots.ne[2] != state.v_snapshots.ne[2])
        invalid("K and V snapshot slot counts must match");
}

void bind_cache(Arguments& args, Validation& check, const CyclicKVCacheLayerView& cache,
                const Tensor&, const PagedKVStorageLayout& layout) {
    if (cache.capacity != kCyclicCapacity || cache.padded_capacity < cache.capacity ||
        cache.padded_capacity > static_cast<unsigned>(std::numeric_limits<int>::max()))
        invalid("invalid cyclic capacity", check.op);
    const int padded = static_cast<int>(cache.padded_capacity);
    auto bind        = [&](Conv& conv, const StateFieldView& data, const StateFieldView& scale,
                    const PagedKVVectorLayout& format) {
        conv.cache =
            check.state<char>(data, format.data_dtype,
                                     {format.data_leading_extent, padded, kKvHeads}, "cyclic cache", true);
        if (format.has_scale()) {
            conv.scales = check.state<char>(scale, format.scale_dtype,
                                                   {format.scale_leading_extent, padded, kKvHeads},
                                                   "cyclic cache scales", true);
            if (conv.scales.offsets == conv.cache.offsets) conv.scales.offsets = nullptr;
        } else if (scale.records.storage.data || scale.records.byte_offsets.data) {
            invalid("unscaled cyclic cache has scales", check.op);
        }
    };
    bind(args.k, cache.k, cache.k_scales, layout.key);
    bind(args.v, cache.v, cache.v_scales, layout.value);
    args.cache_rows = padded;
}

void bind_cache(Arguments& args, Validation& check, const PagedKVBatchLayerView& cache,
                const Tensor& rows, const PagedKVStorageLayout& layout) {
    const auto& key   = cache.k_pages;
    const auto& table = key.records.page_offsets;
    if (key.records.page_tokens != kPagedKVPageSize) invalid("invalid paged capacity");
    check.tensor(rows, DType::I32, {check.sequences}, "KV table rows");
    auto bind = [&](const PagedPlaneView& plane, DType dtype, int leading, const char* name) {
        operand::paged_plane(plane, dtype, {leading, kPagedKVPageSize, kKvHeads}, 16, kOp, name,
                             check.device);
        operand::same_paged_records(key, plane, kOp, name);
        check.record(plane.records.storage, plane.plane.tensor.region, table.data, true);
        return Records<char>{static_cast<char*>(plane.data()), i64(table)};
    };
    auto bind_pair = [&](Conv& conv, const PagedPlaneView& data, const PagedPlaneView& scale,
                         const PagedKVVectorLayout& format) {
        conv.cache = bind(data, format.data_dtype, format.data_leading_extent, "paged cache");
        if (format.has_scale()) {
            conv.scales =
                bind(scale, format.scale_dtype, format.scale_leading_extent, "paged scales");
            conv.scales.offsets = nullptr;
        } else if (scale.records.storage.data || scale.records.page_offsets.data) {
            invalid("unscaled paged cache has scales");
        }
    };
    bind_pair(args.k, key, cache.k_scale_pages, layout.key);
    bind_pair(args.v, cache.v_pages, cache.v_scale_pages, layout.value);
    check.reads[check.num_reads++] = operand::range(table, kOp);
    args.cache_rows                = table.ne[0];
    args.table_rows                = i32(rows);
}

void validate_chunk(Validation& check, const RelativeAttentionChunkKV& chunk, int width, int batch,
                    bool write) {
    const auto layout = paged_kv_storage_layout(chunk.storage, kDim);
    check.tensor(chunk.k, layout.key.data_dtype, {kDim, kKvHeads, width, batch}, "chunk K", write);
    check.tensor(chunk.v, layout.value.data_dtype, {kDim, kKvHeads, width, batch}, "chunk V",
                 write);
    if (is_fp8(chunk.storage)) {
        check.tensor(chunk.k_scales, DType::U8, {1, kKvHeads, width, batch}, "chunk K scales",
                     write);
        check.tensor(chunk.v_scales, DType::U8, {1, kKvHeads, width, batch}, "chunk V scales",
                     write);
    } else if (chunk.k_scales.data || chunk.v_scales.data) {
        invalid("unscaled chunk has scales", check.op);
    }
}

void validate_cache_profile(KvCacheStorage storage, int dim, int heads, const char* op) {
    if ((!is_fp8(storage) && storage != KvCacheStorage::BFloat16) || dim != kDim ||
        heads != kKvHeads)
        invalid("unsupported KV cache profile", op);
}

void commit(const RelativeAttentionChunkKV& chunk, const RaggedSequenceView* ragged,
            const Tensor& positions, const Tensor& counts, CyclicKVCacheLayerView cache,
            CudaExecutionView execution) {
    const auto device = execution.device();
    validate_cache_profile(cache.storage, cache.head_dim, cache.num_kv_heads, kCommitOp);
    if (chunk.storage != cache.storage) invalid("chunk and cyclic storage differ", kCommitOp);
    if (ragged) operand::ragged_sequence(*ragged, kCommitOp, device);
    const int width    = ragged ? ragged->total_tokens() : positions.ne[0];
    const int batch    = ragged ? 1 : positions.ne[1];
    const auto columns = std::int64_t(width) * batch;
    if (width <= 0 || batch <= 0 || columns > std::numeric_limits<int>::max())
        invalid("invalid token extent", kCommitOp);
    Validation check{device, ragged ? ragged->num_seqs() : batch, kCommitOp};
    validate_chunk(check, chunk, width, batch, false);
    check.tensor(positions, DType::I32, {width, batch}, "positions");
    if (counts.data) check.tensor(counts, DType::I32, {batch}, "accepted counts");
    if (ragged)
        for (const auto* t : {&ragged->token_to_seq, &ragged->cu_seqlens, &ragged->seq_lengths})
            check.reads[check.num_reads++] = operand::range(*t, kCommitOp);
    Arguments dest{};
    bind_cache(dest, check, cache, {}, paged_kv_storage_layout(cache.storage, kDim));
    check.disjoint();
    const auto plane = [](const Tensor& data, const Tensor& scales, const Conv& dst) {
        return CommitPlane{static_cast<const char*>(data.data),
                           static_cast<const char*>(scales.data), dst.cache, dst.scales};
    };
    // The host ragged layout provides a launch envelope. Device lengths still
    // choose which slots and source tokens the graph replay actually commits.
    int tail_slots = std::min(width, kCyclicCapacity);
    if (ragged) {
        tail_slots = 0;
        for (int seq = 0; seq < check.sequences && tail_slots < kCyclicCapacity; ++seq)
            tail_slots =
                std::max(tail_slots, std::min(ragged->segment_length(seq), kCyclicCapacity));
    }
    CommitArguments args{plane(chunk.k, chunk.k_scales, dest.k),
                         plane(chunk.v, chunk.v_scales, dest.v),
                         i32(positions),
                         i32(counts),
                         ragged ? i32(ragged->cu_seqlens) : nullptr,
                         ragged ? i32(ragged->seq_lengths) : nullptr,
                         width,
                         check.sequences,
                         tail_slots,
                         dest.cache_rows};
    launch_commit(args, is_fp8(cache.storage), execution);
}

template <class Cache>
void prepare(const RelativeAttentionPrepareInputs& in, const Tensor& valid,
             const RaggedSequenceView* ragged, const Tensor& rows, const State& state, Cache cache,
             const RelativeAttentionPrepareParams& params, RelativeAttentionPrepareOutputs out,
             CudaExecutionView execution) {
    const auto positive = [](float x) { return x > 0 && std::isfinite(x); };
    if (!positive(params.qk_clip_threshold) || !positive(params.rel_clip_threshold) ||
        !positive(params.clip_eps) || !positive(params.v_norm_eps) ||
        (params.use_log_scaling &&
         (params.log_scaling_floor <= 0 || !(params.log_scaling_alpha >= 0) ||
          !std::isfinite(params.log_scaling_alpha))))
        invalid("invalid normalization or log-scaling parameters");
    const bool fp8        = is_fp8(cache.storage);
    constexpr bool Chunk  = std::is_same_v<Cache, RelativeAttentionChunkKV>;
    constexpr bool Cyclic = std::is_same_v<Cache, CyclicKVCacheLayerView>;
    if constexpr (Chunk)
        validate_cache_profile(cache.storage, kDim, kKvHeads, kOp);
    else
        validate_cache_profile(cache.storage, cache.head_dim, cache.num_kv_heads, kOp);
    const auto device = execution.device();
    if (ragged) operand::ragged_sequence(*ragged, kOp, device);
    const int width    = ragged ? ragged->total_tokens() : in.q.ne[2];
    const int batch    = ragged ? 1 : in.q.ne[3];
    const auto columns = std::int64_t(width) * batch;
    if (width <= 0 || batch <= 0 || columns > std::numeric_limits<int>::max())
        invalid("invalid token extent");
    if constexpr (Cyclic) {
        if (width != 1) invalid("direct cyclic prepare requires W == 1");
    }
    Validation check{device, ragged ? ragged->num_seqs() : batch};
    auto features = [&](const Tensor& t, int d, int heads, const char* name, bool write = false) {
        check.tensor(t, DType::BF16, {d, heads, width, batch}, name, write);
    };
    features(in.q, kDim, kQueryHeads, "Q");
    features(in.r, kRelDim, kQueryHeads, "R");
    features(in.k, kDim, kKvHeads, "K");
    features(in.v, kDim, kKvHeads, "V");
    features(out.q_out, kDim, kQueryHeads, "Q output", true);
    features(out.r_out, kRelDim, kQueryHeads, "R output", true);
    check.tensor(in.q_scale, DType::BF16, {kDim, kQueryHeads}, "Q scale");
    check.tensor(in.r_scale, DType::BF16, {kRelDim, kQueryHeads}, "R scale");
    check.tensor(in.k_conv_weight, DType::BF16, {kChannels, kTaps}, "K convolution weight");
    check.tensor(in.v_conv_weight, DType::BF16, {kChannels, kTaps}, "V convolution weight");
    if (bool(in.k_dyn.data) != bool(in.v_dyn.data)) invalid("dynamic K and V must match");
    if (in.k_dyn.data) {
        features(in.k_dyn, kTaps, kKvHeads, "dynamic K");
        features(in.v_dyn, kTaps, kKvHeads, "dynamic V");
    }
    check.tensor(in.positions, DType::I32, {width, batch}, "positions");
    if (valid.data) check.tensor(valid, DType::I32, {batch}, "valid columns");
    if (ragged)
        for (const auto* t : {&ragged->token_to_seq, &ragged->cu_seqlens, &ragged->seq_lengths})
            check.reads[check.num_reads++] = operand::range(*t, kOp);

    const int sink_channels = in.sink_x.data ? in.sink_x.ne[0] : 0;
    if (bool(in.sink_x.data) != bool(in.sink_weight.data) ||
        bool(in.sink_x.data) != bool(out.sink_out.data))
        invalid("sink input, weight and output must be supplied together");
    if (in.sink_x.data) {
        if (sink_channels <= 0) invalid("sink width must be positive");
        check.tensor(in.sink_x, DType::BF16, {sink_channels, width, batch}, "sink input");
        check.tensor(in.sink_weight, DType::BF16, {sink_channels, kQueryHeads}, "sink weight");
        check.tensor(out.sink_out, DType::BF16, {kQueryHeads, width, batch}, "sink output", true);
    }

    Arguments args{};
    args.params        = params;
    args.q             = bf16(in.q);
    args.r             = bf16(in.r);
    args.q_scale       = bf16(in.q_scale);
    args.r_scale       = bf16(in.r_scale);
    args.sink_x        = bf16(in.sink_x);
    args.sink_weight   = bf16(in.sink_weight);
    args.q_out         = bf16(out.q_out);
    args.r_out         = bf16(out.r_out);
    args.sink_out      = bf16(out.sink_out);
    args.k             = {bf16(in.k), bf16(in.k_conv_weight), bf16(in.k_dyn), {}, {}, {}, {}};
    args.v             = {bf16(in.v), bf16(in.v_conv_weight), bf16(in.v_dyn), {}, {}, {}, {}};
    args.positions     = i32(in.positions);
    args.valid         = i32(valid);
    args.width         = width;
    args.columns       = static_cast<int>(columns);
    args.sink_channels = sink_channels;
    if (ragged) {
        args.token_to_seq = i32(ragged->token_to_seq);
        args.cu_seqlens   = i32(ragged->cu_seqlens);
        args.seq_lengths  = i32(ragged->seq_lengths);
    }
    bind_state(args, check, state);
    if constexpr (Chunk) {
        validate_chunk(check, cache, width, batch, true);
        args.k.cache.base  = static_cast<char*>(cache.k.data);
        args.v.cache.base  = static_cast<char*>(cache.v.data);
        args.k.scales.base = static_cast<char*>(cache.k_scales.data);
        args.v.scales.base = static_cast<char*>(cache.v_scales.data);
    } else {
        bind_cache(args, check, cache, rows, paged_kv_storage_layout(cache.storage, kDim));
    }
    check.disjoint();
    constexpr auto destination = Chunk    ? Destination::Chunk
                                 : Cyclic ? Destination::Cyclic
                                          : Destination::Paged;
    launch(args, destination, fp8, execution);
}

} // namespace
} // namespace relative_prepare

using relative_prepare::prepare;

void relative_attention_prepare(const RelativeAttentionPrepareInputs& in, const Tensor& valid,
                                ConstStateFieldView ki, StateFieldView ko, ConstStateFieldView vi,
                                StateFieldView vo, CyclicKVCacheLayerView cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs out, CudaExecutionView execution) {
    prepare(in, valid, nullptr, {}, {ki, vi, ko, vo, nullptr, {}, {}}, cache, params, out,
            execution);
}

void relative_attention_prepare(const RelativeAttentionPrepareInputs& in, const Tensor& valid,
                                ConstStateFieldView ki, StateFieldView ko, ConstStateFieldView vi,
                                StateFieldView vo, RelativeAttentionChunkKV chunk,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs out, CudaExecutionView execution) {
    prepare(in, valid, nullptr, {}, {ki, vi, ko, vo, nullptr, {}, {}}, chunk, params, out,
            execution);
}

void relative_attention_prepare_snapshot(const RelativeAttentionPrepareInputs& in,
                                         const Tensor& valid, const Tensor& slots,
                                         ConstStateFieldView ki, Tensor& ks, ConstStateFieldView vi,
                                         Tensor& vs, RelativeAttentionChunkKV cache,
                                         const RelativeAttentionPrepareParams& params,
                                         RelativeAttentionPrepareOutputs out,
                                         CudaExecutionView execution) {
    prepare(in, valid, nullptr, {}, {ki, vi, {}, {}, &slots, ks, vs}, cache, params, out,
            execution);
}

void relative_attention_prepare(const RelativeAttentionPrepareInputs& in, const Tensor& valid,
                                const Tensor& rows, ConstStateFieldView ki, StateFieldView ko,
                                ConstStateFieldView vi, StateFieldView vo,
                                PagedKVBatchLayerView cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs out, CudaExecutionView execution) {
    prepare(in, valid, nullptr, rows, {ki, vi, ko, vo, nullptr, {}, {}}, cache, params, out,
            execution);
}

void relative_attention_prepare_snapshot(
    const RelativeAttentionPrepareInputs& in, const Tensor& valid, const Tensor& rows,
    const Tensor& slots, ConstStateFieldView ki, Tensor& ks, ConstStateFieldView vi, Tensor& vs,
    PagedKVBatchLayerView cache, const RelativeAttentionPrepareParams& params,
    RelativeAttentionPrepareOutputs out, CudaExecutionView execution) {
    prepare(in, valid, nullptr, rows, {ki, vi, {}, {}, &slots, ks, vs}, cache, params, out,
            execution);
}

void relative_attention_prepare(const RelativeAttentionPrepareInputs& in,
                                const RaggedSequenceView& ragged, ConstStateFieldView ki,
                                StateFieldView ko, ConstStateFieldView vi, StateFieldView vo,
                                RelativeAttentionChunkKV cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs out, CudaExecutionView execution) {
    prepare(in, {}, &ragged, {}, {ki, vi, ko, vo, nullptr, {}, {}}, cache, params, out, execution);
}

void relative_attention_prepare(const RelativeAttentionPrepareInputs& in,
                                const RaggedSequenceView& ragged, const Tensor& rows,
                                ConstStateFieldView ki, StateFieldView ko, ConstStateFieldView vi,
                                StateFieldView vo, PagedKVBatchLayerView cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs out, CudaExecutionView execution) {
    prepare(in, {}, &ragged, rows, {ki, vi, ko, vo, nullptr, {}, {}}, cache, params, out,
            execution);
}

void relative_attention_commit_cyclic_kv(const RelativeAttentionChunkKV& chunk,
                                         const Tensor& positions, const Tensor& counts,
                                         CyclicKVCacheLayerView cache,
                                         CudaExecutionView execution) {
    relative_prepare::commit(chunk, nullptr, positions, counts, cache, execution);
}

void relative_attention_commit_cyclic_kv(const RelativeAttentionChunkKV& chunk,
                                         const RaggedSequenceView& ragged, const Tensor& positions,
                                         CyclicKVCacheLayerView cache,
                                         CudaExecutionView execution) {
    relative_prepare::commit(chunk, &ragged, positions, {}, cache, execution);
}

} // namespace ninfer::ops
