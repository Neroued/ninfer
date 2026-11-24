#include "ops/softmax_attention/qsa/prepare/launch.h"
#include "ops/softmax_attention/qsa/prepare/kernel.cuh"

#include "ops/common/host/operands.h"
#include "core/platform.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops::detail::qsa {
namespace {

template <class T>
PrepareColumns<T> prepare_columns(const Tensor& t) {
    return {static_cast<T*>(t.data), t.nb[1] / std::int64_t(sizeof(T)),
            t.nb[2] / std::int64_t(sizeof(T))};
}

template <class T>
PreparePage<T> prepare_page(const PagedPlaneView& t) {
    return {static_cast<T*>(t.data()), std::int64_t(t.row_stride_bytes() / sizeof(T)),
            std::int64_t(t.head_stride_bytes() / sizeof(T))};
}

template <class T>
PrepareState<T> prepare_state(const ConstStateFieldView& view) {
    return {common::operand::state_field_base<const T>(view),
            static_cast<const std::int64_t*>(view.records.byte_offsets.data),
            view.field.tensor.shape[0]};
}

template <bool Ragged>
PrepareArguments prepare_arguments(const QsaKvProjectionView& p,
                                   const QsaKvPrepareParameters& params, const QsaBatchView& batch,
                                   const Tensor& rope, const ConstQsaTailStateView& before,
                                   QsaCacheView cache, QsaRawRecordWriteView records) {
    const auto bf  = [](const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); };
    const auto i32 = [](const Tensor& t) { return static_cast<const std::int32_t*>(t.data); };
    PrepareArguments a{
        nullptr,
        bf(p.key),
        bf(p.value),
        nullptr,
        bf(p.index_key),
        nullptr,
        bf(params.key_norm),
        nullptr,
        bf(params.index_key_norm),
        nullptr,
        nullptr,
        i32(batch.positions),
        i32(rope),
        i32(batch.valid_columns),
        i32(batch.table_rows),
        i32(records.offsets),
        static_cast<const std::int64_t*>(cache.main.k_pages.records.page_offsets.data),
        cache.main.k_pages.records.page_offsets.nb[1] / 8,
        prepare_state<__nv_bfloat16>(before.raw_keys),
        prepare_state<std::int32_t>(before.first_rope),
        prepare_columns<__nv_bfloat16>(records.raw_keys),
        prepare_columns<std::int32_t>(records.rope_positions),
        prepare_page<__nv_bfloat16>(cache.index.keys),
        prepare_page<std::uint8_t>(cache.main.k_pages),
        prepare_page<std::uint8_t>(cache.main.v_pages),
        prepare_page<std::uint8_t>(cache.main.k_scale_pages),
        prepare_page<std::uint8_t>(cache.main.v_scale_pages),
        p.key.ne[2],
        p.key.ne[3],
        rope.ne[1],
        p.key.ne[1],
        params.rms_epsilon,
        {}};
    if constexpr (Ragged) {
        a.rope_axis_stride = rope.nb[1] / std::int64_t(sizeof(std::int32_t));
        a.rope_row_stride  = rope.nb[2] / std::int64_t(sizeof(std::int32_t));
    }
    // Default model frequencies are constructed once on Host, with no Device allocation.
    static const auto default_frequency = [] {
        std::array<double, 32> result{};
        for (int i = 0; i < 32; ++i) result[i] = std::pow(1e7, -double(i) / 32);
        return result;
    }();
    for (int i = 0; i < 32; ++i)
        a.frequency[i] = params.rope_theta == 1e7F
                             ? default_frequency[i]
                             : std::pow(double(params.rope_theta), -double(i) / 32);
    return a;
}

struct PreparePlan {
    bool token_major, bounded_phase, vector_weights;
    bool single_segment = false;
};

PreparePlan select_prepare_plan(const PrepareArguments& args, float theta, bool queries) {
    const auto columns = std::int64_t(args.tokens) * args.batch;
    bool token_major =
        !queries || (NINFER_TARGET_SPARK ? args.kv_heads == 2 || columns < 256 || columns > 1024
                                         : columns <= 320);
#if NINFER_TARGET_RTX5090
    if (args.kv_heads == 4) token_major = columns <= 256;
#endif
    return {token_major, theta >= 1,
            queries && common::operand::aligned(args.wq, 16) &&
                common::operand::aligned(args.wiq, 8)};
}

template <KvCacheStorage Storage, bool BoundedPhase, bool VectorWeights, int KvHeads, bool Queries,
          bool Ragged>
void submit(const PrepareInput<Ragged>& args, PreparePlan plan, cudaStream_t stream) {
    constexpr int groups = PrepareGeometry<KvHeads, Queries>::groups;
    const auto launch    = [&]<bool TokenMajor>() {
        constexpr bool vector_kv  = KvHeads == 4 || (NINFER_TARGET_RTX5090 && KvHeads == 2);
        constexpr int token_limit = TokenMajor ? INT_MAX / groups : INT_MAX;
        for (std::int64_t row = 0; row < args.batch; row += 65535) {
            for (std::int64_t token = 0; token < args.tokens; token += token_limit) {
                auto window        = args;
                window.row_begin   = static_cast<int>(row);
                window.token_begin = static_cast<int>(token);
                const auto rows    = std::min(65535, args.batch - window.row_begin);
                const auto tokens  = std::min(token_limit, args.tokens - window.token_begin);
                const dim3 grid =
                    TokenMajor ? dim3(tokens * groups, rows) : dim3(tokens, rows, groups);
                if constexpr (Ragged) {
                    if (plan.single_segment)
                        prepare_ragged_kernel<Storage, BoundedPhase, VectorWeights, TokenMajor,
                                              KvHeads, Queries, vector_kv, true>
                            <<<grid, 128, 0, stream>>>(window);
                    else
                        prepare_ragged_kernel<Storage, BoundedPhase, VectorWeights, TokenMajor,
                                              KvHeads, Queries, vector_kv, false>
                            <<<grid, 128, 0, stream>>>(window);
                } else
                    prepare_kernel<Storage, BoundedPhase, VectorWeights, TokenMajor, KvHeads,
                                   Queries, vector_kv, false><<<grid, 128, 0, stream>>>(window);
            }
        }
    };
    if constexpr (!Queries && KvHeads != 4)
        launch.template operator()<true>();
    else if (plan.token_major)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

template <KvCacheStorage Storage, int KvHeads, bool Queries, bool Ragged>
void dispatch_geometry(const PrepareInput<Ragged>& args, PreparePlan plan, cudaStream_t stream) {
    // For theta>=1 every phase is bounded by the nonnegative-I32 coordinate domain.
    if constexpr (!Queries) {
        if (plan.bounded_phase)
            submit<Storage, true, false, KvHeads, false, Ragged>(args, plan, stream);
        else
            submit<Storage, false, false, KvHeads, false, Ragged>(args, plan, stream);
    } else if (plan.bounded_phase) {
        if (plan.vector_weights)
            submit<Storage, true, true, KvHeads, true, Ragged>(args, plan, stream);
        else
            submit<Storage, true, false, KvHeads, true, Ragged>(args, plan, stream);
    } else {
        if (plan.vector_weights)
            submit<Storage, false, true, KvHeads, true, Ragged>(args, plan, stream);
        else
            submit<Storage, false, false, KvHeads, true, Ragged>(args, plan, stream);
    }
}

template <KvCacheStorage Storage, bool Queries, bool Ragged>
void dispatch(const PrepareInput<Ragged>& args, PreparePlan plan, cudaStream_t stream) {
    if (args.kv_heads == 1) dispatch_geometry<Storage, 1, Queries, Ragged>(args, plan, stream);

#if NINFER_TARGET_RTX5090
    else if (args.kv_heads == 4)
        dispatch_geometry<Storage, 4, Queries, Ragged>(args, plan, stream);
#endif
    else
        dispatch_geometry<Storage, 2, Queries, Ragged>(args, plan, stream);
}

template <bool Queries, bool Ragged>
void launch(const PrepareInput<Ragged>& args, PreparePlan plan, KvCacheStorage storage,
            CudaExecutionView execution) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        dispatch<KvCacheStorage::BFloat16, Queries, Ragged>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Int8Group64:
        dispatch<KvCacheStorage::Int8Group64, Queries, Ragged>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        dispatch<KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, Queries, Ragged>(args, plan,
                                                                          execution.stream());
        break;
    case KvCacheStorage::Nvfp4Group16:
        dispatch<KvCacheStorage::Nvfp4Group16, Queries, Ragged>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        dispatch<KvCacheStorage::Fp8KeyNvfp4Value, Queries, Ragged>(args, plan, execution.stream());
        break;
    default:
        throw std::invalid_argument("qsa_prepare: unsupported KV storage");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const ConstQsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                    CudaExecutionView execution) {
    auto args =
        prepare_arguments<false>({projected.key, projected.value, projected.index_key},
                                 parameters.kv, batch, rope_positions, before, cache, records);
    args.q      = static_cast<const __nv_bfloat16*>(projected.query.data);
    args.iq     = static_cast<const __nv_bfloat16*>(projected.index_query.data);
    args.wq     = static_cast<const __nv_bfloat16*>(parameters.query_norm.data);
    args.wiq    = static_cast<const __nv_bfloat16*>(parameters.index_query_norm.data);
    args.out_q  = static_cast<__nv_bfloat16*>(query.data);
    args.out_iq = static_cast<__nv_bfloat16*>(index_query.data);
    launch<true, false>(args, select_prepare_plan(args, parameters.kv.rope_theta, true),
                        cache.main.storage, execution);
}

void launch_prepare_kv(const QsaKvProjectionView& projected,
                       const QsaKvPrepareParameters& parameters, const QsaBatchView& batch,
                       const Tensor& rope_positions, const ConstQsaTailStateView& before,
                       QsaCacheView cache, QsaRawRecordWriteView records,
                       CudaExecutionView execution) {
    const auto args = prepare_arguments<false>(projected, parameters, batch, rope_positions, before,
                                               cache, records);
    launch<false, false>(args, select_prepare_plan(args, parameters.rope_theta, false),
                         cache.main.storage, execution);
}

namespace {

template <class T>
struct TailDestination {
    T* data;
    const std::int64_t* offsets;

    __device__ T* at(int segment) const {
        return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(data) + offsets[segment]);
    }
};

struct PrepareTailArguments {
    PrepareState<std::uint16_t> before_raw;
    PrepareState<int> before_rope;
    TailDestination<std::uint16_t> after_raw;
    TailDestination<int> after_rope;
    const std::uint16_t* index_key;
    const int *positions, *rope, *cu_seqlens, *seq_lengths;
    std::int64_t axis_stride;
    int axes;
};

template <class Word>
__global__ void qsa_prepare_tail_update_kernel(const __grid_constant__ PrepareTailArguments a) {
    const int s    = blockIdx.x;
    const int lane = threadIdx.x, c = lane * (sizeof(Word) / 2);
    const int seg_begin = a.cu_seqlens[s], base = a.positions[seg_begin];
    const int frontier = base + a.seq_lengths[s], r = frontier % 4, begin = frontier - r;
    Word words[3]{};
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (j < r) {
            const auto* source =
                begin + j < base
                    ? a.before_raw.at(c, j, s)
                    : a.index_key + (std::int64_t(seg_begin) + begin + j - base) * 128 + c;
            words[j] = *reinterpret_cast<const Word*>(source);
        }
    }
    int rho = 0;
    if (lane < 3 && r != 0)
        rho = begin < base ? *a.before_rope.at(lane, 0, s)
                           : a.rope[std::int64_t(seg_begin) + begin - base +
                                    a.axis_stride * (a.axes == 1 ? 0 : lane)];
    auto* raw = a.after_raw.at(s);
#pragma unroll
    for (int j = 0; j < 3; ++j) *reinterpret_cast<Word*>(raw + j * 128 + c) = words[j];
    if (lane < 3) a.after_rope.at(s)[lane] = rho;
}

bool pair_aligned(const ConstStateFieldView& field) {
    return field.field.tensor.region.alignment % 4 == 0 &&
           common::operand::aligned(common::operand::state_field_base<const std::byte>(field), 4);
}

void launch_tail_update(const QsaKvProjectionView& p, const QsaRaggedBatchView& batch,
                        const Tensor& rope, const ConstQsaTailStateView& before,
                        QsaTailStateView after, CudaExecutionView execution) {
    const auto i32  = [](const Tensor& t) { return static_cast<const int*>(t.data); };
    const auto i64  = [](const Tensor& t) { return static_cast<const std::int64_t*>(t.data); };
    const auto& seq = batch.ragged;
    PrepareTailArguments a{prepare_state<std::uint16_t>(before.raw_keys),
                           prepare_state<int>(before.first_rope),
                           {common::operand::state_field_base<std::uint16_t>(after.raw_keys),
                            i64(after.raw_keys.records.byte_offsets)},
                           {common::operand::state_field_base<int>(after.first_rope),
                            i64(after.first_rope.records.byte_offsets)},
                           static_cast<const std::uint16_t*>(p.index_key.data),
                           i32(batch.positions),
                           i32(rope),
                           i32(seq.cu_seqlens),
                           i32(seq.seq_lengths),
                           rope.nb[1] / 4,
                           rope.ne[1]};
    const bool paired = pair_aligned(before.raw_keys) && pair_aligned(after.raw_keys) &&
                        common::operand::aligned(p.index_key.data, 4);
    const int rows = seq.num_seqs();
    if (paired)
        qsa_prepare_tail_update_kernel<std::uint32_t><<<rows, 64, 0, execution.stream()>>>(a);
    else
        qsa_prepare_tail_update_kernel<std::uint16_t><<<rows, 128, 0, execution.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

RaggedPrepareArguments ragged_arguments(const QsaKvProjectionView& p,
                                        const QsaKvPrepareParameters& params,
                                        const QsaRaggedBatchView& batch, const Tensor& rope,
                                        const ConstQsaTailStateView& before, QsaCacheView cache) {
    return {prepare_arguments<true>(p, params, {batch.positions, {}, batch.table_rows}, rope,
                                    before, cache, {}),
            static_cast<const int*>(batch.ragged.token_to_seq.data),
            static_cast<const int*>(batch.ragged.cu_seqlens.data)};
}
} // namespace

void launch_prepare_update(const QsaProjectionView& p, const QsaPrepareParameters& params,
                           const QsaRaggedBatchView& batch, const Tensor& rope,
                           const ConstQsaTailStateView& before, QsaCacheView cache,
                           QsaTailStateView after, Tensor& query, Tensor& index_query,
                           CudaExecutionView execution) {
    const QsaKvProjectionView kv{p.key, p.value, p.index_key};
    auto args           = ragged_arguments(kv, params.kv, batch, rope, before, cache);
    args.q              = static_cast<const __nv_bfloat16*>(p.query.data);
    args.iq             = static_cast<const __nv_bfloat16*>(p.index_query.data);
    args.wq             = static_cast<const __nv_bfloat16*>(params.query_norm.data);
    args.wiq            = static_cast<const __nv_bfloat16*>(params.index_query_norm.data);
    args.out_q          = static_cast<__nv_bfloat16*>(query.data);
    args.out_iq         = static_cast<__nv_bfloat16*>(index_query.data);
    auto plan           = select_prepare_plan(args, params.kv.rope_theta, true);
    plan.single_segment = batch.ragged.num_seqs() == 1;
    launch<true, true>(args, plan, cache.main.storage, execution);
    launch_tail_update(kv, batch, rope, before, after, execution);
}

void launch_prepare_kv_update(const QsaKvProjectionView& p, const QsaKvPrepareParameters& params,
                              const QsaRaggedBatchView& batch, const Tensor& rope,
                              const ConstQsaTailStateView& before, QsaCacheView cache,
                              QsaTailStateView after, CudaExecutionView execution) {
    const auto args     = ragged_arguments(p, params, batch, rope, before, cache);
    auto plan           = select_prepare_plan(args, params.rope_theta, false);
    plan.single_segment = batch.ragged.num_seqs() == 1;
    launch<false, true>(args, plan, cache.main.storage, execution);
    launch_tail_update(p, batch, rope, before, after, execution);
}

} // namespace ninfer::ops::detail::qsa
