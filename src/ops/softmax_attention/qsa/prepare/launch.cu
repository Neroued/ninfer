#include "ops/softmax_attention/qsa/prepare/launch.h"
#include "ops/softmax_attention/qsa/prepare/kernel.cuh"

#include "ops/common/host/operands.h"
#include "core/platform.h"

#include <array>
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

template <KvCacheStorage Storage, bool BoundedPhase, bool VectorWeights, int KvHeads, bool Queries>
void submit(const PrepareArguments& args, PreparePlan plan, cudaStream_t stream) {
    constexpr int groups = PrepareGeometry<KvHeads, Queries>::groups;
    const auto launch    = [&]<bool TokenMajor>() {
        const dim3 grid = TokenMajor ? dim3(args.tokens * groups, args.batch)
                                        : dim3(args.tokens, args.batch, groups);
        prepare_kernel<Storage, BoundedPhase, VectorWeights, TokenMajor, KvHeads, Queries>
            <<<grid, 128, 0, stream>>>(args);
    };
    if constexpr (!Queries && KvHeads != 4)
        launch.template operator()<true>();
    else if (plan.token_major)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

template <KvCacheStorage Storage, int KvHeads, bool Queries>
void dispatch_geometry(const PrepareArguments& args, PreparePlan plan, cudaStream_t stream) {
    // For theta>=1 every phase is bounded by the nonnegative-I32 coordinate domain.
    if constexpr (!Queries) {
        if (plan.bounded_phase)
            submit<Storage, true, false, KvHeads, false>(args, plan, stream);
        else
            submit<Storage, false, false, KvHeads, false>(args, plan, stream);
    } else if (plan.bounded_phase) {
        if (plan.vector_weights)
            submit<Storage, true, true, KvHeads, true>(args, plan, stream);
        else
            submit<Storage, true, false, KvHeads, true>(args, plan, stream);
    } else {
        if (plan.vector_weights)
            submit<Storage, false, true, KvHeads, true>(args, plan, stream);
        else
            submit<Storage, false, false, KvHeads, true>(args, plan, stream);
    }
}

template <KvCacheStorage Storage, bool Queries>
void dispatch(const PrepareArguments& args, PreparePlan plan, cudaStream_t stream) {
    if (args.kv_heads == 1) dispatch_geometry<Storage, 1, Queries>(args, plan, stream);

#if NINFER_TARGET_RTX5090
    else if (args.kv_heads == 4) {
        if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256 ||
                      Storage == KvCacheStorage::BFloat16)
            dispatch_geometry<Storage, 4, Queries>(args, plan, stream);
    }
#endif
    else
        dispatch_geometry<Storage, 2, Queries>(args, plan, stream);
}

template <bool Queries>
void launch(const PrepareArguments& args, PreparePlan plan, KvCacheStorage storage,
            CudaExecutionView execution) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        dispatch<KvCacheStorage::BFloat16, Queries>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Int8Group64:
        dispatch<KvCacheStorage::Int8Group64, Queries>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        dispatch<KvCacheStorage::Fp8E4M3Row256, Queries>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Nvfp4Group16:
        dispatch<KvCacheStorage::Nvfp4Group16, Queries>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        dispatch<KvCacheStorage::Fp8KeyNvfp4Value, Queries>(args, plan, execution.stream());
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
    auto args   = prepare_arguments({projected.key, projected.value, projected.index_key},
                                    parameters.kv, batch, rope_positions, before, cache, records);
    args.q      = static_cast<const __nv_bfloat16*>(projected.query.data);
    args.iq     = static_cast<const __nv_bfloat16*>(projected.index_query.data);
    args.wq     = static_cast<const __nv_bfloat16*>(parameters.query_norm.data);
    args.wiq    = static_cast<const __nv_bfloat16*>(parameters.index_query_norm.data);
    args.out_q  = static_cast<__nv_bfloat16*>(query.data);
    args.out_iq = static_cast<__nv_bfloat16*>(index_query.data);
    launch<true>(args, select_prepare_plan(args, parameters.kv.rope_theta, true),
                 cache.main.storage, execution);
}

void launch_prepare_kv(const QsaKvProjectionView& projected,
                       const QsaKvPrepareParameters& parameters, const QsaBatchView& batch,
                       const Tensor& rope_positions, const ConstQsaTailStateView& before,
                       QsaCacheView cache, QsaRawRecordWriteView records,
                       CudaExecutionView execution) {
    const auto args =
        prepare_arguments(projected, parameters, batch, rope_positions, before, cache, records);
    launch<false>(args, select_prepare_plan(args, parameters.rope_theta, false), cache.main.storage,
                  execution);
}

} // namespace ninfer::ops::detail::qsa
