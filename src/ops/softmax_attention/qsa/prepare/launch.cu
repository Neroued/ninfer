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
PreparePage<T> prepare_page(const Tensor& t) {
    return {static_cast<T*>(t.data), t.nb[1] / std::int64_t(sizeof(T)),
            t.nb[2] / std::int64_t(sizeof(T)), t.nb[3] / std::int64_t(sizeof(T))};
}

PrepareArguments prepare_arguments(const QsaProjectionView& p, const QsaPrepareParameters& params,
                                   const QsaBatchView& batch, const Tensor& rope,
                                   const QsaTailStateView& before, QsaCacheView cache,
                                   QsaRawRecordWriteView records, Tensor& q, Tensor& iq) {
    const auto bf  = [](const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); };
    const auto i32 = [](const Tensor& t) { return static_cast<const std::int32_t*>(t.data); };
    PrepareArguments a{bf(p.query),
                       bf(p.key),
                       bf(p.value),
                       bf(p.index_query),
                       bf(p.index_key),
                       bf(params.query_norm),
                       bf(params.key_norm),
                       bf(params.index_query_norm),
                       bf(params.index_key_norm),
                       static_cast<__nv_bfloat16*>(q.data),
                       static_cast<__nv_bfloat16*>(iq.data),
                       i32(batch.positions),
                       i32(rope),
                       i32(batch.valid_columns),
                       i32(batch.table_rows),
                       i32(before.slots),
                       i32(records.offsets),
                       i32(cache.main.block_tables),
                       cache.main.block_tables.nb[1] / 4,
                       prepare_columns<const __nv_bfloat16>(before.raw_keys),
                       prepare_columns<const std::int32_t>(before.first_rope),
                       prepare_columns<__nv_bfloat16>(records.raw_keys),
                       prepare_columns<std::int32_t>(records.rope_positions),
                       prepare_page<__nv_bfloat16>(cache.index.keys),
                       prepare_page<std::uint8_t>(cache.main.k_pages),
                       prepare_page<std::uint8_t>(cache.main.v_pages),
                       prepare_page<std::uint8_t>(cache.main.k_scale_pages),
                       prepare_page<std::uint8_t>(cache.main.v_scale_pages),
                       p.query.ne[2],
                       p.query.ne[3],
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

PreparePlan select_prepare_plan(const PrepareArguments& args, float theta) {
    const auto columns     = std::int64_t(args.tokens) * args.batch;
    const bool token_major = NINFER_TARGET_SPARK
                                 ? args.kv_heads == 2 || columns < 256 || columns > 1024
                                 : columns <= 320;
    return {token_major, theta >= 1,
            common::operand::aligned(args.wq, 16) && common::operand::aligned(args.wiq, 8)};
}

template <KvCacheStorage Storage, bool BoundedPhase, bool VectorWeights, int KvHeads>
void submit(const PrepareArguments& args, PreparePlan plan, cudaStream_t stream) {
    constexpr int groups = PrepareGeometry<KvHeads>::groups;
    if (plan.token_major)
        prepare_kernel<Storage, BoundedPhase, VectorWeights, true, KvHeads>
            <<<dim3(args.tokens * groups, args.batch), 128, 0, stream>>>(args);
    else
        prepare_kernel<Storage, BoundedPhase, VectorWeights, false, KvHeads>
            <<<dim3(args.tokens, args.batch, groups), 128, 0, stream>>>(args);
}

template <KvCacheStorage Storage, int KvHeads>
void dispatch_geometry(const PrepareArguments& args, PreparePlan plan, cudaStream_t stream) {
    // For theta>=1 every phase is bounded by the nonnegative-I32 coordinate domain.
    if (plan.bounded_phase) {
        if (plan.vector_weights)
            submit<Storage, true, true, KvHeads>(args, plan, stream);
        else
            submit<Storage, true, false, KvHeads>(args, plan, stream);
    } else {
        if (plan.vector_weights)
            submit<Storage, false, true, KvHeads>(args, plan, stream);
        else
            submit<Storage, false, false, KvHeads>(args, plan, stream);
    }
}

template <KvCacheStorage Storage>
void dispatch(const PrepareArguments& args, PreparePlan plan, cudaStream_t stream) {
    if (args.kv_heads == 1)
        dispatch_geometry<Storage, 1>(args, plan, stream);
    else
        dispatch_geometry<Storage, 2>(args, plan, stream);
}

} // namespace

void launch_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const QsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                    CudaExecutionView execution) {
    const auto args = prepare_arguments(projected, parameters, batch, rope_positions, before, cache,
                                        records, query, index_query);
    const auto plan = select_prepare_plan(args, parameters.rope_theta);
    switch (cache.main.storage) {
    case KvCacheStorage::BFloat16:
        dispatch<KvCacheStorage::BFloat16>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Int8Group64:
        dispatch<KvCacheStorage::Int8Group64>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        dispatch<KvCacheStorage::Fp8E4M3Row256>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Nvfp4Group16:
        dispatch<KvCacheStorage::Nvfp4Group16>(args, plan, execution.stream());
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        dispatch<KvCacheStorage::Fp8KeyNvfp4Value>(args, plan, execution.stream());
        break;
    default:
        throw std::invalid_argument("qsa_prepare: unsupported KV storage");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::qsa
