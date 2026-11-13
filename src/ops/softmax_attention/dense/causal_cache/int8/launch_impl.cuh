#pragma once
// Template bodies of the INT8 causal attention dispatch. Our split; upstream keeps them in
// launch.cu, where every geometry, token width, batch/mask variant and kernel family was one
// translation unit that nvcc compiles on one thread (3-5 min after any kernel change). Each
// (geometry, entry) pair is instantiated in its own unit
// (launch_<geometry>_{append,cached,prefill}.cu) and launch.cu only dispatches.
#include "ops/softmax_attention/dense/causal_cache/int8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/template_launch.cuh"
#include <stdexcept>

namespace ninfer::ops::detail::int8_kv_launch {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache, Input input,
             CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G, class Input, bool Writable>
void grouped_instance(const CausalAttentionOperands& p, Int8KvCacheView<Writable> cache,
                      Input input, CausalKvPartition partition, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (p.width) {
#define NINFER_INT8_GROUPED(T)                                                                     \
    case T:                                                                                        \
        return grouped<G, T>(p, cache, input, partition, partial, stream)
        NINFER_INT8_GROUPED(1);
        NINFER_INT8_GROUPED(2);
        NINFER_INT8_GROUPED(3);
        NINFER_INT8_GROUPED(4);
        NINFER_INT8_GROUPED(5);
        NINFER_INT8_GROUPED(6);
        NINFER_INT8_GROUPED(7);
        NINFER_INT8_GROUPED(8);
#undef NINFER_INT8_GROUPED
    }
    throw std::logic_error("INT8 grouped plan exceeds the selected token tile");
}

template <class G>
void parallel_grouped(const CausalAttentionOperands& p, Int8KvReadView cache,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Int8KvGroupedInstance<G, Int8KvCausalPlan::kTokenTile>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_int8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                   CausalCachedInput, true>(p, cache, {}, partition, partial,
                                                            stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G>
void tiled(const CausalAttentionOperands& p, Int8KvReadView cache, cudaStream_t stream) {
    launch_int8_kv_tiled_mma<G, Int8KvTiledInstance>(p, cache, stream);
}

// Instantiated only in the per-geometry units.
#define NINFER_INT8_KV_LAUNCH_UNITS(G, EXTERN)                                                     \
    EXTERN template void grouped_instance<G, CausalAppendInput, true>(                             \
        const CausalAttentionOperands&, Int8KvCacheView<true>, CausalAppendInput,                  \
        CausalKvPartition, CausalPartialView, cudaStream_t);                                       \
    EXTERN template void grouped_instance<G, CausalCachedInput, false>(                            \
        const CausalAttentionOperands&, Int8KvCacheView<false>, CausalCachedInput,                 \
        CausalKvPartition, CausalPartialView, cudaStream_t);                                       \
    EXTERN template void parallel_grouped<G>(const CausalAttentionOperands&, Int8KvReadView,       \
                                             CausalKvPartition, CausalPartialView, cudaStream_t);  \
    EXTERN template void tiled<G>(const CausalAttentionOperands&, Int8KvReadView, cudaStream_t);

NINFER_INT8_KV_LAUNCH_UNITS(CausalD256H24Kv4, extern)
NINFER_INT8_KV_LAUNCH_UNITS(CausalD256H12Kv2, extern)
NINFER_INT8_KV_LAUNCH_UNITS(CausalD256H16Kv2, extern)

} // namespace ninfer::ops::detail::int8_kv_launch
