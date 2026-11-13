// CausalD256H12Kv2: cached entry of the INT8 attention dispatch (launch_impl.cuh).
#include "ops/softmax_attention/dense/causal_cache/int8/launch_impl.cuh"

namespace ninfer::ops::detail::int8_kv_launch {

template void grouped_instance<CausalD256H12Kv2, CausalCachedInput, false>(
    const CausalAttentionOperands&, Int8KvCacheView<false>, CausalCachedInput, CausalKvPartition,
    CausalPartialView, cudaStream_t);

} // namespace ninfer::ops::detail::int8_kv_launch
