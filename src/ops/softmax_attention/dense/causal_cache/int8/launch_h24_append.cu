// CausalD256H24Kv4: append entry of the INT8 attention dispatch (launch_impl.cuh).
#include "ops/softmax_attention/dense/causal_cache/int8/launch_impl.cuh"

namespace ninfer::ops::detail::int8_kv_launch {

template void grouped_instance<CausalD256H24Kv4, CausalAppendInput, true>(
    const CausalAttentionOperands&, Int8KvCacheView<true>, CausalAppendInput, CausalKvPartition,
    CausalPartialView, cudaStream_t);

} // namespace ninfer::ops::detail::int8_kv_launch
