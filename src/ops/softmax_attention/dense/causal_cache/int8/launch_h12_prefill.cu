// CausalD256H12Kv2: prefill entry of the INT8 attention dispatch (launch_impl.cuh).
#include "ops/softmax_attention/dense/causal_cache/int8/launch_impl.cuh"

namespace ninfer::ops::detail::int8_kv_launch {

template void parallel_grouped<CausalD256H12Kv2>(const CausalAttentionOperands&, Int8KvReadView,
                                                 CausalKvPartition, CausalPartialView,
                                                 cudaStream_t);
template void tiled<CausalD256H12Kv2>(const CausalAttentionOperands&, Int8KvReadView, cudaStream_t);

} // namespace ninfer::ops::detail::int8_kv_launch
