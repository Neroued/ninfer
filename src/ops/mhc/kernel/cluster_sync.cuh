#pragma once

#include "ops/common/cuda/memory.cuh"

namespace ninfer::ops::detail::mhc {
// Invocation-local flags publish shared data to other CTAs in the same cluster.
// The enclosing kernels own initialization and the cluster lifetime barriers.
__device__ __forceinline__ void publish_cluster_value(unsigned* ready) {
    asm volatile("st.release.cluster.shared::cta.u32 [%0], 1;" ::"r"(common::smem_addr(ready))
                 : "memory");
}

__device__ __forceinline__ void wait_cluster_value(const unsigned* ready) {
    unsigned seen;
    do {
        asm volatile("ld.acquire.cluster.shared::cluster.u32 %0, [%1];"
                     : "=r"(seen)
                     : "r"(common::smem_addr(ready))
                     : "memory");
    } while (!seen);
}
} // namespace ninfer::ops::detail::mhc
