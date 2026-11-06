#pragma once

#include "ops/common/cuda/memory.cuh"

namespace ninfer::ops::detail::mhc {
// Barriers live in the consumer CTA. Producers arrive remotely after writing
// the consumer's shared data; consumers wait locally before reading that data.
// The enclosing cluster barriers own initialization and shared-memory lifetime.
using ClusterBarrier = unsigned long long;

__device__ __forceinline__ void init_cluster_barrier(ClusterBarrier* barrier, unsigned arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(common::smem_addr(barrier)),
                 "r"(arrivals)
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void arrive_cluster_barrier(ClusterBarrier* barrier) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(
                     common::smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ void wait_cluster_barrier(ClusterBarrier* barrier, unsigned phase) {
    unsigned done;
    do {
        asm volatile("{ .reg .pred p; "
                     "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; "
                     "selp.u32 %0, 1, 0, p; }"
                     : "=r"(done)
                     : "r"(common::smem_addr(barrier)), "r"(phase)
                     : "memory");
    } while (!done);
}
} // namespace ninfer::ops::detail::mhc
