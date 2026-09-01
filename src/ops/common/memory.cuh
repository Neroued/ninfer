#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// cp.async (and commit_group/wait_group) is Ampere+ only (sm_80). Volta has
// no async-copy hardware, so below sm_80 every op here decomposes into a
// synchronous vectorized load+store: cp_async becomes an ordinary copy, and
// cp_commit()/cp_wait() become no-ops since the data has already landed by
// the time cp_async() returns. See docs/v100.md.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

#else // __CUDA_ARCH__ < 800

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Bytes == 4) {
        *static_cast<unsigned*>(smem_dst) = *static_cast<const unsigned*>(gmem_src);
    } else if constexpr (Bytes == 8) {
        *static_cast<uint2*>(smem_dst) = *static_cast<const uint2*>(gmem_src);
    } else {
        *static_cast<uint4*>(smem_dst) = *static_cast<const uint4*>(gmem_src);
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    auto* dst       = static_cast<unsigned char*>(smem_dst);
    const auto* src = static_cast<const unsigned char*>(gmem_src);
#pragma unroll
    for (int i = 0; i < Bytes; ++i) { dst[i] = (i < src_bytes) ? src[i] : static_cast<unsigned char>(0); }
}

__device__ __forceinline__ void cp_commit() {}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
}

#endif // __CUDA_ARCH__ >= 800

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninfer::ops
