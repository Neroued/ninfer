#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {
enum class Cache { ca, cg };
}

namespace ninfer::ops::common {

__device__ __forceinline__ void prefetch_l2(const void* data, std::size_t bytes,
                                            std::size_t offset) {
    if (data && offset < bytes) {
        const auto* address = static_cast<const char*>(data) + offset;
        asm volatile("prefetch.global.L2 [%0];" ::"l"(address));
    }
}

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

__device__ __forceinline__ uint4 streaming_load(const uint4* input) {
    uint4 value;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(input));
    return value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ std::uint64_t l2_streaming_policy() {
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;"
                 : "=l"(policy));
    return policy;
}

__device__ __forceinline__ std::uint64_t l2_retained_policy() {
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %0, 1.0;"
                 : "=l"(policy));
    return policy;
}

// Permute eight-element vectors within each 64-element row sector.
__device__ __forceinline__ int swizzle_xor_64(int row, int col) { return col ^ ((row & 7) << 3); }

// Permute sixteen-element vectors within each 128-element row sector.
__device__ __forceinline__ int swizzle_xor_128(int row, int col) { return col ^ ((row & 7) << 4); }

// Compose an MMA shared-memory lane base with precomputed column/matrix and row XOR bits.
__device__ __forceinline__ unsigned swizzle_xor_address(unsigned base, unsigned column,
                                                        unsigned matrix, unsigned row) {
    return base + ((column | matrix) ^ row);
}

// Storage and launch must select the same static/dynamic representation.
template <int Bytes>
__device__ __forceinline__ unsigned char* shared_storage() {
    static_assert(Bytes > 0 && Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char dynamic_storage[];
        return dynamic_storage;
    } else {
        __shared__ __align__(16) unsigned char static_storage[Bytes];
        return static_storage;
    }
}

} // namespace ninfer::ops::common
