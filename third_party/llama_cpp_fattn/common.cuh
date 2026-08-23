#pragma once

// Compatibility shim standing in for llama.cpp's ggml/src/ggml-cuda/common.cuh.
//
// The vendored flash-attention kernel (see README.md) includes "common.cuh",
// which upstream is 1,669 lines pulling in ggml.h, ggml-impl.h, ggml-cuda.h and
// ggml-common.h. The device side of the kernel actually needs about a dozen
// symbols from it, none of which involve a ggml type. Providing them under the
// upstream file name means fattn-mma-f16.cuh, mma.cuh and cp-async.cuh are
// vendored byte-for-byte with no include rewriting.
//
// Every definition below is copied verbatim from llama.cpp at 62bf73d25 unless
// the comment says otherwise. Do not "improve" them: the point is that the
// vendored kernel sees exactly the semantics it was written against.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <limits>

// ---------------------------------------------------------------------------
// ggml.h / ggml-impl.h
// ---------------------------------------------------------------------------

#define GGML_ABORT(...)                                    \
    do {                                                   \
        fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);    \
        fprintf(stderr, __VA_ARGS__);                      \
        fprintf(stderr, "\n");                             \
        abort();                                           \
    } while (0)

#define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)

#define GGML_UNUSED(x) (void)(x)

#ifdef __CUDACC__
template<typename... Args>
__host__ __device__ constexpr void ggml_unused_vars_impl(Args&&...) noexcept {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)
#else
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while(0)
#endif // __CUDACC__

// mma.cuh references GGML_TYPE_MXFP4 in one tag-dispatch specialisation that no
// flash-attention configuration instantiates. Only the enumerator is needed.
enum ggml_type {
    GGML_TYPE_MXFP4 = 39,
};

// ---------------------------------------------------------------------------
// Architecture constants and availability macros
// ---------------------------------------------------------------------------

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32

#define GGML_CUDA_CC_PASCAL          600
#define GGML_CUDA_CC_DP4A            610
#define GGML_CUDA_CC_VOLTA           700
#define GGML_CUDA_CC_TURING          750
#define GGML_CUDA_CC_AMPERE          800
#define GGML_CUDA_CC_ADA_LOVELACE    890
#define GGML_CUDA_CC_HOPPER          900
#define GGML_CUDA_CC_BLACKWELL      1200

// The Volta instructions are in principle available on Turing or newer but they are effectively unusable:
#if __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#define FLASH_ATTN_AVAILABLE

// AMD_MFMA_AVAILABLE / AMD_WMMA_AVAILABLE / BLACKWELL_MMA_AVAILABLE are
// deliberately never defined: this vendoring targets NVIDIA sm_70 only.

// ---------------------------------------------------------------------------
// Host-side capability predicates
// ---------------------------------------------------------------------------

// ninfer: upstream resolves this against __CUDA_ARCH_LIST__. This vendoring is
// compiled for exactly one architecture, so the identity is correct here.
static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}

static bool volta_mma_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
}

static bool turing_mma_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

static bool ampere_mma_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

static bool cp_async_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

static bool amd_mfma_available(const int /*cc*/) { return false; }
static bool amd_wmma_available(const int /*cc*/) { return false; }

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
    return 32;
}

static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
}

static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. Compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

// PDL is Hopper+; on sm_70 both of these are inert, matching upstream's guards.
static __device__ __forceinline__ void ggml_cuda_pdl_sync() {}
static __device__ __forceinline__ void ggml_cuda_pdl_lc() {}

// From fattn-common.cuh:9,11,19 -- the constants the kernel body uses. The rest
// of that header is quantized-KV dot products and the ggml launcher, neither of
// which this vendoring needs.
#define FATTN_KQ_STRIDE       256
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return powf(base, exph);
}

#define GGML_CUDA_RESTRICT __restrict__

// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// ---------------------------------------------------------------------------
// Fast integer division (the kernel takes pre-computed uint3 magic values)
// ---------------------------------------------------------------------------

// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}
