// Implements: ops/linear/q8/q8_a8.h
#include "ops/linear/q8/q8_a8.h"

#include "core/device.h"
#include "ops/launcher/kernel_attr_once.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <string_view>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// ---- activation quantization
// ---------------------------------------------------------------------------------- One warp per
// (token, 128 values); lane l holds values 4l..4l+3. A G-value group is G / 4 lanes. Tokens past
// `tokens` (up to the padded width) are written as zero codes with scale 0.
template <int G>
__global__ void __launch_bounds__(256)
    q8_a8_quantize_kernel(const __nv_bfloat16* __restrict__ x, int tokens, int padded, int k,
                          std::int8_t* __restrict__ xq, float* __restrict__ sx) {
    constexpr int kLanes = G / 4;
    const int chunks     = k / 128;
    const std::int64_t item =
        static_cast<std::int64_t>(blockIdx.x) * 8 + static_cast<std::int64_t>(threadIdx.x >> 5);
    if (item >= static_cast<std::int64_t>(padded) * chunks) return;
    const int t           = static_cast<int>(item / chunks);
    const int chunk       = static_cast<int>(item - static_cast<std::int64_t>(t) * chunks);
    const int lane        = static_cast<int>(threadIdx.x & 31U);
    const std::int64_t at = static_cast<std::int64_t>(t) * k + chunk * 128 + lane * 4;
    float v[4]            = {0.0F, 0.0F, 0.0F, 0.0F};
    if (t < tokens) {
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(x + at));
        v[0]            = __uint_as_float(raw.x << 16);
        v[1]            = __uint_as_float(raw.x & 0xFFFF0000U);
        v[2]            = __uint_as_float(raw.y << 16);
        v[3]            = __uint_as_float(raw.y & 0xFFFF0000U);
    }
    // amax as the largest |bits|: a NaN (above Inf) wins, so it cannot hide behind fmaxf.
    unsigned amax_bits = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) amax_bits = max(amax_bits, __float_as_uint(v[i]) & 0x7FFFFFFFU);
#pragma unroll
    for (int off = 1; off < kLanes; off <<= 1)
        amax_bits = max(amax_bits, __shfl_xor_sync(0xFFFFFFFFU, amax_bits, off));
    float scale          = 0.0F;
    std::uint32_t packed = 0;
    if (amax_bits >= 0x7F800000U) {
        scale = __uint_as_float(0x7FC00000U);
    } else if (amax_bits >= 0x00800000U) {
        const float amax = __uint_as_float(amax_bits);
        const float inv  = __fdiv_rn(127.0F, amax);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = max(-127, min(127, __float2int_rn(__fmul_rn(v[i], inv))));
            packed |= (static_cast<std::uint32_t>(q) & 0xFFU) << (8 * i);
        }
        scale = __fdiv_rn(amax, 127.0F);
    }
    *reinterpret_cast<std::uint32_t*>(xq + at) = packed;
    if ((lane & (kLanes - 1)) == 0) {
        const int group                                   = (chunk * 128 + lane * 4) / G;
        sx[static_cast<std::int64_t>(group) * padded + t] = scale;
    }
}

// ---- GEMM
// -----------------------------------------------------------------------------------------------------
// CTA 128 rows x 128 tokens, 8 warps (4 row warps x 2 token warps, 32 x 64 each), K in stages of 64
// (two weight groups) through a 4-stage cp.async ring; rows padded to 80 bytes (conflict-free
// ldmatrix). One MMA m16n8k32 per fragment and weight group, converted and scaled right after it.
// Prototype on the RTX 5070 Ti, 8192 x 2560, T = 8192: G = 32 157.5 TOPS, G = 128 188.7 TOPS (int32
// over all K, no scales: 219).
constexpr int kBm = 128, kBn = 128, kBk = 64, kStages = 4, kPitch = 80, kThreads = 256;

struct __align__(16) Q8A8Stage {
    signed char a[kBm * kPitch];
    signed char b[kBn * kPitch];
    __half sw[kBm * 2];
    float sx[2 * kBn];
};

constexpr int kSmem = kStages * static_cast<int>(sizeof(Q8A8Stage));
static_assert(kSmem <= 99 * 1024, "the A8 ring must fit the 99 KiB opt-in limit");
static_assert(kBn * (kBm + 4) * 4 <= kSmem, "the FP32 output tile reuses the ring");

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                      unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_s8_16832(int (&c)[4], const unsigned (&a)[4], unsigned b0,
                                             unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <class Out>
__device__ __forceinline__ Out q8_a8_store_value(float v);

template <>
__device__ __forceinline__ __nv_bfloat16 q8_a8_store_value<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <>
__device__ __forceinline__ float q8_a8_store_value<float>(float v) {
    return v;
}

// codes [M][K] int8, ws [M][K / 32] binary16, xq [padded][K] int8, sx [K / G][padded] FP32, out
// [T][M] (the piece's first token at out). grid (padded / 128, M / 128).
template <int G, class Out>
__global__ void __launch_bounds__(kThreads, 1)
    q8_a8_gemm_kernel(const signed char* __restrict__ codes, const __half* __restrict__ ws,
                      const signed char* __restrict__ xq, const float* __restrict__ sx,
                      Out* __restrict__ out, int m, int k, int tokens, int padded) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto* st       = reinterpret_cast<Q8A8Stage*>(smem);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31, gid = lane >> 2, lid = lane & 3;
    const int wr = warp >> 1, wt = warp & 1;
    const int m0 = static_cast<int>(blockIdx.y) * kBm, t0 = static_cast<int>(blockIdx.x) * kBn;
    const int wgroups = k / 32, stages_k = k / kBk;
    constexpr int kSxRows = G == 32 ? 2 : 1; // activation-scale rows per 64-value stage
    auto load             = [&](int s, int kt) {
        Q8A8Stage& S = st[s];
        const int k0 = kt * kBk;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * kThreads; // 512 chunks of 16 B: row c / 4, part c % 4
            const int r = c >> 2, p = c & 3;
            cp16(&S.a[r * kPitch + p * 16],
                 codes + static_cast<std::int64_t>(m0 + r) * k + k0 + p * 16);
            cp16(&S.b[r * kPitch + p * 16],
                 xq + static_cast<std::int64_t>(t0 + r) * k + k0 + p * 16);
        }
        if (tid < kBm)
            cp4(&S.sw[tid * 2], ws + static_cast<std::int64_t>(m0 + tid) * wgroups + kt * 2);
        if (tid < 32 * kSxRows) {
            const int row = tid >> 5, q = tid & 31;
            const int g = G == 32 ? kt * 2 + row : (kt * kBk) / 128;
            cp16(&S.sx[row * kBn + q * 4], sx + static_cast<std::int64_t>(g) * padded + t0 + q * 4);
        }
    };
    float acc[2][8][4], tmp[2][8][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 8; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[a][b][e] = 0.0F;
                tmp[a][b][e] = 0.0F;
            }
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < stages_k) load(s, s);
        cp_commit();
    }
    for (int kt = 0; kt < stages_k; ++kt) {
        cp_wait<kStages - 2>();
        __syncthreads();
        const int nk = kt + kStages - 1;
        if (nk < stages_k) load(nk % kStages, nk);
        cp_commit();
        const Q8A8Stage& S = st[kt % kStages];
#pragma unroll
        for (int gi = 0; gi < 2; ++gi) {
            unsigned af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row = wr * 32 + mi * 16 + (lane & 15);
                ldsm4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                      smem_u32(&S.a[row * kPitch + gi * 32 + (lane >> 4) * 16]));
            }
            float swr[2][2], sxt[8][2];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wr * 32 + mi * 16 + gid;
                swr[mi][0]  = __half2float(S.sw[r * 2 + gi]);
                swr[mi][1]  = __half2float(S.sw[(r + 8) * 2 + gi]);
            }
            const int sxrow = G == 32 ? gi : 0;
#pragma unroll
            for (int ni = 0; ni < 8; ++ni) {
                const int t = wt * 64 + ni * 8 + 2 * lid;
                sxt[ni][0]  = S.sx[sxrow * kBn + t];
                sxt[ni][1]  = S.sx[sxrow * kBn + t + 1];
            }
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                unsigned b0, b1, b2, b3;
                const int tok = wt * 64 + nj * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm4(b0, b1, b2, b3,
                      smem_u32(&S.b[tok * kPitch + gi * 32 + ((lane >> 3) & 1) * 16]));
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        int c4[4] = {0, 0, 0, 0};
                        mma_s8_16832(c4, af[mi], h ? b2 : b0, h ? b3 : b1);
                        const int ni = 2 * nj + h;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            if constexpr (G == 32) {
                                const float p = __fmul_rn(swr[mi][e >> 1], sxt[ni][e & 1]);
                                acc[mi][ni][e] =
                                    __fmaf_rn(p, static_cast<float>(c4[e]), acc[mi][ni][e]);
                            } else {
                                tmp[mi][ni][e] = __fmaf_rn(static_cast<float>(c4[e]),
                                                           swr[mi][e >> 1], tmp[mi][ni][e]);
                            }
                        }
                    }
                }
            }
            if constexpr (G == 128) {
                if (((kt * 2 + gi) & 3) == 3) {
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                        for (int ni = 0; ni < 8; ++ni)
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                acc[mi][ni][e] =
                                    __fmaf_rn(tmp[mi][ni][e], sxt[ni][e & 1], acc[mi][ni][e]);
                                tmp[mi][ni][e] = 0.0F;
                            }
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
    // The output tile as [token][row] in shared memory, then 16-byte stores of whole rows per
    // token.
    constexpr int kTp = kBm + 16 / static_cast<int>(sizeof(Out));
    Out* tile         = reinterpret_cast<Out*>(smem);
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 8; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r       = wr * 32 + mi * 16 + gid + (e >> 1) * 8;
                const int t       = wt * 64 + ni * 8 + 2 * lid + (e & 1);
                tile[t * kTp + r] = q8_a8_store_value<Out>(acc[mi][ni][e]);
            }
    __syncthreads();
    constexpr int kPer = 16 / static_cast<int>(sizeof(Out));
    for (int c = tid; c < kBn * (kBm / kPer); c += kThreads) {
        const int t = c / (kBm / kPer), p = c % (kBm / kPer);
        if (t0 + t >= tokens) continue;
        *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(t0 + t) * m + m0 + p * kPer) =
            *reinterpret_cast<const uint4*>(&tile[t * kTp + p * kPer]);
    }
}

std::int32_t round_up(std::int32_t v, std::int32_t to) { return (v + to - 1) / to * to; }

std::atomic<int> g_forced{-1};

template <int G, class Out>
void launch_piece(const __nv_bfloat16* x, const Weight& w, Out* out, std::int32_t tokens,
                  std::int8_t* xq, float* sx, cudaStream_t stream) {
    const std::int32_t k = w.k, padded = round_up(tokens, kBn);
    const std::int64_t items = static_cast<std::int64_t>(padded) * (k / 128);
    q8_a8_quantize_kernel<G>
        <<<static_cast<unsigned>((items + 7) / 8), 256, 0, stream>>>(x, tokens, padded, k, xq, sx);
    CUDA_CHECK(cudaGetLastError());
    static FuncAttrPerDevice attr;
    attr.ensure(q8_a8_gemm_kernel<G, Out>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    const dim3 grid(static_cast<unsigned>(padded / kBn), static_cast<unsigned>(w.n / kBm));
    q8_a8_gemm_kernel<G, Out><<<grid, kThreads, kSmem, stream>>>(
        static_cast<const signed char*>(w.qdata), static_cast<const __half*>(w.scales), xq, sx, out,
        w.n, k, tokens, padded);
    CUDA_CHECK(cudaGetLastError());
}

template <int G>
void run_mode(const Tensor& x, const Weight& w, Tensor& out, void* workspace, cudaStream_t stream) {
    const std::int32_t k = w.k, t = x.ne[1];
    const std::int32_t piece_cap = round_up(std::min(t, kQ8A8PieceTokens), kBn);
    auto* xq                     = static_cast<std::int8_t*>(workspace);
    auto* sx                     = reinterpret_cast<float*>(static_cast<unsigned char*>(workspace) +
                                                            static_cast<std::size_t>(piece_cap) * k);
    for (std::int32_t p0 = 0; p0 < t; p0 += kQ8A8PieceTokens) {
        const std::int32_t n = std::min(kQ8A8PieceTokens, t - p0);
        const auto* xp =
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(p0) * k;
        if (out.dtype == DType::FP32) {
            launch_piece<G, float>(
                xp, w, static_cast<float*>(out.data) + static_cast<std::int64_t>(p0) * w.n, n, xq,
                sx, stream);
        } else {
            launch_piece<G, __nv_bfloat16>(
                xp, w, static_cast<__nv_bfloat16*>(out.data) + static_cast<std::int64_t>(p0) * w.n,
                n, xq, sx, stream);
        }
    }
}

void warn_once(std::atomic<bool>& flag, const char* why) {
    if (!flag.exchange(true)) {
        std::fprintf(stderr, "q8 A8: %s; these calls take the A16 tiles\n", why);
    }
}

} // namespace

Q8A8Mode q8_a8_mode() {
    const int forced = g_forced.load(std::memory_order_relaxed);
    if (forced >= 0) return static_cast<Q8A8Mode>(forced);
    static const Q8A8Mode mode = [] {
        const char* v = std::getenv("NINFER_Q8_A8");
        if (v == nullptr || v[0] == '\0') return Q8A8Mode::Off;
        const std::string_view s(v);
        if (s == "128") return Q8A8Mode::G128;
        if (s == "32" || s == "1") return Q8A8Mode::G32;
        if (s != "0") {
            std::fprintf(stderr, "NINFER_Q8_A8=%s not recognised (32, 128 or 0): A8 stays off\n",
                         v);
        }
        return Q8A8Mode::Off;
    }();
    return mode;
}

void q8_a8_force_mode_for_test(Q8A8Mode mode) {
    g_forced.store(static_cast<int>(mode), std::memory_order_relaxed);
}

bool q8_a8_admitted(std::int32_t n, std::int32_t k) {
    return (k == 2560 && (n == 8192 || n == 6656 || n == 1280)) || (n == 2560 && k == 3072);
}

std::size_t q8_a8_workspace_bytes(std::int32_t k, std::int32_t tokens, Q8A8Mode mode) {
    if (mode == Q8A8Mode::Off || k <= 0 || tokens <= 0) return 0;
    const auto piece = static_cast<std::size_t>(round_up(std::min(tokens, kQ8A8PieceTokens), kBn));
    return piece * static_cast<std::size_t>(k) +
           static_cast<std::size_t>(k / static_cast<int>(mode)) * piece * 4;
}

void q8_a8_run(const Tensor& x, const Weight& w, Tensor& out, Q8A8Mode mode, void* workspace,
               std::size_t bytes, cudaStream_t stream) {
    if (mode == Q8A8Mode::Off) throw std::invalid_argument("q8 A8: mode Off");
    if (w.qtype != QType::Q8_G32_FP16 || w.n % kBm != 0 || w.k % 128 != 0 ||
        w.padded_shape[1] != w.k) {
        throw std::invalid_argument(
            "q8 A8: needs Q8_G32 with N % 128 == 0 and K % 128 == 0, unpadded");
    }
    if (x.dtype != DType::BF16 || x.ne[0] != w.k || x.ne[1] <= 0 || out.ne[0] != w.n ||
        out.ne[1] != x.ne[1] || (out.dtype != DType::BF16 && out.dtype != DType::FP32)) {
        throw std::invalid_argument("q8 A8: expected BF16 x [K, T] -> BF16 or FP32 out [N, T]");
    }
    if (workspace == nullptr || reinterpret_cast<std::uintptr_t>(workspace) % 256 != 0 ||
        bytes < q8_a8_workspace_bytes(w.k, x.ne[1], mode)) {
        throw std::invalid_argument("q8 A8: workspace too small or misaligned");
    }
    if (mode == Q8A8Mode::G32) {
        run_mode<32>(x, w, out, workspace, stream);
    } else {
        run_mode<128>(x, w, out, workspace, stream);
    }
}

bool q8_a8_try_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena* arena,
                        cudaStream_t stream) {
    const Q8A8Mode mode = q8_a8_mode();
    if (mode == Q8A8Mode::Off || x.ne[1] < kQ8A8MinTokens || !q8_a8_admitted(w.n, w.k) ||
        w.padded_shape[1] != w.k || (out.dtype != DType::BF16 && out.dtype != DType::FP32)) {
        return false;
    }
    // The kernel reads the codes with 16-byte and the scales with 4-byte cp.async.
    if ((reinterpret_cast<std::uintptr_t>(w.qdata) & 15U) != 0 ||
        (reinterpret_cast<std::uintptr_t>(w.scales) & 3U) != 0) {
        static std::atomic<bool> misaligned{false};
        warn_once(misaligned, "misaligned weight planes");
        return false;
    }
    static std::atomic<bool> no_arena{false}, small_arena{false};
    if (arena == nullptr) {
        warn_once(no_arena, "no workspace");
        return false;
    }
    const std::size_t bytes = q8_a8_workspace_bytes(w.k, x.ne[1], mode);
    const auto at           = reinterpret_cast<std::uintptr_t>(arena->base()) + arena->used();
    const auto pad          = static_cast<std::size_t>((256 - (at & 255U)) & 255U);
    if (arena->used() + pad + bytes > arena->capacity()) {
        warn_once(small_arena, "the workspace has no room for the quantized activations");
        return false;
    }
    const auto scope = arena->scope();
    const auto span  = arena->alloc_bytes(bytes, 256);
    q8_a8_run(x, w, out, mode, span.data, bytes, stream);
    return true;
}

} // namespace ninfer::ops::detail
