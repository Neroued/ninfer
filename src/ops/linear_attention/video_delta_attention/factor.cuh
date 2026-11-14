// SPDX-License-Identifier: Apache-2.0
// Adapted from SGLang vdn_delta_factors.cuh, commit 5792cf6e5e11aebc62d41cf19828f615a8fa7e0a.
// https://github.com/sgl-project/sglang/blob/5792cf6e5e11aebc62d41cf19828f615a8fa7e0a/python/sglang/kernels/jit/csrc/diffusion/vdn_delta_factors.cuh
// Changes: explicit NInfer operands, in-place factors with a read/store barrier,
// symmetric Gram load, head waves, batch addressing and direct state-init output.
#pragma once

namespace ninfer::ops::detail::vda::factor_internal {
constexpr int kDim            = 128; // dk == dv == 128
constexpr int kBlockSize      = 256; // 16 x 16 tiles of 8 x 8
constexpr int kMinBlocksPerSm = 2;   // 128 registers per thread
constexpr int kXsBytes        = kDim * kDim * static_cast<int>(sizeof(float)); // 64 KB dynamic smem
constexpr unsigned kFullMask  = 0xffffffffu;

// thread tj's tile columns 8tj..8tj+3 land at 4tj.., 8tj+4..8tj+7 at 64+4tj.., so a quarter warp
// reads 8 consecutive 16-byte chunks
__device__ __forceinline__ int swz_lo(int tj) { return 4 * tj; }

__device__ __forceinline__ int swz_hi(int tj) { return 64 + 4 * tj; }

__device__ __forceinline__ float rcp_nr(float p) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(p));
    return fmaf(r, fmaf(-p, r, 1.f), r); // one Newton step: ~0.5 ulp
}

__device__ __forceinline__ float4 ld4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ void st4(float* p, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}

__device__ __forceinline__ void st4(float* p, float2 a, float2 b) {
    *reinterpret_cast<float4*>(p) = make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float2 f2(float a) { return make_float2(a, a); }

// packed fma: (a.x*b.x+c.x, a.y*b.y+c.y); one FFMA2 on sm_100+ (the first operand is a scalar
// broadcast in SASS), two FFMA elsewhere.  Bitwise identical results either way.
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    return __ffma2_rn(a, b, c);
#else
    return make_float2(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y));
#endif
}

__device__ __forceinline__ float get(const float4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

struct Smem {
    float row[2][2][kDim];  // [buffer][band row m][swizzled col]   scaled band rows R'
    float col[2][kDim * 2]; // [buffer][row*2 + m]                   multipliers G
};

/// Prepare the band of pivot block {k, k+1}, k = 8*kt1 + R1 (R1 even).  Column owners (tj == kt1)
/// publish G = C (0 for the band rows) and zero their band columns; row owners (ti == kt1) replace
/// P by I, scale R' = P^-1 R and publish it.  Executed by all threads (P is broadcast with
/// shuffles).
template <int R1, int BUF>
__device__ __forceinline__ void prepare(float2 (&t)[8][4], Smem& sm, int kt1, int ti, int tj,
                                        int i0) {
    constexpr int CP = R1 >> 1;
    const int src  = ((kt1 & 1) << 4) | kt1; // lane of the diagonal tile inside the row-owner warp
    const float pa = __shfl_sync(kFullMask, t[R1][CP].x, src);
    const float pb = __shfl_sync(kFullMask, t[R1][CP].y, src);
    const float pc = __shfl_sync(kFullMask, t[R1 + 1][CP].x, src);
    const float pd = __shfl_sync(kFullMask, t[R1 + 1][CP].y, src);
    if (tj == kt1) {
        const bool diag = (ti == kt1);
        float* dst      = &sm.col[BUF][i0 * 2];
        float2 g[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const bool band = diag && (r == R1 || r == R1 + 1);
            g[r]            = band ? make_float2(0.f, 0.f) : t[r][CP];
            if (!band) t[r][CP] = make_float2(0.f, 0.f);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) st4(dst + 4 * q, g[2 * q], g[2 * q + 1]);
    }
    if (ti == kt1) {
        const float det  = fmaf(pa, pd, -pb * pc);
        const float idet = rcp_nr(det);
        const float ia = pd * idet, ib = -pb * idet, ic = -pc * idet, id = pa * idet;
        if (tj == kt1) {
            t[R1][CP]     = make_float2(1.f, 0.f);
            t[R1 + 1][CP] = make_float2(0.f, 1.f);
        }
#pragma unroll
        for (int cp = 0; cp < 4; ++cp) {
            const float2 x = t[R1][cp], y = t[R1 + 1][cp];
            t[R1][cp]     = make_float2(fmaf(ia, x.x, ib * y.x), fmaf(ia, x.y, ib * y.y));
            t[R1 + 1][cp] = make_float2(fmaf(ic, x.x, id * y.x), fmaf(ic, x.y, id * y.y));
        }
        st4(&sm.row[BUF][0][swz_lo(tj)], t[R1][0], t[R1][1]);
        st4(&sm.row[BUF][0][swz_hi(tj)], t[R1][2], t[R1][3]);
        st4(&sm.row[BUF][1][swz_lo(tj)], t[R1 + 1][0], t[R1 + 1][1]);
        st4(&sm.row[BUF][1][swz_hi(tj)], t[R1 + 1][2], t[R1 + 1][3]);
    }
}

/// Rank-2 update for pivots (8kt + KR, 8kt + KR + 1); the next block's band is prepared in the
/// middle.
template <int KR>
__device__ __forceinline__ void step(float2 (&t)[8][4], Smem& sm, int kt, int ti, int tj, int i0) {
    constexpr int CUR = (KR >> 1) & 1, NXT = CUR ^ 1;
    constexpr int R1 = (KR + 2) & 7, CP = R1 >> 1; // next block's band rows / column pair
    float2 rk0[4], rk1[4];
    float ck0[8], ck1[8]; // negated multipliers (scalar broadcast operands of FFMA2)
    {
        const float4 a = ld4(&sm.row[CUR][0][swz_lo(tj)]), b = ld4(&sm.row[CUR][0][swz_hi(tj)]);
        const float4 c = ld4(&sm.row[CUR][1][swz_lo(tj)]), d = ld4(&sm.row[CUR][1][swz_hi(tj)]);
        rk0[0]          = make_float2(a.x, a.y);
        rk0[1]          = make_float2(a.z, a.w);
        rk0[2]          = make_float2(b.x, b.y);
        rk0[3]          = make_float2(b.z, b.w);
        rk1[0]          = make_float2(c.x, c.y);
        rk1[1]          = make_float2(c.z, c.w);
        rk1[2]          = make_float2(d.x, d.y);
        rk1[3]          = make_float2(d.z, d.w);
        const float* cp = &sm.col[CUR][i0 * 2];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 v = ld4(cp + 4 * q);
            ck0[2 * q]     = -v.x;
            ck1[2 * q]     = -v.y;
            ck0[2 * q + 1] = -v.z;
            ck1[2 * q + 1] = -v.w;
        }
    }
    // part 1: the next band (rows R1, R1+1 fully; column pair CP in the other rows)
#pragma unroll
    for (int cp = 0; cp < 4; ++cp) {
        t[R1][cp] = fma2(f2(ck1[R1]), rk1[cp], fma2(f2(ck0[R1]), rk0[cp], t[R1][cp]));
        t[R1 + 1][cp] =
            fma2(f2(ck1[R1 + 1]), rk1[cp], fma2(f2(ck0[R1 + 1]), rk0[cp], t[R1 + 1][cp]));
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r == R1 || r == R1 + 1) continue;
        t[r][CP] = fma2(f2(ck1[r]), rk1[CP], fma2(f2(ck0[r]), rk0[CP], t[r][CP]));
    }
    if (KR != 6 || kt != 15) prepare<R1, NXT>(t, sm, kt + (KR == 6 ? 1 : 0), ti, tj, i0);
    __syncthreads();
    // part 2: everything else (registers only; overlaps the barrier skew and the next step's loads)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r == R1 || r == R1 + 1) continue;
#pragma unroll
        for (int cp = 0; cp < 4; ++cp) {
            if (cp == CP) continue;
            t[r][cp] = fma2(f2(ck1[r]), rk1[cp], fma2(f2(ck0[r]), rk0[cp], t[r][cp]));
        }
    }
}

// Invert I + A in FP32. The Gram buffer becomes diag(alpha) (I + A)^-1,
// packed into two BF16 planes; the write buffer becomes B (I + A)^-1 in FP32.
// State initialization skips the transition and writes scaled injection directly
// to the caller's FP32 state. Scratch matrices are indexed by frame and wave group.
__global__ void __launch_bounds__(kBlockSize, kMinBlocksPerSm)
    factor_kernel(Arguments args, float* A, float* B, int head_begin, int wave_heads, int batch,
                  bool init) {
    extern __shared__ __align__(16) float Xs[]; // [kDim][kDim], column-swizzled copy of (I + A)^-1
    __shared__ __align__(16) Smem sm;

    const size_t n = static_cast<size_t>(blockIdx.x) * wave_heads + blockIdx.y;
    const int tid  = threadIdx.x;
    const int ti = tid >> 4, tj = tid & 15;
    const int i0 = ti * 8, j0 = tj * 8;
    const size_t mat = static_cast<size_t>(n) * kDim * kDim;
    const float* An  = A + mat;
    const float* Bn  = B + mat;

    float2 t[8][4];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const float4 v0 = ld4(An + (i0 + r) * kDim + j0), v1 = ld4(An + (i0 + r) * kDim + j0 + 4);
        t[r][0] = make_float2(0.5f * (v0.x + An[j0 * kDim + i0 + r]),
                              0.5f * (v0.y + An[(j0 + 1) * kDim + i0 + r]));
        t[r][1] = make_float2(0.5f * (v0.z + An[(j0 + 2) * kDim + i0 + r]),
                              0.5f * (v0.w + An[(j0 + 3) * kDim + i0 + r]));
        t[r][2] = make_float2(0.5f * (v1.x + An[(j0 + 4) * kDim + i0 + r]),
                              0.5f * (v1.y + An[(j0 + 5) * kDim + i0 + r]));
        t[r][3] = make_float2(0.5f * (v1.z + An[(j0 + 6) * kDim + i0 + r]),
                              0.5f * (v1.w + An[(j0 + 7) * kDim + i0 + r]));
    }
    if (ti == tj) { // M = I + A
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r & 1) {
                t[r][r >> 1].y += 1.f;
            } else {
                t[r][r >> 1].x += 1.f;
            }
        }
    }

    prepare<0, 0>(t, sm, 0, ti, tj, i0);
    __syncthreads();
#pragma unroll 1
    for (int kt = 0; kt < kDim / 8; ++kt) {
        step<0>(t, sm, kt, ti, tj, i0);
        step<2>(t, sm, kt, ti, tj, i0);
        step<4>(t, sm, kt, ti, tj, i0);
        step<6>(t, sm, kt, ti, tj, i0);
    }

    // transition = diag(alpha) X; every A input was consumed before this in-place store.
    if (!init) {
        const int f         = blockIdx.x;
        const auto location = resolve_group(args, batch, head_begin + blockIdx.y);
        const float* al =
            args.alpha +
            (static_cast<size_t>(location.batch) * args.frames + f) * args.heads * kDim +
            location.head * kDim + i0;
        float av[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) av[r] = al[r];
        auto* transition_hi = reinterpret_cast<__nv_bfloat16*>(A + mat);
        auto* transition_lo = transition_hi + kDim * kDim;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
#pragma unroll
            for (int cp = 0; cp < 4; ++cp) {
                const float x = av[r] * t[r][cp].x, y = av[r] * t[r][cp].y;
                const auto high = __floats2bfloat162_rn(x, y);
                const auto low =
                    __floats2bfloat162_rn(x - __low2float(high), y - __high2float(high));
                const int index = (i0 + r) * kDim + j0 + 2 * cp;
                *reinterpret_cast<__nv_bfloat162*>(transition_hi + index) = high;
                *reinterpret_cast<__nv_bfloat162*>(transition_lo + index) = low;
            }
        }
    }
    // stage X in smem (swizzled columns) for the GEMM
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        float* row = Xs + (i0 + r) * kDim;
        st4(row + swz_lo(tj), t[r][0], t[r][1]);
        st4(row + swz_hi(tj), t[r][2], t[r][3]);
    }
    __syncthreads();

    // injection = B X  (register-tiled GEMM; B rows from global through L1, X from smem)
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int cp = 0; cp < 4; ++cp) t[r][cp] = make_float2(0.f, 0.f);
    const float* Bp = Bn + i0 * kDim;
#pragma unroll 1
    for (int k = 0; k < kDim; k += 4) {
        float4 b[8];
#pragma unroll
        for (int r = 0; r < 8; ++r)
            b[r] = __ldg(reinterpret_cast<const float4*>(Bp + r * kDim + k));
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const float* xr = Xs + (k + kk) * kDim;
            const float4 x0 = ld4(xr + swz_lo(tj)), x1 = ld4(xr + swz_hi(tj));
            const float2 xv[4] = {make_float2(x0.x, x0.y), make_float2(x0.z, x0.w),
                                  make_float2(x1.x, x1.y), make_float2(x1.z, x1.w)};
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float2 bv = f2(get(b[r], kk));
#pragma unroll
                for (int cp = 0; cp < 4; ++cp) t[r][cp] = fma2(bv, xv[cp], t[r][cp]);
            }
        }
    }
    __syncthreads(); // all threads finish reading B before any in-place injection store
    {
        float* Jn = init ? args.state_out +
                               (static_cast<size_t>(batch) * args.heads + head_begin + blockIdx.y) *
                                   kDim * kDim
                         : B + mat;
        if (init) {
#pragma unroll
            for (int r = 0; r < 8; ++r)
#pragma unroll
                for (int cp = 0; cp < 4; ++cp) {
                    t[r][cp].x *= args.state_scale;
                    t[r][cp].y *= args.state_scale;
                }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            float* row = Jn + (i0 + r) * kDim + j0;
            if ((reinterpret_cast<std::uintptr_t>(row) & 15) == 0) {
                st4(row, t[r][0], t[r][1]);
                st4(row + 4, t[r][2], t[r][3]);
            } else {
#pragma unroll
                for (int cp = 0; cp < 4; ++cp) {
                    row[2 * cp]     = t[r][cp].x;
                    row[2 * cp + 1] = t[r][cp].y;
                }
            }
        }
    }
}


} // namespace ninfer::ops::detail::vda::factor_internal
