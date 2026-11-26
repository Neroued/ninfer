#include "ops/moe/routed_swiglu/cuda/fp8/rtx5090/softcap_mma.cuh"

#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/launch.h"
#include "ops/common/quant/fp8.cuh"
#include "ops/linear/kernels/fp8/fp8_a8_tma_mma.cuh"

namespace ninfer::ops::detail::moe_routed::mxfp8_softcap {
namespace {

struct Maps {
    CUtensorMap weights, input, scales;
};

template <bool Gate, bool A8, int Columns, int Features, int RowWarps, int TokenWarps,
          bool Bulk = false, bool Prefetch = false, bool StageInputScales = false,
          bool Swap = false, bool ExtraProducers = false>
struct Tile {
    static constexpr bool gate = Gate, a8 = A8, bulk = Bulk;
    static constexpr int k_tile    = 128;
    static constexpr bool prefetch = Prefetch, swapped = Swap,
                          stage_input_scales = StageInputScales;
    static constexpr int columns = Columns, features = Features, row_warps = RowWarps;
    static constexpr int warps        = RowWarps * TokenWarps;
    static constexpr int warp_columns = Columns / TokenWarps, warp_features = Features / RowWarps;
    static constexpr int fragment_columns = Swap ? 8 : 16, fragment_features = Swap ? 16 : 8;
    static constexpr int mmas_tokens             = warp_columns / fragment_columns;
    static constexpr int mmas_rows               = warp_features / fragment_features;
    static constexpr int branches                = Gate ? 2 : 1;
    static constexpr int rows                    = Gate ? intermediate : hidden;
    static constexpr int k                       = Gate ? hidden : intermediate;
    static constexpr int stages                  = A8 && Columns <= 32 ? 3 : 2;
    static constexpr bool redistribute_registers = warps == 16;
    template <bool Tma>
    static constexpr int producers = Tma && (redistribute_registers || ExtraProducers) ? 128 : 32;
    template <bool Tma>
    static constexpr int threads = producers<Tma> + 32 * warps;
    static_assert(rows % Features == 0 && warp_columns % 16 == 0);
    static_assert(!Gate || warp_features % 32 == 0);
    static_assert(!Swap || (!Gate && A8));
    static_assert(!StageInputScales || (Gate && A8 && Bulk));
    static_assert(!prefetch || a8);
};

template <class S>
struct Shared {
    alignas(128) std::uint8_t weights[S::stages][S::branches][S::features * S::k_tile];
    alignas(128) std::uint8_t input[S::stages][S::columns * S::k_tile * (S::a8 ? 1 : 2)];
    alignas(16) std::uint8_t weight_scales[S::bulk ? 1 : S::stages][S::branches]
                                          [S::features * (S::bulk ? S::k / 32 : S::k_tile / 32)];
    alignas(16) std::uint8_t
        input_scales[S::a8
                         ? (S::bulk && !S::stage_input_scales
                                ? S::columns * (S::gate ? input_scale_stride : product_scale_stride)
                                : S::stages * S::columns * 4)
                         : 1];
    std::uint64_t full[S::stages], empty[S::stages];
    int tokens[S::columns];
    float coefficients[S::gate ? 1 : S::columns];
};

template <class S>
__device__ __forceinline__ int code_offset(int row, int col) {
    return row * S::k_tile + (col ^ ((row & 7) * 16));
}

template <class S>
__device__ __forceinline__ int bf16_offset(int row, int col) {
    return row * S::k_tile + (col ^ ((row & 7) * 8));
}

__device__ __forceinline__ void bulk_load(void* destination, const void* source, int bytes,
                                          std::uint64_t* barrier) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(common::smem_addr(destination)), "l"(source), "r"(bytes),
          "r"(common::smem_addr(barrier))
        : "memory");
}

template <class S, bool Tma>
__device__ void produce(Shared<S>& tile, const Maps& maps, const Matrix& weight, int weight_row,
                        int row_begin, int input_begin, int columns, const __nv_bfloat16* x,
                        Mxfp8A8Workspace input, bool tma_input, bool async_lanes) {
    const int tid                     = threadIdx.x;
    constexpr int producer_threads    = S::template producers<Tma>;
    constexpr int weight_scale_stride = S::k / 32;
    constexpr int scale_stride        = S::gate ? input_scale_stride : product_scale_stride;
    const auto input_row              = [&](int col) {
        return col >= columns ? 0 : S::gate ? tile.tokens[col] : input_begin + col;
    };
#pragma unroll 1
    for (int kt = 0; kt < S::k / S::k_tile; ++kt) {
        const int stage   = kt % S::stages;
        auto* input_codes = tile.input[stage];
        common::cta_mbarrier_wait(&tile.empty[stage], 1 ^ ((kt / S::stages) & 1));
        auto* full = &tile.full[stage];
        if constexpr (Tma || S::bulk) {
            if (!tid) {
                constexpr int weight_scale_bytes = S::branches * S::features * weight_scale_stride;
                const int input_scale_bytes =
                    tma_input && !S::stage_input_scales ? S::columns * scale_stride : 0;
                const int code_bytes =
                    Tma ? (S::branches * S::features + (tma_input ? S::columns : 0)) * S::k_tile
                        : 0;
                const int bytes =
                    code_bytes + (S::bulk && !kt ? weight_scale_bytes + input_scale_bytes : 0);
                if (bytes)
                    common::cta_mbarrier_arrive_expect_tx(full, bytes);
                else
                    common::cta_mbarrier_arrive(full);
                if constexpr (S::bulk) {
                    if (!kt) {
#pragma unroll
                        for (int b = 0; b < S::branches; ++b)
                            bulk_load(tile.weight_scales[0][b],
                                      weight.scales +
                                          (row_begin + b * S::rows) * weight_scale_stride,
                                      S::features * weight_scale_stride, full);
                        if constexpr (Tma && !S::stage_input_scales)
                            if (tma_input)
                                common::tma_load_2d(tile.input_scales, &maps.scales, 0, input_begin,
                                                    full);
                    }
                }
                if constexpr (Tma) {
#pragma unroll
                    for (int b = 0; b < S::branches; ++b)
                        common::tma_load_2d(tile.weights[stage][b], &maps.weights, kt * S::k_tile,
                                            weight_row + row_begin + b * S::rows, full);
                    if (tma_input)
                        common::tma_load_2d(input_codes, &maps.input, kt * S::k_tile, input_begin,
                                            full);
                }
            }
        }
        if constexpr (!Tma) {
            for (int i = tid; i < S::branches * S::features * (S::k_tile / 16);
                 i += producer_threads) {
                const int row    = i / (S::k_tile / 16) % S::features;
                const int branch = i / (S::k_tile / 16) / S::features;
                const int col    = i % (S::k_tile / 16) * 16;
                const int source = row_begin + branch * S::rows + row;
                common::cp_async<16, Cache::cg>(
                    tile.weights[stage][branch] + code_offset<S>(row, col),
                    weight.codes + source * S::k + kt * S::k_tile + col);
            }
        }
        if constexpr (!S::bulk) {
            for (int i = tid; i < S::branches * S::features; i += producer_threads) {
                const int row = i % S::features, branch = i / S::features;
                const int source = row_begin + branch * S::rows + row;
                common::cp_async<4>(tile.weight_scales[stage][branch] + row * 4,
                                    weight.scales + source * weight_scale_stride + kt * 4);
            }
        }
        if constexpr (S::a8) {
            if constexpr (!S::bulk || S::stage_input_scales) {
                for (int row = tid; row < S::columns; row += producer_threads)
                    common::cp_async_zfill<4>(tile.input_scales + (stage * S::columns + row) * 4,
                                              input.scales + input_row(row) * scale_stride + kt * 4,
                                              row < columns ? 4 : 0);
            } else if constexpr (!S::stage_input_scales) {
                if (!tma_input && !kt)
                    for (int i = tid; i < S::columns * (scale_stride / 16); i += producer_threads) {
                        const int row = i / (scale_stride / 16);
                        const int col = i % (scale_stride / 16) * 16;
                        common::cp_async_zfill<16>(tile.input_scales + i * 16,
                                                   input.scales + input_row(row) * scale_stride +
                                                       col,
                                                   row < columns ? 16 : 0);
                    }
            }
            if (!tma_input) {
                for (int i = tid; i < S::columns * (S::k_tile / 16); i += producer_threads) {
                    const int t = i / (S::k_tile / 16), col = i % (S::k_tile / 16) * 16;
                    common::cp_async_zfill<16, Cache::cg>(input_codes + code_offset<S>(t, col),
                                                          input.codes + input_row(t) * S::k +
                                                              kt * S::k_tile + col,
                                                          t < columns ? 16 : 0);
                }
            }
        } else {
            auto* dest = reinterpret_cast<__nv_bfloat16*>(input_codes);
            for (int i = tid; i < S::columns * (S::k_tile / 8); i += producer_threads) {
                const int t = i / (S::k_tile / 8), col = i % (S::k_tile / 8) * 8;
                common::cp_async_zfill<16>(dest + bf16_offset<S>(t, col),
                                           x + input_row(t) * S::k + kt * S::k_tile + col,
                                           t < columns ? 16 : 0);
            }
        }
        // TMA contributes one arrival. Gathered inputs and staged BF16 use an
        // additional arrival for each issuing lane, ordered after its copies.
        if (async_lanes)
            asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"
                         :
                         : "r"(common::smem_addr(full))
                         : "memory");
    }
}

template <class S>
__device__ __forceinline__ void
consume(Shared<S>& tile, int lane, int warp, int columns,
        float (&acc)[S::mmas_tokens][S::branches][S::mmas_rows][4]) {
    const int first_token      = warp / S::row_warps * S::warp_columns;
    const int first_row        = warp % S::row_warps * S::warp_features;
    const int a_row            = (lane & 7) + (lane & 8);
    constexpr int scale_tile   = S::k_tile / 32;
    constexpr int scale_stride = S::gate ? input_scale_stride : product_scale_stride;
    constexpr int steps        = S::k / S::k_tile;
    int stage = 0, phase = 0;
#pragma unroll 1
    for (int kt = 0; kt < steps; ++kt) {
        common::cta_mbarrier_wait(&tile.full[stage], phase);
        if constexpr (S::a8) {
            unsigned sx[S::mmas_tokens], sw[S::branches][S::mmas_rows];
#pragma unroll
            for (int m = 0; m < S::mmas_tokens; ++m) {
                const int col = first_token + m * S::fragment_columns + lane / 4 +
                                (S::swapped ? 0 : ((lane & 3) == 1 ? 8 : 0));
                const int offset = S::bulk && !S::stage_input_scales
                                       ? col * scale_stride + kt * scale_tile
                                       : (stage * S::columns + col) * scale_tile;
                sx[m]            = common::load_vec<unsigned>(tile.input_scales + offset);
            }
#pragma unroll
            for (int b = 0; b < S::branches; ++b)
#pragma unroll
                for (int n = 0; n < S::mmas_rows; ++n) {
                    const int row = first_row + n * S::fragment_features + lane / 4 +
                                    (S::swapped && (lane & 3) == 1 ? 8 : 0);
                    const int offset = S::bulk ? row * (S::k / 32) + kt * 4 : row * 4;
                    sw[b][n]         = common::load_vec<unsigned>(
                        tile.weight_scales[S::bulk ? 0 : stage][b] + offset);
                }

            constexpr int input_words = S::swapped ? 2 : 4, weight_words = S::swapped ? 4 : 2;
            constexpr int buffers = S::prefetch ? 2 : 1;
            unsigned x[buffers][S::mmas_tokens][input_words];
            const auto load_input = [&](int step, int slot) {
#pragma unroll
                for (int m = 0; m < S::mmas_tokens; ++m) {
                    const int row =
                        first_token + m * S::fragment_columns + (S::swapped ? lane & 7 : a_row);
                    const int col = step * 32 + (S::swapped ? (lane & 8) * 2 : (lane / 16) * 16);
                    const auto address =
                        common::smem_addr(tile.input[stage] + code_offset<S>(row, col));
                    auto& frag = x[slot][m];
                    if constexpr (S::swapped)
                        common::ldmatrix_x2(frag[0], frag[1], address);
                    else
                        common::ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], address);
                }
            };
            const auto load_weight = [&](int step, int branch, int n,
                                         unsigned(&frag)[weight_words]) {
                const int row =
                    first_row + n * S::fragment_features + (S::swapped ? a_row : lane & 7);
                const int col = step * 32 + (S::swapped ? (lane / 16) * 16 : (lane & 8) * 2);
                const auto address =
                    common::smem_addr(tile.weights[stage][branch] + code_offset<S>(row, col));
                if constexpr (S::swapped)
                    common::ldmatrix_x4(frag[0], frag[1], frag[2], frag[3], address);
                else
                    common::ldmatrix_x2(frag[0], frag[1], address);
            };
            if constexpr (S::prefetch) load_input(0, 0);
#pragma unroll
            for (int step = 0; step < scale_tile; ++step) {
                const int x_slot = step % buffers;
                if constexpr (!S::prefetch)
                    load_input(step, x_slot);
                else if (step + 1 < scale_tile)
                    load_input(step + 1, x_slot ^ 1);
#pragma unroll
                for (int b = 0; b < S::branches; ++b) {
                    unsigned w[buffers][weight_words];
                    if constexpr (S::prefetch) load_weight(step, b, 0, w[0]);
#pragma unroll
                    for (int n = 0; n < S::mmas_rows; ++n) {
                        const int w_slot = n % buffers;
                        if constexpr (!S::prefetch)
                            load_weight(step, b, n, w[w_slot]);
                        else if (n + 1 < S::mmas_rows)
                            load_weight(step, b, n + 1, w[w_slot ^ 1]);
                        const auto& wf = w[w_slot];
#pragma unroll
                        for (int m = 0; m < S::mmas_tokens; ++m) {
                            auto& v        = acc[m][b][n];
                            const auto& xf = x[x_slot][m];
                            if constexpr (S::swapped)
                                common::mma_fp8_e4m3_scaled(v[0], v[1], v[2], v[3], wf[0], wf[1],
                                                            wf[2], wf[3], xf[0], xf[1], sw[b][n],
                                                            sx[m], step);
                            else
                                common::mma_fp8_e4m3_scaled(v[0], v[1], v[2], v[3], xf[0], xf[1],
                                                            xf[2], xf[3], wf[0], wf[1], sx[m],
                                                            sw[b][n], step);
                        }
                    }
                }
            }
        } else {
            const auto* input = reinterpret_cast<const __nv_bfloat16*>(tile.input[stage]);
#pragma unroll
            for (int step = 0; step < S::k_tile / 16; ++step) {
                unsigned a[S::mmas_tokens][4];
#pragma unroll
                for (int m = 0; m < S::mmas_tokens; ++m) {
                    const int row = first_token + m * 16 + a_row;
                    const int col = step * 16 + (lane / 16) * 8;
                    common::ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                        common::smem_addr(input + bf16_offset<S>(row, col)));
                }
#pragma unroll
                for (int b = 0; b < S::branches; ++b) {
#pragma unroll
                    for (int n = 0; n < S::mmas_rows; ++n) {
                        const int row = first_row + n * 8 + lane / 4;
                        const int col = step * 16 + (lane & 3) * 2;
                        const auto scale =
                            tile.weight_scales[stage][b][row * scale_tile + step / 2];
                        const auto* ptr = tile.weights[stage][b] + code_offset<S>(row, col);
                        const auto w0 =
                            common::fp8_dequant_bf16x2(common::load_vec<std::uint16_t>(ptr), scale);
                        const auto w1 = common::fp8_dequant_bf16x2(
                            common::load_vec<std::uint16_t>(ptr + 8), scale);
#pragma unroll
                        for (int m = 0; m < S::mmas_tokens; ++m) {
                            if (first_token + m * 16 >= columns) break;
                            auto& v = acc[m][b][n];
                            common::mma_bf16(v[0], v[1], v[2], v[3], a[m][0], a[m][1], a[m][2],
                                             a[m][3], w0, w1);
                        }
                    }
                }
            }
        }

        // Dependent MMA instructions have consumed every synchronous shared load.
        __syncwarp();
        if (!lane) common::cta_mbarrier_arrive(&tile.empty[stage]);
        if (++stage == S::stages) {
            stage = 0;
            phase ^= 1;
        }
    }
}

template <int Mmas>
__device__ __forceinline__ void emit_product(float2 (&values)[Mmas], Storage storage, int row,
                                             int packed, bool live, int lane) {
    if (storage.product.codes) {
#pragma unroll
        for (int n = 0; n < Mmas; n += 4) {
            float largest = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                largest = fmaxf(largest, fmaxf(fabsf(values[n + i].x), fabsf(values[n + i].y)));
            largest          = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, 1));
            largest          = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, 2));
            const auto scale = common::fp8_ue8m0_scale(largest);
            if (live) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const auto v = values[n + i];
                    common::store_vec(storage.product.codes + packed * intermediate + row +
                                          (n + i) * 8,
                                      common::fp8_quant_code2(v.x, v.y, scale.inverse_scale));
                }
                if (!(lane & 3))
                    storage.product.scales[packed * product_scale_stride + (row + n * 8) / 32] =
                        scale.scale;
            }
        }
    } else if (live) {
#pragma unroll
        for (int n = 0; n < Mmas; ++n) {
            const auto v = values[n];
            common::store_vec(storage.bf16_product + packed * intermediate + row + n * 8,
                              __floats2bfloat162_rn(v.x, v.y));
        }
    }
}

__device__ __forceinline__ void add_output(float* out, float x, float y, float z, float w) {
    asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(out), "f"(x), "f"(y), "f"(z), "f"(w)
                 : "memory");
}

template <class S>
__device__ __forceinline__ void
emit_transposed(const float (&acc)[S::mmas_tokens][S::branches][S::mmas_rows][4],
                const Shared<S>& tile, const Storage& storage, int columns, int first_token,
                int first_row, int lane) {
#pragma unroll
    for (int m = 0; m < S::mmas_tokens; ++m) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int col     = first_token + m * 8 + (lane & 3) * 2 + p;
            const bool live   = col < columns;
            const float coeff = live ? tile.coefficients[col] : 0;
            const int token   = live ? tile.tokens[col] : 0;
#pragma unroll
            for (int n = 0; n < S::mmas_rows; ++n) {
#pragma unroll
                for (int high = 0; high < 2; ++high) {
                    const float x0 = acc[m][0][n][p + 2 * high] * coeff;
                    const float x1 = __shfl_down_sync(0xffffffff, x0, 4);
                    const float x2 = __shfl_down_sync(0xffffffff, x0, 8);
                    const float x3 = __shfl_down_sync(0xffffffff, x0, 12);
                    if (live && !(lane & 12))
                        add_output(storage.sum + token * hidden + first_row + n * 16 + high * 8 +
                                       lane / 4,
                                   x0, x1, x2, x3);
                }
            }
        }
    }
}

template <class S, bool Tma, int Threads = S::template threads<Tma>>
__global__ __launch_bounds__(Threads, 1) void grouped(const __grid_constant__ Maps maps,
                                                      ExpertMatrixDeviceView bank,
                                                      const __nv_bfloat16* x,
                                                      const float* coefficients,
                                                      SoftcappedActivation activation, int tokens,
                                                      const __grid_constant__ Storage storage) {
    constexpr int row_tiles = S::rows / S::features;
    const int job = blockIdx.x / row_tiles, row_begin = blockIdx.x % row_tiles * S::features;
    const auto& g = storage.groups;
    if (job >= *g.jobs) return;
    const int expert = g.job_experts[job], local = g.job_columns[job];
    const int packed        = g.offsets[expert] + local;
    const int columns       = min(S::columns, g.counts[expert] - local);
    const auto weight       = resolve(bank, expert);
    const bool shared_input = S::gate && S::a8 && g.load_range[0] == tokens;
    const bool tma_input    = Tma && S::a8 && (!S::gate || shared_input);
    const bool async_lanes  = !S::bulk || !Tma || !tma_input || S::stage_input_scales;
    const int input_begin   = shared_input ? local : packed;
    extern __shared__ __align__(128) std::uint8_t shared[];
    auto& tile                 = *reinterpret_cast<Shared<S>*>(shared);
    const int active_consumers = ((columns + S::warp_columns - 1) / S::warp_columns) * S::row_warps;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < S::stages; ++s) {
            common::cta_mbarrier_init(&tile.full[s],
                                      (async_lanes ? S::template producers<Tma> : 0) +
                                          int(Tma || S::bulk));
            common::cta_mbarrier_init(&tile.empty[s], active_consumers);
        }
        common::cta_mbarrier_fence_init();
    }
    for (int c = threadIdx.x; c < columns; c += blockDim.x) {
        const int assignment = g.assignments[packed + c];
        tile.tokens[c]       = assignment / top_k;
        if constexpr (!S::gate) tile.coefficients[c] = coefficients[assignment];
    }
    __syncthreads();
    if (threadIdx.x < S::template producers<Tma>) {
        if constexpr (S::redistribute_registers && Tma)
            asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" : : : "memory");
        // Only the issuing lane observes the stage barriers on the TMA-only path.
        if (!async_lanes && threadIdx.x != 0) return;
        const int weight_row = Tma ? weight.physical * (bank.code_stride / S::k) : 0;
        produce<S, Tma>(tile, maps, weight, weight_row, row_begin, input_begin, columns, x,
                        S::gate ? storage.input : storage.product, tma_input, async_lanes);
        return;
    }
    const int warp = (threadIdx.x - S::template producers<Tma>) / 32, lane = threadIdx.x % 32;
    const int first_token = warp / S::row_warps * S::warp_columns;
    // The register allocation instruction is collective across four warps.
    if constexpr (S::redistribute_registers && Tma)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;" : : : "memory");
    if (first_token >= columns) return;
    const int row = row_begin + warp % S::row_warps * S::warp_features + (lane & 3) * 2;
    float acc[S::mmas_tokens][S::branches][S::mmas_rows][4]{};
    consume<S>(tile, lane, warp, columns, acc);
    if constexpr (S::swapped) {
        emit_transposed<S>(acc, tile, storage, columns, first_token,
                           row_begin + warp % S::row_warps * S::warp_features, lane);
    } else {
#pragma unroll
        for (int m = 0; m < S::mmas_tokens; ++m) {
            if (first_token + m * 16 >= columns) break;
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int col   = first_token + m * 16 + p * 8 + lane / 4;
                const bool live = col < columns;
                if constexpr (S::gate) {
                    float2 product[S::mmas_rows];
#pragma unroll
                    for (int n = 0; n < S::mmas_rows; ++n)
                        product[n] = make_float2(
                            activation(acc[m][0][n][2 * p], acc[m][1][n][2 * p]),
                            activation(acc[m][0][n][2 * p + 1], acc[m][1][n][2 * p + 1]));
                    emit_product(product, storage, row, packed + col, live, lane);
                } else {
                    const float coeff = live ? tile.coefficients[col] : 0;
                    const int token   = live ? tile.tokens[col] : 0;
#pragma unroll
                    for (int n = 0; n < S::mmas_rows; ++n) {
                        const float x0 = acc[m][0][n][p * 2] * coeff;
                        const float x1 = acc[m][0][n][p * 2 + 1] * coeff;
                        const float x2 = __shfl_xor_sync(0xffffffff, x0, 1);
                        const float x3 = __shfl_xor_sync(0xffffffff, x1, 1);
                        if (live && !(lane & 1))
                            add_output(storage.sum + token * hidden + row + n * 8, x0, x1, x2, x3);
                    }
                }
            }
        }
    }
}

CUtensorMap activation_map(const std::uint8_t* pointer, int rows, int stride, int box_width,
                           int box_rows, CUtensorMapSwizzle swizzle,
                           CUtensorMapL2promotion promotion) {
    CUtensorMap map{};
    const std::uint64_t dimensions[]{std::uint64_t(stride), std::uint64_t(rows)};
    const std::uint64_t strides[]{std::uint64_t(stride)};
    const std::uint32_t box[]{std::uint32_t(box_width), std::uint32_t(box_rows)}, steps[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<std::uint8_t*>(pointer), dimensions,
        strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, promotion,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(
            std::string("routed SwiGLU: failed to encode MXFP8 activation tensor map: ") +
            (name ? name : "CUDA error"));
    }
    return map;
}

template <class S, class Indexed = S>
void launch_stage(const ExpertMatrixDeviceView& bank, const __nv_bfloat16* x,
                  const float* coefficients, SoftcappedActivation activation, Plan plan,
                  const Storage& storage, int tokens, cudaStream_t stream) {
    constexpr int rows   = S::rows * S::branches;
    const int jobs       = job_capacity(plan, tokens * top_k);
    const auto* codes    = reinterpret_cast<const std::uint8_t*>(bank.codes);
    const auto code_rows = bank.offsets ? 0 : (bank.physical_count - 1) * (bank.code_stride / S::k);
    if (!bank.offsets && bank.code_stride % S::k == 0 && code_rows <= INT_MAX - rows) {
        Maps maps{};
        maps.weights =
            fp8_tma_map(codes, static_cast<int>(code_rows) + rows, S::k, S::features, S::k_tile);
        if constexpr (S::a8) {
            const auto data          = S::gate ? storage.input : storage.product;
            const int input_rows     = S::gate ? tokens : tokens * top_k;
            constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
            maps.input = activation_map(data.codes, input_rows, S::k, S::k_tile, S::columns,
                                        CU_TENSOR_MAP_SWIZZLE_128B, promotion);
            if constexpr (S::bulk && !S::stage_input_scales) {
                constexpr int stride = S::gate ? input_scale_stride : product_scale_stride;
                maps.scales = activation_map(data.scales, input_rows, stride, stride, S::columns,
                                             CU_TENSOR_MAP_SWIZZLE_NONE, promotion);
            }
        }
        const int shared = common::prepare_shared<sizeof(Shared<S>), grouped<S, true>, true>();
        grouped<S, true>
            <<<jobs*(S::rows / S::features), S::template threads<true>, shared, stream>>>(
                maps, bank, x, coefficients, activation, tokens, storage);
    } else {
        const int shared =
            common::prepare_shared<sizeof(Shared<Indexed>), grouped<Indexed, false>, true>();
        grouped<Indexed, false>
            <<<jobs*(Indexed::rows / Indexed::features), Indexed::template threads<false>, shared,
               stream>>>({}, bank, x, coefficients, activation, tokens, storage);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_mma(const __nv_bfloat16* x, const ExpertBankDeviceView& bank, const float* coefficients,
                SoftcappedActivation activation, Plan plan, const Storage& storage, int tokens,
                cudaStream_t stream) {
    const auto gate = [&]<bool A8, int C, int F, int R, int T, bool Bulk, bool Prefetch = false,
                          bool StageInputScales = false, bool ExtraProducers = false>() {
        using S =
            Tile<true, A8, C, F, R, T, Bulk, Prefetch, StageInputScales, false, ExtraProducers>;
        launch_stage<S>(bank.gate_up, x, coefficients, activation, plan, storage, tokens, stream);
    };
    const auto down = [&]<bool A8, int C, int F, int R, int T, bool Bulk, bool Prefetch = false>() {
        using S = Tile<false, A8, C, F, R, T, Bulk, Prefetch>;
        launch_stage<S>(bank.down, reinterpret_cast<const __nv_bfloat16*>(storage.bf16_product),
                        coefficients, activation, plan, storage, tokens, stream);
    };
    const auto clear_sum = [&] {
        CUDA_CHECK(
            cudaMemsetAsync(storage.sum, 0, std::size_t(tokens) * hidden * sizeof(float), stream));
    };
    const auto mixed =
        [&]<int C, int GF, int GR, int GT, int DF, int DR, int DT, bool Bulk = true>() {
            if (plan.gate_a8)
                gate.template operator()<true, C, GF, GR, GT, Bulk>();
            else
                gate.template operator()<false, C, GF, GR, GT, false>();
            clear_sum();
            if (plan.down_a8)
                down.template operator()<true, C, DF, DR, DT, Bulk>();
            else
                down.template operator()<false, C, DF, DR, DT, false>();
        };
    switch (plan.schedule) {
    case Schedule::Grouped16:
        return mixed.template operator()<16, 64, 2, 1, 128, 4, 1>();
    case Schedule::Grouped32:
        return mixed.template operator()<32, 64, 2, 2, 128, 4, 2>();
    case Schedule::Grouped64:
        return mixed.template operator()<64, 128, 4, 2, 128, 4, 2, false>();
    case Schedule::Grouped128:
        gate.template operator()<true, 128, 64, 2, 8, true>();
        clear_sum();
        down.template operator()<true, 128, 128, 4, 4, true>();
        return;
    case Schedule::Wide128:
    case Schedule::Wide128Transposed: {
        // Stage four scale bytes per input row alongside each K128 code tile.
        // The complete 72-byte scale rows for Gate/Up weights fit in shared memory.
        gate.template operator()<true, 128, 96, 3, 4, true, true, true, true>();
        clear_sum();
        if (plan.schedule == Schedule::Wide128Transposed) {
            using Down        = Tile<false, true, 128, 192, 3, 4, true, true, false, true>;
            using IndexedDown = Tile<false, true, 128, 128, 2, 4, true, true>;
            launch_stage<Down, IndexedDown>(bank.down, nullptr, coefficients, activation, plan,
                                            storage, tokens, stream);
        } else {
            down.template operator()<true, 128, 128, 2, 4, true, true>();
        }
        return;
    }
    case Schedule::Vector:
        throw std::logic_error("routed SwiGLU: vector schedule passed to grouped MXFP8 launcher");
    }
}

} // namespace ninfer::ops::detail::moe_routed::mxfp8_softcap
