#pragma once
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/grouped_tma.cuh"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/csf_scale_tile.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::grouped_scale_cache {
using grouped_tma::Maps;
using grouped_tma::make_map;

template <class S>
struct Decoder : csf_scale_tile::Shape<S::intermediate, S::gate, 2> {
    static constexpr int threads = S::threads, warps = threads / 32;
};

template <int I, bool Gate, int C, bool Csf>
struct Shape : grouped_tma::Shape<I, Gate, C, Csf> {
    static constexpr int stages           = 2;
    static constexpr int producer_threads = 64, producer_warps = 1, consumer_threads = 256,
                         consumer_warps = 8;
    static constexpr int threads = 320, min_blocks = C == 128 && !Gate ? 2 : 1,
                         warp_columns = C / 4, atoms = warp_columns / 8;
};

// Scale restoration, operand transfers and MMA have independent warp roles.
// Only two consumed row tiles are expanded, in CTA-owned shared storage.
template <class S>
struct Shared : grouped_tma::OutputMetadata<S> {
    csf_scale_tile::Tile<Decoder<S>> recovery[2];
    alignas(512) std::uint8_t weights[2][64 * 64];
    alignas(512) std::uint8_t inputs[2][S::columns * 64];
    alignas(128) unsigned input_scales[2][S::columns * 4];
    alignas(8) std::uint64_t full[2], empty[2], scale_full[2], scale_empty[2];
    float factors[2], product_factors[2];
};

template <class S>
__device__ __forceinline__ void produce(Shared<S>& s, const Maps& maps, int row, int begin,
                                        int rotation) {
    const int lane = threadIdx.x % 32;
#pragma unroll 1
    for (int kt = 0; kt < S::k / 128; ++kt) {
        const int stage = kt % 2, physical = (kt + rotation) % (S::k / 128);
        common::cta_mbarrier_wait(&s.empty[stage], 1U ^ ((kt / 2) & 1U));
        if (!lane) {
            common::cta_mbarrier_expect_tx(
                &s.full[stage], (64 + S::columns) * 64 + ((kt & 1) ? 0 : S::columns * 16));
            common::tma_load_2d(s.weights[stage], &maps.weights, physical * 64, row,
                                &s.full[stage]);
            if constexpr (S::gate) {
                common::tma_load_2d(s.weights[stage] + 32 * 64, &maps.weights, physical * 64,
                                    row + S::intermediate, &s.full[stage]);
            }
            common::tma_load_2d(s.inputs[stage], &maps.inputs, physical * 64, begin,
                                &s.full[stage]);
            if (!(kt & 1))
                common::tma_load_2d(s.input_scales[(kt / 2) % 2], &maps.scales, physical / 2 * 16,
                                    begin, &s.full[stage]);
        }
        __syncwarp();
        common::cta_mbarrier_arrive(&s.full[stage]);
    }
}

template <class S>
__device__ __forceinline__ void consume(Shared<S>& s, int columns, int rotation, int scale_slot,
                                        float (&acc)[2][S::atoms][4]) {
    const int thread = threadIdx.x - S::producer_threads, lane = thread % 32, warp = thread / 32,
              wm = warp % 2, wn = warp / 2;
#pragma unroll 1
    for (int kt = 0; kt < S::k / 128; ++kt) {
        const int stage = kt % 2, physical = (kt + rotation) % (S::k / 128);
        common::cta_mbarrier_wait(&s.full[stage], (kt / 2) & 1U);
#pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            unsigned pb0[2], pb1[2], pbs[2];
            const auto load_b = [&](int n, int slot) {
                const int column = wn * S::warp_columns + n * 8;
                common::ldmatrix_x2(
                    pb0[slot], pb1[slot],
                    common::smem_addr(s.inputs[stage] +
                                      grouped_tma::address<S>(column + (lane & 7),
                                                              ki * 32 + ((lane >> 3) & 1) * 16)));
                pbs[slot] =
                    s.input_scales[(kt / 2) % 2][(column + (lane >> 2)) * 4 + (kt & 1) * 2 + ki];
            };
            if (wn * S::warp_columns < columns) load_b(0, 0);
            unsigned a[2][4], as[2];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int base = S::gate ? wm * 16 + m * 32 : wm * 32 + m * 16;
                common::ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                    common::smem_addr(s.weights[stage] +
                                                      grouped_tma::address<S>(
                                                          base + (lane & 7) + ((lane >> 3 & 1) * 8),
                                                          ki * 32 + (lane >> 4) * 16)));
                as[m] = s.recovery[scale_slot].data.scales[csf_scale_tile::scale_word(
                    base + ((lane & 1) << 3) + (lane >> 2), physical * 2 + ki)];
            }
#pragma unroll
            for (int n = 0; n < S::atoms; ++n) {
                if (n + 1 < S::atoms && (wn * S::warp_columns + (n + 1) * 8 < columns))
                    load_b(n + 1, (n + 1) & 1);
                if (wn * S::warp_columns + n * 8 < columns) {
#pragma unroll
                    for (int m = 0; m < 2; ++m)
                        common::mma_nvfp4_e4m3(acc[m][n][0], acc[m][n][1], acc[m][n][2],
                                               acc[m][n][3], a[m][0], a[m][1], a[m][2], a[m][3],
                                               pb0[n & 1], pb1[n & 1], as[m], pbs[n & 1]);
                }
            }
        }
        common::cta_mbarrier_arrive(&s.empty[stage]);
    }
}

template <class S>
__global__ __launch_bounds__(S::threads, S::min_blocks) void contract(
    const __grid_constant__ Maps maps, ExpertMatrixDeviceView bank, ActivationScale input_scale,
    ActivationScale product_scale, Groups groups, const float* coefficients, std::uint8_t* product,
    std::uint8_t* product_scales, float* output, int minimum, int maximum, float limit) {
    const int population_load = balanced_group_load(groups);
    if (groups.load_range[1] <= minimum ||
        (population_load ? population_load : groups.load_range[0]) > maximum)
        return;
    extern __shared__ __align__(512) unsigned char memory[];
    auto& s       = *reinterpret_cast<Shared<S>*>(memory);
    const int tid = threadIdx.x, consumer = tid - S::producer_threads, lane = tid % 32;
    const int wm = (consumer / 32) % 2, wn = (consumer / 32) / 2;
    if (!tid) {
        for (int stage = 0; stage < 2; ++stage) {
            common::cta_mbarrier_init(&s.full[stage], 32);
            common::cta_mbarrier_init(&s.empty[stage], S::consumer_threads);
            common::cta_mbarrier_init(&s.scale_full[stage], 32);
            common::cta_mbarrier_init(&s.scale_empty[stage], S::consumer_threads);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    constexpr int row_tiles = (S::gate ? S::intermediate : kHidden) / S::features;
    constexpr int window    = S::gate ? row_tiles : 8;
    const int jobs = *groups.jobs, tasks = jobs * row_tiles,
              rotation = (int(blockIdx.x) % (S::k / 256)) * 2;
    const auto process = [&]<int Role>() {
        // A job can contain one tile. Destination slots follow transfers across
        // jobs so the next producer cannot overwrite the preceding epilogue.
        int sequence = 0, transfer = 0;
        for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
            const int row_tile =
                S::gate ? task % row_tiles : task / (jobs * window) * window + task % window;
            const int job     = S::gate ? task / row_tiles : task / window % jobs;
            const int column0 = groups.job_columns[job];
            if (column0 % (4 * S::columns)) continue;
            const int expert = groups.job_experts[job], offset = groups.offsets[expert],
                      count          = groups.offsets[expert + 1] - offset;
            const int dispatch_count = population_load ? population_load : count;
            if (dispatch_count <= minimum || dispatch_count > maximum) continue;
            const unsigned physical =
                bank.expert_indices ? __ldg(bank.expert_indices + expert) : expert;
            if (physical == UINT32_MAX) continue;
            const int row0 = row_tile * S::features, slot = sequence & 1;
            if constexpr (Role == 0) {
                common::cta_mbarrier_wait(&s.scale_empty[slot], 1U ^ ((sequence / 2) & 1U));
                const auto weight = common::nvfp4_mul_expert_matrix<S::csf>(bank, expert);
                if constexpr (S::csf) {
                    csf_scale_tile::restore_warp<S::intermediate, S::gate, Decoder<S>>(
                        s.recovery[slot], weight, row0);
                } else {
                    for (int index = lane; index < 64 * (S::k / 64); index += 32) {
                        const int row = index % 64, g4 = index / 64,
                                  wr = row0 +
                                       (S::gate && row >= 32 ? row - 32 + S::intermediate : row);
                        s.recovery[slot].data.scales[csf_scale_tile::scale_word(row, g4)] =
                            *reinterpret_cast<const unsigned*>(
                                weight.scales + common::nvfp4_scale_byte_offset(wr, g4 * 4, S::k));
                    }
                }
                if (!lane) {
                    s.factors[slot]         = input_scale.get(expert) * weight.multiplier;
                    s.product_factors[slot] = product_scale.get(expert);
                }
                __syncwarp();
                common::cta_mbarrier_arrive(&s.scale_full[slot]);
            } else {
                if constexpr (Role == 2) {
                    common::cta_mbarrier_wait(&s.scale_full[slot], (sequence / 2) & 1U);
                }
                const int physical_row =
                    int(std::uint64_t(physical) * bank.code_stride / (S::k / 2)) + row0;
                for (int tile = 0; tile < 4; ++tile) {
                    const int column_begin = column0 + tile * S::columns,
                              columns      = min(S::columns, count - column_begin);
                    if (columns <= 0) break;
                    const int begin = offset + column_begin;
                    if constexpr (Role == 1) {
                        if constexpr (!S::gate) {
                            for (int c = lane; c < columns; c += 32) {
                                const int a = __ldg(groups.assignments + begin + c);
                                s.destinations[transfer & 1][c] = {a / kTopK,
                                                                   __ldg(coefficients + a)};
                            }
                        }
                        produce<S>(s, maps, physical_row, begin, rotation);
                    } else {
                        float acc[2][S::atoms][4]{};
                        consume<S>(s, columns, rotation, slot, acc);
                        const float factor = s.factors[slot];
#pragma unroll
                        for (int n = 0; n < S::atoms; ++n) {
                            const int c = wn * S::warp_columns + n * 8 + 2 * (lane & 3);
                            if constexpr (S::gate) {
#pragma unroll
                                for (int t = 0; t < 2; ++t) {
                                    const float a = acc[0][n][t] * factor,
                                                b = acc[0][n][t + 2] * factor;
                                    const float lo =
                                        common::silu(fminf(a, limit)) *
                                        fmaxf(-limit, fminf(acc[1][n][t] * factor, limit));
                                    const float hi =
                                        common::silu(fminf(b, limit)) *
                                        fmaxf(-limit, fminf(acc[1][n][t + 2] * factor, limit));
                                    store_product<S::intermediate, true>(
                                        lo, hi, s.product_factors[slot], product, product_scales,
                                        c + t < columns ? begin + c + t : -1, row0 + wm * 16);
                                }
                            } else {
#pragma unroll
                                for (int t = 0; t < 2; ++t)
                                    if (c + t < columns) {
                                        const auto d = s.destinations[transfer & 1][c + t];
                                        float* out   = output + std::int64_t(d.token) * kHidden;
#pragma unroll
                                        for (int m = 0; m < 2; ++m)
#pragma unroll
                                            for (int v = 0; v < 2; ++v)
                                                atomicAdd(out + row0 + wm * 32 + m * 16 +
                                                              (lane >> 2) + v * 8,
                                                          (acc[m][n][t + v * 2] * factor) *
                                                              d.coefficient);
                                    }
                            }
                        }
                    }
                    if constexpr (!S::gate) ++transfer;
                }
                if constexpr (Role == 2) {
                    __syncwarp();
                    common::cta_mbarrier_arrive(&s.scale_empty[slot]);
                }
            }
            ++sequence;
        }
    };
    if (tid < 32)
        process.template operator()<0>();
    else if (tid < 64)
        process.template operator()<1>();
    else
        process.template operator()<2>();
}

template <int I, bool Gate, int Columns, bool Csf>
void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            const Plan& plan, const Storage& storage, CudaExecutionView execution, int minimum,
            int maximum) {
    static_assert((Gate && Columns == 256) || (!Gate && Csf && I == 2048 && Columns == 128),
                  "unregistered full-scale expert tile");
    using S         = Shape<I, Gate, Columns, Csf>;
    const auto bank = Gate ? experts.bank.gate_up : experts.bank.down;
    Maps maps;
    maps.weights = make_map(const_cast<std::byte*>(bank.codes), S::k / 2,
                            std::uint64_t(bank.physical_count) * bank.code_stride / (S::k / 2), 64,
                            Gate ? 32 : 64, CU_TENSOR_MAP_SWIZZLE_64B);
    maps.inputs  = make_map(Gate ? storage.input_codes : storage.product_codes, S::k / 2,
                           kTopK * x.ne[1], 64, Columns, CU_TENSOR_MAP_SWIZZLE_64B);
    maps.scales  = make_map(Gate ? storage.input_scales : storage.product_scales, S::k / 16,
                           kTopK * x.ne[1], 16, Columns, CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto calibration = [](const ActivationCalibrationView& v) {
        return v.values().size() == 1
                   ? ActivationScale{nullptr, v.values().front()}
                   : ActivationScale{static_cast<const float*>(v.device_values.data), 1};
    };
    constexpr auto kernel = contract<S>;
    const int bytes       = common::prepare_shared<sizeof(Shared<S>), kernel, true>();
    const int jobs =
        kTopK * x.ne[1] / plan.job_columns() + std::min(kExperts, kTopK * x.ne[1] / (minimum + 1));
    const int blocks = std::min(jobs * ((Gate ? I : kHidden) / S::features),
                                execution.multiprocessor_count() * S::min_blocks);
    kernel<<<blocks, S::threads, bytes, execution.stream()>>>(
        maps, bank,
        calibration(Gate ? experts.activation_calibration.gate_up
                         : experts.activation_calibration.down),
        calibration(experts.activation_calibration.down), storage.groups,
        static_cast<const float*>(routes.coefficients.data), storage.product_codes,
        static_cast<std::uint8_t*>(storage.product_scales), storage.routed_sum, minimum, maximum,
        plan.clamp_limit);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::grouped_scale_cache
