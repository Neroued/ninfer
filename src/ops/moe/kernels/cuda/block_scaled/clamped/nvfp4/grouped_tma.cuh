#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/compute.cuh"
#include "ops/common/host/launch.h"
#include <cuda.h>

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::grouped_tma {

struct alignas(128) Maps {
    CUtensorMap weights, inputs, scales;
};

template <int I, bool Gate, int Columns, bool Csf>
struct Shape {
    static constexpr int k = Gate ? kHidden : I, intermediate = I, columns = Columns;
    static constexpr bool gate = Gate, csf = Csf;
    static constexpr int step   = Columns <= 64 ? 256 : 128;
    static constexpr int stages = Columns <= 64 ? 2 : 3, code_bytes = step / 2, scale_slots = 2;
    static constexpr int swizzle_alignment = step * 4;
    static constexpr int producer_threads  = Columns >= 256 ? 128 : 64,
                         producer_warps    = producer_threads / 32;
    static constexpr int consumer_threads  = Columns >= 256 ? 256 : 128;
    static constexpr int consumer_warps    = consumer_threads / 32;
    static constexpr int threads           = producer_threads + consumer_threads;
    static constexpr int warp_columns = Columns / (consumer_warps / 2), atoms = warp_columns / 8;
    static constexpr int features   = Gate ? 32 : 64;
    static constexpr int min_blocks = Columns <= 16    ? 3
                                      : Columns <= 64  ? 2
                                      : Columns == 256 ? 1
                                                       : 2;
    static_assert(k % (2 * step) == 0);
};

template <class S>
struct Metadata {};

template <int I, bool Gate, int Columns>
struct Metadata<Shape<I, Gate, Columns, true>> {
    common::Csf32RowBases128 bases;
    using S = Shape<I, Gate, Columns, true>;
    common::Csf32EncodedTile128x4 encoded[S::stages][S::step / 64];
};

template <class S, bool Down = !S::gate>
struct OutputMetadata {};

template <class S>
struct OutputMetadata<S, true> {
    struct Destination {
        int token;
        float coefficient;
    };

    // The next job can be produced while consumers finish the previous output.
    Destination destinations[2][S::columns];
};

template <class S>
struct Shared : Metadata<S>, OutputMetadata<S> {
    alignas(S::swizzle_alignment) std::uint8_t weights[S::stages][64 * S::code_bytes];
    alignas(S::swizzle_alignment) std::uint8_t inputs[S::stages][S::columns * S::code_bytes];
    alignas(16) unsigned weight_scales[S::stages][S::step * 2];
    // A 16-byte scale row covers two K128 stages. Two separate slots prevent
    // the next even stage from overwriting the preceding odd stage's scales.
    alignas(16) unsigned input_scales[S::scale_slots][S::columns * 4];
    float factors[2], product_factors[2];
    alignas(8) std::uint64_t full[S::stages], empty[S::stages];
};

template <class S>
__device__ __forceinline__ int address(int row, int byte) {
    if constexpr (S::step == 256) {
        return row * 128 + (byte ^ ((row & 7) << 4));
    } else {
        return row * 64 + (byte ^ (((row >> 1) & 3) << 4));
    }
}

template <class S>
__device__ __forceinline__ void produce(Shared<S>& tile, const Maps& maps,
                                        common::Nvfp4MultiplierMatrix weight, int physical_row,
                                        int row0, int begin, int rotation, int sequence) {
    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    const auto rows = S::gate ? make_uint4(row0, row0 + S::intermediate, 0, 0)
                              : make_uint4(row0, row0 + 32, 0, 0);
    if constexpr (S::csf) {
        if (warp == 0) common::csf32_copy_row_bases_async<2>(weight.csf, rows, tile.bases);
        common::cp_commit();
        common::cp_wait<0>();
        asm volatile("bar.sync 1, %0;" ::"n"(S::producer_threads) : "memory");
    }
    for (int kt = 0; kt < S::k / S::step; ++kt) {
        const int tick  = sequence * (S::k / S::step) + kt;
        const int stage = tick % S::stages, physical = (kt + rotation) % (S::k / S::step);
        common::cta_mbarrier_wait(&tile.empty[stage], 1U ^ ((tick / S::stages) & 1U));
        if (!tid) {
            const bool scale_pair = S::step == 256 || (kt & 1) == 0;
            const unsigned bytes =
                (64 + S::columns) * S::code_bytes + (scale_pair ? S::columns * 16 : 0);
            common::cta_mbarrier_arrive_expect_tx(&tile.full[stage], bytes);
            common::tma_load_2d(tile.weights[stage], &maps.weights, physical * S::code_bytes,
                                physical_row, &tile.full[stage]);
            if constexpr (S::gate)
                common::tma_load_2d(tile.weights[stage] + 32 * S::code_bytes, &maps.weights,
                                    physical * S::code_bytes, physical_row + S::intermediate,
                                    &tile.full[stage]);
            common::tma_load_2d(tile.inputs[stage], &maps.inputs, physical * S::code_bytes, begin,
                                &tile.full[stage]);
            if (scale_pair)
                common::tma_load_2d(
                    tile.input_scales[S::step == 256 ? stage : (kt / 2) % S::scale_slots],
                    &maps.scales, (S::step == 256 ? physical : physical / 2) * 16, begin,
                    &tile.full[stage]);
        }
        if constexpr (S::csf) {
            if constexpr (S::step == 256) {
#pragma unroll
                for (int group4 = warp; group4 < 4; group4 += S::producer_warps) {
                    common::csf32_copy_tile128x4_async<S::gate ? 2 * S::intermediate : kHidden,
                                                       S::k / 16, 2>(
                        weight.csf, rows, physical * 16 + group4 * 4, tile.encoded[stage][group4]);
                }
                common::cp_commit();
                common::cp_wait<0>();
                __syncwarp();
#pragma unroll
                for (int group4 = warp; group4 < 4; group4 += S::producer_warps) {
#pragma unroll
                    for (int part = 0; part < 2; ++part) {
                        common::csf32_decode_word<256, S::gate ? 2 * S::intermediate : kHidden,
                                                  S::k / 16, true>(
                            weight.csf, rows, physical * 16 + group4 * 4, tile.bases,
                            tile.encoded[stage][group4], part,
                            reinterpret_cast<std::uint8_t*>(tile.weight_scales[stage]), part * 32,
                            group4 * 4);
                    }
                }
            } else {
                const int group4 = S::producer_warps == 4 ? warp / 2 : warp;
                if (S::producer_warps == 2 || !(warp & 1)) {
                    common::csf32_copy_tile128x4_async<S::gate ? 2 * S::intermediate : kHidden,
                                                       S::k / 16, 2>(
                        weight.csf, rows, physical * 8 + group4 * 4, tile.encoded[stage][group4]);
                }
                common::cp_commit();
                common::cp_wait<0>();
                if constexpr (S::producer_warps == 4) {
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                } else {
                    __syncwarp();
                }
#pragma unroll
                for (int part = S::producer_warps == 4 ? warp % 2 : 0; part < 2;
                     part += S::producer_warps / 2) {
                    common::csf32_decode_word<128, S::gate ? 2 * S::intermediate : kHidden,
                                              S::k / 16, true>(
                        weight.csf, rows, physical * 8 + group4 * 4, tile.bases,
                        tile.encoded[stage][group4], part,
                        reinterpret_cast<std::uint8_t*>(tile.weight_scales[stage]), part * 32,
                        group4 * 4);
                }
            }
        } else {
            for (int item = tid; item < S::step; item += S::producer_threads) {
                const int row = item % 64, group4 = item / 64;
                const int wr = row0 + (S::gate && row >= 32 ? row - 32 + S::intermediate : row);
                tile.weight_scales[stage][group4 * 128 + row] = *reinterpret_cast<const unsigned*>(
                    weight.scales + common::nvfp4_scale_byte_offset(
                                        wr, physical * (S::step / 16) + group4 * 4, S::k));
            }
        }
        __syncwarp();
        if (!lane) common::cta_mbarrier_arrive(&tile.full[stage]);
    }
    // Neither warp may overwrite the bases until both have decoded the final
    // tile of this expert. Consumers no longer access the compressed metadata.
    if constexpr (S::csf) asm volatile("bar.sync 1, %0;" ::"n"(S::producer_threads) : "memory");
}

template <class S>
__device__ __forceinline__ void consume(Shared<S>& tile, int columns, float (&acc)[2][S::atoms][4],
                                        int sequence) {
    const int consumer = threadIdx.x - S::producer_threads, lane = consumer & 31;
    const int warp = consumer / 32, wm = warp % 2, wn = warp / 2;
#pragma unroll 1
    for (int kt = 0; kt < S::k / S::step; ++kt) {
        const int tick  = sequence * (S::k / S::step) + kt;
        const int stage = tick % S::stages;
        common::cta_mbarrier_wait(&tile.full[stage], (tick / S::stages) & 1U);
#pragma unroll
        for (int ki = 0; ki < S::step / 64; ++ki) {
            unsigned pb0[2], pb1[2], pbs[2];
            const auto load_b = [&](int n, int slot) {
                const int column = wn * S::warp_columns + n * 8;
                common::ldmatrix_x2(
                    pb0[slot], pb1[slot],
                    common::smem_addr(
                        tile.inputs[stage] +
                        address<S>(column + (lane & 7), ki * 32 + ((lane >> 3) & 1) * 16)));
                pbs[slot] = tile.input_scales[S::step == 256 ? stage : (kt / 2) % S::scale_slots]
                                             [(column + (lane >> 2)) * 4 +
                                              (S::step == 256 ? 0 : (kt & 1) * 2) + ki];
            };
            if (wn * S::warp_columns < columns) load_b(0, 0);
            unsigned a[2][4], as[2];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int base = S::gate ? wm * 16 + m * 32 : wm * 32 + m * 16;
                const int row  = base + (lane & 7) + ((lane >> 3 & 1) * 8),
                          col  = ki * 32 + (lane >> 4) * 16;
                common::ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                    common::smem_addr(tile.weights[stage] + address<S>(row, col)));
                as[m] =
                    tile.weight_scales[stage][ki * 128 + base + ((lane & 1) << 3) + (lane >> 2)];
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
        if (!lane) common::cta_mbarrier_arrive(&tile.empty[stage]);
    }
}

template <class S>
__global__ __launch_bounds__(S::threads, S::min_blocks) void contract(
    const __grid_constant__ Maps maps, ExpertMatrixDeviceView bank, ActivationScale input_scale,
    ActivationScale product_scale, Groups groups, const float* coefficients, std::uint8_t* product,
    std::uint8_t* product_scales, float* sum, int minimum, int maximum, int subtiles, float limit) {
    const int population_load = balanced_group_load(groups);
    if (groups.load_range[1] <= minimum ||
        (population_load ? population_load : groups.load_range[0]) > maximum)
        return;
    extern __shared__ __align__(1024) unsigned char memory[];
    auto& tile    = *reinterpret_cast<Shared<S>*>(memory);
    const int tid = threadIdx.x, consumer = tid - S::producer_threads, lane = tid & 31;
    const int wm = (consumer / 32) % 2, wn = (consumer / 32) / 2;
    if (!tid) {
#pragma unroll
        for (int stage = 0; stage < S::stages; ++stage) {
            common::cta_mbarrier_init(&tile.full[stage], 1 + S::producer_warps);
            common::cta_mbarrier_init(&tile.empty[stage], S::consumer_warps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    constexpr int row_tiles  = (S::gate ? S::intermediate : kHidden) / S::features;
    constexpr int row_window = S::gate ? row_tiles : 8;
    const int jobs = *groups.jobs * subtiles, tasks = jobs * row_tiles;
    const int rotation = (int(blockIdx.x) % (S::k / (S::step * 2))) * 2;
    // Down keeps a 512-row output window in L2 while traversing every expert.
    // This bounds the reduction working set without a second accumulator tile
    // or compute on tokens which never selected the expert.
    const auto process = [&]<bool Producer>() {
        int sequence = 0;
        for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
            const int row_tile = S::gate
                                     ? task % row_tiles
                                     : task / (jobs * row_window) * row_window + task % row_window;
            const int item     = S::gate ? task / row_tiles : task / row_window % jobs;
            const int job = item / subtiles, subtile = item % subtiles,
                      row0   = row_tile * S::features;
            const int expert = groups.job_experts[job], offset = groups.offsets[expert],
                      end            = groups.offsets[expert + 1];
            const int count          = end - offset;
            const int dispatch_count = population_load ? population_load : count;
            if (dispatch_count <= minimum || dispatch_count > maximum) continue;
            const int begin   = offset + groups.job_columns[job] + subtile * S::columns;
            const int columns = min(S::columns, end - begin);
            if (columns <= 0) continue;
            const unsigned physical =
                bank.expert_indices ? __ldg(bank.expert_indices + expert) : expert;
            if (physical == UINT32_MAX) continue;
            if constexpr (Producer) {
                const auto weight = common::nvfp4_mul_expert_matrix<S::csf>(bank, expert);
                if (!tid) {
                    tile.factors[sequence & 1] = input_scale.get(expert) * weight.multiplier;
                    tile.product_factors[sequence & 1] = product_scale.get(expert);
                }
                if constexpr (!S::gate) {
                    for (int column = tid; column < columns; column += S::producer_threads) {
                        const int assignment = __ldg(groups.assignments + begin + column);
                        tile.destinations[sequence & 1][column] = {
                            assignment / kTopK, __ldg(coefficients + assignment)};
                    }
                }
                const int physical_row =
                    int((weight.codes - reinterpret_cast<const std::uint8_t*>(bank.codes)) /
                        (S::k / 2)) +
                    row0;
                produce<S>(tile, maps, weight, physical_row, row0, begin, rotation, sequence);
            } else {
                float acc[2][S::atoms][4]{};
                consume<S>(tile, columns, acc, sequence);
                const float factor         = tile.factors[sequence & 1];
                const float product_factor = tile.product_factors[sequence & 1];
#pragma unroll
                for (int n = 0; n < S::atoms; ++n) {
                    const int column = wn * S::warp_columns + n * 8 + 2 * (lane & 3);
                    if constexpr (S::gate) {
#pragma unroll
                        for (int t = 0; t < 2; ++t) {
                            const float g0 = acc[0][n][t] * factor, g1 = acc[0][n][t + 2] * factor;
                            const float h0 = common::silu(fminf(g0, limit)) *
                                             fmaxf(-limit, fminf(acc[1][n][t] * factor, limit));
                            const float h1 = common::silu(fminf(g1, limit)) *
                                             fmaxf(-limit, fminf(acc[1][n][t + 2] * factor, limit));
                            store_product<S::intermediate, true>(
                                h0, h1, product_factor, product, product_scales,
                                column + t < columns ? begin + column + t : -1, row0 + wm * 16);
                        }
                    } else {
#pragma unroll
                        for (int t = 0; t < 2; ++t)
                            if (column + t < columns) {
                                const auto metadata = tile.destinations[sequence & 1][column + t];
                                float* destination  = sum + std::int64_t(metadata.token) * kHidden;
                                const float coefficient = metadata.coefficient;
#pragma unroll
                                for (int m = 0; m < 2; ++m)
#pragma unroll
                                    for (int v = 0; v < 2; ++v) {
                                        const int row =
                                            row0 + wm * 32 + m * 16 + (lane >> 2) + v * 8;
                                        atomicAdd(destination + row,
                                                  (acc[m][n][t + v * 2] * factor) * coefficient);
                                    }
                            }
                    }
                }
            }
            // K/stages is even for all supported matrices: full/empty parity returns
            // to its initial value. Producers may start the next task while consumers
            // finish their epilogue; stage ownership protects every shared operand.
            ++sequence;
        }
    };
    if (tid < S::producer_threads)
        process.template operator()<true>();
    else
        process.template operator()<false>();
}

inline CUtensorMap make_map(void* pointer, std::uint64_t columns, std::uint64_t rows,
                            std::uint32_t box_columns, std::uint32_t box_rows,
                            CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    const std::uint64_t dimensions[]{columns, rows}, strides[]{columns};
    const std::uint32_t boxes[]{box_columns, box_rows}, steps[]{1, 1};
    const auto result =
        cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, pointer, dimensions, strides,
                               boxes, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) throw std::runtime_error("native expert TMA descriptor failed");
    return map;
}

template <int I, bool Gate, int Columns, bool Csf>
void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            const Plan& plan, const Storage& storage, CudaExecutionView execution, int minimum,
            int maximum) {
    using S         = Shape<I, Gate, Columns, Csf>;
    const auto bank = Gate ? experts.bank.gate_up : experts.bank.down;
    Maps maps;
    constexpr auto swizzle =
        S::step == 256 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
    maps.weights = make_map(const_cast<std::byte*>(bank.codes), S::k / 2,
                            std::uint64_t(bank.physical_count) * bank.code_stride / (S::k / 2),
                            S::code_bytes, Gate ? 32 : 64, swizzle);
    maps.inputs  = make_map(Gate ? storage.input_codes : storage.product_codes, S::k / 2,
                           kTopK * x.ne[1], S::code_bytes, Columns, swizzle);
    maps.scales  = make_map(Gate ? storage.input_scales : storage.product_scales, S::k / 16,
                           kTopK * x.ne[1], 16, Columns, CU_TENSOR_MAP_SWIZZLE_NONE);
    const auto calibration = [](const ActivationCalibrationView& v) {
        return v.values().size() == 1
                   ? ActivationScale{nullptr, v.values().front()}
                   : ActivationScale{static_cast<const float*>(v.device_values.data), 1};
    };
    constexpr auto kernel = contract<S>;
    const int bytes       = common::prepare_shared<sizeof(Shared<S>), kernel, true>();
    const int subtiles = (std::min({plan.job_columns(), maximum, x.ne[1]}) + Columns - 1) / Columns;
    const int jobs     = (kTopK * x.ne[1] / plan.job_columns() +
                      std::min(kExperts, kTopK * x.ne[1] / (minimum + 1))) *
                     subtiles;
    int blocks = std::min(jobs * ((Gate ? I : kHidden) / S::features),
                           execution.multiprocessor_count() * S::min_blocks);
    // A stride divisible by window*subtiles can pin entire persistent CTAs
    // to empty column tails. Rotate through every residue of that power-of-two
    // period, keeping one fewer CTA than the resident limit when necessary.
    if constexpr (!Gate) {
        if (subtiles > 1 && blocks > 1 && !(blocks & 1)) --blocks;
    }
    kernel<<<blocks, S::threads, bytes, execution.stream()>>>(
        maps, bank,
        calibration(Gate ? experts.activation_calibration.gate_up
                         : experts.activation_calibration.down),
        calibration(experts.activation_calibration.down), storage.groups,
        static_cast<const float*>(routes.coefficients.data), storage.product_codes,
        static_cast<std::uint8_t*>(storage.product_scales), storage.routed_sum, minimum, maximum,
        subtiles, plan.clamp_limit);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4::grouped_tma
