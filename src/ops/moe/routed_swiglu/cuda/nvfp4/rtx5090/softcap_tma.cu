#include "ops/moe/routed_swiglu/cuda/nvfp4/rtx5090/softcap_tma.cuh"

#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/activation_quant.cuh"

#include <numeric>

namespace ninfer::ops::detail::moe_routed::softcap::tma {
namespace {
using clamped::nvfp4::ActivationQuantizer;
using clamped::nvfp4::ActivationScale;
using moe_cuda::block_scaled::SoftcappedActivation;
using moe_cuda::block_scaled::scale_accumulator_wide;

constexpr int kStep = 128, code_bytes = kStep / 2;
constexpr int scale_box_rows = 64, scale_box_bytes = 16 * scale_box_rows;

struct alignas(128) Maps {
    CUtensorMap codes, weights, input_scales, weight_scales;
};

template <bool Gate, int Columns, int Features, int Stages, int RowWarps, int TokenWarps,
          bool Persistent = false>
struct Shape {
    static constexpr bool gate       = Gate;
    static constexpr bool persistent = Persistent;
    static_assert(!Persistent || Gate);
    static_assert(Stages >= 2);
    static constexpr int columns = Columns, features = Features, stages = Stages;
    static constexpr int k     = Gate ? hidden : intermediate;
    static constexpr int steps = k / kStep;
    static_assert(!Persistent || steps % 2 == 0);
    static constexpr int rows     = Gate ? intermediate : hidden;
    static constexpr int branches = Gate ? 2 : 1, row_warps = RowWarps;
    static constexpr int warps                   = RowWarps * TokenWarps;
    static constexpr bool redistribute_registers = warps == 16;
    static constexpr int producer_warps          = redistribute_registers ? 4 : 1;
    static constexpr int threads                 = (warps + producer_warps) * 32;
    static constexpr int warp_columns = Columns / TokenWarps, warp_features = Features / RowWarps;
    static constexpr int mmas_tokens = warp_columns / 16, mmas_rows = warp_features / 8;
    // A feature tile can start partway through a 128-row block of weight scales.
    static constexpr int scale_tiles = (Features + 127 - std::gcd(Features, 128) + 128) / 128;
    static constexpr int input_scale_stride = Gate ? hidden / 16 : product_scale_stride;
    static_assert(rows % Features == 0 && warp_columns % 16 == 0);
    static_assert(warp_features % (Gate ? 16 : 8) == 0);
};

template <class S>
struct alignas(128) Tile {
    alignas(128) std::uint8_t input[S::stages][S::columns * code_bytes];
    alignas(128) std::uint8_t weights[S::stages][S::branches][S::features * code_bytes];
    alignas(128) unsigned input_scales[S::stages][S::columns][4];
    alignas(128) unsigned weight_scales[S::stages][S::branches][S::scale_tiles][256];
    std::uint64_t full[S::stages], empty[S::stages];
};

template <class S>
struct Destinations {
    int tokens[S::gate ? 1 : S::columns];
    float coefficients[S::gate ? 1 : S::columns];
};

CUtensorMap make_map(const void* storage, std::uint64_t columns, std::uint64_t rows,
                     std::uint32_t box_columns, std::uint32_t box_rows, CUtensorMapSwizzle swizzle,
                     CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
    CUtensorMap map{};
    const std::uint64_t dimensions[]{columns, rows}, strides[]{columns};
    const std::uint32_t box[]{box_columns, box_rows}, steps[]{1, 1};
    if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(storage),
                               dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               swizzle, promotion,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        throw std::runtime_error("routed SwiGLU: failed to encode SM120a TMA descriptor");
    return map;
}

bool affine(const ExpertMatrixDeviceView& bank, int k, int rows) {
    return !bank.offsets && bank.physical_count && bank.code_stride % (k / 2) == 0 &&
           bank.scale_stride % 16 == 0 &&
           (bank.physical_count - 1) * (bank.code_stride / (k / 2)) + rows <= INT_MAX &&
           (bank.physical_count - 1) * (bank.scale_stride / 16) + rows * k / 256 <= INT_MAX;
}

__device__ __forceinline__ int swizzle(int row, int column) {
    return row * code_bytes + (((column / 16) ^ ((row / 2) & 3)) * 16);
}

template <class S>
__device__ void produce(Tile<S>& tile, const Maps& maps, int input_begin, int code_row,
                        int scale_row, int row_begin, int epoch) {
    constexpr int bytes = (S::columns + S::branches * S::features) * code_bytes +
                          S::branches * S::scale_tiles * scale_box_bytes;
#pragma unroll 1
    for (int kt = 0; kt < S::steps; ++kt) {
        const int step       = (S::persistent ? epoch * S::steps : 0) + kt;
        const int stage      = step % S::stages;
        const int scale_slot = (step / 2) % S::stages;
        const int phase      = 1 ^ ((step / S::stages) & 1);
        common::cta_mbarrier_wait(&tile.empty[stage], phase);
        const bool read_scales = (kt & 1) == 0;
        auto* full             = &tile.full[stage];
        common::cta_mbarrier_arrive_expect_tx(full, bytes + (read_scales ? S::columns * 16 : 0));
        common::tma_load_2d(tile.input[stage], &maps.codes, kt * code_bytes, input_begin, full);
#pragma unroll
        for (int b = 0; b < S::branches; ++b) {
            const int row = row_begin + b * S::rows;
            common::tma_load_2d(tile.weights[stage][b], &maps.weights, kt * code_bytes,
                                code_row + row, full);
#pragma unroll
            for (int m = 0; m < S::scale_tiles; ++m) {
                const int scale = ((row / 128 + m) * (S::k / 64) + kt * 2) * 32;
                common::tma_load_2d(tile.weight_scales[stage][b][m], &maps.weight_scales, 0,
                                    scale_row + scale, full);
            }
        }
        // Each scale slot belongs to a pair of K steps. Its next write is 2*stages
        // steps later, so waiting for this code stage also protects the old odd reader.
        if (read_scales)
            common::tma_load_2d(tile.input_scales[scale_slot], &maps.input_scales, (kt / 2) * 16,
                                input_begin, full);
    }
}

template <class S>
__device__ void consume(Tile<S>& tile, int row_begin, int lane, int warp, int columns,
                        float (&acc)[S::mmas_tokens][S::branches][S::mmas_rows][4], int epoch) {
    const int warp_m = warp / S::row_warps, warp_n = warp % S::row_warps;
    const int a_row  = (lane & 7) + (((lane >> 3) & 1) * 8);
    const int a_byte = (lane >> 4) * 16;
    const int b_row = lane & 7, b_byte = ((lane >> 3) & 1) * 16;
    const int a_scale_row  = ((lane & 1) * 8) + (lane >> 2);
    const int b_scale_row  = lane >> 2;
    const int column_begin = warp_m * S::warp_columns;
#pragma unroll 1
    for (int kt = 0; kt < S::steps; ++kt) {
        const int step       = (S::persistent ? epoch * S::steps : 0) + kt;
        const int stage      = step % S::stages;
        const int scale_slot = (step / 2) % S::stages;
        common::cta_mbarrier_wait(&tile.full[stage], (step / S::stages) & 1);
        if (column_begin >= columns) {
            if (!lane) common::cta_mbarrier_arrive(&tile.empty[stage]);
            continue;
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            unsigned a[S::mmas_tokens][4], ax[S::mmas_tokens];
#pragma unroll
            for (int m = 0; m < S::mmas_tokens; ++m) {
                const int base = column_begin + m * 16;
                if (base >= columns) continue;
                const auto* ptr = tile.input[stage] + swizzle(base + a_row, half * 32 + a_byte);
                common::ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3], common::smem_addr(ptr));
                ax[m] = tile.input_scales[scale_slot][base + a_scale_row][(kt & 1) * 2 + half];
            }
#pragma unroll
            for (int b = 0; b < S::branches; ++b) {
                const auto load_weight = [&](int n) {
                    uint3 fragment;
                    const int base = warp_n * S::warp_features + n * 8;
                    const auto* ptr =
                        tile.weights[stage][b] + swizzle(base + b_row, half * 32 + b_byte);
                    common::ldmatrix_x2(fragment.x, fragment.y, common::smem_addr(ptr));
                    const int row   = (row_begin & 127) + base + b_scale_row;
                    const int index = (half * 32 + (row & 31)) * 4 + ((row & 127) / 32);
                    fragment.z      = tile.weight_scales[stage][b][row / 128][index];
                    return fragment;
                };
                auto weight = load_weight(0);
#pragma unroll
                for (int n = 0; n < S::mmas_rows; ++n) {
                    const auto current = weight;
                    if (n + 1 < S::mmas_rows) weight = load_weight(n + 1);
#pragma unroll
                    for (int m = 0; m < S::mmas_tokens; ++m) {
                        if (column_begin + m * 16 >= columns) continue;
                        auto& d = acc[m][b][n];
                        common::mma_nvfp4_e4m3(d[0], d[1], d[2], d[3], a[m][0], a[m][1], a[m][2],
                                               a[m][3], current.x, current.y, ax[m], current.z);
                    }
                }
            }
        }
        __syncwarp();
        if (!lane) common::cta_mbarrier_arrive(&tile.empty[stage]);
    }
}

struct ProductGroup {
    uint2 codes;
    std::uint8_t scale;
};

__device__ __forceinline__ ProductGroup encode_product(float2 a, float2 b,
                                                       ActivationQuantizer quantizer, int lane) {
    float largest     = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
    largest           = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, 1));
    largest           = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, 2));
    const auto scale  = quantizer.scale(largest);
    const auto a_code = quantizer.pair(a, scale), b_code = quantizer.pair(b, scale);
    unsigned pair = (a_code | (b_code << 16)) << ((lane & 3) * 8 % 16);
    // Each four-lane subgroup holds two groups of eight features in the two MMAs.
    pair |= __shfl_xor_sync(0xffffffff, pair, 1);
    const unsigned other = __shfl_xor_sync(0xffffffff, pair, 2);
    const auto lo        = (pair & 65535) | (other << 16);
    const auto hi        = (pair >> 16) | (other & 0xffff0000);
    return {make_uint2(lo, hi), scale};
}

__device__ __forceinline__ void add_output(float* destination, float x, float y, float z, float w) {
    asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(destination), "f"(x), "f"(y), "f"(z), "f"(w)
                 : "memory");
}

struct Job {
    int expert, relative, packed, columns, row_begin;
    unsigned physical;
};

template <class S>
__device__ __forceinline__ Job get_job(clamped::Groups groups, ExpertMatrixDeviceView bank,
                                       int task) {
    constexpr int row_tiles = S::rows / S::features;
    const int j             = task / row_tiles;
    const int e = groups.job_experts[j], count = groups.counts[e];
    int relative = groups.job_columns[j], row = task % row_tiles;
    if constexpr (S::persistent) {
        // Within an expert, let adjacent work share a weight row tile.
        const int grouped_tiles = (count + S::columns - 1) / S::columns;
        const int index         = (relative / S::columns) * row_tiles + row;
        row                     = index / grouped_tiles;
        relative                = index % grouped_tiles * S::columns;
    }
    return {e,
            relative,
            groups.offsets[e] + relative,
            min(S::columns, count - relative),
            row * S::features,
            bank.expert_indices ? bank.expert_indices[e] : unsigned(e)};
}

template <class S>
__device__ __forceinline__ void
emit_tile(const Job job, const ExpertMatrixDeviceView& bank, const Destinations<S>& destinations,
          ActivationScale input_scale, ActivationScale product_scale,
          SoftcappedActivation activation, std::uint8_t* product, std::uint8_t* scales,
          float* output, int lane, int warp,
          const float (&acc)[S::mmas_tokens][S::branches][S::mmas_rows][4]) {
    const auto [expert, relative, packed, columns, row_begin, physical] = job;
    const int warp_column = (warp / S::row_warps) * S::warp_columns + lane / 4;
    const int warp_row    = row_begin + (warp % S::row_warps) * S::warp_features;
    if ((warp / S::row_warps) * S::warp_columns >= columns) return;
    const float input_multiplier  = input_scale.get(expert);
    const float weight_multiplier = bank.matrix_scales[physical * bank.matrix_scale_stride];
    const float factor            = input_multiplier * weight_multiplier;
    if constexpr (S::gate) {
        const ActivationQuantizer quantizer(product_scale.get(expert));
        const auto emit_products = [&]<bool Wide>() {
            const auto rescale = [&](float value) {
                if constexpr (Wide)
                    return scale_accumulator_wide(value, input_multiplier, weight_multiplier);
                return value * factor;
            };
#pragma unroll
            for (int m = 0; m < S::mmas_tokens; ++m)
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int column = warp_column + m * 16 + p * 8;
                    static_assert(S::mmas_rows % 4 == 0);
#pragma unroll
                    for (int n = 0; n < S::mmas_rows; n += 4) {
                        const auto pair = [&](int row) {
                            const auto& gate = acc[m][0][row];
                            const auto& up   = acc[m][1][row];
                            return make_float2(
                                activation(rescale(gate[2 * p]), rescale(up[2 * p])),
                                activation(rescale(gate[2 * p + 1]), rescale(up[2 * p + 1])));
                        };
                        const auto a = encode_product(pair(n), pair(n + 1), quantizer, lane);
                        const auto b = encode_product(pair(n + 2), pair(n + 3), quantizer, lane);
                        if (!(lane & 3) && column < columns) {
                            const int out = packed + column;
                            const int row = warp_row + n * 8;
                            common::store_vec(
                                product + out * (intermediate / 2) + row / 2,
                                make_uint4(a.codes.x, a.codes.y, b.codes.x, b.codes.y));
                            common::store_vec(scales + out * product_scale_stride + row / 16,
                                              std::uint16_t(a.scale | (b.scale << 8)));
                        }
                    }
                }
        };
        if (__builtin_expect(isfinite(factor), true))
            emit_products.template operator()<false>();
        else
            emit_products.template operator()<true>();
    } else {
#pragma unroll
        for (int m = 0; m < S::mmas_tokens; ++m)
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int column           = warp_column + m * 16 + p * 8;
                const bool live            = column < columns;
                const int token            = live ? destinations.tokens[column] : 0;
                const float coefficient    = live ? destinations.coefficients[column] : 0;
                const float weighted       = live ? coefficient * factor : 0;
                const bool finite_weighted = isfinite(weighted);
                const auto emit_rows       = [&]<bool Wide>() {
                    const auto rescale = [&](float value) {
                        if constexpr (Wide)
                            if (!finite_weighted)
                                return scale_accumulator_wide(value, input_multiplier,
                                                                    weight_multiplier, coefficient);
                        return value * weighted;
                    };
#pragma unroll
                    for (int n = 0; n < S::mmas_rows; ++n) {
                        const float x = rescale(acc[m][0][n][2 * p]);
                        const float y = rescale(acc[m][0][n][2 * p + 1]);
                        const int row = warp_row + n * 8 + (lane & 3) * 2;
                        const float z = __shfl_xor_sync(0xffffffff, x, 1);
                        const float w = __shfl_xor_sync(0xffffffff, y, 1);
                        if (live && !(lane & 1))
                            add_output(output + token * hidden + row, x, y, z, w);
                    }
                };
                if (__builtin_expect(__all_sync(0xffffffff, finite_weighted), true))
                    emit_rows.template operator()<false>();
                else
                    emit_rows.template operator()<true>();
            }
    }
}

template <class S>
__global__ __launch_bounds__(S::threads, 1) void kernel(
    const __grid_constant__ Maps maps, const __grid_constant__ ExpertMatrixDeviceView bank,
    const __grid_constant__ clamped::Groups groups,
    const __grid_constant__ ActivationScale input_scale,
    const __grid_constant__ ActivationScale product_scale,
    const __grid_constant__ SoftcappedActivation activation, const float* coefficients,
    std::uint8_t* product, std::uint8_t* scales, float* output, int tokens, bool broadcast) {
    constexpr int row_tiles = S::rows / S::features;
    const int tasks         = *groups.jobs * row_tiles;
    constexpr int chunk     = S::persistent ? 2 : 1;
    const auto task_at      = [=](int item) {
        return (blockIdx.x + item / chunk * gridDim.x) * chunk + item % chunk;
    };
    if (task_at(0) >= tasks) return;
    extern __shared__ __align__(128) std::uint8_t buffer[];
    auto& tile = *reinterpret_cast<Tile<S>*>(buffer);
    __shared__ Destinations<S> destinations;
    if constexpr (!S::gate) {
        const auto job = get_job<S>(groups, bank, task_at(0));
        if (job.physical != UINT32_MAX && threadIdx.x < job.columns) {
            const int assignment                   = groups.assignments[job.packed + threadIdx.x];
            destinations.tokens[threadIdx.x]       = assignment / top_k;
            destinations.coefficients[threadIdx.x] = coefficients[assignment];
        }
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S::stages; ++stage) {
            common::cta_mbarrier_init(&tile.full[stage], 1);
            common::cta_mbarrier_init(&tile.empty[stage], S::warps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    const bool shared_input = broadcast && groups.load_range[0] == tokens;
    if (threadIdx.x < S::producer_warps * 32) {
        if constexpr (S::redistribute_registers)
            asm volatile("setmaxnreg.dec.sync.aligned.u32 24;" : : : "memory");
        if (threadIdx.x != 0) return;
        int epoch = 0;
        // Keep the stage phase across jobs so this warp can fill the next tile
        // as soon as a consumer finishes a stage, including during the epilogue.
        for (int i = 0, task; (task = task_at(i)) < tasks; ++i) {
            const auto job = get_job<S>(groups, bank, task);
            if (job.physical != UINT32_MAX)
                produce<S>(tile, maps, shared_input ? job.relative : job.packed,
                           job.physical * (bank.code_stride / (S::k / 2)),
                           job.physical * (bank.scale_stride / 16), job.row_begin, epoch++);
            if constexpr (!S::persistent) break;
        }
        return;
    }
    if constexpr (S::redistribute_registers)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;" : : : "memory");
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32 - S::producer_warps;
    int epoch = 0;
    for (int i = 0, task; (task = task_at(i)) < tasks; ++i) {
        const auto job = get_job<S>(groups, bank, task);
        if (job.physical != UINT32_MAX) {
            float acc[S::mmas_tokens][S::branches][S::mmas_rows][4]{};
            consume<S>(tile, job.row_begin, lane, warp, job.columns, acc, epoch++);
            emit_tile<S>(job, bank, destinations, input_scale, product_scale, activation, product,
                         scales, output, lane, warp, acc);
        } else if constexpr (S::gate) {
            constexpr int threads = S::warps * 32;
            const int tid         = threadIdx.x - S::producer_warps * 32;
            for (int v = tid; v < job.columns * (S::features / 16); v += threads) {
                const int c = job.packed + v / (S::features / 16);
                const int r = job.row_begin / 16 + v % (S::features / 16);
                *reinterpret_cast<uint2*>(product + c * (intermediate / 2) + r * 8) = {};
                scales[c * product_scale_stride + r]                                = 0;
            }
        }
        if constexpr (!S::persistent) break;
    }
}

template <class S>
void launch(const ExpertMatrixDeviceView& bank, ActivationScale input_scale,
            ActivationScale product_scale, SoftcappedActivation activation,
            const clamped::Storage& storage, const float* coefficients, int jobs, int tokens,
            bool broadcast, CudaExecutionView execution) {
    constexpr int k = S::k, rows = S::rows * S::branches;
    const auto slots         = bank.physical_count - 1;
    const int assignments    = tokens * top_k;
    constexpr auto promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    Maps maps{make_map(S::gate ? storage.input_codes : storage.product_codes, k / 2, assignments,
                       code_bytes, S::columns, CU_TENSOR_MAP_SWIZZLE_64B),
              make_map(bank.codes, k / 2, slots * (bank.code_stride / (k / 2)) + rows, code_bytes,
                       S::features, CU_TENSOR_MAP_SWIZZLE_64B, promotion),
              make_map(S::gate ? storage.input_scales : storage.product_scales,
                       S::input_scale_stride, assignments, 16, S::columns,
                       CU_TENSOR_MAP_SWIZZLE_NONE),
              make_map(bank.scales, 16, slots * (bank.scale_stride / 16) + rows * k / 256, 16,
                       scale_box_rows, CU_TENSOR_MAP_SWIZZLE_NONE, promotion)};
    const int shared = common::prepare_shared<sizeof(Tile<S>), kernel<S>, true>();
    const int tasks  = jobs * (S::rows / S::features);
    const int blocks =
        S::persistent ? std::min(common::div_up(tasks, 2), 2 * execution.multiprocessor_count())
                      : tasks;
    kernel<S><<<blocks, S::threads, shared, execution.stream()>>>(
        maps, bank, storage.groups, input_scale, product_scale, activation, coefficients,
        storage.product_codes, static_cast<std::uint8_t*>(storage.product_scales),
        storage.routed_sum, tokens, broadcast);
}
} // namespace

bool supports(const ExpertBankDeviceView& bank) {
    return affine(bank.gate_up, hidden, 2 * intermediate) &&
           affine(bank.down, intermediate, hidden);
}

void contract(bool gate, Schedule schedule, const ExpertMatrixDeviceView& bank,
              ActivationScale input_scale, ActivationScale product_scale,
              SoftcappedActivation activation, const clamped::Storage& storage,
              const float* coefficients, int jobs, int tokens, bool broadcast,
              CudaExecutionView execution) {
    const auto run = [&]<class S>() {
        launch<S>(bank, input_scale, product_scale, activation, storage, coefficients, jobs, tokens,
                  gate && broadcast, execution);
    };
    if (schedule == Schedule::Sm120Tma64) {
        if (gate)
            run.template operator()<Shape<true, 64, 128, 3, 4, 2>>();
        else
            run.template operator()<Shape<false, 64, 128, 4, 4, 2>>();
    } else if (gate) {
        // Short prefills benefit from more CTAs on the few experts with work.
        constexpr int persistent_gate_tokens = 1024;
        if (tokens >= persistent_gate_tokens)
            run.template operator()<Shape<true, 128, 128, 3, 4, 4, true>>();
        else
            run.template operator()<Shape<true, 128, 96, 3, 3, 4>>();
    } else {
        run.template operator()<Shape<false, 128, 192, 3, 3, 4>>();
    }
}

} // namespace ninfer::ops::detail::moe_routed::softcap::tma
