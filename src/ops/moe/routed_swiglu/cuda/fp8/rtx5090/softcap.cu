#include "ops/moe/routed_swiglu/cuda/fp8/rtx5090/softcap_mma.cuh"

#include "ops/common/cuda/reduce.cuh"
#include "ops/common/host/launch.h"
#include "ops/common/quant/fp8.cuh"
#include "ops/moe/kernels/cuda/block_scaled/grouping.cuh"
#include "ops/moe/kernels/cuda/output.cuh"

namespace ninfer::ops::detail::moe_routed::mxfp8_softcap {
namespace {
using moe_cuda::block_scaled::clamped::Groups;
using moe_cuda::expert_in_subset;

struct Routes {
    const int* ids;
    const float* coefficients;
    const int* mask;
    ExpertSubsetKind subset;

    __device__ bool includes(int expert) const { return expert_in_subset(subset, mask, expert); }
};

struct Addressing {
    const std::uint64_t* offsets;
    const std::uint32_t* indices;
    unsigned count;

    __device__ bool contains(int expert) const {
        if (offsets) return offsets[expert] != kExpertAbsentOffset;
        return !indices || indices[expert] < count;
    }
};

Addressing addressing(const ExpertMatrixDeviceView& bank) {
    return {bank.offsets, bank.expert_indices, bank.physical_count};
}

__device__ int scan_experts(int* warp_totals, int value) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        const int other = __shfl_up_sync(0xffffffff, value, delta);
        if (lane >= delta) value += other;
    }
    if (lane == 31) warp_totals[warp] = value;
    __syncthreads();
    for (int previous = 0; previous < warp; ++previous) value += warp_totals[previous];
    __syncthreads();
    return value;
}

__device__ bool active(Routes routes, Addressing gate, Addressing down, int expert) {
    return routes.includes(expert) && gate.contains(expert) && down.contains(expert);
}

__device__ void publish_groups(Groups groups, int count, int columns, int* scan) {
    const int expert      = threadIdx.x;
    groups.counts[expert] = count;
    moe_cuda::block_scaled::publish_load_range<experts>(groups.load_range, count);
    const int end          = scan_experts(scan, count);
    groups.offsets[expert] = end - count;
    if (groups.cursors) groups.cursors[expert] = end - count;
    if (expert == experts - 1) groups.offsets[experts] = end;
    const int jobs       = (count + columns - 1) / columns;
    const int after_jobs = scan_experts(scan, jobs);
    for (int j = 0; j < jobs; ++j) {
        groups.job_experts[after_jobs - jobs + j] = expert;
        groups.job_columns[after_jobs - jobs + j] = j * columns;
    }
    if (expert == experts - 1) *groups.jobs = after_jobs;
}

__global__ void group_small(Routes routes, Addressing gate, Addressing down, Groups groups,
                            int assignments, int columns) {
    __shared__ int counts[experts], scan[experts / 32], ranks[top_k * single_block_tokens];
    const int expert = threadIdx.x;
    counts[expert]   = 0;
    __syncthreads();
    for (int a = expert; a < assignments; a += experts) {
        const int e = routes.ids[a];
        ranks[a]    = active(routes, gate, down, e) ? atomicAdd(counts + e, 1) : -1;
    }
    __syncthreads();
    publish_groups(groups, counts[expert], columns, scan);
    __syncthreads();
    for (int a = expert; a < assignments; a += experts)
        if (ranks[a] >= 0) {
            const int e = routes.ids[a];
            const int rank =
                groups.load_range && counts[e] == assignments / top_k ? a / top_k : ranks[a];
            groups.assignments[groups.offsets[e] + rank] = a;
        }
}

__global__ void count_routes(Routes routes, Addressing gate, Addressing down, int* counts,
                             int assignments) {
    __shared__ int local[experts];
    local[threadIdx.x] = 0;
    __syncthreads();
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < assignments) {
        const int expert = routes.ids[a];
        if (active(routes, gate, down, expert)) atomicAdd(local + expert, 1);
    }
    __syncthreads();
    const int count = local[threadIdx.x];
    if (count) atomicAdd(counts + threadIdx.x, count);
}

__global__ void scan_routes(Groups groups, int columns) {
    __shared__ int scan[experts / 32];
    publish_groups(groups, groups.counts[threadIdx.x], columns, scan);
}

__global__ void pack_routes(Routes routes, Addressing gate, Addressing down, Groups groups,
                            int assignments) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= assignments) return;
    const int expert = routes.ids[a];
    if (active(routes, gate, down, expert)) {
        const int packed = groups.load_range && groups.counts[expert] == assignments / top_k
                               ? groups.offsets[expert] + a / top_k
                               : atomicAdd(groups.cursors + expert, 1);
        groups.assignments[packed] = a;
    }
}

template <bool Prescale>
__device__ __forceinline__ float bf16_dot_8(const uint2 weight, const uint4 x, std::uint8_t scale) {
    const unsigned words[]{x.x, x.y, x.z, x.w};
    const unsigned raw[]{weight.x, weight.y};
    if constexpr (Prescale) {
        // The represented weights can exceed BF16's exponent range in either direction.
        const double factor = __hiloint2double((int(scale) + 896) << 20, 0);
        double sum          = 0;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const auto w   = common::fp8_code2_to_float2(raw[pair / 2] >> (pair % 2 * 16));
            const double a = __uint_as_float(words[pair] << 16);
            const double b = __uint_as_float(words[pair] & 0xffff0000);
            sum            = fma(w.x * factor, a, sum);
            sum            = fma(w.y * factor, b, sum);
        }
        return static_cast<float>(sum);
    } else {
        float low = 0, high = 0;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const unsigned w = common::fp8_e4m3x2_to_bf16x2_bits(raw[pair / 2] >> (pair % 2 * 16));
            asm("{ .reg .b16 w0, w1, x0, x1;\n"
                "  mov.b32 {w0,w1}, %2; mov.b32 {x0,x1}, %3;\n"
                "  fma.rn.f32.bf16 %0, w0, x0, %0;\n"
                "  fma.rn.f32.bf16 %1, w1, x1, %1; }"
                : "+f"(low), "+f"(high)
                : "r"(w), "r"(words[pair]));
        }
        return (low + high) * common::ue8m0_to_float(scale);
    }
}

constexpr int vector_gate_threads = 256;

__global__ __launch_bounds__(vector_gate_threads) void vector_gate(const __nv_bfloat16* x,
                                                                   ExpertMatrixDeviceView bank,
                                                                   Addressing down, Routes routes,
                                                                   SoftcappedActivation activation,
                                                                   float* product) {
    const int assignment = blockIdx.y, expert = routes.ids[assignment];
    if (!routes.includes(expert) || !down.contains(expert)) return;
    const auto weight = resolve(bank, expert);
    if (!weight.codes) return;
    __shared__ uint4 input[hidden / 8];
    const auto* source = reinterpret_cast<const uint4*>(x + assignment / top_k * hidden);
    // BF16 magnitudes below 2^116 are safe for unscaled eight-value dots.
    // Test the top two exponent bits of each staged value, which covers all
    // magnitudes at or above that limit.
    const auto large = [](uint4 values) {
        const auto high_exponent = [](unsigned pair) { return pair & (pair >> 1); };
        return (high_exponent(values.x) | high_exponent(values.y) | high_exponent(values.z) |
                high_exponent(values.w)) &
               0x20002000;
    };
    static_assert(hidden / 8 > vector_gate_threads && hidden / 8 <= 2 * vector_gate_threads);
    const int tid    = threadIdx.x;
    const auto first = source[tid];
    input[tid]       = first;
    unsigned extreme = large(first);
    if (tid < hidden / 8 - vector_gate_threads) {
        const int next    = tid + vector_gate_threads;
        const auto values = source[next];
        input[next]       = values;
        extreme |= large(values);
    }
    const bool prescale = __syncthreads_or(extreme != 0);
    const int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    const auto project = [&]<bool Prescale>() {
        float gate = 0, up = 0;
        for (int v = lane; v < hidden / 8; v += 32) {
            const uint4 value = input[v];
            const auto dot    = [&](int r) {
                const auto codes = common::load_vec<uint2>(weight.codes + r * hidden + v * 8);
                const auto scale = weight.scales[r * (hidden / 32) + v / 4];
                return bf16_dot_8<Prescale>(codes, value, scale);
            };
            gate += dot(row);
            up += dot(row + intermediate);
        }
        gate = common::warp_reduce_sum(gate);
        up   = common::warp_reduce_sum(up);
        if (!lane) product[assignment * intermediate + row] = activation(gate, up);
    };
    if (prescale)
        project.template operator()<true>();
    else
        project.template operator()<false>();
}

__device__ __forceinline__ float fp32_dot_8(uint2 weight, float4 x0, float4 x1) {
    const float values[]{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const unsigned raw[]{weight.x, weight.y};
    float low = 0, high = 0;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const auto w = common::fp8_e4m3x2_to_bf16x2_bits(raw[pair / 2] >> (pair % 2 * 16));
        low          = fmaf(__uint_as_float(w << 16), values[pair * 2], low);
        high         = fmaf(__uint_as_float(w & 0xffff0000), values[pair * 2 + 1], high);
    }
    return low + high;
}

template <int Rows, class Output>
__global__ __launch_bounds__(256) void vector_down(const float* product,
                                                   ExpertMatrixDeviceView bank, Addressing gate,
                                                   Routes routes, Output output) {
    const int token = blockIdx.y, route = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int assignment = token * top_k + route, expert = routes.ids[assignment];
    const int row_begin = blockIdx.x * Rows;
    float result[Rows]{};
    if (routes.includes(expert) && gate.contains(expert)) {
        const auto weight = resolve(bank, expert);
        if (weight.codes) {
            constexpr int vectors = intermediate / 8;
            for (int v = lane; v < vectors; v += 32) {
                const auto* source = product + assignment * intermediate + v * 8;
                const auto x0      = common::load_vec<float4>(source);
                const auto x1      = common::load_vec<float4>(source + 4);
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    const int row = row_begin + r;
                    const auto w =
                        common::load_vec<uint2>(weight.codes + row * intermediate + v * 8);
                    const auto scale = weight.scales[row * (intermediate / 32) + v / 4];
                    result[r] =
                        fmaf(fp32_dot_8(w, x0, x1), common::ue8m0_to_float(scale), result[r]);
                }
            }
            const float coefficient = routes.coefficients[assignment];
#pragma unroll
            for (int r = 0; r < Rows; ++r)
                result[r] = common::warp_reduce_sum(result[r]) * coefficient;
        }
    }
    __shared__ float partial[top_k][Rows];
    if (!lane)
        for (int r = 0; r < Rows; ++r) partial[route][r] = result[r];
    __syncthreads();
    if (threadIdx.x < Rows) {
        float value = 0;
#pragma unroll
        for (int j = 0; j < top_k; ++j) value += partial[j][threadIdx.x];
        const int index = token * hidden + row_begin + threadIdx.x;
        output.store(index, output.initial(index, true) + value, true);
    }
}

template <class Output>
__global__ void finish(const float* sums, int elements, Output output) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (index >= elements) return;
    auto value = common::load_vec<float4>(sums + index);
    if constexpr (std::is_same_v<Output, moe_cuda::BfloatResult>) {
        if (output.add_residual) {
            const auto lo =
                __bfloat1622float2(common::load_vec<__nv_bfloat162>(output.data + index));
            const auto hi =
                __bfloat1622float2(common::load_vec<__nv_bfloat162>(output.data + index + 2));
            value.x += lo.x;
            value.y += lo.y;
            value.z += hi.x;
            value.w += hi.y;
        }
        alignas(8) const __nv_bfloat162 converted[]{__floats2bfloat162_rn(value.x, value.y),
                                                    __floats2bfloat162_rn(value.z, value.w)};
        common::store_vec(output.data + index, common::load_vec<uint2>(converted));
    } else {
        common::store_vec(output.data + index, value);
    }
}

__global__ void quantize_input(const __nv_bfloat16* x, Mxfp8A8Workspace input, int tokens) {
    const int index      = blockIdx.x * blockDim.x + threadIdx.x;
    const int group      = index / 8;
    const bool live      = group < tokens * (hidden / 32);
    const unsigned lanes = __ballot_sync(0xffffffff, live);
    if (!live) return;
    const int offset  = index * 4;
    const auto a      = __bfloat1622float2(common::load_vec<__nv_bfloat162>(x + offset));
    const auto b      = __bfloat1622float2(common::load_vec<__nv_bfloat162>(x + offset + 2));
    float largest     = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
    largest           = fmaxf(largest, __shfl_xor_sync(lanes, largest, 4));
    largest           = fmaxf(largest, __shfl_xor_sync(lanes, largest, 2));
    largest           = fmaxf(largest, __shfl_xor_sync(lanes, largest, 1));
    const auto scale  = common::fp8_ue8m0_scale(largest);
    const unsigned lo = common::fp8_quant_code2(a.x, a.y, scale.inverse_scale);
    const unsigned hi = common::fp8_quant_code2(b.x, b.y, scale.inverse_scale);
    common::store_vec(input.codes + offset, lo | (hi << 16));
    if (!(index & 7))
        input.scales[group / (hidden / 32) * input_scale_stride + group % (hidden / 32)] =
            scale.scale;
}

} // namespace

void launch(const Tensor& input, const SwiGLUProfile& profile, const CudaSwiGLUExperts& weights,
            const ExpertRoutes& routes, ExpertSubset subset, MoeEpilogue epilogue, Tensor& output,
            Plan plan, const Storage& storage, CudaExecutionView execution) {
    const auto stream = execution.stream();
    const int tokens = input.ne[1], assignments = tokens * top_k;
    const auto* x    = static_cast<const __nv_bfloat16*>(input.data);
    const auto& bank = weights.bank;
    const Routes r{static_cast<const int*>(routes.ids.data),
                   static_cast<const float*>(routes.coefficients.data),
                   static_cast<const int*>(subset.mask.data), subset.kind};
    const SoftcappedActivation activation(*profile.softcap);
    if (plan.grouped()) {
        const auto gate = addressing(bank.gate_up), down = addressing(bank.down);
        if (plan.single_block) {
            group_small<<<1, experts, 0, stream>>>(r, gate, down, storage.groups, assignments,
                                                   plan.columns());
        } else {
            CUDA_CHECK(cudaMemsetAsync(storage.groups.counts, 0, experts * sizeof(int), stream));
            const int blocks = common::div_up(assignments, experts);
            count_routes<<<blocks, experts, 0, stream>>>(r, gate, down, storage.groups.counts,
                                                         assignments);
            scan_routes<<<1, experts, 0, stream>>>(storage.groups, plan.columns());
            pack_routes<<<blocks, experts, 0, stream>>>(r, gate, down, storage.groups, assignments);
        }
        if (plan.gate_a8)
            quantize_input<<<common::div_up(tokens * (hidden / 4), 256), 256, 0, stream>>>(
                x, storage.input, tokens);
        CUDA_CHECK(cudaGetLastError());
        launch_mma(x, bank, r.coefficients, activation, plan, storage, tokens, stream);
    }
    moe_cuda::with_output(output, epilogue, [&](auto destination) {
        if (plan.grouped()) {
            const int elements = tokens * hidden;
            finish<<<common::div_up(elements / 4, 256), 256, 0, stream>>>(storage.sum, elements,
                                                                          destination);
        } else {
            vector_gate<<<dim3(intermediate / 8, assignments), vector_gate_threads, 0, stream>>>(
                x, bank.gate_up, addressing(bank.down), r, activation, storage.vector_product);
            constexpr int rows = 4;
            vector_down<rows><<<dim3(hidden / rows, tokens), 256, 0, stream>>>(
                storage.vector_product, bank.down, addressing(bank.gate_up), r, destination);
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail::moe_routed::mxfp8_softcap
