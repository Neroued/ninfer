#include "ops/moe/routed_swiglu/cuda/nvfp4/rtx5090/softcap.h"

#include "ops/common/host/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/a16.cuh"
#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/compute.cuh"
#include "ops/moe/kernels/cuda/block_scaled/grouping.cuh"
#include "ops/moe/routed_swiglu/cuda/nvfp4/rtx5090/softcap_tma.cuh"

#include <bit>

namespace ninfer::ops::detail::moe_routed::softcap {
namespace {
using moe_cuda::expert_in_subset;
using moe_cuda::block_scaled::SoftcappedActivation;
using moe_cuda::block_scaled::scale_accumulator_wide;
using moe_cuda::block_scaled::publish_load_range;
namespace clamped = moe_cuda::block_scaled::clamped;
namespace nvfp4   = clamped::nvfp4;
using clamped::Groups;

__device__ int scan_experts(int* totals, int value) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
#pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        const int carry = __shfl_up_sync(0xffffffff, value, delta);
        if (lane >= delta) value += carry;
    }
    if (lane == 31) totals[warp] = value;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < experts / 32 - 1; ++i) {
        if (i < warp) value += totals[i];
    }
    __syncthreads();
    return value;
}

__device__ void publish_groups(Groups groups, int count, int columns, int* scan) {
    const int e      = threadIdx.x;
    groups.counts[e] = count;
    publish_load_range<experts>(groups.load_range, count);
    const int end     = scan_experts(scan, count);
    const int begin   = end - count;
    groups.offsets[e] = begin;
    if (groups.cursors) groups.cursors[e] = begin;
    if (e == experts - 1) groups.offsets[experts] = end;
    const int jobs    = (count + columns - 1) / columns;
    const int end_job = scan_experts(scan, jobs);
    for (int j = 0; j < jobs; ++j) {
        groups.job_experts[end_job - jobs + j] = e;
        groups.job_columns[end_job - jobs + j] = j * columns;
    }
    if (e == experts - 1) *groups.jobs = end_job;
}

__device__ void choose_grouped(Groups groups, int* use_grouped) {
    if (use_grouped && threadIdx.x == 0) {
        const int active = groups.load_range[2];
        const bool grouped =
            active && groups.offsets[experts] * 2 >= active * minimum_a4_reuse_twice;
        *use_grouped = grouped;
        if (!grouped) *groups.jobs = 0;
    }
}

__global__ void group_small(const int* ids, ExpertSubsetKind subset, const int* mask, Groups groups,
                            int assignments, int columns, int* use_grouped) {
    __shared__ int counts[experts], scan[experts / 32], ranks[top_k * single_block_tokens];
    const int e = threadIdx.x;
    counts[e]   = 0;
    __syncthreads();
    for (int a = e; a < assignments; a += experts) {
        const int expert = ids[a];
        ranks[a] = expert_in_subset(subset, mask, expert) ? atomicAdd(counts + expert, 1) : -1;
    }
    __syncthreads();
    const int tokens = assignments / top_k;
    publish_groups(groups, counts[e], columns, scan);
    __syncthreads();
    choose_grouped(groups, use_grouped);
    for (int a = e; a < assignments; a += experts)
        if (ranks[a] >= 0) {
            const int expert = ids[a];
            const int rank   = counts[expert] == tokens ? a / top_k : ranks[a];
            groups.assignments[groups.offsets[expert] + rank] = a;
        }
}

__global__ void count_routes(const int* ids, ExpertSubsetKind subset, const int* mask, int* counts,
                             int assignments) {
    __shared__ int local[experts];
    local[threadIdx.x] = 0;
    __syncthreads();
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < assignments) {
        const int e = ids[a];
        if (expert_in_subset(subset, mask, e)) atomicAdd(local + e, 1);
    }
    __syncthreads();
    const int count = local[threadIdx.x];
    if (count) atomicAdd(counts + threadIdx.x, count);
}

__global__ void scan_routes(Groups groups, int columns) {
    __shared__ int scan[experts / 32];
    publish_groups(groups, groups.counts[threadIdx.x], columns, scan);
}

__global__ void pack_routes(const int* ids, ExpertSubsetKind subset, const int* mask, Groups groups,
                            int assignments) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= assignments) return;
    const int e = ids[a];
    if (expert_in_subset(subset, mask, e)) {
        // Distinct IDs give a full-load expert exactly one assignment per token.
        // Dense down consumes that expert's packed rows in token order.
        const int packed           = groups.counts[e] == assignments / top_k
                                         ? groups.offsets[e] + a / top_k
                                         : atomicAdd(groups.cursors + e, 1);
        groups.assignments[packed] = a;
    }
}

nvfp4::ActivationScale calibration(const ActivationCalibrationView& value) {
    return value.values().size() == 1
               ? nvfp4::ActivationScale{nullptr, value.values().front()}
               : nvfp4::ActivationScale{static_cast<const float*>(value.device_values.data), 1};
}

__global__ void quantize_input(const __nv_bfloat16* x, const int* ids, Groups groups,
                               nvfp4::ActivationScale scale, std::uint8_t* codes,
                               std::uint8_t* scales, int broadcast_tokens,
                               const int* use_grouped = nullptr, float* clear_sum = nullptr) {
    if (use_grouped && !*use_grouped) return;
    const int packed = blockIdx.x, group = threadIdx.x;
    if (clear_sum)
        reinterpret_cast<float2*>(clear_sum)[packed * (hidden / 16) + group] = make_float2(0, 0);
    const int live       = groups.offsets[experts];
    const bool broadcast = broadcast_tokens && groups.load_range[0] == broadcast_tokens;
    const int input_rows = broadcast ? broadcast_tokens : live;
    if (packed >= input_rows) return;
    const int a            = broadcast ? packed * top_k : groups.assignments[packed];
    const float multiplier = broadcast ? scale.scalar : scale.get(ids[a]);
    const auto q =
        nvfp4::ActivationQuantizer(multiplier).block(x + (a / top_k) * hidden + group * 16);
    *reinterpret_cast<uint2*>(codes + packed * (hidden / 2) + group * 8) =
        make_uint2(q.codes_lo, q.codes_hi);
    scales[packed * (hidden / 16) + group] = q.scale;
}

struct EncodedGroupPair {
    uint4 codes;
    std::uint16_t scales;
};

template <int K>
__device__ __forceinline__ EncodedGroupPair load_group_pair(common::Nvfp4MultiplierMatrix weight,
                                                            int row, int group) {
    EncodedGroupPair result;
    const auto* codes = weight.codes + row * (K / 2) + group * 8;
    asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.codes.x), "=r"(result.codes.y), "=r"(result.codes.z), "=r"(result.codes.w)
        : "l"(codes));
    result.scales = *reinterpret_cast<const std::uint16_t*>(
        weight.scales + common::nvfp4_scale_byte_offset(row, group, K));
    return result;
}

#if __CUDACC_VER_MAJOR__ < 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 2)
template <int Part>
__device__ __forceinline__ common::Nvfp4F16x16 decode_group_pair(EncodedGroupPair encoded) {
    const auto code = Part == 0 ? make_int2(encoded.codes.x, encoded.codes.y)
                                : make_int2(encoded.codes.z, encoded.codes.w);
    return common::nvfp4_dequant_f16x16(code, (encoded.scales >> (8 * Part)) & 255);
}

__device__ __forceinline__ float dot_group(float sum, common::Nvfp4F16x16 weight, uint4 x0,
                                           uint4 x1) {
    const unsigned input[]{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const int halves[]{weight.lo.x, weight.lo.y, weight.lo.z, weight.lo.w,
                       weight.hi.x, weight.hi.y, weight.hi.z, weight.hi.w};
#    pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const auto x = common::bf16x2_bits_to_float2(input[pair]);
        const auto w = __half22float2(*reinterpret_cast<const __half2*>(halves + pair));
        sum          = fmaf(x.x, w.x, sum);
        sum          = fmaf(x.y, w.y, sum);
    }
    return sum;
}

#endif

template <int Part>
__device__ __forceinline__ float dot_bf16_group(float sum, EncodedGroupPair weight, uint4 x0,
                                                uint4 x1) {
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
    const unsigned x[]{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const unsigned w[]{Part == 0 ? weight.codes.x : weight.codes.z,
                       Part == 0 ? weight.codes.y : weight.codes.w};
    const unsigned code             = (weight.scales >> (8 * Part)) & 255;
    const unsigned short scale_pair = code * 0x0101;
    unsigned scale;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(scale) : "h"(scale_pair));
    // The E2M1 × E4M3 product is exactly representable in BF16.
    float even = sum, odd = 0;
#    pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned packed = (w[i / 4] >> ((i % 4) * 8)) & 255;
        asm("{ .reg .b8 code; .reg .b16 x0, x1, w0, w1; .reg .b32 w;\n"
            "  cvt.u8.u32 code, %3;\n"
            "  cvt.rn.bf16x2.e2m1x2 w, code;\n"
            "  mul.rn.bf16x2 w, w, %4;\n"
            "  mov.b32 {x0,x1}, %2;\n"
            "  mov.b32 {w0,w1}, w;\n"
            "  fma.rn.f32.bf16 %0, x0, w0, %0;\n"
            "  fma.rn.f32.bf16 %1, x1, w1, %1; }"
            : "+f"(even), "+f"(odd)
            : "r"(x[i]), "r"(packed), "r"(scale));
    }
    return even + odd;
#else
    return dot_group(sum, decode_group_pair<Part>(weight), x0, x1);
#endif
}

__global__ void vector_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                            ExpertSubsetKind subset, const int* mask, float* activation,
                            SoftcappedActivation swiglu, const int* use_grouped = nullptr) {
    if (use_grouped && *use_grouped) return;
    const int assignment = blockIdx.y, expert = ids[assignment];
    if (!expert_in_subset(subset, mask, expert)) return;
    const auto weight = common::nvfp4_mul_expert_matrix(bank, expert);
    if (!weight.valid) return;
    constexpr int warps     = 8;
    constexpr int x_vectors = hidden / 8, vectors_per_warp = 4 * 32;
    constexpr int padded_vectors =
        (x_vectors + vectors_per_warp - 1) / vectors_per_warp * vectors_per_warp;
    __shared__ uint4 input[padded_vectors];
    const auto* source = x + (assignment / top_k) * hidden;
    for (int i = threadIdx.x; i < x_vectors; i += blockDim.x) {
        // Each stream of 16-byte loads traverses consecutive shared-memory banks.
        const int to = i / vectors_per_warp * vectors_per_warp + i % 4 * 32 + i / 4 % 32;
        input[to]    = common::load_vec<uint4>(source + i * 8);
    }
    __syncthreads();
    const int row = blockIdx.x * warps + threadIdx.x / 32, lane = threadIdx.x % 32;
    float gate = 0, up = 0;
    for (int group = lane * 2; group < hidden / 16; group += 64) {
        const auto g       = load_group_pair<hidden>(weight, row, group);
        const auto u       = load_group_pair<hidden>(weight, row + intermediate, group);
        const auto* values = input + group / 64 * vectors_per_warp + lane;
        const auto a = values[0], b = values[32], c = values[64], d = values[96];
        gate = dot_bf16_group<1>(dot_bf16_group<0>(gate, g, a, b), g, c, d);
        up   = dot_bf16_group<1>(dot_bf16_group<0>(up, u, a, b), u, c, d);
    }
    gate = common::warp_reduce_sum(gate) * weight.multiplier;
    up   = common::warp_reduce_sum(up) * weight.multiplier;
    if (!lane) {
        constexpr int partition_size = intermediate / down_k_partitions;
        constexpr int groups         = partition_size / 16;
        const int in_partition       = row % partition_size;
        // Keep the four FP32 vectors of each K16 block in separate contiguous planes.
        const int output_row = row / partition_size * partition_size +
                               (in_partition % 16 / 4 * groups + in_partition / 16) * 4 +
                               in_partition % 4;
        activation[assignment * intermediate + output_row] = swiglu(gate, up);
    }
}

__device__ __forceinline__ float dot_group(float sum, common::Nvfp4F16x16 weight,
                                           const float4 (&packed)[4]) {
    const float x[]{packed[0].x, packed[0].y, packed[0].z, packed[0].w, packed[1].x, packed[1].y,
                    packed[1].z, packed[1].w, packed[2].x, packed[2].y, packed[2].z, packed[2].w,
                    packed[3].x, packed[3].y, packed[3].z, packed[3].w};
    const int halves[]{weight.lo.x, weight.lo.y, weight.lo.z, weight.lo.w,
                       weight.hi.x, weight.hi.y, weight.hi.z, weight.hi.w};
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const auto w = __half22float2(*reinterpret_cast<const __half2*>(halves + pair));
        sum          = fmaf(x[pair * 2], w.x, sum);
        sum          = fmaf(x[pair * 2 + 1], w.y, sum);
    }
    return sum;
}

template <int Rows, class Output = std::nullptr_t>
__global__ __launch_bounds__(top_k * 32) void vector_down_partitioned(
    const float* activation, ExpertMatrixDeviceView bank, const int* ids, const float* coefficients,
    ExpertSubsetKind subset, const int* mask, float* output, int tokens,
    const int* use_grouped = nullptr, Output destination = nullptr) {
    constexpr bool complete = !std::is_same_v<Output, std::nullptr_t>;
    if (use_grouped && *use_grouped) return;
    constexpr int groups_per_cta = intermediate / (16 * down_k_partitions);
    __shared__ float values[top_k][Rows];
    const int token = blockIdx.y, route = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int assignment = token * top_k + route, expert = ids[assignment];
    float sums[Rows]{};
    if (expert_in_subset(subset, mask, expert)) {
        const auto weight = common::nvfp4_mul_expert_matrix(bank, expert);
        if (weight.valid) {
            if (lane < groups_per_cta) {
#pragma unroll
                for (int p = 0; p < (complete ? down_k_partitions : 1); ++p) {
                    const int partition  = complete ? p : blockIdx.z;
                    const int group      = partition * groups_per_cta + lane;
                    constexpr int stride = groups_per_cta * 4;
                    const float* x       = activation + assignment * intermediate +
                                     partition * (stride * 4) + lane * 4;
                    const float4 source[4]{common::load_vec<float4>(x),
                                           common::load_vec<float4>(x + stride),
                                           common::load_vec<float4>(x + 2 * stride),
                                           common::load_vec<float4>(x + 3 * stride)};
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
                        const int row = blockIdx.x * Rows + r;
                        const auto w  = common::nvfp4_dequant_f16x16(
                            weight.codes + row * (intermediate / 2) + group * 8,
                            weight
                                .scales[common::nvfp4_scale_byte_offset(row, group, intermediate)]);
                        sums[r] = dot_group(sums[r], w, source);
                    }
                }
            }
            const float coefficient = coefficients[assignment];
            const float factor      = weight.multiplier * coefficient;
#pragma unroll
            for (int r = 0; r < Rows; ++r) sums[r] = common::warp_reduce_sum(sums[r]);
            if (__builtin_expect(isfinite(factor), true)) {
#pragma unroll
                for (int r = 0; r < Rows; ++r) sums[r] *= factor;
            } else {
#pragma unroll
                for (int r = 0; r < Rows; ++r)
                    sums[r] = scale_accumulator_wide(sums[r], weight.multiplier, coefficient);
            }
        }
    }
    if (!lane) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) values[route][r] = sums[r];
    }
    __syncthreads();
    if (threadIdx.x < Rows) {
        float result = 0;
#pragma unroll
        for (int j = 0; j < top_k; ++j) result += values[j][threadIdx.x];
        const int index = token * hidden + blockIdx.x * Rows + threadIdx.x;
        if constexpr (complete)
            destination.store(index, destination.initial(index, true) + result, true);
        else
            output[std::size_t(blockIdx.z) * tokens * hidden + index] = result;
    }
}

template <class Output>
__global__ void store_partitions(const float* partitions, Output output, int elements,
                                 const int* use_grouped   = nullptr,
                                 const float* grouped_sum = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    float sum = 0;
    if (use_grouped && *use_grouped) {
        sum = grouped_sum[i];
    } else {
#pragma unroll
        for (int j = 0; j < down_k_partitions; ++j) sum += partitions[j * elements + i];
    }
    output.store(i, output.initial(i, true) + sum, true);
}

template <class Output>
__global__ void store_sum(const float* sum, Output output, int elements) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) output.store(i, output.initial(i, true) + sum[i], true);
}

template <bool Gate, int Columns>
void launch_bf16(const Tensor& x, const CudaSwiGLUExperts& w, const ExpertRoutes& routes,
                 const clamped::Storage& s, SoftcappedActivation activation, int jobs,
                 CudaExecutionView execution, int job_columns = Columns, int minimum = 0,
                 int maximum = INT_MAX) {
    constexpr int rows = 128, step = 64, stages = 2;
    using S =
        nvfp4::a16::Schedule<intermediate, Gate, rows, Columns, step, stages, false, false, hidden>;
    constexpr auto kernel = nvfp4::a16::grouped<intermediate, Gate, rows, Columns, step, stages,
                                                false, false, false, hidden, SoftcappedActivation>;
    const int bytes       = common::prepare_shared<sizeof(nvfp4::a16::Tile<S>), kernel>();
    const int subtiles    = (std::min(job_columns, maximum) + Columns - 1) / Columns;
    const int blocks =
        std::min(jobs * subtiles * ((Gate ? intermediate : hidden) / S::feature_rows),
                 execution.multiprocessor_count() * 4);
    kernel<<<blocks, S::threads, bytes, execution.stream()>>>(
        static_cast<const __nv_bfloat16*>(x.data), Gate ? w.bank.gate_up : w.bank.down, s.groups,
        s.a16_product, nullptr, static_cast<const float*>(routes.coefficients.data), s.routed_sum,
        job_columns, activation, minimum, maximum, 0, clamped::Nvfp4A16Dispatch{});
}

template <bool Gate, int Columns>
void launch_fp4(const CudaSwiGLUExperts& w, const ExpertRoutes& routes, const clamped::Storage& s,
                SoftcappedActivation activation, int jobs, CudaExecutionView execution,
                int job_columns = Columns) {
    constexpr int rows = 128, step = 128, stages = 2;
    using S =
        nvfp4::Schedule<Columns, Gate, stages, step, 0, intermediate, true, false, rows, hidden>;
    constexpr auto kernel =
        nvfp4::grouped_mma<Columns, Gate, true, stages, step, 0, true, false, intermediate, true,
                           false, rows, hidden, SoftcappedActivation, true>;
    const int bytes    = common::prepare_shared<sizeof(nvfp4::TileStorage<S>), kernel>();
    const int subtiles = std::bit_ceil(unsigned((job_columns + Columns - 1) / Columns));
    const int blocks   = std::min(jobs * subtiles * ((Gate ? intermediate : hidden) / S::rows),
                                  execution.multiprocessor_count() * 4);
    kernel<<<blocks, S::threads, bytes, execution.stream()>>>(
        Gate ? s.input_codes : s.product_codes,
        static_cast<const std::uint8_t*>(Gate ? s.input_scales : s.product_scales),
        Gate ? w.bank.gate_up : w.bank.down,
        calibration(Gate ? w.activation_calibration.gate_up : w.activation_calibration.down),
        Gate ? calibration(w.activation_calibration.down) : nvfp4::ActivationScale{}, s.groups,
        s.product_codes, static_cast<std::uint8_t*>(s.product_scales), nullptr,
        static_cast<const float*>(routes.coefficients.data), s.routed_sum,
        std::countr_zero(unsigned(subtiles)), activation, 0, 0, 0, INT_MAX,
        clamped::ExpertLoadRegime::All, 0, 0, 0, 0, false, 0, 0, true);
}

} // namespace

void launch(const Tensor& x, const SwiGLUProfile& profile, const CudaSwiGLUExperts& w,
            const ExpertRoutes& routes, ExpertSubset subset, MoeEpilogue mode, Tensor& destination,
            Plan plan, const Storage& storage, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const auto* ids          = static_cast<const int*>(routes.ids.data);
    const auto* mask         = static_cast<const int*>(subset.mask.data);
    const auto* input        = static_cast<const __nv_bfloat16*>(x.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    const auto activation    = SoftcappedActivation{*profile.softcap};
    const int tokens = x.ne[1], assignments = top_k * tokens;

    if (direct_vector(plan.schedule)) {
        vector_gate<<<dim3(intermediate / 8, assignments), 256, 0, stream>>>(
            input, w.bank.gate_up, ids, subset.kind, mask, storage.activation, activation);
        moe_cuda::with_output(destination, mode, [&](auto output) {
            if (plan.schedule == Schedule::VectorComplete) {
                constexpr int rows = 4;
                vector_down_partitioned<rows>
                    <<<dim3(hidden / rows, tokens), top_k * 32, 0, stream>>>(
                        storage.activation, w.bank.down, ids, coefficients, subset.kind, mask,
                        nullptr, tokens, nullptr, output);
            } else {
                constexpr int rows = 8;
                vector_down_partitioned<rows>
                    <<<dim3(hidden / rows, tokens, down_k_partitions), top_k * 32, 0, stream>>>(
                        storage.activation, w.bank.down, ids, coefficients, subset.kind, mask,
                        storage.grouped.contributions, tokens);
                const int elements = tokens * hidden;
                store_partitions<<<(elements + 255) / 256, 256, 0, stream>>>(
                    storage.grouped.contributions, output, elements);
            }
        });
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const auto& s = storage.grouped;
    const auto& g = s.groups;
    if (plan.single_block_grouping) {
        group_small<<<1, experts, 0, stream>>>(ids, subset.kind, mask, g, assignments, plan.columns,
                                               storage.use_grouped);
    } else {
        CUDA_CHECK(cudaMemsetAsync(g.counts, 0, experts * sizeof(int), stream));
        count_routes<<<(assignments + experts - 1) / experts, experts, 0, stream>>>(
            ids, subset.kind, mask, g.counts, assignments);
        scan_routes<<<1, experts, 0, stream>>>(g, plan.columns);
        pack_routes<<<(assignments + 255) / 256, 256, 0, stream>>>(ids, subset.kind, mask, g,
                                                                   assignments);
    }
    const auto clear_sum = [&] {
        CUDA_CHECK(cudaMemsetAsync(s.routed_sum, 0, destination.numel() * sizeof(float), stream));
    };
    const int jobs = assignments / plan.columns + std::min(experts, assignments);
    if (!native_tma(plan.schedule) && !storage.use_grouped) clear_sum();
    switch (plan.schedule) {
    case Schedule::DecodeA4Reuse: {
        constexpr int rows      = 8;
        const auto* use_grouped = storage.use_grouped;
        vector_gate<<<dim3(intermediate / rows, assignments), rows * 32, 0, stream>>>(
            input, w.bank.gate_up, ids, subset.kind, mask, storage.activation, activation,
            use_grouped);
        vector_down_partitioned<rows>
            <<<dim3(hidden / rows, tokens, down_k_partitions), top_k * 32, 0, stream>>>(
                storage.activation, w.bank.down, ids, coefficients, subset.kind, mask,
                s.contributions, tokens, use_grouped);
        quantize_input<<<assignments, hidden / 16, 0, stream>>>(
            input, ids, g, calibration(w.activation_calibration.gate_up), s.input_codes,
            static_cast<std::uint8_t*>(s.input_scales), 0, use_grouped, s.routed_sum);
        launch_fp4<true, 32>(w, routes, s, activation, jobs, execution);
        launch_fp4<false, 32>(w, routes, s, activation, jobs, execution);
        break;
    }
    case Schedule::Bf16Short16:
        launch_bf16<true, 16>(x, w, routes, s, activation, jobs, execution);
        launch_bf16<false, 16>(x, w, routes, s, activation, jobs, execution);
        break;
    case Schedule::Bf16Mma:
        launch_bf16<true, 32>(x, w, routes, s, activation, jobs, execution);
        launch_bf16<false, 32>(x, w, routes, s, activation, jobs, execution);
        break;
    case Schedule::Bf16Bucketed: {
        const int bound = 2, medium = 128;
        const int blocks = execution.multiprocessor_count() * 6;
        nvfp4::a16::vector_gate<intermediate, 2, false, hidden, SoftcappedActivation>
            <<<blocks, 128, 0, stream>>>(input, w.bank.gate_up, g, s.a16_product, activation, bound,
                                         clamped::Nvfp4A16Dispatch{});
        launch_bf16<true, 32>(x, w, routes, s, activation, jobs, execution, plan.columns, bound,
                              medium);
        launch_bf16<true, 64>(x, w, routes, s, activation, jobs, execution, plan.columns, medium);
        nvfp4::a16::vector_down<intermediate, 2, false, hidden, SoftcappedActivation>
            <<<blocks, 128, 0, stream>>>(w.bank.down, g, s.a16_product, nullptr, coefficients,
                                         s.routed_sum, bound, clamped::Nvfp4A16Dispatch{});
        launch_bf16<false, 32>(x, w, routes, s, activation, jobs, execution, plan.columns, bound,
                               medium);
        launch_bf16<false, 64>(x, w, routes, s, activation, jobs, execution, plan.columns, medium);
        break;
    }
    case Schedule::Fp4Short:
    case Schedule::Fp4Mma: {
        quantize_input<<<assignments, hidden / 16, 0, stream>>>(
            input, ids, g, calibration(w.activation_calibration.gate_up), s.input_codes,
            static_cast<std::uint8_t*>(s.input_scales), 0);
        if (plan.schedule == Schedule::Fp4Short) {
            launch_fp4<true, 8>(w, routes, s, activation, jobs, execution);
            launch_fp4<false, 8>(w, routes, s, activation, jobs, execution);
        } else {
            launch_fp4<true, 32>(w, routes, s, activation, jobs, execution);
            launch_fp4<false, 32>(w, routes, s, activation, jobs, execution);
        }
        break;
    }
    case Schedule::Sm120TmaDense:
    case Schedule::Sm120Tma64: {
        const auto gu       = calibration(w.activation_calibration.gate_up);
        const auto down     = calibration(w.activation_calibration.down);
        const bool native   = tma::supports(w.bank);
        const int broadcast = native && !gu.values ? tokens : 0;
        quantize_input<<<assignments, hidden / 16, 0, stream>>>(
            input, ids, g, gu, s.input_codes, static_cast<std::uint8_t*>(s.input_scales),
            broadcast);
        if (native) {
            tma::contract(true, plan.schedule, w.bank.gate_up, gu, down, activation, s,
                          coefficients, jobs, tokens, broadcast != 0, execution);
            clear_sum();
            tma::contract(false, plan.schedule, w.bank.down, down, {}, activation, s, coefficients,
                          jobs, tokens, false, execution);
        } else {
            launch_fp4<true, 32>(w, routes, s, activation, jobs, execution, plan.columns);
            clear_sum();
            launch_fp4<false, 32>(w, routes, s, activation, jobs, execution, plan.columns);
        }
        break;
    }
    default:
        throw std::logic_error("invalid softcap grouped schedule");
    }
    moe_cuda::with_output(destination, mode, [&](auto output) {
        const int elements = tokens * hidden;
        if (storage.use_grouped)
            store_partitions<<<(elements + 255) / 256, 256, 0, stream>>>(
                s.contributions, output, elements, storage.use_grouped, s.routed_sum);
        else
            store_sum<<<(elements + 255) / 256, 256, 0, stream>>>(s.routed_sum, output, elements);
    });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::moe_routed::softcap
