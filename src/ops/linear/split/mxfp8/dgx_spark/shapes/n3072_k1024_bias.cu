#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_bias_vector_epilogue.cuh"

#include <algorithm>
#include <array>
#include <limits>

namespace ninfer::ops::detail::linear_split {
namespace {
template <class S>
using Static  = NativeFp8ScheduleInstance<S, 1024>;
using Small   = Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>;
using A8Near  = Static<Mxfp8A8TmaSchedule<64, 64, 256, 32, 16, 2, 1, 2>>;
using A8Bulk  = Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>>;
using A16Bulk = Static<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>;

using Output   = LinearBf16SegmentedOutput<1024, 1024, 1024>;
using Epilogue = LinearBiasEpilogue;
using Launch   = void (*)(const Tensor&, const Weight&, Output, Epilogue, Mxfp8A8Workspace,
                        cudaStream_t);

struct Route {
    int last_tokens;
    Launch launch;
    bool quantize;
    int code_alignment = 1;
    bool tensor_core   = true;

    int code_tokens(int t) const { return mxfp8_a8_padded_tokens(t, code_alignment); }
};

struct StreamingBiasEpilogue : Bf16MmaStreamingWarpStoreEpilogue {
    LinearBiasEpilogue bias;

    template <class Out>
    __device__ __forceinline__ void store_fragment(const Out& output, int row, int token,
                                                   float4 value, int rows, int tokens) const {
        Bf16MmaStreamingWarpStoreEpilogue::store_fragment(
            output, row, token,
            make_float4(bias.apply(row, token, value.x), bias.apply(row, token + 1, value.y),
                        bias.apply(row + 8, token, value.z),
                        bias.apply(row + 8, token + 1, value.w)),
            rows, tokens);
    }
};

template <class S>
void sliced(const Tensor& x, const Weight& weight, Output output, Epilogue epilogue,
            Mxfp8A8Workspace, cudaStream_t stream) {
    launch_native_fp8_a16_sliced_k<S>(mxfp8_a16_operands(x, weight), output, epilogue, stream);
}

template <class S>
void mma(const Tensor& x, const Weight& weight, Output output, Epilogue epilogue, Mxfp8A8Workspace,
         cudaStream_t stream) {
    launch_native_fp8_a16_mma<S>(mxfp8_a16_operands(x, weight), output, epilogue, stream);
}

struct StreamingSegment : LinearBf16SegmentOutput {
    __device__ __forceinline__ void store_vector(int row, int token, uint4 v) const {
        auto* ptr = data + static_cast<std::int64_t>(token) * rows + row - parent_row_begin;
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(v.x), "r"(v.y),
                     "r"(v.z), "r"(v.w)
                     : "memory");
    }
};

struct StreamingOutput : Output {
    __device__ __forceinline__ void store_vector(int row, int token, uint4 v) const {
        StreamingSegment{segment(row)}.store_vector(row, token, v);
    }

    template <int Rows>
    __device__ __forceinline__ auto bind_tile(int row) const {
        if constexpr (1024 % Rows == 0)
            return StreamingSegment{segment(row)};
        else
            return *this;
    }
};

using VectorBiasEpilogue = Bf16BiasVectorEpilogue<3072>;

template <class S, bool Padded = false, bool Vector = false>
void tma(const Tensor& x, const Weight& w, Output out, Epilogue epi, Mxfp8A8Workspace scratch,
         cudaStream_t stream) {
    int codes = Padded ? mxfp8_a8_padded_tokens(x.ne[1], S::kBlockTokens) : x.ne[1];
    launch_mxfp8_a8_quantize<__nv_bfloat16, Padded>(static_cast<const __nv_bfloat16*>(x.data), 1024,
                                                    x.ne[1], scratch, stream, codes);
    if constexpr (Vector)
        launch_mxfp8_a8_tma<S, Padded>(mxfp8_a8_operands(w, scratch, x.ne[1]), StreamingOutput{out},
                                       VectorBiasEpilogue{{}, epi.bias}, stream);
    else
        launch_mxfp8_a8_tma<S, Padded>(mxfp8_a8_operands(w, scratch, x.ne[1]), out,
                                       StreamingBiasEpilogue{{}, epi}, stream);
}

template <class S>
void simt(const Tensor& x, const Weight& w, Output out, Epilogue epi, Mxfp8A8Workspace,
          cudaStream_t stream) {
    launch_native_fp8_a16_simt<S>(mxfp8_a16_operands(x, w), out, epi, stream);
}

template <class S, bool Vector = false>
constexpr Route padded(int last) {
    return {last, tma<S, true, Vector>, true, S::kBlockTokens};
}

template <class S>
struct RegisterWeights : S {
    static constexpr bool kRegisterWeights = true;
};

template <class S>
void warp_mma(const Tensor& x, const Weight& w, Output out, Epilogue epi, Mxfp8A8Workspace,
              cudaStream_t stream) {
    launch_native_fp8_a16_mma<S>(mxfp8_a16_operands(x, w), out, StreamingBiasEpilogue{{}, epi},
                                 stream);
}

using Near      = RegisterWeights<Static<NativeFp8MmaSchedule<32, 32, 128, 16, 8, 2, 2, 8>>>;
using Mid64     = RegisterWeights<Static<NativeFp8MmaSchedule<32, 64, 128, 16, 16, 2, 2, 8>>>;
using A8BulkG2  = Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2, 1, 2>>;
using A8WideG8  = Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 8, true, 1>>;
using A8WideG16 = Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 16, true, 1>>;
constexpr int kAllTokens = std::numeric_limits<int>::max();
constexpr std::array<Route, 2> kA16{{
    {32, sliced<Small>, false},
    {kAllTokens, mma<A16Bulk>, false},
}};
constexpr std::array<Route, 15> kA8{{
    {1, simt<Static<NativeFp8A16SimtSchedule<1, 8, 1, 16, 4, 2>>>, false, 1, false},
    {4, sliced<Static<NativeFp8A16SlicedKSchedule<8, 4, 8, 128, 1>>>, false},
    {8, sliced<Small>, false},
    {32, warp_mma<Near>, false},
    {64, warp_mma<Mid64>, false},
    padded<A8Near, true>(128),
    padded<A8Bulk, true>(256),
    padded<A8WideG16, true>(384),
    padded<A8BulkG2, true>(512),
    padded<A8WideG8, true>(768),
    padded<A8Bulk, true>(1024),
    padded<A8WideG8, true>(1152),
    padded<A8Bulk, true>(3072),
    padded<A8WideG16, true>(kAllTokens / A8WideG16::kBlockTokens * A8WideG16::kBlockTokens),
    {kAllTokens, tma<A8WideG16, false, true>, true},
}};

std::span<const Route> routes(LinearPolicy policy) {
    return allows_a8(policy) ? std::span<const Route>{kA8} : std::span<const Route>{kA16};
}

const Route& select_route(LinearPolicy policy, int tokens) {
    for (const auto& route : routes(policy))
        if (tokens <= route.last_tokens) return route;
    throw std::invalid_argument("MXFP8 linear_split bias: unsupported token count");
}

bool uses_tensor_cores(LinearPolicy policy, int tokens) {
    return select_route(policy, tokens).tensor_core;
}

std::size_t capacity(LinearPolicy policy, std::int32_t first, std::int32_t last) {
    std::size_t peak = 0;
    int begin        = 1;
    for (const auto& route : routes(policy)) {
        const int end = std::min(route.last_tokens, last);
        if (route.quantize && begin <= last && end >= first)
            peak = std::max(peak,
                            mxfp8_a8_workspace_capacity_bytes(end, 1024, route.code_tokens(end)));
        if (route.last_tokens >= last) return peak;
        begin = route.last_tokens + 1;
    }
    throw std::invalid_argument("MXFP8 linear_split bias: unsupported token interval");
}

void launch(const Tensor& x, const Weight& weight, const Tensor& bias,
            std::span<const Tensor> outputs, LinearPolicy policy, MemoryArena* workspace,
            cudaStream_t stream) {
    const Output out{{static_cast<__nv_bfloat16*>(outputs[0].data),
                      static_cast<__nv_bfloat16*>(outputs[1].data),
                      static_cast<__nv_bfloat16*>(outputs[2].data)}};
    const Epilogue epilogue{static_cast<const __nv_bfloat16*>(bias.data)};
    const auto& route  = select_route(policy, x.ne[1]);
    const auto scratch = route.quantize ? allocate_mxfp8_a8_workspace(*workspace, x.ne[1], 1024,
                                                                      route.code_tokens(x.ne[1]))
                                        : Mxfp8A8Workspace{};
    route.launch(x, weight, out, epilogue, scratch, stream);
}
} // namespace

const SplitBiasShape kMxfp8SplitBiasN3072K1024{3072,   1024,     kSplitRows<1024, 1024, 1024>,
                                               launch, capacity, uses_tensor_cores};
} // namespace ninfer::ops::detail::linear_split
