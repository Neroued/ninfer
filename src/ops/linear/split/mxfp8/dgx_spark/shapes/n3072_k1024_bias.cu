#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/fp8/native_a16.cuh"
#include "ops/linear/kernels/fp8/native_a16_sliced_k.cuh"
#include "ops/linear/kernels/mxfp8/a8.cuh"
#include "ops/linear/kernels/mxfp8/a8_tma.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

#include <algorithm>
#include <array>
#include <limits>

namespace ninfer::ops::detail::linear_split {
namespace {
template <class S>
using Static   = NativeFp8ScheduleInstance<S, 1024>;
using Small    = Static<NativeFp8A16SlicedKSchedule<32, 8, 4, 128, 2>>;
using A8Medium = Static<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 4>>;
using A8Large  = Static<NativeFp8MmaSchedule<64, 32, 128, 32, 16, 3>>;
using A8Bulk   = Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>>;
using A16Bulk  = Static<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>;

using Output   = LinearBf16SegmentedOutput<1024, 1024, 1024>;
using Epilogue = LinearBiasEpilogue;
using Launch   = void (*)(const Tensor&, const Weight&, Output, Epilogue, Mxfp8A8Workspace,
                        cudaStream_t);

struct Route {
    int last_tokens;
    Launch launch;
    bool quantize;
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

template <class S>
void a8(const Tensor& x, const Weight& weight, Output output, Epilogue epilogue,
        Mxfp8A8Workspace scratch, cudaStream_t stream) {
    launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), 1024, x.ne[1], scratch,
                             stream);
    launch_mxfp8_a8_mma<S>(mxfp8_a8_operands(weight, scratch, x.ne[1]), output, epilogue, stream);
}

template <class S>
void tma(const Tensor& x, const Weight& weight, Output output, Epilogue epilogue,
         Mxfp8A8Workspace scratch, cudaStream_t stream) {
    launch_mxfp8_a8_quantize(static_cast<const __nv_bfloat16*>(x.data), 1024, x.ne[1], scratch,
                             stream);
    launch_mxfp8_a8_tma<S>(mxfp8_a8_operands(weight, scratch, x.ne[1]), output,
                           StreamingBiasEpilogue{{}, epilogue}, stream);
}

constexpr int kAllTokens = std::numeric_limits<int>::max();
constexpr std::array<Route, 2> kA16{{
    {32, sliced<Small>, false},
    {kAllTokens, mma<A16Bulk>, false},
}};
constexpr std::array<Route, 4> kA8{{
    kA16[0],
    {64, a8<A8Medium>, true},
    {128, a8<A8Large>, true},
    {kAllTokens, tma<A8Bulk>, true},
}};

std::span<const Route> routes(LinearPolicy policy) {
    return allows_a8(policy) ? std::span<const Route>{kA8} : std::span<const Route>{kA16};
}

const Route& select_route(LinearPolicy policy, int tokens) {
    for (const auto& route : routes(policy))
        if (tokens <= route.last_tokens) return route;
    throw std::invalid_argument("MXFP8 linear_split bias: unsupported token count");
}

std::size_t capacity(LinearPolicy policy, std::int32_t first, std::int32_t last) {
    std::size_t peak = 0;
    int begin        = 1;
    for (const auto& route : routes(policy)) {
        const int end = std::min(route.last_tokens, last);
        if (route.quantize && begin <= last && end >= first)
            peak = std::max(peak, mxfp8_a8_workspace_capacity_bytes(end, 1024));
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
    const auto scratch = route.quantize ? allocate_mxfp8_a8_workspace(*workspace, x.ne[1], 1024)
                                        : Mxfp8A8Workspace{};
    route.launch(x, weight, out, epilogue, scratch, stream);
}
} // namespace

const SplitBiasShape kMxfp8SplitBiasN3072K1024{3072, 1024, kSplitRows<1024, 1024, 1024>, launch,
                                               capacity};
} // namespace ninfer::ops::detail::linear_split
