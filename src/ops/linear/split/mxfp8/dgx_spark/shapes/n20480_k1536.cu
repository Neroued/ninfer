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
using Static    = NativeFp8ScheduleInstance<S, 1536>;
using Small     = Static<NativeFp8A16SlicedKSchedule<16, 16, 4, 128, 2>>;
using Medium    = Static<NativeFp8MmaSchedule<32, 32, 128, 16, 16, 2>>;
using A16Bulk   = Static<NativeFp8MmaSchedule<64, 128, 128, 32, 32, 2>>;
using A8Medium  = Static<Mxfp8A8TmaSchedule<64, 64, 256, 32, 16, 2>>;
using A8Large   = Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 64, 2>>;
using A8Bulk    = Static<Mxfp8A8TmaSchedule<128, 128, 128, 64, 32, 2>>;
using A8Largest = Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 4, true, 1>>;

using Output   = LinearBf16SegmentedOutput<16384, 4096>;
using Epilogue = LinearIdentityEpilogue;
using Launch   = void (*)(const Tensor&, const Weight&, Output, Epilogue, Mxfp8A8Workspace,
                        cudaStream_t);

struct Route {
    int last_tokens;
    Launch launch;
    bool quantize;
    int code_alignment = 1;

    int code_tokens(int tokens) const { return mxfp8_a8_padded_tokens(tokens, code_alignment); }
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

struct StreamingOutput : Output {
    __device__ __forceinline__ void store_vector(int row, int token, uint4 values) const {
        const auto destination = segment(row);
        auto* ptr = destination.data + static_cast<std::int64_t>(token) * destination.rows + row -
                    destination.parent_row_begin;
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(values.x),
                     "r"(values.y), "r"(values.z), "r"(values.w)
                     : "memory");
    }
};

struct VectorEpilogue : Bf16TmaVectorEpilogue {
    // A partial grid still contains complete tiles. Check each tile once before
    // traversing its output vectors, keeping the tail's write domain unchanged.
    template <class S, bool Full, class O>
    __device__ __forceinline__ void finish_tile(const O& out, unsigned char* scratch,
                                                float (&acc)[S::kMmaRows][S::kMmaTokens][4],
                                                int row, int token, int rows, int tokens) const {
        if constexpr (20480 % S::kBlockRows == 0) {
            if (Full || tokens - token >= S::kBlockTokens)
                Bf16TmaVectorEpilogue::template finish_tile<S, true, false>(out, scratch, acc, row,
                                                                            token, rows, tokens);
            else
                Bf16TmaVectorEpilogue::template finish_tile<S, false, false>(out, scratch, acc, row,
                                                                             token, rows, tokens);
        } else if (row + S::kBlockRows <= rows) {
            if (tokens - token >= S::kBlockTokens)
                Bf16TmaVectorEpilogue::template finish_tile<S, true, false>(out, scratch, acc, row,
                                                                            token, rows, tokens);
            else
                Bf16TmaVectorEpilogue::template finish_tile<S, false, false>(out, scratch, acc, row,
                                                                             token, rows, tokens);
        } else {
            if (tokens - token >= S::kBlockTokens)
                Bf16TmaVectorEpilogue::template finish_tile<S, true, true>(out, scratch, acc, row,
                                                                           token, rows, tokens);
            else
                Bf16TmaVectorEpilogue::template finish_tile<S, false, true>(out, scratch, acc, row,
                                                                            token, rows, tokens);
        }
    }
};

template <class S, bool Streaming = false, bool PaddedCodes = false, bool Vector = false>
void tma(const Tensor& x, const Weight& weight, Output output, Epilogue, Mxfp8A8Workspace scratch,
         cudaStream_t stream) {
    const int code_tokens =
        PaddedCodes ? mxfp8_a8_padded_tokens(x.ne[1], S::kBlockTokens) : x.ne[1];
    launch_mxfp8_a8_quantize<__nv_bfloat16, PaddedCodes>(
        static_cast<const __nv_bfloat16*>(x.data), 1536, x.ne[1], scratch, stream, code_tokens);
    if constexpr (Vector) {
        launch_mxfp8_a8_tma<S, PaddedCodes>(mxfp8_a8_operands(weight, scratch, x.ne[1]),
                                            StreamingOutput{output}, VectorEpilogue{}, stream);
    } else {
        using Store = std::conditional_t<Streaming, Bf16MmaStreamingWarpStoreEpilogue, Epilogue>;
        launch_mxfp8_a8_tma<S, PaddedCodes>(mxfp8_a8_operands(weight, scratch, x.ne[1]), output,
                                            Store{}, stream);
    }
}

template <class S, bool Vector = false>
constexpr Route padded(int last) {
    return {last, tma<S, true, true, Vector>, true, S::kBlockTokens};
}

constexpr int kAllTokens = std::numeric_limits<int>::max();
constexpr std::array<Route, 3> kA16{{
    {32, sliced<Small>, false},
    {64, mma<Medium>, false},
    {kAllTokens, mma<A16Bulk>, false},
}};
constexpr std::array<Route, 13> kA8{{
    {1, sliced<Static<NativeFp8A16SlicedKSchedule<8, 1, 8, 256, 1>>>, false},
    {2, sliced<Static<NativeFp8A16SlicedKSchedule<8, 2, 8, 256, 2>>>, false},
    {4, sliced<Static<NativeFp8A16SlicedKSchedule<8, 4, 8, 256, 1>>>, false},
    {8, sliced<Static<NativeFp8A16SlicedKSchedule<8, 8, 8, 256, 1>>>, false},
    kA16[0],
    {64, tma<A8Medium>, true},
    padded<Static<Mxfp8A8TmaSchedule<32, 128, 256, 16, 32, 2, 1, 2>>>(128),
    {192, tma<Static<Mxfp8A8TmaSchedule<128, 64, 128, 64, 16, 3, 1, 2, true, 2>>, true>, true},
    {384, tma<A8Large, true>, true},
    {512, tma<Static<Mxfp8A8TmaSchedule<192, 128, 128, 48, 64, 2, 1, 16, true, 1>>, true>, true},
    padded<A8Bulk, true>(3072),
    padded<A8Largest, true>(kAllTokens / A8Largest::kBlockTokens * A8Largest::kBlockTokens),
    // The final partial int-range tile cannot represent its rounded-up code extent.
    {kAllTokens, tma<A8Largest, true, false, true>, true},
}};

std::span<const Route> routes(LinearPolicy policy) {
    return allows_a8(policy) ? std::span<const Route>{kA8} : std::span<const Route>{kA16};
}

const Route& select_route(LinearPolicy policy, int tokens) {
    for (const auto& route : routes(policy))
        if (tokens <= route.last_tokens) return route;
    throw std::invalid_argument("MXFP8 linear_split: unsupported token count");
}

std::size_t capacity(LinearPolicy policy, std::int32_t first, std::int32_t last) {
    std::size_t peak = 0;
    int begin        = 1;
    for (const auto& route : routes(policy)) {
        const int end = std::min(route.last_tokens, last);
        if (route.quantize && begin <= last && end >= first)
            peak = std::max(peak,
                            mxfp8_a8_workspace_capacity_bytes(end, 1536, route.code_tokens(end)));
        if (route.last_tokens >= last) return peak;
        begin = route.last_tokens + 1;
    }
    throw std::invalid_argument("MXFP8 linear_split: unsupported token interval");
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    const Output out{{static_cast<__nv_bfloat16*>(outputs[0].data),
                      static_cast<__nv_bfloat16*>(outputs[1].data)}};
    const auto& route  = select_route(weights.policy, x.ne[1]);
    const auto scratch = route.quantize ? allocate_mxfp8_a8_workspace(*workspace, x.ne[1], 1536,
                                                                      route.code_tokens(x.ne[1]))
                                        : Mxfp8A8Workspace{};
    route.launch(x, weights.first, out, Epilogue{}, scratch, stream);
}
} // namespace

const SplitShape kMxfp8SplitN20480K1536{20480, 1536, kSplitRows<16384, 4096>, launch, capacity};
} // namespace ninfer::ops::detail::linear_split
