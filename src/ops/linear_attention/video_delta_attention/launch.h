#pragma once
#include "ninfer/ops/video_delta_attention.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

namespace ninfer::ops::detail::vda {
constexpr int D             = 128;
constexpr int HeadWave      = 28;
constexpr int StateHeadWave = 56;

// Partition only the additive text statistics. All partitions are reduced before
// the single implicit solve, so this does not introduce a causal text recurrence.
inline int state_statistics_partitions(int heads, int tokens) {
    if (!tokens) return 0;
    const auto tiles = (std::int64_t(tokens) + 31) / 32;
    return static_cast<int>(std::min<std::int64_t>(tiles, std::max(1, 168 / heads)));
}

struct Vector {
    const __nv_bfloat16* data{};
    std::int64_t head_stride{}, token_stride{}, batch_stride{}; // elements
};

struct Arguments {
    Vector q, k, v, gate;
    const float* alpha{};
    const float* beta{};
    const float* initial{};
    const __nv_bfloat16* weight{};
    __nv_bfloat16* out{};
    float* state_out{};
    int heads{}, frames{}, per_frame{}, tokens{}, batch{};
    VideoDeltaAttentionWindow window{};
    float eps{}, state_scale{};
};

#if defined(__CUDACC__)
struct BatchHead {
    int batch, head;
};

// A launch can span several samples. Keep the common single-sample path free
// of division, while all tensors retain their public [batch,token,head,dim] layout.
__device__ __forceinline__ BatchHead resolve_group(const Arguments& args, int first_batch,
                                                   int linear_head) {
    if (linear_head < args.heads) return {first_batch, linear_head};
    return {first_batch + linear_head / args.heads, linear_head % args.heads};
}
#endif

void launch(const Arguments&, float* p, float* j, float* boundary, int head_begin, int heads,
            int batch, bool init, cudaStream_t);
} // namespace ninfer::ops::detail::vda
