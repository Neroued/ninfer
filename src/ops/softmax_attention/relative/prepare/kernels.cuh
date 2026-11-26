#pragma once

#include "ninfer/ops/relative_attention.h"

#include <cuda_bf16.h>

namespace ninfer::ops::relative_prepare {

inline constexpr int kDim            = 128;
inline constexpr int kRelDim         = 16;
inline constexpr int kQueryHeads     = 24;
inline constexpr int kKvHeads        = 8;
inline constexpr int kChannels       = kKvHeads * kDim;
inline constexpr int kHistory        = 3;
inline constexpr int kTaps           = kHistory + 1;
inline constexpr int kCyclicCapacity = 1024;

using Bf16 = __nv_bfloat16;

template <class T>
struct Records {
    T* base;
    const std::int64_t* offsets;
};

struct Conv {
    const Bf16* input;
    const Bf16* weight;
    const Bf16* dynamic;
    Records<const Bf16> before;
    Records<Bf16> after;
    Records<char> cache;
    Records<char> scales;
};

struct Arguments {
    RelativeAttentionPrepareParams params;
    const Bf16 *q, *r, *q_scale, *r_scale, *sink_x, *sink_weight;
    Bf16 *q_out, *r_out, *sink_out;
    Conv k, v;
    const int *positions, *valid, *snapshot_slots, *table_rows;
    const int *token_to_seq, *cu_seqlens, *seq_lengths;
    int width, columns, sink_channels, cache_rows;
    std::int64_t snapshot_stride_k, snapshot_stride_v;
};

enum class Destination { Cyclic, Chunk, Paged };

struct CommitPlane {
    const char* data;
    const char* scales;
    Records<char> cache;
    Records<char> cache_scales;
};

struct CommitArguments {
    CommitPlane k, v;
    const int *positions, *valid, *cu_seqlens, *seq_lengths;
    int width, sequences, tail_slots, cache_rows;
};

void launch(const Arguments& args, Destination destination, bool fp8, CudaExecutionView execution);
void launch_commit(const CommitArguments& args, bool fp8, CudaExecutionView execution);

} // namespace ninfer::ops::relative_prepare
