// Included for the dense and packed kernel entry points.
#if defined(NINFER_CAUSAL_RAGGED)
template <typename Geometry, class Schedule, bool MultiBatch, bool Masked>
__launch_bounds__(Schedule::kThreads) __global__
    void bf16_kv_merge_ragged_kernel(const float* partial_acc, const float* partial_m,
                                     const float* partial_l, const std::int32_t* positions,
                                     const std::int32_t* valid_columns, std::int32_t tokens,
                                     std::int32_t batch_size, Bf16KvPartition partition,
                                     __nv_bfloat16* out,
                                     const __grid_constant__ CausalShortMetadata metadata) {
    constexpr int DChunk = Schedule::kDChunk;
    static_assert(DChunk <= Geometry::kHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    const int batch       = flat_column / tokens;
    const int token       = flat_column - batch * tokens;
    const int tid         = threadIdx.x;
    const int segment     = metadata.segments.entry(batch);
    if (token >= metadata.seq_lengths[segment]) return;
    const int output_column = metadata.cu_seqlens[segment] + token;
    const auto slots        = causal_ragged_merge(partition, metadata, positions, batch);
    {
        const auto stat = static_cast<std::int64_t>(slots.offset) * Geometry::QHeads * tokens;
        partial_acc += stat * Geometry::kHeadDim;
        partial_m += stat;
        partial_l += stat;
    }
#else
template <typename Geometry, class Schedule, bool MultiBatch, bool Masked>
__launch_bounds__(Schedule::kThreads) __global__
    void bf16_kv_merge_kernel(const float* partial_acc, const float* partial_m,
                              const float* partial_l, const std::int32_t* positions,
                              const std::int32_t* valid_columns, std::int32_t tokens,
                              std::int32_t batch_size, Bf16KvPartition partition,
                              __nv_bfloat16* out) {
    constexpr int DChunk = Schedule::kDChunk;
    static_assert(DChunk <= Geometry::kHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    const int live_columns = Masked ? valid_columns[batch] : tokens;
    int output_column      = token;
    if constexpr (MultiBatch) { output_column += batch * tokens; }
    if constexpr (Masked) {
        const int absolute_column = token;
        if (absolute_column >= live_columns) {
            if (tid < DChunk && d_start + tid < Geometry::kHeadDim)
                out[causal_q_index<Geometry>(q_head, d_start + tid, output_column)] =
                    __float2bfloat16(0.0f);
            return;
        }
    }

    const auto slots = causal_merge_splits<MultiBatch, Masked>(
        partition, positions, valid_columns, tokens, live_columns, batch, batch_size);
    if constexpr (MultiBatch) {
        const auto stat = static_cast<std::int64_t>(slots.offset) * Geometry::QHeads * tokens;
        partial_acc += stat * Geometry::kHeadDim;
        partial_m += stat;
        partial_l += stat;
    }
#endif
    const int active_split_count =
        slots.count < 0 ? partition.live(slots.window).splits
                        : (slots.count == 0 ? 0 : partition.live(slots.window, slots.count).splits);

    __shared__ float weights[Schedule::kThreads], warp_sums[Schedule::kWarps], scalars[2];
    const float head_l = bf16_kv_merge_statistics<Geometry, Schedule>(
        partial_m, partial_l, q_head, token, tokens, active_split_count, weights, warp_sums,
        scalars);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= Geometry::kHeadDim) return;
    float numerator = 0.0f;
    for (int split = 0; split < active_split_count; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[causal_partial_index<Geometry>(q_head, d, token, split, tokens)] *
                weights[split];
    }

    const float value = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[causal_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(value);
}
