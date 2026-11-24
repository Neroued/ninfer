// Included for the dense and packed kernel entry points.
#if defined(NINFER_CAUSAL_RAGGED)
template <class Geometry, int DChunk, bool MultiBatch, bool Masked, bool InverseRotation>
__launch_bounds__(256) __global__
    void causal_natural_merge_ragged_kernel(const float* partial_acc, const float* partial_m,
                                            const float* partial_l, const std::int32_t* positions,
                                            const std::int32_t* valid_columns, std::int32_t tokens,
                                            std::int32_t batch_size, std::int32_t logical_capacity,
                                            CausalKvPartition partition, __nv_bfloat16* out,
                                            const __grid_constant__ CausalShortMetadata metadata) {
    static_assert(Geometry::kHeadDim == kCausalHeadDim);
    static_assert(DChunk > 0 && DChunk <= kCausalHeadDim);
    static_assert(!InverseRotation || DChunk == kCausalHeadDim);
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
template <class Geometry, int DChunk, bool MultiBatch, bool Masked, bool InverseRotation>
__launch_bounds__(256) __global__
    void causal_natural_merge_kernel(const float* partial_acc, const float* partial_m,
                                     const float* partial_l, const std::int32_t* positions,
                                     const std::int32_t* valid_columns, std::int32_t tokens,
                                     std::int32_t batch_size, std::int32_t logical_capacity,
                                     CausalKvPartition partition, __nv_bfloat16* out) {
    static_assert(Geometry::kHeadDim == kCausalHeadDim);
    static_assert(DChunk > 0 && DChunk <= kCausalHeadDim);
    static_assert(!InverseRotation || DChunk == kCausalHeadDim);
    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = static_cast<int>(threadIdx.x);
    if (q_head >= Geometry::QHeads || token >= tokens) return;
    if constexpr (MultiBatch) {
        if (batch >= batch_size) return;
    }
    int output_column = token;
    if constexpr (MultiBatch) output_column += batch * tokens;
    const int live_columns = Masked ? valid_columns[batch] : tokens;
    if constexpr (Masked) {
        if (token >= live_columns) {
            if (tid < DChunk && d_start + tid < 256)
                out[causal_q_index<Geometry>(q_head, d_start + tid, output_column)] =
                    __float2bfloat16(0.0f);
            return;
        }
    }

    const auto slots =
        causal_merge_splits<MultiBatch, Masked>(partition, positions, valid_columns, tokens,
                                                live_columns, batch, batch_size, logical_capacity);
    if constexpr (MultiBatch) {
        const auto stat = static_cast<std::int64_t>(slots.offset) * Geometry::QHeads * tokens;
        partial_acc += stat * Geometry::kHeadDim;
        partial_m += stat;
        partial_l += stat;
    }
#endif
    const int active_splits = slots.count < 0 ? partition.active(slots.window) : slots.count;
    __shared__ float weights[256], warp_sums[8], scalars[2];
    const float head_l = causal_merge_natural_statistics<Geometry>(
        partial_m, partial_l, q_head, token, tokens, active_splits, weights, warp_sums, scalars);
    const int d = d_start + tid;
    if (tid >= DChunk || d >= 256) return;
    float numerator = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[causal_partial_index<Geometry>(q_head, d, token, split, tokens)] *
                weights[split];
    }

    const float value = head_l > 0.0F ? numerator / head_l : 0.0F;
    if constexpr (InverseRotation) {
        __shared__ float normalized[kCausalHeadDim];
        normalized[tid] = value;
        __syncthreads();
        if (tid < 32)
            causal_store_inverse_rotated_row<Geometry>(normalized, out, q_head, output_column);
    } else {
        causal_store_output(out + causal_q_index<Geometry>(q_head, d, output_column), value);
    }
}
