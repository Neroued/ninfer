// Included for the dense and packed kernel entry points.
#if defined(NINFER_CAUSAL_RAGGED)
template <class G, bool InverseRotation>
__launch_bounds__(256) __global__
    void causal_tiled_merge_ragged_kernel(CausalPartialView partial, const std::int32_t* positions,
                                          const std::int32_t* valid_columns, int width,
                                          CausalKvPartition partition, __nv_bfloat16* out,
                                          const __grid_constant__ CausalTiledMetadata metadata) {
    const int lane         = threadIdx.x & 31;
    const int token        = blockIdx.x;
    const int head         = blockIdx.y * 8 + (threadIdx.x >> 5);
    const int index        = metadata.template find<&CausalTiledSegment::token_end>(token);
    const int segment      = metadata.segments.entry(index).segment;
    const int before       = index ? metadata.segments.entry(index - 1).token_end : 0;
    const int begin        = metadata.cu_seqlens[segment];
    const int output_token = begin + (token - before);
    const int visible      = positions[begin + metadata.seq_lengths[segment] - 1] + 1;
    const int splits       = partition.active(visible);
#else
template <class G, bool InverseRotation>
__launch_bounds__(256) __global__
    void causal_tiled_merge_kernel(CausalPartialView partial, const std::int32_t* positions,
                                   const std::int32_t* valid_columns, int width,
                                   CausalKvPartition partition, __nv_bfloat16* out) {
    const int lane  = threadIdx.x & 31;
    const int row   = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int head  = row % G::QHeads;
    const int token = row / G::QHeads;
    if (token >= width) return;
    if (valid_columns && token >= valid_columns[0]) {
#pragma unroll
        for (int d = lane; d < 256; d += 32)
            causal_store_output(out + causal_q_index<G>(head, d, token), 0.0F);
        return;
    }
    const int splits        = partition.active(positions[width - 1] + 1);
    const int output_token = token;
#endif
    const auto stat         = causal_stat_index<G>(head, token, lane, width);
    const float m           = lane < splits ? partial.maximum[stat] : -CUDART_INF_F;
    const float l           = lane < splits ? partial.sum[stat] : 0.0F;
    const float maximum     = common::warp_max(m);
    const float weight      = l > 0.0F ? expf(m - maximum) : 0.0F;
    const float denominator = common::warp_sum(l * weight);
    float values[8]{};
    for (int split = 0; split < splits; ++split) {
        const float w     = __shfl_sync(0xffffffffU, weight, split);
        const auto offset = causal_partial_index<G>(head, lane, token, split, width);
#pragma unroll
        for (int i = 0; i < 8; ++i) values[i] += partial.acc[offset + i * 32] * w;
    }
    const float inverse = denominator > 0.0F ? __frcp_rn(denominator) : 0.0F;
#pragma unroll
    for (float& value : values) value *= inverse;
    if constexpr (InverseRotation) kv::normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        causal_store_output(out + causal_q_index<G>(head, lane + i * 32, output_token), values[i]);
}
