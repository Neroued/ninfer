#pragma once

// ninfer::ops::detail - private launch prototypes for gqa_attention policies.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class GqaAttentionRoute { SmallT, ChunkedSmallT, Prompt, VoltaFlash };

struct GqaSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope);

bool gqa_attention_uses_small_t(std::int32_t tokens);

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                              std::int32_t batch_size, DType cache_dtype,
                                              GqaExecutionEnvelope envelope);

const char* gqa_attention_route_name(GqaAttentionRoute route);

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream);

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream);

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream);

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream);

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// Volta (sm_70) flash-attention prefill route. Defined in
// gqa_attention_volta_flash.cu, the only translation unit that sees the vendored
// llama.cpp kernel. See the V100 performance summary.

// Q-block width. Bounds mask memory, which is O(tokens * visible_keys): a full
// 12K prompt mask would be 288 MB and grows quadratically, one Q-block's is
// ~25 MB at that length. Phase 4 sweeps this.
inline constexpr std::int32_t kVoltaFlashQBlockTokens = 1024;

// Below this width the chunked route's smaller fixed cost wins, and the staging
// buffers are not worth allocating.
inline constexpr std::int32_t kVoltaFlashMinimumWidth = 64;

// Mask rows are padded to this multiple so the kernel's row tiles never read past
// the buffer on a Q-block whose token count is not a whole number of tiles.
inline constexpr std::int32_t kVoltaFlashMaskRowPad = 64;

// The key extent is padded to this multiple (FATTN_KQ_STRIDE upstream). The kernel
// is only ever exercised with a padded key count in llama.cpp, and its last key
// tile reads a whole tile past the true count.
inline constexpr std::int32_t kVoltaFlashKeyPad = 256;

// Upper bound on the stream-K fixup metadata, in float2 elements, for one Q-block
// of the given token count. Depends on the device's SM count and the kernel's
// achieved occupancy, so it is a runtime query rather than a constant.
std::size_t gqa_attention_volta_flash_meta_elements(std::int32_t q_heads,
                                                   std::int32_t tokens);

void gqa_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& positions, const Tensor& table_rows,
                                      float scale, PagedKVBatchLayerView cache,
                                      GqaExecutionEnvelope envelope, std::int32_t q_block_tokens,
                                      Tensor& k_gathered, Tensor& v_gathered, Tensor& mask,
                                      Tensor& q_f32, Tensor& out_f32, Tensor& dst_meta, Tensor& out,
                                      cudaStream_t stream);
#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
