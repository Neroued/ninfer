#include "ninfer/ops/kv_cache_append.h"

#include "core/paged_kv_storage.h"
#include "ops/kv_cache/append/launch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim       = 128;
constexpr std::int32_t kKVHeads       = 8;
constexpr std::uint32_t kWindow       = 4096;
constexpr std::int32_t kFullHeadDim   = 256;
constexpr const char* kAppendOp       = "kv_cache_append";
constexpr const char* kPrefixAppendOp = "kv_cache_append_prefix";

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous and non-null");
    }
}

std::uint32_t validate_full_cache(const PagedKVLayerView& cache, std::int32_t kv_heads) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kFullHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("kv_cache_append: invalid cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kFullHeadDim) {
        throw std::invalid_argument("kv_cache_append: invalid cache geometry or storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("kv_cache_append: invalid cache capacity");
    }

    if (cache.k_pages.dtype != layout.key.data_dtype ||
        cache.v_pages.dtype != layout.value.data_dtype) {
        throw std::invalid_argument("kv_cache_append: invalid cache data dtype");
    }
    require_shape(cache.k_pages, layout.key.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, kAppendOp, "cache k pages");
    require_shape(cache.v_pages, layout.value.data_leading_extent, kPagedKVPageSize, kv_heads,
                  physical_pages, kAppendOp, "cache v pages");
    require_contiguous_nonnull(cache.k_pages, kAppendOp, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, kAppendOp, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument("kv_cache_append: block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, kAppendOp, "block table");
    require_contiguous_nonnull(cache.block_table, kAppendOp, "block table");

    const auto validate_scale = [&](const Tensor& tensor, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (tensor.data != nullptr) {
                throw std::invalid_argument("kv_cache_append: unscaled cache must not have scales");
            }
            return;
        }
        if (tensor.dtype != vector.scale_dtype) {
            throw std::invalid_argument("kv_cache_append: invalid cache scale dtype");
        }
        require_shape(tensor, vector.scale_leading_extent, kPagedKVPageSize, kv_heads,
                      physical_pages, kAppendOp, name);
        require_contiguous_nonnull(tensor, kAppendOp, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache k scale pages");
    validate_scale(cache.v_scale_pages, layout.value, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

void require_vector_aligned(const Tensor& tensor, const char* name) {
    require_contiguous_nonnull(tensor, kPrefixAppendOp, name);
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & 15u) != 0u) {
        throw std::invalid_argument("kv_cache_append_prefix: " + std::string(name) +
                                    " must be 16-byte aligned");
    }
}

detail::KVCacheAppendPrefixPlan validate_inputs(const Tensor& k, const Tensor& v,
                                                const Tensor& positions, const Tensor& counts,
                                                const Tensor& selectors,
                                                KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("kv_cache_append_prefix: k/v must be BF16");
    }
    if (positions.dtype != DType::I32 || counts.dtype != DType::I32 ||
        selectors.dtype != DType::I32) {
        throw std::invalid_argument(
            "kv_cache_append_prefix: positions/counts/selectors must be I32");
    }
    const std::int32_t tokens = k.ne[2];
    const std::int32_t batch  = k.ne[3];
    if (tokens < 1) { throw std::invalid_argument("kv_cache_append_prefix: T must be positive"); }
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("kv_cache_append_prefix: B must be 1..8");
    }
    require_shape(k, kHeadDim, kKVHeads, tokens, batch, kPrefixAppendOp, "k");
    require_shape(v, kHeadDim, kKVHeads, tokens, batch, kPrefixAppendOp, "v");
    require_shape(positions, tokens, batch, 1, 1, kPrefixAppendOp, "positions");
    require_shape(counts, batch, 1, 1, 1, kPrefixAppendOp, "counts");
    require_shape(selectors, batch, 1, 1, 1, kPrefixAppendOp, "selectors");
    require_vector_aligned(k, "k");
    require_vector_aligned(v, "v");
    require_contiguous_nonnull(positions, kPrefixAppendOp, "positions");
    require_contiguous_nonnull(counts, kPrefixAppendOp, "counts");
    require_contiguous_nonnull(selectors, kPrefixAppendOp, "selectors");
    return detail::kv_cache_append_prefix_resolve_plan(tokens, envelope);
}

void validate_paged_cache(const PagedKVBatchLayerView& cache,
                          KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (cache.storage != KvCacheStorage::BFloat16 || cache.num_kv_heads != kKVHeads ||
        cache.head_dim != kHeadDim || cache.k_pages.ne[2] <= 0 ||
        cache.k_pages.ne[2] != cache.v_pages.ne[2] || cache.block_tables.ne[0] <= 0 ||
        envelope.max_count >
            static_cast<std::uint32_t>(cache.block_tables.ne[0]) * kPagedKVPageSize) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid paged cache");
    }
    const std::int32_t physical_pages = cache.k_pages.ne[2];
    if (cache.k_pages.dtype != DType::BF16 || cache.v_pages.dtype != DType::FP16 ||
        cache.k_pages.ne[0] != kHeadDim || cache.k_pages.ne[1] != kPagedKVPageSize ||
        cache.k_pages.ne[3] != kKVHeads || cache.v_pages.ne[0] != kHeadDim ||
        cache.v_pages.ne[1] != kPagedKVPageSize || cache.v_pages.ne[2] != physical_pages ||
        cache.v_pages.ne[3] != kKVHeads || cache.k_scale_pages.data != nullptr ||
        cache.v_scale_pages.data != nullptr || cache.block_tables.dtype != DType::I32 ||
        cache.block_tables.ne[1] <= 0 || cache.block_tables.ne[2] != 1 ||
        cache.block_tables.ne[3] != 1) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid paged cache tensors");
    }
    require_vector_aligned(cache.k_pages, "cache k pages");
    require_vector_aligned(cache.v_pages, "cache v pages");
    require_contiguous_nonnull(cache.block_tables, kPrefixAppendOp, "cache block tables");
}

void validate_cyclic_cache(const CyclicKVCacheLayerView& cache,
                           KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (cache.num_kv_heads != kKVHeads || cache.head_dim != kHeadDim || cache.capacity != kWindow ||
        cache.padded_capacity < cache.capacity ||
        cache.padded_capacity >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        envelope.max_count > cache.capacity) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid cyclic cache");
    }
    const auto padded = static_cast<std::int32_t>(cache.padded_capacity);
    if (cache.k.dtype != DType::BF16 || cache.v.dtype != DType::FP16 || cache.k.ne[0] != kHeadDim ||
        cache.k.ne[1] != padded || cache.k.ne[2] != kKVHeads || cache.v.ne[0] != kHeadDim ||
        cache.v.ne[1] != padded || cache.v.ne[2] != kKVHeads ||
        cache.v.ne[3] != cache.lane_capacity || cache.lane_capacity <= 0 ||
        cache.k.ne[3] != cache.lane_capacity) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid cyclic cache tensors");
    }
    require_vector_aligned(cache.k, "cache k");
    require_vector_aligned(cache.v, "cache v");
}

} // namespace

void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, cudaStream_t stream) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("kv_cache_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("kv_cache_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    if (kv_heads != 4 && kv_heads != 2) {
        throw std::invalid_argument("kv_cache_append: unsupported KV head geometry");
    }
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("kv_cache_append: T must be positive"); }
    require_shape(k, kFullHeadDim, kv_heads, tokens, 1, kAppendOp, "k");
    require_shape(v, kFullHeadDim, kv_heads, tokens, 1, kAppendOp, "v");
    require_shape(positions, tokens, 1, 1, 1, kAppendOp, "positions");
    require_contiguous_nonnull(k, kAppendOp, "k");
    require_contiguous_nonnull(v, kAppendOp, "v");
    require_contiguous_nonnull(positions, kAppendOp, "positions");
    const std::uint32_t capacity = validate_full_cache(cache, kv_heads);
    if (static_cast<std::uint32_t>(tokens) > capacity) {
        throw std::invalid_argument("kv_cache_append: T exceeds cache capacity");
    }
    if (cache.storage == KvCacheStorage::Fp8KeyNvfp4Value) {
        detail::kv_cache_append_k8v4_launch(k, v, positions, cache, stream);
    } else if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::kv_cache_append_nvfp4_launch(k, v, positions, cache, stream);
    } else {
        detail::kv_cache_append_launch(k, v, positions, cache, stream);
    }
}

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto plan = validate_inputs(k, v, positions, counts, table_rows, envelope);
    validate_paged_cache(cache, envelope);
    detail::kv_cache_append_prefix_launch(k, v, positions, counts, table_rows, cache, plan, stream);
}

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& lanes,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, cudaStream_t stream) {
    const auto plan = validate_inputs(k, v, positions, counts, lanes, envelope);
    validate_cyclic_cache(cache, envelope);
    detail::kv_cache_append_prefix_launch(k, v, positions, counts, lanes, cache, plan, stream);
}

} // namespace ninfer::ops
