#include "ninfer/ops/kv_cache_append.h"
#include "core/paged_kv_storage.h"
#include "ops/common/host/operands.h"
#include "ops/kv_cache/append/dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim       = 128;
constexpr std::int32_t kKVHeads       = 8;
constexpr std::int32_t kFullHeadDim   = 256;
constexpr const char* kAppendOp       = "kv_cache_append";
constexpr const char* kPrefixAppendOp = "kv_cache_append_prefix";

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name,
                                Device device) {
    common::operand::device(tensor.device, device, op, name);
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name +
                                    " must be contiguous and non-null");
    }
}

std::uint32_t validate_full_cache(const PagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                  Device device, const char* op) {
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kFullHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kFullHeadDim ||
        cache.k_pages.records.page_tokens != kPagedKVPageSize) {
        throw std::invalid_argument(std::string(op) + ": invalid cache geometry or storage");
    }

    const auto& offsets         = cache.k_pages.records.page_offsets;
    const std::int64_t capacity = static_cast<std::int64_t>(offsets.ne[0]) * kPagedKVPageSize;
    if (capacity <= 0 || capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid cache capacity");
    }
    common::operand::paged_plane(cache.k_pages, layout.key.data_dtype,
                                 {layout.key.data_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                 op, "cache K", device);
    common::operand::paged_plane(cache.v_pages, layout.value.data_dtype,
                                 {layout.value.data_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                 op, "cache V", device);
    common::operand::same_paged_records(cache.k_pages, cache.v_pages, op, "cache K/V");
    const auto validate_scale = [&](const PagedPlaneView& plane, const PagedKVVectorLayout& vector,
                                    const char* name) {
        if (!vector.has_scale()) {
            if (plane.data() != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled cache must not have scales");
            }
            return;
        }
        common::operand::paged_plane(plane, vector.scale_dtype,
                                     {vector.scale_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                     op, name, device);
        common::operand::same_paged_records(cache.k_pages, plane, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache K scale");
    validate_scale(cache.v_scale_pages, layout.value, "cache V scale");
    return static_cast<std::uint32_t>(capacity);
}

void require_vector_aligned(const Tensor& tensor, const char* name, Device device) {
    require_contiguous_nonnull(tensor, kPrefixAppendOp, name, device);
    if (!common::operand::aligned(tensor.data, 16)) {
        throw std::invalid_argument("kv_cache_append_prefix: " + std::string(name) +
                                    " must be 16-byte aligned");
    }
}

void validate_inputs(const Tensor& k, const Tensor& v, const Tensor& positions,
                     const Tensor& counts, const Tensor* selectors,
                     KVCacheAppendPrefixExecutionEnvelope envelope, Device device,
                     std::int32_t head_dim, std::int32_t kv_heads) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("kv_cache_append_prefix: k/v must be BF16");
    }
    if (positions.dtype != DType::I32 || counts.dtype != DType::I32 ||
        (selectors && selectors->dtype != DType::I32)) {
        throw std::invalid_argument(
            "kv_cache_append_prefix: positions/counts/selectors must be I32");
    }
    const std::int32_t tokens = k.ne[2];
    const std::int32_t batch  = k.ne[3];
    if (tokens < 1) { throw std::invalid_argument("kv_cache_append_prefix: T must be positive"); }
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("kv_cache_append_prefix: B must be 1..8");
    }
    require_shape(k, head_dim, kv_heads, tokens, batch, kPrefixAppendOp, "k");
    require_shape(v, head_dim, kv_heads, tokens, batch, kPrefixAppendOp, "v");
    require_shape(positions, tokens, batch, 1, 1, kPrefixAppendOp, "positions");
    require_shape(counts, batch, 1, 1, 1, kPrefixAppendOp, "counts");
    if (selectors) require_shape(*selectors, batch, 1, 1, 1, kPrefixAppendOp, "selectors");
    require_vector_aligned(k, "k", device);
    require_vector_aligned(v, "v", device);
    require_contiguous_nonnull(positions, kPrefixAppendOp, "positions", device);
    require_contiguous_nonnull(counts, kPrefixAppendOp, "counts", device);
    if (selectors) require_contiguous_nonnull(*selectors, kPrefixAppendOp, "selectors", device);
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid execution envelope");
    }
}

void validate_paged_cache(const PagedKVBatchLayerView& cache,
                          KVCacheAppendPrefixExecutionEnvelope envelope, Device device) {
    if (cache.storage != KvCacheStorage::BFloat16 || cache.num_kv_heads != kKVHeads ||
        cache.head_dim != kHeadDim || cache.k_pages.records.page_tokens != kPagedKVPageSize ||
        envelope.max_count > static_cast<std::uint32_t>(cache.k_pages.records.page_offsets.ne[0]) *
                                 kPagedKVPageSize) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid paged cache");
    }
    common::operand::paged_plane(cache.k_pages, DType::BF16, {kHeadDim, kPagedKVPageSize, kKVHeads},
                                 16, kPrefixAppendOp, "cache K", device);
    common::operand::paged_plane(cache.v_pages, DType::FP16, {kHeadDim, kPagedKVPageSize, kKVHeads},
                                 16, kPrefixAppendOp, "cache V", device);
    common::operand::same_paged_records(cache.k_pages, cache.v_pages, kPrefixAppendOp, "cache K/V");
    if (cache.k_scale_pages.data() || cache.v_scale_pages.data()) {
        throw std::invalid_argument("kv_cache_append_prefix: unscaled cache must not have scales");
    }
}

void validate_cyclic_cache(const CyclicKVCacheLayerView& cache, std::int32_t batch,
                           KVCacheAppendPrefixExecutionEnvelope envelope, Device device) {
    const bool registered_capacity = cache.capacity == 2048 || cache.capacity == 4096;
    if (cache.num_kv_heads != kKVHeads || cache.head_dim != kHeadDim || !registered_capacity ||
        cache.padded_capacity < cache.capacity ||
        cache.padded_capacity >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        envelope.max_count > cache.capacity) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid cyclic cache");
    }
    const auto padded = static_cast<std::int32_t>(cache.padded_capacity);
    common::operand::state_field(cache.k, DType::BF16, {kHeadDim, padded, kKVHeads}, 16,
                                 kPrefixAppendOp, "cache K", device);
    common::operand::state_field(cache.v, DType::FP16, {kHeadDim, padded, kKVHeads}, 16,
                                 kPrefixAppendOp, "cache V", device);
    const auto& offsets = cache.k.records.byte_offsets;
    if (offsets.ne[0] != batch || cache.v.records.byte_offsets.ne[0] != batch ||
        cache.k.field.layers != 1 || cache.v.field.layers != 1 ||
        cache.v.records.byte_offsets.data != offsets.data ||
        cache.v.records.storage.data != cache.k.records.storage.data) {
        throw std::invalid_argument("kv_cache_append_prefix: inconsistent state records");
    }
}

} // namespace

void kv_cache_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                     PagedKVLayerView cache, CudaExecutionView execution) {
    const auto stream = execution.stream();
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
    require_contiguous_nonnull(k, kAppendOp, "k", execution.device());
    require_contiguous_nonnull(v, kAppendOp, "v", execution.device());
    require_contiguous_nonnull(positions, kAppendOp, "positions", execution.device());
    const std::uint32_t capacity = validate_full_cache(single_row_paged_kv_batch_view(cache),
                                                       kv_heads, execution.device(), kAppendOp);
    if (static_cast<std::uint32_t>(tokens) > capacity) {
        throw std::invalid_argument("kv_cache_append: T exceeds cache capacity");
    }
    detail::kv_cache_append_dispatch(k, v, positions, cache, stream);
}

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            PagedKVBatchLayerView cache, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const bool full_geometry = k.ne[0] == kFullHeadDim && (k.ne[1] == 2 || k.ne[1] == 4);
    validate_inputs(k, v, positions, counts, &table_rows, envelope, execution.device(),
                    full_geometry ? kFullHeadDim : kHeadDim, full_geometry ? k.ne[1] : kKVHeads);
    if (full_geometry) {
        const auto capacity =
            validate_full_cache(cache, k.ne[1], execution.device(), kPrefixAppendOp);
        if (envelope.max_count > capacity) {
            throw std::invalid_argument("kv_cache_append_prefix: prefix exceeds cache capacity");
        }
    } else {
        validate_paged_cache(cache, envelope, execution.device());
    }
    detail::kv_cache_append_prefix_dispatch(k, v, positions, counts, table_rows, cache, envelope,
                                            stream);
}

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, CudaExecutionView execution) {
    const auto stream = execution.stream();
    validate_inputs(k, v, positions, counts, nullptr, envelope, execution.device(), kHeadDim,
                    kKVHeads);
    validate_cyclic_cache(cache, k.ne[3], envelope, execution.device());
    detail::kv_cache_append_prefix_dispatch(k, v, positions, counts, cache, envelope, stream);
}

} // namespace ninfer::ops
