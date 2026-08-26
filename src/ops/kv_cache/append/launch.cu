#include "ops/kv_cache/append/launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/kernel.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

template <typename Geometry, typename CacheView, typename Metadata>
void launch_full(const Tensor& k, const Tensor& v, const Tensor& positions, CacheView cache,
                 Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    if (cache.dtype == DType::FP8_E4M3FN) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int TokensPerTile = 8;
            const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads));
            kv_cache_append_full_fp8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int FillWarps       = kBlock / 32;
            const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
            kv_cache_append_full_fp8_kernel<Geometry, Metadata><<<fill_grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::uint8_t*>(cache_k.data), static_cast<std::uint8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (cache.dtype == DType::I8) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        if (tokens >= 128 && Geometry::KVHeads == 2) {
            constexpr int TokensPerTile = 8;
            const int max_tiles         = div_up(tokens + TokensPerTile - 1, TokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads));
            kv_cache_append_full_i8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int FillWarps       = kBlock / 32;
            const std::int64_t fill_units = static_cast<std::int64_t>(tokens) * Geometry::KVHeads;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(FillWarps)));
            kv_cache_append_full_i8_kernel<Geometry, Metadata><<<fill_grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::int8_t*>(cache_k.data), static_cast<std::int8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    constexpr int Block         = Geometry::KVHeads == 4 ? 128 : 96;
    constexpr int VecElems      = 8;
    const std::int64_t elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                  (kKVCacheAppendFullHeadDim / VecElems);
    const int fill_grid = static_cast<int>(div_up(elements, static_cast<std::int64_t>(Block)));
    kv_cache_append_full_bf16_kernel<Geometry, Metadata><<<fill_grid, Block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data), metadata,
        static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
        tokens);
    CUDA_CHECK(cudaGetLastError());
}

void validate_plan(const Tensor& k, const KVCacheAppendPrefixPlan& plan) {
    if (plan.tokens != k.ne[2] || plan.min_count < 0 || plan.max_count < plan.min_count ||
        plan.max_count > plan.tokens) {
        throw std::invalid_argument("kv_cache_append_prefix: inconsistent plan");
    }
}

void launch_paged(const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
                  const Tensor& table_rows, PagedKVBatchLayerView cache,
                  const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    validate_plan(k, plan);
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<__nv_bfloat16*>(cache.k_pages.data);
    auto* cache_v       = static_cast<__nv_bfloat16*>(cache.v_pages.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(counts.data);
    const auto* rows    = static_cast<const std::int32_t*>(table_rows.data);
    const auto* tables  = static_cast<const std::int32_t*>(cache.block_tables.data);

    const dim3 grid(1 + (plan.max_count - 1) / 4, k.ne[3], 1);
    kv_cache_append_prefix_paged_kernel<<<grid, kBlock, 0, stream>>>(
        input_k, input_v, pos, count, rows, cache_k, cache_v, tables, cache.k_pages.ne[2],
        cache.block_tables.ne[0], plan.min_count, plan.max_count, plan.tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cyclic(const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
                   const Tensor& lanes, CyclicKVCacheLayerView cache,
                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    validate_plan(k, plan);
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<__nv_bfloat16*>(cache.k.data);
    auto* cache_v       = static_cast<__nv_bfloat16*>(cache.v.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(counts.data);
    const auto* lane    = static_cast<const std::int32_t*>(lanes.data);
    const int padded    = static_cast<int>(cache.padded_capacity);

    const dim3 grid(1 + (plan.max_count - 1) / 4, k.ne[3], 1);
    kv_cache_append_prefix_cyclic_kernel<<<grid, kBlock, 0, stream>>>(
        input_k, input_v, pos, count, lane, cache_k, cache_v, plan.min_count, plan.max_count,
        plan.tokens, padded);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void kv_cache_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                            PagedKVLayerView cache, cudaStream_t stream) {
    const KVCacheAppendDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
        launch_full<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
        return;
    }
    launch_full<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
}

void kv_cache_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  PagedKVBatchLayerView cache, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const KVCacheAppendBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (k.ne[1] == KVCacheAppendD256Kv4::KVHeads) {
            launch_full<KVCacheAppendD256Kv4>(k, v, positions, cache, metadata, stream);
            return;
        }
        launch_full<KVCacheAppendD256Kv2>(k, v, positions, cache, metadata, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (tokens < 1) {
        throw std::invalid_argument("kv_cache_append_prefix plan: T must be positive");
    }
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument("kv_cache_append_prefix plan: invalid execution envelope");
    }
    return {
        .tokens    = tokens,
        .min_count = static_cast<std::int32_t>(envelope.min_count),
        .max_count = static_cast<std::int32_t>(envelope.max_count),
    };
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& table_rows,
                                   PagedKVBatchLayerView cache, const KVCacheAppendPrefixPlan& plan,
                                   cudaStream_t stream) {
    launch_paged(k, v, positions, counts, table_rows, cache, plan, stream);
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& lanes,
                                   CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    launch_cyclic(k, v, positions, counts, lanes, cache, plan, stream);
}

} // namespace ninfer::ops::detail
