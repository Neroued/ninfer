#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "ops/common/host/graph_profile.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/internal.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/kernels/causal/operands.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct CausalGraphWorkspaceLayout {
    WorkspaceLayoutBuilder layout;
    std::array<std::size_t, 3> offsets{};
    std::size_t next = 0;

    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape) {
        offsets[next++] = layout.peak_bytes();
        return layout.alloc(dtype, shape);
    }
};

std::uint64_t partition_graph_word(std::initializer_list<int> fields) {
    // Six BF16 fields fit in one word. Each field is at most 256.
    std::uint64_t word = 0;
    for (const int field : fields) word = (word << 9) | std::uint64_t(field);
    return word;
}

template <class Plan>
detail::CausalSlicePlan
quantized_slice_plan(const Plan& plan, int kv_heads, bool append, bool tiled_partials,
                     detail::kv::PagedWriteInstance (*select_write)(int, int, int)) {
    using Family     = decltype(plan.family);
    const bool tiled = plan.family == Family::Tiled;
    const auto& p    = plan.partition;
    const auto partition =
        !tiled || tiled_partials
            ? partition_graph_word({p.capacity, p.target, p.key_shift, p.single_target})
            : 0;
    const int slots   = (tiled ? tiled_partials : p.capacity > 1) ? p.capacity : 0;
    const int visible = tiled ? 0 : int(context_bucket_tokens(plan.bucket));
    const int writer  = append && plan.family != Family::Grouped
                            ? 1 + int(select_write(kv_heads, plan.width, plan.batch))
                            : 0;
    return {int(plan.family), plan.query_tile, visible, writer, partition, slots};
}

void require_causal_extent(ContextBucket bucket, int batch, int width, bool masked,
                           const char* op) {
    ContextBucketRange{bucket}.validate(ContextBucket::K256, op);
    if (width <= 0 || batch <= 0 || (batch > 1 && width > detail::kCausalMaximumVerify)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    if (!masked && context_bucket_tokens(bucket) < static_cast<std::uint32_t>(width)) {
        throw std::invalid_argument(std::string(op) + ": context bucket is shorter than W");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                AttentionHeadGeometry geometry, const ConstPagedKVLayerView& cache,
                                ContextBucket bucket, float scale, const char* op, Device device) {
    detail::require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - detail::kCausalExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t tokens   = q.ne[2];
    require_causal_extent(bucket, 1, tokens, false, op);
    common::operand::shape(q, {detail::kCausalHeadDim, q_heads, tokens, 1}, op, "q");
    common::operand::shape(positions, {tokens, 1, 1, 1}, op, "positions");
    common::operand::shape(out, {detail::kCausalHeadDim, q_heads, tokens, 1}, op, "out");
    common::operand::contiguous_nonnull(q, op, "q", device);
    common::operand::contiguous_nonnull(positions, op, "positions", device);
    common::operand::contiguous_nonnull(out, op, "out", device);
    const auto& offsets = cache.k_pages.records.page_offsets;
    common::operand::shape(offsets, {offsets.ne[0], 1, 1, 1}, op, "block table");
    common::operand::contiguous_nonnull(offsets, op, "block table", device);
    detail::validate_causal_batch_cache(single_row_paged_kv_batch_view(cache), kv_heads, op,
                                        device);
}

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const ConstPagedKVBatchLayerView& cache,
                                        AttentionHeadGeometry geometry, ContextBucket bucket,
                                        float scale, const char* op, Device device) {
    detail::require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - detail::kCausalExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    require_causal_extent(bucket, batch, width, masked, op);
    common::operand::shape(q, {detail::kCausalHeadDim, q_heads, width, batch}, op, "q");
    common::operand::shape(positions, {width, batch, 1, 1}, op, "positions");
    if (masked) { common::operand::shape(valid_columns, {batch, 1, 1, 1}, op, "valid columns"); }
    common::operand::shape(kv_table_rows, {batch, 1, 1, 1}, op, "KV table rows");
    common::operand::shape(out, {detail::kCausalHeadDim, q_heads, width, batch}, op, "out");
    common::operand::contiguous_nonnull(q, op, "q", device);
    common::operand::contiguous_nonnull(positions, op, "positions", device);
    if (masked) { common::operand::contiguous_nonnull(valid_columns, op, "valid columns", device); }
    common::operand::contiguous_nonnull(kv_table_rows, op, "KV table rows", device);
    common::operand::contiguous_nonnull(out, op, "out", device);
    detail::validate_causal_batch_cache(cache, kv_heads, op, device);
}

} // namespace

namespace detail {

CausalSliceGraphWords CausalSlicePlan::graph_words(int heads, int width, int batch) const {
    CausalGraphWorkspaceLayout layout;
    if (partial_slots > 0)
        (void)allocate_causal_partials(layout, heads, width, partial_slots, batch);
    return {std::uint64_t(route),
            std::uint64_t(query_tile),
            std::uint64_t(visible),
            std::uint64_t(writer),
            partition,
            layout.offsets[1],
            layout.offsets[2]};
}

void require_causal_geometry(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kCausalHeadDim ||
        !((geometry.query_heads == 24 && geometry.kv_heads == 4) ||
          (geometry.query_heads == 16 && geometry.kv_heads == 2))) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

int causal_rows_per_launch(int batch, int width) {
    // The merge launch places W*B in grid.z. Single-row prefill has its own routes.
    return batch == 1 ? 1 : std::min(batch, kCausalMaximumRows / width);
}

void validate_causal_batch_cache(const ConstPagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                 const char* op, Device device) {
    common::operand::device(cache.device(), device, op, "cache");
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kCausalHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kCausalHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }

    const auto& offsets = cache.k_pages.records.page_offsets;
    if (offsets.ne[0] <= 0 || offsets.ne[1] <= 0 ||
        cache.k_pages.records.page_tokens != kPagedKVPageSize) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }
    common::operand::paged_plane(cache.k_pages, layout.key.data_dtype,
                                 {layout.key.data_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                 op, "cache K", device);
    common::operand::paged_plane(cache.v_pages, layout.value.data_dtype,
                                 {layout.value.data_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                 op, "cache V", device);
    common::operand::same_paged_records(cache.k_pages, cache.v_pages, op, "cache K/V");
    const auto validate_scale = [&](const ConstPagedPlaneView& plane,
                                    const PagedKVVectorLayout& vector, const char* name) {
        if (!vector.has_scale()) {
            if (plane.records.storage.data != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled KV cache must not have scales");
            }
            return;
        }
        common::operand::paged_plane(plane, vector.scale_dtype,
                                     {vector.scale_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                     op, name, device);
        common::operand::same_paged_records(cache.k_pages, plane, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache K scales");
    validate_scale(cache.v_scale_pages, layout.value, "cache V scales");
}

CausalSlicePlan causal_slice_plan(AttentionHeadGeometry g, KvCacheStorage storage,
                                  ContextBucket bucket, int rows, int width, int sms, bool append) {
    switch (storage) {
    case KvCacheStorage::BFloat16: {
        const auto plan = select_bf16_kv_causal_plan(g.query_heads, width, rows, bucket, sms);
        const auto& p   = plan.partition;
        const auto description = bf16_kv_instance_description(plan.instance);
        const auto partition =
            description.grouped
                ? partition_graph_word({p.capacity, p.normal_target, p.long_target, p.key_rows,
                                        p.single_normal_target, p.single_long_target})
                : 0;
        const int writer =
            append && !description.grouped ? 1 + int(select_bf16_kv_write(g.kv_heads)) : 0;
        return {int(plan.instance), description.query_rows,         0, writer,
                partition,          p.capacity > 1 ? p.capacity : 0};
    }
    case KvCacheStorage::Fp8E4m3fnGroup128Ue8m0:
        return quantized_slice_plan(
            select_fp8_kv_causal_plan(g.query_heads, width, rows, bucket, sms), g.kv_heads, append,
            true, select_fp8_kv_write);
    case KvCacheStorage::Int8Group64:
        return quantized_slice_plan(
            select_int8_kv_causal_plan(g.query_heads, width, rows, bucket, sms), g.kv_heads, append,
            false, select_int8_kv_write);
    case KvCacheStorage::Nvfp4Group16:
        return quantized_slice_plan(
            select_nvfp4_kv_causal_plan(g.query_heads, width, rows, bucket, sms), g.kv_heads,
            append, false, select_nvfp4_kv_write);
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return quantized_slice_plan(
            select_k8v4_kv_causal_plan(g.query_heads, width, rows, bucket, sms), g.kv_heads, append,
            true, select_k8v4_kv_write);
    }
    throw std::invalid_argument("causal_softmax_attention: unsupported KV cache storage");
}

} // namespace detail

OpGraphProfile causal_softmax_attention_graph_profile(AttentionHeadGeometry geometry,
                                                      KvCacheStorage cache_storage,
                                                      ContextBucket bucket, std::int32_t batch_size,
                                                      std::int32_t tokens,
                                                      std::int32_t multiprocessor_count,
                                                      bool append, bool masked) {
    constexpr const char* op = "causal_softmax_attention graph";
    detail::require_causal_geometry(geometry, op);
    require_causal_extent(bucket, batch_size, tokens, masked, op);
    if (multiprocessor_count <= 0 || (!append && (batch_size != 1 || masked))) {
        throw std::invalid_argument("causal_softmax_attention graph: invalid profile");
    }
    const int rows     = detail::causal_rows_per_launch(batch_size, tokens);
    const int tail     = batch_size % rows;
    const auto profile = [&](int slice_rows) {
        return detail::causal_slice_plan(geometry, cache_storage, bucket, slice_rows, tokens,
                                         multiprocessor_count, append)
            .graph_words(geometry.query_heads, tokens, slice_rows);
    };
    const auto full = profile(rows);
    const auto last = tail ? profile(tail) : detail::CausalSliceGraphWords{};
    const std::array header{
        std::uint64_t(geometry.head_dim),
        std::uint64_t(geometry.query_heads),
        std::uint64_t(geometry.kv_heads),
        std::uint64_t(cache_storage),
        std::uint64_t(batch_size),
        std::uint64_t(tokens),
        std::uint64_t(append),
        std::uint64_t(masked),
        std::uint64_t(rows),
        std::uint64_t(tail),
    };
    std::array<std::uint64_t, header.size() + 2 * full.size()> words{};
    auto end = std::copy(header.begin(), header.end(), words.begin());
    end      = std::copy(full.begin(), full.end(), end);
    std::copy(last.begin(), last.end(), end);
    return detail::make_op_graph_profile(words);
}

std::size_t causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage, ContextBucketRange buckets,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width,
    std::int32_t multiprocessor_count) {
    detail::require_causal_geometry(geometry, "causal_softmax_attention workspace");
    buckets.validate(ContextBucket::K256, "causal_softmax_attention workspace");
    const std::int32_t q_heads = geometry.query_heads;
    (void)paged_kv_storage_layout(cache_storage, detail::kCausalHeadDim);
    if (multiprocessor_count <= 0 || batch_size <= 0 || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > detail::kCausalMaximumVerify)) {
        throw std::invalid_argument(
            "causal_softmax_attention workspace: invalid profile or interval");
    }

    const auto capacity = [&](ContextBucket bucket, int rows, int first_width, int last_width) {
        if (cache_storage == KvCacheStorage::BFloat16)
            return detail::bf16_kv_workspace_bytes(q_heads, rows, first_width, last_width, bucket,
                                                   multiprocessor_count);

        if (cache_storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0)
            return detail::fp8_kv_workspace_bytes(q_heads, rows, first_width, last_width, bucket,
                                                  multiprocessor_count);

        if (cache_storage == KvCacheStorage::Int8Group64)
            return detail::int8_kv_workspace_bytes(q_heads, rows, first_width, last_width, bucket,
                                                   multiprocessor_count);

        if (cache_storage == KvCacheStorage::Nvfp4Group16)
            return detail::nvfp4_kv_workspace_bytes(q_heads, rows, first_width, last_width, bucket,
                                                    multiprocessor_count);

        return detail::k8v4_kv_workspace_bytes(q_heads, rows, first_width, last_width, bucket,
                                               multiprocessor_count);
    };
    std::size_t maximum = 0;
    buckets.for_each([&](ContextBucket bucket) {
        if (detail::causal_rows_per_launch(batch_size, max_width) == batch_size) {
            maximum = std::max(maximum, capacity(bucket, batch_size, min_width, max_width));
            return;
        }
        for (int width = min_width; width <= max_width; ++width) {
            const int rows = detail::causal_rows_per_launch(batch_size, width);
            maximum        = std::max(maximum, capacity(bucket, rows, width, width));
            const int tail = batch_size % rows;
            if (tail != 0) maximum = std::max(maximum, capacity(bucket, tail, width, width));
        }
    });
    return maximum;
}

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "causal_softmax_attention: workspace belongs to another device");
    }
    constexpr const char* op = "causal_softmax_attention";
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       geometry, bucket, scale, op, execution.device());
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("causal_softmax_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = geometry.kv_heads;
    common::operand::shape(k, {detail::kCausalHeadDim, kv_heads, width, batch}, op, "k");
    common::operand::shape(v, {detail::kCausalHeadDim, kv_heads, width, batch}, op, "v");
    common::operand::contiguous_nonnull(k, op, "k", execution.device());
    common::operand::contiguous_nonnull(v, op, "v", execution.device());

    const auto dispatch = [&](const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, Tensor& out) {
        if (cache.storage == KvCacheStorage::BFloat16) {
            detail::bf16_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows,
                                             scale, cache, bucket, workspace, out, execution);
            return;
        }

        if (cache.storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
            detail::fp8_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                            cache, bucket, workspace, out, execution);
            return;
        }

        if (cache.storage == KvCacheStorage::Int8Group64) {
            detail::int8_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows,
                                             scale, cache, bucket, workspace, out, execution);
            return;
        }

        if (cache.storage == KvCacheStorage::Nvfp4Group16) {
            detail::nvfp4_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows,
                                              scale, cache, bucket, workspace, out, execution);
            return;
        }

        detail::k8v4_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                         cache, bucket, workspace, out, execution);
    };
    // The merge launch places W*B in grid.z. Slice physical rows without changing validity.
    const int rows_per_launch = detail::causal_rows_per_launch(batch, width);
    for (std::int64_t begin = 0; begin < batch; begin += rows_per_launch) {
        const int rows = std::min<std::int64_t>(rows_per_launch, batch - begin);
        auto output    = out.slice(3, begin, rows);
        dispatch(q.slice(3, begin, rows), k.slice(3, begin, rows), v.slice(3, begin, rows),
                 positions.slice(1, begin, rows),
                 valid_columns.data ? valid_columns.slice(0, begin, rows) : Tensor{},
                 kv_table_rows.slice(0, begin, rows), output);
    }
}

void causal_softmax_attention_cached(const Tensor& q, const Tensor& positions,
                                     AttentionHeadGeometry geometry, float scale,
                                     const ConstPagedKVLayerView& cache, ContextBucket bucket,
                                     MemoryArena& workspace, Tensor& out,
                                     CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "causal_softmax_attention_cached: workspace belongs to another device");
    }
    constexpr const char* op = "causal_softmax_attention_cached";
    validate_attention_tensors(q, positions, out, geometry, cache, bucket, scale, op,
                               execution.device());

    if (cache.storage == KvCacheStorage::BFloat16) {
        detail::bf16_kv_cached_attention(q, positions, scale, cache, bucket, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        detail::fp8_kv_cached_attention(q, positions, scale, cache, bucket, workspace, out,
                                        execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Int8Group64) {
        detail::int8_kv_cached_attention(q, positions, scale, cache, bucket, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::nvfp4_kv_cached_attention(q, positions, scale, cache, bucket, workspace, out,
                                          execution);
        return;
    }

    detail::k8v4_kv_cached_attention(q, positions, scale, cache, bucket, workspace, out, execution);
}

} // namespace ninfer::ops
