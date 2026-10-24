#pragma once

#include "core/layout.h"
#include "core/state_image.h"
#include "ninfer/ops/kpool_mla.h"

#include <cstdint>

namespace ninfer::ops {

// Address-free storage geometry. These helpers describe and bind caller-owned resources;
// they do not allocate Device/Host backing, choose requests or establish compute support.
// Positive dimensions are required; FP8 latent_dim is a multiple of 128. Execution and its
// workspace queries separately admit registered contraction geometries.
struct KpoolMlaCacheSpec {
    KpoolMlaStorage storage = KpoolMlaStorage::BFloat16;
    std::int32_t latent_dim = 0, index_dim = 0, layers = 0;
};

// One page group contains every layer, each in data / optional scales / index order. Plane
// offsets and total size come from plan_kv_page_layout(), with 64 tokens and 256-byte plane
// alignment. A group may reside anywhere in the caller's Context backing. Main/MTP or other
// distinct histories use separate layouts/tables according to their owner's schedule.
struct KpoolMlaCacheLayout {
    KpoolMlaCacheSpec spec;
    KVPageLayout page;

    // Bind one layer to a common arena and I64 page-offset table. No payload or table entries
    // are read; layer must be in [0,spec.layers), and records.page_tokens must match the layout.
    [[nodiscard]] KpoolMlaCacheView layer(PagedRecordView records, std::int32_t layer) const;
    [[nodiscard]] ConstKpoolMlaCacheView layer(ConstPagedRecordView records,
                                               std::int32_t layer) const;
};

[[nodiscard]] KpoolMlaCacheLayout plan_kpool_mla_cache_layout(const KpoolMlaCacheSpec& spec);

struct KpoolTailFields {
    StateField raw_keys;
    StateField raw_gates;
};

// Append two BF16 [index_dim,3] fields with the given positive layer count to the caller's
// StateImageBuilder. Field starts/layer strides use the builder's 256-byte layout. The caller
// binds these IDs using StateImageStorage::field(..., offsets), then selects layer(l) for
// Prepare. The same field schema can describe a frame-local temporary tail without a StateId.
[[nodiscard]] KpoolTailFields append_kpool_tail_fields(StateImageBuilder& builder,
                                                       std::int32_t index_dim, std::int32_t layers);

// Frame records have [Di,W,R,L] shape; dimensions/strides in these regions are authoritative.
// Shared Device metadata (base positions, available prefixes and per-call offsets) is supplied
// separately so all layers can use one set. The frame retains records through accepted-prefix
// extraction's device completion; these are not Op-local scratch. Planning appends to the
// caller's larger frame LayoutBuilder.
struct KpoolRawRecordLayout {
    TensorRegion raw_keys;
    TensorRegion raw_gates;

    // Single layer, first B rows, with the original W/R/L strides retained; 1<=B<=R.
    [[nodiscard]] KpoolRawRecordWriteView write_view(MemorySpan backing, std::int32_t layer,
                                                     std::int32_t batch_size,
                                                     const Tensor& offsets) const;
    [[nodiscard]] KpoolRawRecordReadView read_view(MemorySpan backing, const Tensor& base_positions,
                                                   const Tensor& valid_columns) const;
};

// Positive Di/W/L and 1<=R<=8; checked Tensor extents and size arithmetic. Each plane starts
// at 256-byte alignment, feature contiguous. Metadata and final frame padding are excluded.
[[nodiscard]] KpoolRawRecordLayout
plan_kpool_raw_records(LayoutBuilder& builder, std::int32_t index_dim, std::int32_t width,
                       std::int32_t record_capacity, std::int32_t layers);

// The three regions together form one selection table. Its row count is independent of a
// later consumer's T/B: selection_rows maps that consumer to the retained producer table.
// Metadata mappings, positions and lengths remain caller-owned frame inputs.
struct KpoolBlockSelectionLayout {
    TensorRegion pool_ids;
    TensorRegion pool_counts;
    TensorRegion selection_end;

    [[nodiscard]] KpoolBlockSelection bind(MemorySpan backing) const;
};

// Positive Nselection; append I32 [512,Nselection], [Nselection], [Nselection] planes, each
// at 256-byte alignment. Payload is 2056*Nselection bytes; allocation includes alignment gaps.
[[nodiscard]] KpoolBlockSelectionLayout plan_kpool_block_selection(LayoutBuilder& builder,
                                                                   std::int32_t selection_rows);

} // namespace ninfer::ops
