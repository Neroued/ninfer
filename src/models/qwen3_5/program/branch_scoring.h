#pragma once

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/arena.h"
#include "models/common/context/transfer.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/types.h"

#include <cstdint>
#include <array>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5 {

inline constexpr std::uint32_t kBranchBatch = 8;
inline constexpr std::uint32_t kBranchWidth = 16;

// A non-owning overlay of the general workspace; Context records remain Session-owned.
struct BranchLayout {
    std::uint32_t batch_capacity = 0;
    std::uint32_t prefill_chunk  = 0;
    KVExecutionTableLayout tables;
    GdnReplayRecordLayout records;
    TensorRegion ids, positions, rope_positions, counts, rows, offsets, hidden, tails, logits;
    LayoutRegion scratch;
    std::size_t bytes = 0;

    [[nodiscard]] std::size_t pinned_bytes() const noexcept {
        return std::size_t(tables.spec.logical_page_capacity) * tables.spec.table_rows *
               sizeof(std::int64_t);
    }
};

namespace detail {
// Borrows the native Context payload, with separate tables inside the idle general workspace.
// Mappings survive individual BranchWork frames; each newly bound history republishes its row.
struct BranchWorkspace {
    BranchWorkspace(MemorySpan, const BranchLayout&, PagedKVCache&, MemorySpan, DeviceContext&);
    PagedKVCache cache;
    context::ContextStorage storage;
    context::ContextMigration actions;
};

struct BranchExecution {
    BranchExecution(MemorySpan, const BranchLayout&, LinearAttentionStateView&);
    GdnReplayRecords records;
    std::optional<ops::GdnReplayFoldPlan> fold;
    Tensor ids, positions, rope_positions, counts, rows, offsets, hidden, tails, logits;
    MemoryArena work;
    std::vector<std::vector<std::uint16_t>> readback;
    std::vector<TokenId> lows;
    std::array<TokenId, kBranchWidth * kBranchBatch> host_ids{};
    std::array<std::int32_t, kBranchWidth * kBranchBatch> host_positions{};
    std::array<std::int32_t, kBranchWidth * kBranchBatch> host_rope{};
    std::array<std::int32_t, kBranchBatch> host_counts{}, host_rows{};
    std::array<std::int64_t, kBranchBatch> host_offsets{};
};
} // namespace detail

} // namespace ninfer::models::qwen3_5
