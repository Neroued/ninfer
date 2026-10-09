#pragma once

#include "ninfer/branch_score.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/storage/kv_address_space.h"
#include "ninfer/ops/gdn_replay.h"

namespace ninfer::models::qwen3_5::detail {

inline constexpr std::int32_t kBranchBatch = 8;
inline constexpr std::int32_t kBranchWidth = 16;

struct BranchLayout {
    std::int32_t batch_capacity = 0;
    std::uint32_t prefill_chunk = 0;
    KVExecutionTableLayout tables;
    LinearAttentionStatePoolLayout states;
    GdnReplayRecordLayout records;
    TensorRegion ids, positions, counts, rows, slots, hidden, tails, logits;
    LayoutRegion scratch;
    std::size_t bytes = 0;
};

// Temporary overlay of the Program's existing workspace, not a startup reservation.
// Released before generation reuses that workspace; no weights or device allocation of its own.
// Slot 0: static prefix; 1..B: immutable context trunks; B+1..2B: mutable branches.
struct BranchStorage {
    qwen3_5::PagedKVCache kv;
    const std::int32_t batch_capacity;
    LinearAttentionStatePool states;
    GdnReplayRecords records;
    ops::GdnReplayFoldPlan fold;
    KVAddressSpaceStore addresses;
    Tensor ids, positions, counts, rows, slots, hidden, tails, logits;
    WorkspaceArena work;
    std::vector<TokenId> cached_prefix;
    std::optional<KVAddressSpaceHandle> cached_address;

    BranchStorage(DeviceSpan backing, const BranchLayout& layout, qwen3_5::PagedKVCache& kv,
                  LogicalKVPageStore& pages);
    ~BranchStorage();
    void release_cache();
};

} // namespace ninfer::models::qwen3_5::detail
