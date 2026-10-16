#pragma once

#include "ninfer/branch_score.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/common/context/kv_history.h"
#include "models/qwen3_5/state/state_image.h"
#include "ninfer/ops/gdn_replay.h"

namespace ninfer::models::qwen3_5::detail {

inline constexpr std::int32_t kBranchBatch = 8;
inline constexpr std::int32_t kBranchWidth = 16;

struct BranchLayout {
    std::int32_t batch_capacity = 0;
    std::uint32_t prefill_chunk = 0;
    KVExecutionTableLayout tables;
    GdnReplayRecordLayout records;
    TensorRegion ids, positions, counts, rows, slots, hidden, tails, logits;
    LayoutRegion scratch;
    std::size_t bytes = 0;
};

// Temporary workspace overlay plus leases from the Program's existing StateImage store.
// Released before generation resumes; no weights or device allocation of its own.
// Local slots 0, 1..B and B+1..2B map to reserved physical root/trunk/branch slots.
struct BranchStorage {
    qwen3_5::PagedKVCache kv;
    const std::int32_t batch_capacity;
    LinearAttentionStateView& states;
    GdnReplayRecords records;
    ops::GdnReplayFoldPlan fold;
    context::KVHistoryStore addresses;
    Tensor ids, positions, counts, rows, slots, hidden, tails, logits;
    MemoryArena work;
    std::vector<TokenId> cached_prefix;
    std::optional<context::KVHistory> cached_address;
    StateImageStore& state_store;
    std::vector<StateImageHandle> state_handles;

    BranchStorage(MemorySpan backing, const BranchLayout& layout, qwen3_5::PagedKVCache& kv,
                  context::LogicalKVPageStore& pages, qwen3_5::StateImageViews& images,
                  StateImageStore& store, std::int32_t batch, CudaExecutionView execution);
    ~BranchStorage();
    void release_cache();

    [[nodiscard]] std::int32_t state_slot(std::int32_t local) const {
        return state_store.physical_slot(state_handles.at(local));
    }
};

} // namespace ninfer::models::qwen3_5::detail
