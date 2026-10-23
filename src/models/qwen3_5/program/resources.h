#pragma once

#include "core/arena.h"
#include "core/buffer.h"
#include "core/context_pool.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "models/common/moe/schedule.h"
#include "models/common/qwen/vision/vision.h"
#include "models/qwen3_5/program/frames.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/state/state_image.h"
#include "ninfer/ops/gdn_replay.h"

#include <cstddef>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5 {

struct DFlashPersistentSpec {
    std::uint32_t full_layers = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    std::int32_t feature_rows = 0;
};

struct ResourceSpec {
    DecoderStateSpec decoder;
    StateImageSpec state_images;
    std::optional<GdnReplayRecordSpec> replay_records;
    std::optional<DFlashPersistentSpec> dflash;
    FrameSpec frame;
    std::size_t context_bytes      = 0;
    std::size_t host_context_bytes = 0;
};

struct DFlashPersistentLayout {
    std::optional<PagedKVCacheLayout> full;
    TensorRegion prefill_features;
    TensorRegion prefill_positions;
    TensorRegion pending_features;
};

struct ResourceLayout {
    DecoderStateLayout decoder;
    StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    FrameLayout frame;
    std::optional<TensorRegion> token_counts;
    std::size_t context_bytes      = 0;
    std::size_t host_context_bytes = 0;
    std::size_t bytes              = 0;

    [[nodiscard]] std::size_t pinned_bytes() const;
    [[nodiscard]] std::vector<RecordSpec> context_records() const;
};

struct WorkspaceSpec {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<qwen::VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

[[nodiscard]] ResourceLayout plan_resources(const ResourceSpec& spec);

struct DFlashPersistentState {
    CyclicKVCacheView& local;
    std::optional<PagedKVCache> full;
    Tensor prefill_features, prefill_positions, pending_features;

    DFlashPersistentState(MemorySpan arena, MemorySpan backing,
                          const DFlashPersistentLayout& layout, CyclicKVCacheView& local_state);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer,
                                                     const Tensor& offsets) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
};

// Rank-owned arenas and execution storage; the Directory assigns each Context record.
class Resources {
public:
    Resources(DeviceContext& device, const ResourceLayout& layout, const WorkspaceSpec& workspace,
              const StartupObserver& observer = {});
    Resources(const Resources&)            = delete;
    Resources& operator=(const Resources&) = delete;

    [[nodiscard]] PagedKVCache* backend_kv() noexcept;
    [[nodiscard]] const PagedKVCache* backend_kv() const noexcept;

    const std::vector<RecordSpec> context_records;
    DeviceBuffer context;
    DeviceBuffer persistent;
    DeviceBuffer workspace_storage;
    PinnedHostBuffer host_context;
    MemoryArena work;
    DecoderState decoder;
    StateImageViews state_images;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    std::optional<DFlashPersistentState> dflash;
    Tensor token_counts;
    ExecutionFrame frame;
    std::unique_ptr<moe::Schedule> moe;
};

} // namespace ninfer::models::qwen3_5
