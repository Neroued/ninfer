#pragma once

#include "models/qwen4_exp/execution/workspace.h"
#include "models/qwen4_exp/state/state_image.h"
#include "models/qwen4_exp/state/kv_cache.h"
#include "models/qwen4_exp/program/buffers.h"
#include "models/qwen4_exp/ple/disk/prepare.h"
#include "models/qwen4_exp/ple/upload.h"
#include "runtime/engine/kv_capacity.h"
#include "models/common/moe/schedule.h"

namespace ninfer::models::qwen4_exp {
struct PersistentLayout {
    StateLayout state;
    QsaCacheLayout main;
    std::optional<QsaCacheLayout> mtp;
    BufferLayout buffers;
    std::size_t bytes;
};

struct ExecutionShape {
    std::uint32_t max_context, max_concurrency, prefill_chunk, draft_tokens;
    KvCacheStorage kv_storage;
    EnginePurpose purpose;
    bool context_cache = false;
    bool cuda_graph    = false;
};

struct ExecutionPlan {
    std::shared_ptr<const execution::ModelProfile> profile;
    ExecutionShape shape;
    PersistentLayout persistent;
    execution::WorkspacePlan workspace;
    std::shared_ptr<const moe::Plan> moe;
    ple::disk::PrepareOptions ple;
    ple::UploadMemory ple_upload{};
    std::size_t pinned_output_bytes   = 0;
    std::size_t ple_prepare_bytes     = 0;
    std::size_t host_context_bytes    = 0;
    std::size_t pinned_io_bytes       = 0;
    std::size_t pinned_sampling_bytes = 0;
    std::size_t pinned_mtp_bytes      = 0;
    runtime::KvCapacityResolution capacity;
    bool graph_probe = false;

    [[nodiscard]] std::size_t device_bytes() const;

    [[nodiscard]] std::size_t device_reservation_bytes() const;

    [[nodiscard]] std::size_t host_capacity_bytes() const { return host_context_bytes; }

    [[nodiscard]] std::uint32_t kv_capacity() const { return capacity.resolved_tokens; }
};

[[nodiscard]] std::shared_ptr<const moe::Plan> describe_moe_plan(const execution::ModelProfile&,
                                                                 const MoeOptions&);

[[nodiscard]] std::span<const std::uint32_t> default_expert_priority(const TextConfig&,
                                                                     std::size_t layer);

// Available bytes exclude immutable GPU weights and the caller's startup headroom.
// Shared-memory targets include planned Host preparation storage in the physical reservation.
[[nodiscard]] std::optional<ExecutionPlan>
plan_execution(std::shared_ptr<const execution::ModelProfile>, ple::disk::TableGeometry,
               const EngineOptions&, DeviceContext&, std::size_t available_bytes,
               std::shared_ptr<const moe::Plan> prototype = {});

[[nodiscard]] ExecutionPlan plan_graph_probe(std::shared_ptr<const execution::ModelProfile>,
                                             ple::disk::TableGeometry, const EngineOptions&,
                                             DeviceContext&, std::shared_ptr<const moe::Plan>);
} // namespace ninfer::models::qwen4_exp
