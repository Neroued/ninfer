#pragma once

#include "models/common/context/transfer.h"
#include "models/glm5next/program/planning.h"
#include "ninfer/ops/kda_replay.h"

namespace ninfer::models::glm5next {

struct ContextCommunication {
    const distributed::Group* group = nullptr;
    std::span<const std::uint32_t> members;
};

// Fixed rank-owned storage; the coordinating ContextDirectory owns record identities,
// offsets, row leases and publication. DeviceContext outlives this owner.
class Resources {
public:
    Resources(ExecutionPlan, DeviceContext&, ContextCommunication = {});
    ~Resources();
    Resources(const Resources&)            = delete;
    Resources& operator=(const Resources&) = delete;

    void synchronize() const;

    [[nodiscard]] MemorySpan context_backing() { return context_.span(); }

    [[nodiscard]] MemorySpan fixed() { return fixed_.span(); }

    [[nodiscard]] MemorySpan workspace_storage() { return workspace_.span(); }

    [[nodiscard]] MemorySpan ingress() { return ingress_.span(); }

    [[nodiscard]] MemorySpan sampling_ingress() { return sampling_.span(); }

    [[nodiscard]] MemorySpan egress() { return egress_.span(); }

    [[nodiscard]] MemorySpan context_counts() { return context_counts_.span(); }

    [[nodiscard]] Tensor tensor(const TensorRegion&);
    [[nodiscard]] Tensor packed(const TensorRegion&, std::initializer_list<std::int32_t>);
    [[nodiscard]] ops::KpoolTailStateView mtp_tail(std::uint32_t slot, std::int32_t rows);
    [[nodiscard]] RankMemoryUsage observe() const;

    const ExecutionPlan plan;

private:
    DeviceContext& device_;
    DeviceBuffer context_, fixed_, workspace_;
    PinnedHostBuffer ingress_, sampling_, egress_, context_counts_;

public:
    std::optional<models::vision::Storage> vision;
    MemoryArena workspace;
    StateViews states;
    CacheViews main;
    std::unique_ptr<CacheViews> mtp;
    std::unique_ptr<context::ContextStorage> context;
    std::unique_ptr<context::ContextMigration> compute_actions, context_actions;
    std::optional<KdaReplayRecords> kda_records;
    std::optional<ops::KdaReplayFoldPlan> kda_fold;
};

} // namespace ninfer::models::glm5next
