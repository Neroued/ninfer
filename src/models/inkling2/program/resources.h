#pragma once

#include "models/common/context/transfer.h"
#include "models/inkling2/program/planning.h"

#include <initializer_list>
#include <memory>

namespace ninfer::models::inkling2 {

class Resources {
public:
    Resources(ExecutionPlan, DeviceContext&);
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
    [[nodiscard]] RankMemoryUsage observe() const;

    const ExecutionPlan plan;

private:
    DeviceContext& device_;
    DeviceBuffer context_, fixed_, workspace_;
    PinnedHostBuffer ingress_, sampling_, egress_, context_counts_;

public:
    MemoryArena workspace;
    StateViews states;
    CacheViews main;
    std::optional<CacheViews> dspark;
    std::optional<models::vision::Storage> vision;
    std::unique_ptr<context::ContextStorage> context;
    std::unique_ptr<context::ContextMigration> compute_actions, context_actions;
};

} // namespace ninfer::models::inkling2
