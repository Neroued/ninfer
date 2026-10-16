#pragma once

#include "models/qwen4_exp/program/planning.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/execution/batch.h"
#include "models/common/context/migration.h"
#include "models/common/moe/schedule.h"
#include "ninfer/ops/gdn_replay.h"

namespace ninfer::models::qwen4_exp {
struct ContextPool {
    QsaCache cache;
    context::LogicalKVPageStore pages;
    context::KVHistoryStore histories;

    ContextPool(MemorySpan, const QsaCacheLayout&, std::size_t host_bytes, std::uint32_t histories,
                context::HostKVExtentStore*);
};

// One Program's physical resources. Model/Parameters and DeviceContext outlive this owner;
// the Program retires snapshots, histories and migration batches before destroying it.
class Resources {
public:
    Resources(const execution::Parameters&, ExecutionPlan, DeviceContext&);
    ~Resources();
    Resources(const Resources&)            = delete;
    Resources& operator=(const Resources&) = delete;

    [[nodiscard]] MemorySpan persistent() { return persistent_.span(); }

    [[nodiscard]] MemorySpan ingress() { return ingress_.span(); }

    [[nodiscard]] MemorySpan egress() { return egress_.span(); }

    [[nodiscard]] MemorySpan sampling_ingress() { return sampling_.span(); }

    [[nodiscard]] MemorySpan mtp_ingress() { return mtp_ingress_.span(); }

    [[nodiscard]] MemorySpan workspace_storage() { return workspace_.span(); }

    [[nodiscard]] CudaExecutionView ple_transfer() const { return ple_transfer_; }

    void synchronize() const;

    [[nodiscard]] std::uint32_t pool_count() const { return mtp ? 2 : 1; }

    [[nodiscard]] ContextPool& pool(std::uint32_t index) const { return index == 0 ? *main : *mtp; }

    [[nodiscard]] Tensor tensor(const TensorRegion&);
    [[nodiscard]] Tensor packed(const TensorRegion&, std::initializer_list<std::int32_t>);
    [[nodiscard]] execution::BatchView main_batch(std::int32_t tokens, std::int32_t batch,
                                                  std::uint32_t visible);
    [[nodiscard]] execution::BatchView mtp_batch(const MtpBatchLayout&, std::int32_t tokens,
                                                 std::int32_t batch, std::uint32_t visible);
    [[nodiscard]] ops::QsaTailStateView mtp_working_tail(std::uint32_t slot, std::int32_t batch);

    const ExecutionPlan plan;
private:
    DeviceContext& device_;
    DeviceBuffer persistent_, workspace_;
    PinnedHostBuffer ingress_, egress_, sampling_, mtp_ingress_;
    CudaStream ple_stream_;
    CudaExecutionView ple_transfer_;
public:
    MemoryArena workspace;
    std::unique_ptr<HostContextArena> host_context;
    std::unique_ptr<HostKVArena> host_kv;
    std::unique_ptr<context::HostKVExtentStore> host_extents;
    std::unique_ptr<StateViews> states;
    std::unique_ptr<context::StateImageStore> state_store;
    std::unique_ptr<ContextPool> main, mtp;
    std::unique_ptr<context::MigrationContext> migration;
    std::optional<GdnReplayRecords> gdn_records;
    std::optional<ops::GdnReplayFoldPlan> gdn_fold;
    std::unique_ptr<moe::Schedule> experts;
    std::unique_ptr<ple::UploadSlots> ple_upload;
    std::unique_ptr<ple::disk::PrepareService> ple_prepare;
};
} // namespace ninfer::models::qwen4_exp
