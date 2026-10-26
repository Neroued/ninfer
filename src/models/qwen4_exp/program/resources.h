#pragma once

#include "models/qwen4_exp/program/planning.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/execution/batch.h"
#include "models/common/context/storage.h"
#include "models/common/context/transfer.h"
#include "models/common/moe/schedule.h"
#include "ninfer/ops/gdn_replay.h"

namespace ninfer::models::qwen4_exp {
// Fixed rank-local backing. Directory identities and allocation belong to the Session.
// Parameters and DeviceContext outlive this owner and all submitted consumers retire first.
class Resources {
public:
    Resources(const execution::Parameters&, ExecutionPlan, DeviceContext&,
              const StartupObserver& observer = {});
    ~Resources();
    Resources(const Resources&)            = delete;
    Resources& operator=(const Resources&) = delete;

    [[nodiscard]] MemorySpan context_backing() { return context_.span(); }

    [[nodiscard]] MemorySpan persistent() { return persistent_.span(); }

    [[nodiscard]] MemorySpan ingress() { return ingress_.span(); }

    [[nodiscard]] MemorySpan egress() { return egress_.span(); }

    [[nodiscard]] MemorySpan sampling_ingress() { return sampling_.span(); }

    [[nodiscard]] MemorySpan ple_handoff() { return ple_handoff_.span(); }

    [[nodiscard]] MemorySpan compute_aux() { return compute_aux_.span(); }

    [[nodiscard]] MemorySpan context_counts() { return context_counts_.span(); }

    [[nodiscard]] MemorySpan workspace_storage() { return workspace_.span(); }

    [[nodiscard]] CudaExecutionView ple_transfer() const { return ple_transfer_; }

    void synchronize() const;

    [[nodiscard]] std::uint32_t pool_count() const { return mtp ? 2 : 1; }

    [[nodiscard]] QsaCache& pool(std::uint32_t index);
    [[nodiscard]] const QsaCache& pool(std::uint32_t index) const;

    [[nodiscard]] Tensor tensor(const TensorRegion&);
    [[nodiscard]] Tensor packed(const TensorRegion&, std::initializer_list<std::int32_t>);
    [[nodiscard]] execution::BatchView main_batch(std::int32_t tokens, std::int32_t batch,
                                                  std::uint32_t visible, bool sequence = false);
    [[nodiscard]] execution::BatchView mtp_batch(const MtpBatchLayout&, std::int32_t tokens,
                                                 std::int32_t batch, std::uint32_t visible);
    [[nodiscard]] ops::QsaTailStateView mtp_working_tail(std::uint32_t slot, std::int32_t batch);

    const ExecutionPlan plan;
private:
    DeviceContext& device_;
    DeviceBuffer context_, persistent_, workspace_;
    PinnedHostBuffer ingress_, egress_, sampling_, ple_handoff_, compute_aux_, context_counts_;
    PinnedHostBuffer host_context_;
    CudaStream ple_stream_;
    CudaExecutionView ple_transfer_;
public:
    std::optional<vision::Storage> vision;
    MemoryArena workspace;
    std::unique_ptr<StateViews> states;
    std::unique_ptr<QsaCache> main, mtp;
    std::unique_ptr<context::ContextStorage> context;
    std::unique_ptr<context::ContextMigration> compute_actions, context_actions;
    std::optional<GdnReplayRecords> gdn_records;
    std::optional<ops::GdnReplayFoldPlan> gdn_fold;
    std::unique_ptr<moe::Schedule> experts;
    std::unique_ptr<ple::UploadSlots> ple_upload;
    std::unique_ptr<ple::disk::PrepareService> ple_prepare;
};
} // namespace ninfer::models::qwen4_exp
