#pragma once

#include "core/buffer.h"
#include "models/common/moe/schedule.h"
#include "core/expert_cache.h"
#include "models/common/moe/offload/execution.h"

#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::moe::offload {

struct Plan;

// A layer snapshot shared by CPU and GPU. Updates and consumers use one compute stream.
class Partition {
public:
    Partition(ExpertBankId bank, std::int32_t experts, std::span<const std::int32_t> gpu_experts,
              const DeviceContext& device);
    Partition(const Partition&)            = delete;
    Partition& operator=(const Partition&) = delete;

    [[nodiscard]] bool try_update(std::span<const ExpertKey> ready, CudaExecutionView execution);
    [[nodiscard]] PartitionView view() const;

    [[nodiscard]] std::span<const ExpertKey> gpu_keys() const noexcept { return keys_; }

    [[nodiscard]] ops::ExpertSubsetKind gpu_kind() const noexcept { return kind_; }

private:
    ExpertBankId bank_;
    std::int32_t experts_;
    ops::ExpertSubsetKind kind_;
    PinnedHostBuffer mask_;
    DeviceBuffer device_mask_;
    CudaCompletionEvent mask_uploaded_;
    bool upload_pending_ = false;
    std::vector<ExpertKey> keys_;
};

// Program-owned CPU team and one reusable bridge frame.
// Only explicit synchronize/teardown waits for final GPU output; ordinary calls remain ordered
// by the borrowed compute stream. All external calls end before destruction.
class Bridge {
public:
    Bridge(const ExecutionRequirements& requirements, const CpuTopology& topology, CpuTeamPlan team,
           const DeviceContext& device);
    ~Bridge();
    Bridge(const Bridge&)            = delete;
    Bridge& operator=(const Bridge&) = delete;

    [[nodiscard]] ExecutionView execution_view() noexcept;
    void synchronize() const;

private:
    struct Host;
    CudaExecutionView cuda_;
    std::unique_ptr<Host> host_;
};

// Program-owned fixed experts, reusable streaming windows and CPU/GPU execution resources.
// Layer parameters and immutable sources outlive this owner; traversals have one Host caller.
class Resources {
public:
    // Startup warmup exercises the real schedule without binding the serving caller.
    class Preparation {
    public:
        explicit Preparation(Resources&);
        ~Preparation();
        Preparation(const Preparation&)            = delete;
        Preparation& operator=(const Preparation&) = delete;
    private:
        Resources& owner_;
    };

    Resources(std::shared_ptr<const Plan>, std::span<const Parameters* const> layers,
              std::span<const ExpertBankSource> sources, const DeviceContext&,
              ResourcePurpose = ResourcePurpose::Execution);
    ~Resources();
    Resources(const Resources&)            = delete;
    Resources& operator=(const Resources&) = delete;

    // Retain the activity through the traversal; GPU-only streaming does not wake the team.
    [[nodiscard]] std::optional<CpuExecutionView::ActiveScope> begin(std::int32_t tokens);
    [[nodiscard]] bool try_before_layer(ExpertBankId bank);
    [[nodiscard]] bool settled() const noexcept;
    // Immutable bindings for capture. No cache acquisition or submission is performed.
    [[nodiscard]] PreparedExecution describe_layer(ExpertBankId bank, std::int32_t tokens,
                                                   ops::ExpertSubsetKind kind);
    [[nodiscard]] std::optional<PreparedExecution> try_prepare_layer(ExpertBankId bank);
    [[nodiscard]] bool calibrating() const noexcept;
    // Calibration retains the candidate's execution variants but streams one full bank.
    [[nodiscard]] PreparedExecution prepare_calibration_layer(ExpertBankId, ops::ExpertSubsetKind);
    void gpu_submitted(ExpertBankId bank);
    void progress();
    void finish_layer(bool cpu_work) noexcept;
    void cancel();
    void restore_cache_after_preparation();
    // The plan fixes each layer's Write/AddResidual semantics.
    [[nodiscard]] bool start(ExpertBankId bank, const Tensor& hidden, Tensor& output,
                             MemoryArena& workspace, const ops::MoeExecutionHints& hints);
    [[nodiscard]] bool poll();
    void synchronize() const;
    [[nodiscard]] std::uint64_t uploaded_bytes() const noexcept;
    [[nodiscard]] ExpertCacheUsage cache_usage() const;
    [[nodiscard]] const Plan& plan() const;

private:
    friend class LayerGraphExecutable;
    void begin_call();
    void end_call() noexcept;
    void fail_call() noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::moe::offload
