#pragma once

#include "models/common/moe/execution.h"
#include "core/expert_cache.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::moe {

struct HybridPolicy {
    std::int32_t max_tokens         = 0;
    std::uint32_t reference_workers = 0;
};

struct LayerSpec {
    ExpertBankId bank;
    ops::MoeProfile profile;
    ops::MoeEpilogue epilogue = ops::MoeEpilogue::Write;
    std::vector<std::uint32_t> priority;
    HybridPolicy hybrid;
};

struct LayerDescription {
    ExpertBankId bank;
    std::int32_t hybrid_max_tokens;
    std::uint32_t fixed_experts, cache_experts;
    std::span<const std::uint32_t> priority;
};

struct ScheduleDescription {
    std::vector<LayerDescription> layers;
    std::size_t device_bytes = 0, cache_bytes = 0, fixed_bytes = 0, target_bytes = 0;
    std::size_t bridge_device_bytes = 0, bridge_pinned_bytes = 0, cpu_workspace_bytes = 0;
    std::uint32_t window_count = 0, cpu_workers = 0;
};

struct GraphPlan {
    std::array<bool, 3> subsets{};
    std::optional<ops::ExpertSubsetKind> steady;
    friend bool operator==(const GraphPlan&, const GraphPlan&) = default;
};

// Borrowed synchronous capture call. The model supplies its normalization/input preparation;
// the schedule supplies the GPU body without exposing CPU or cache handoff resources.
using GpuCall =
    std::function<void(const Tensor&, Tensor&, MemoryArena&, const ops::MoeExecutionHints&)>;
using GpuBody = std::function<void(const GpuCall&)>;

class LayerGraphDefinition;

class LayerGraphExecutable {
public:
    virtual ~LayerGraphExecutable()                                         = default;
    virtual void update(const LayerGraphDefinition&)                        = 0;
    virtual void upload(cudaStream_t)                                       = 0;
    virtual void replay(Tensor&, CudaExecutionView,
                        const std::function<void()>& after_gpu_submit = {}) = 0;
};

class LayerGraphDefinition {
public:
    virtual ~LayerGraphDefinition()                                                 = default;
    [[nodiscard]] virtual std::unique_ptr<LayerGraphExecutable> instantiate() const = 0;
};

enum class ResourcePurpose { Execution, GraphCalibration };
class Schedule;

// Immutable offload planning. A null plan selects the complete resident GPU path.
// Concrete cache/team planning and resources are compiled only by targets providing offload.
class Plan {
public:
    virtual ~Plan()                                                               = default;
    [[nodiscard]] virtual const ScheduleDescription& description() const noexcept = 0;
    [[nodiscard]] virtual std::shared_ptr<const Plan>
    resolve(std::size_t available, std::int32_t max_tokens, std::uint32_t cpu_workers) const = 0;
    [[nodiscard]] virtual std::size_t workspace_bytes(ExpertBankId, std::int32_t first,
                                                      std::int32_t last) const               = 0;
    [[nodiscard]] virtual GraphPlan graph_plan(ExpertBankId, std::int32_t tokens) const      = 0;
    [[nodiscard]] virtual std::unique_ptr<Schedule> create(std::span<const Parameters* const>,
                                                           std::span<const ExpertBankSource>,
                                                           const DeviceContext&,
                                                           ResourcePurpose) const            = 0;
};

[[nodiscard]] std::shared_ptr<const Plan> describe_plan(std::span<const ExpertBankCacheSpec>,
                                                        std::span<const LayerSpec>);

// One Program owns each schedule. Models retain mathematical traversal order; the schedule
// owns expert availability, CPU activity, transfers and their completion.
class Schedule {
public:
    class Activity {
    public:
        ~Activity() {
            if (owner_) owner_->end_activity();
        }

        Activity(Activity&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)) {}

        Activity(const Activity&)            = delete;
        Activity& operator=(const Activity&) = delete;
    private:
        friend class Schedule;

        explicit Activity(Schedule& owner) : owner_(&owner) {}

        Schedule* owner_;
    };

    class Preparation {
    public:
        explicit Preparation(Schedule& owner) : owner_(owner) { owner_.begin_preparation(); }

        ~Preparation() { owner_.end_preparation(); }

        Preparation(const Preparation&)            = delete;
        Preparation& operator=(const Preparation&) = delete;
    private:
        Schedule& owner_;
    };

    virtual ~Schedule() = default;

    [[nodiscard]] std::optional<Activity> begin(std::int32_t tokens) {
        begin_activity(tokens);
        return Activity(*this);
    }

    virtual void before_layer(ExpertBankId)             = 0;
    virtual void execute(ExpertBankId, const Tensor&, Tensor&, MemoryArena&,
                         const ops::MoeExecutionHints&) = 0;
    [[nodiscard]] virtual std::shared_ptr<const LayerGraphDefinition>
    capture_layer(ExpertBankId, CudaExecutionView, std::int32_t tokens,
                  const std::function<void()>& mixer, const GpuBody&,
                  const std::function<void()>& feature,
                  const std::shared_ptr<const LayerGraphDefinition>& shared)      = 0;
    virtual void restore_cache_after_preparation()                                = 0;
    virtual void synchronize() const                                              = 0;
    [[nodiscard]] virtual std::uint64_t uploaded_bytes() const noexcept           = 0;
    [[nodiscard]] virtual ExpertCacheUsage cache_usage() const                    = 0;
    [[nodiscard]] virtual const ScheduleDescription& description() const noexcept = 0;

private:
    virtual void begin_activity(std::int32_t) = 0;
    virtual void end_activity() noexcept      = 0;
    virtual void begin_preparation()          = 0;
    virtual void end_preparation() noexcept   = 0;
};

} // namespace ninfer::models::moe
