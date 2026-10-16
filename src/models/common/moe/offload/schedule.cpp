#include "models/common/moe/offload/plan.h"
#include "models/common/moe/offload/graph.h"

#include <algorithm>

namespace ninfer::models::moe::offload {
namespace {

ScheduleDescription describe(const Plan& plan) {
    ScheduleDescription out{.device_bytes        = plan.device_bytes,
                            .cache_bytes         = plan.cache.config.capacity_bytes,
                            .fixed_bytes         = plan.cache.config.fixed_bytes,
                            .target_bytes        = plan.target_bytes,
                            .bridge_device_bytes = plan.bridge.device_frame_bytes,
                            .bridge_pinned_bytes = plan.bridge.pinned_frame_bytes,
                            .cpu_workspace_bytes = plan.bridge.cpu_workspace_bytes,
                            .window_count        = plan.window_count,
                            .cpu_workers =
                                static_cast<std::uint32_t>(plan.team.worker_cpus.size())};
    for (const auto& layer : plan.layers)
        out.layers.push_back({layer.spec.bank, layer.hybrid_max_tokens, layer.fixed_experts,
                              layer.cache_experts, layer.spec.priority});
    return out;
}

class Definition;

class Executable final : public moe::LayerGraphExecutable {
public:
    explicit Executable(const Definition&);
    void update(const moe::LayerGraphDefinition&) override;

    void upload(cudaStream_t stream) override { executable_.upload(stream); }

    void replay(Tensor& output, CudaExecutionView cuda,
                const std::function<void()>& after_gpu_submit) override {
        executable_.replay(resources_, bank_, output, cuda, after_gpu_submit);
    }
private:
    Resources& resources_;
    ExpertBankId bank_;
    offload::LayerGraphExecutable executable_;
};

class Definition final : public moe::LayerGraphDefinition {
public:
    Resources& resources;
    ExpertBankId bank;
    offload::LayerGraphDefinition definition;

    Definition(Resources& owner, ExpertBankId id) : resources(owner), bank(id) {}

    std::unique_ptr<moe::LayerGraphExecutable> instantiate() const override {
        return std::make_unique<Executable>(*this);
    }
};

Executable::Executable(const Definition& source)
    : resources_(source.resources), bank_(source.bank), executable_(source.definition) {}

void Executable::update(const moe::LayerGraphDefinition& source) {
    const auto& definition = dynamic_cast<const Definition&>(source);
    if (&definition.resources != &resources_ || definition.bank != bank_)
        throw std::logic_error("MoE graph update changed its execution owner");
    executable_.update(definition.definition);
}

class Execution final : public moe::Schedule {
public:
    Execution(std::shared_ptr<const Plan> plan, std::span<const Parameters* const> layers,
              std::span<const ExpertBankSource> sources, const DeviceContext& device,
              ResourcePurpose purpose)
        : description_(describe(*plan)),
          resources_(std::move(plan), layers, sources, device, purpose) {}

    void before_layer(ExpertBankId bank) override { resources_.before_layer(bank); }

    void execute(ExpertBankId bank, const Tensor& hidden, Tensor& output, MemoryArena& workspace,
                 const ops::MoeExecutionHints& hints) override {
        resources_.execute(bank, hidden, output, workspace, hints);
    }

    std::shared_ptr<const moe::LayerGraphDefinition>
    capture_layer(ExpertBankId bank, CudaExecutionView cuda, std::int32_t tokens,
                  const std::function<void()>& mixer, const GpuBody& body,
                  const std::function<void()>& feature,
                  const std::shared_ptr<const moe::LayerGraphDefinition>& shared) override {
        auto out = std::make_shared<Definition>(resources_, bank);
        if (shared) {
            const auto& existing = dynamic_cast<const Definition&>(*shared);
            if (&existing.resources != &resources_ || existing.bank != bank)
                throw std::logic_error("MoE graph sharing changed its execution owner");
            out->definition.moe = existing.definition.moe;
        } else {
            out->definition.moe = std::make_shared<SubsetGraphDefinitions>();
        }
        offload::capture_layer(
            out->definition, resources_, bank, cuda, tokens, mixer,
            [&](const PreparedExecution& call) {
                body([&](const Tensor& hidden, Tensor& output, MemoryArena& workspace,
                         const ops::MoeExecutionHints& hints) {
                    enqueue_gpu(hidden, call, output, workspace, GpuSubmission::Capture, hints);
                });
            },
            feature);
        return out;
    }

    void restore_cache_after_preparation() override {
        resources_.restore_cache_after_preparation();
    }

    void synchronize() const override { resources_.synchronize(); }

    std::uint64_t uploaded_bytes() const noexcept override { return resources_.uploaded_bytes(); }

    ExpertCacheUsage cache_usage() const override { return resources_.cache_usage(); }

    const ScheduleDescription& description() const noexcept override { return description_; }

private:
    void begin_activity(std::int32_t tokens) override {
        if (active_) throw std::logic_error("MoE traversal is already active");
        if (auto activity = resources_.begin(tokens)) activity_.emplace(std::move(*activity));
        active_ = true;
    }

    void end_activity() noexcept override {
        activity_.reset();
        active_ = false;
    }

    void begin_preparation() override { preparation_.emplace(resources_); }

    void end_preparation() noexcept override { preparation_.reset(); }

    ScheduleDescription description_;
    Resources resources_;
    std::optional<Resources::Preparation> preparation_;
    std::optional<CpuExecutionView::ActiveScope> activity_;
    bool active_ = false;
};

class Planned final : public moe::Plan {
public:
    Planned(offload::Plan plan, std::vector<ExpertBankCacheSpec> banks)
        : plan_(std::make_shared<const offload::Plan>(std::move(plan))), banks_(std::move(banks)),
          description_(describe(*plan_)) {}

    const ScheduleDescription& description() const noexcept override { return description_; }

    std::shared_ptr<const moe::Plan> resolve(std::size_t available, std::int32_t max_tokens,
                                             std::uint32_t workers) const override {
        CpuTopology topology;
        auto resolved = resolve_plan(*plan_, banks_, available, max_tokens, topology, workers);
        return resolved ? std::make_shared<Planned>(std::move(*resolved), banks_) : nullptr;
    }

    std::size_t workspace_bytes(ExpertBankId bank, std::int32_t first,
                                std::int32_t last) const override {
        const auto& layer = plan_->layer(bank);
        const auto end    = std::min(last, layer.hybrid_max_tokens);
        if (first > end) return 0;
        return describe_execution(layer.spec.profile, ops::ExpertSubsetKind::MaskOnes,
                                  layer.spec.epilogue, first, end)
            .gpu_workspace_bytes;
    }

    GraphPlan graph_plan(ExpertBankId bank, std::int32_t tokens) const override {
        return offload::graph_plan(*plan_, bank, tokens);
    }

    std::unique_ptr<moe::Schedule> create(std::span<const Parameters* const> layers,
                                          std::span<const ExpertBankSource> sources,
                                          const DeviceContext& device,
                                          ResourcePurpose purpose) const override {
        return std::make_unique<Execution>(plan_, layers, sources, device, purpose);
    }
private:
    std::shared_ptr<const offload::Plan> plan_;
    std::vector<ExpertBankCacheSpec> banks_;
    ScheduleDescription description_;
};

} // namespace

std::shared_ptr<const moe::Plan> describe_schedule(std::span<const ExpertBankCacheSpec> banks,
                                                   std::span<const LayerSpec> layers) {
    return std::make_shared<Planned>(offload::describe_plan(banks, layers),
                                     std::vector<ExpertBankCacheSpec>(banks.begin(), banks.end()));
}

} // namespace ninfer::models::moe::offload
