#pragma once

#include "models/common/moe/weights.h"
#include "models/qwen4_exp/model.h"

namespace ninfer::models::qwen4_exp {
class ResolvedLoadPlan;

// Reads metadata and small resources. Main expert payload is materialized only after placement.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const LoadOptions& options() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const qwen::FrontendResources& resources() const;
    [[nodiscard]] const ple::disk::Table& ple_table() const;
    [[nodiscard]] ops::WeightInput input(WeightId) const;
    [[nodiscard]] std::span<const moe::BankDescription> expert_banks() const;
    [[nodiscard]] artifact::MaterializationPlan placement(Device, Device main_experts) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl>);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend ResolvedLoadPlan resolve_load(LoadPlan&&, Device, Device main_experts);
    friend std::unique_ptr<Model> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
};

class ResolvedLoadPlan {
public:
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const {
        return placement_;
    }
private:
    friend ResolvedLoadPlan resolve_load(LoadPlan&&, Device, Device main_experts);
    friend std::unique_ptr<Model> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    ResolvedLoadPlan(LoadPlan&&, artifact::MaterializationPlan, std::vector<Device>,
                     std::vector<Device>);
    LoadPlan logical_;
    artifact::MaterializationPlan placement_;
    std::vector<Device> weights_, banks_;
};

[[nodiscard]] LoadPlan plan_load(const artifact::Reader&, LoadOptions = {});
[[nodiscard]] ResolvedLoadPlan resolve_load(LoadPlan&&, Device, Device main_experts);
[[nodiscard]] std::unique_ptr<Model> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                       const StartupObserver* = nullptr);
} // namespace ninfer::models::qwen4_exp
