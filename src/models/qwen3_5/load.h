#pragma once

#include "models/qwen3_5/model.h"

#include <filesystem>
#include <memory>
#include <span>

namespace ninfer::artifact {
class Reader;
struct ParameterReference;
} // namespace ninfer::artifact

namespace ninfer::models::qwen3_5 {

// Cold load plan borrows its Reader until materialization. Its selected Host bytes and parsed
// resources already have owners; no Program or device allocation is needed for plan_load.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    // Page-locked bytes the Engine allocates after the weights within the same page-locked limit
    // (artifact::MaterializationPlan::host_pinned_reserved_bytes).
    void reserve_host_pinned_bytes(std::uint64_t bytes);
    [[nodiscard]] std::size_t parameter_count() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, ExecutionContext&,
                                                    const StartupObserver*);
};

// options.tp == 2 plans one backing per device from the logical tensor-parallel placement in
// load/sharding.h; it rejects a masked draft that reads the full output head.
[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
// Requires a plan for one device.
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
// Requires execution.tp == options.tp. Device i of the Model is execution.dev[i]; at tp 1 this is
// the DeviceContext overload on execution.primary().
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, ExecutionContext& execution,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::qwen3_5
