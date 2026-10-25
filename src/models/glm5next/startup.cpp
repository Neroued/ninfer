#include "models/glm5next/startup.h"
#include "core/platform.h"

namespace ninfer::models::glm5next {

RankPreparation::RankPreparation(const artifact::Reader& reader,
                                 std::shared_ptr<const ModelDefinition> definition, RankPlan rank,
                                 const EngineOptions& options, DeviceContext& device)
    : device_(device) {
    load_.emplace(
        resolve_rank_load(reader, std::move(definition), std::move(rank), device.device()));
    weights_ = {WeightPlacement::Resident, load_->materialization().device_capacity_bytes,
                load_->materialization().host_capacity_bytes};
    auto profile =
        execution::describe_model(load_->definition(), load_->weight_data(), load_->rank_plan());
    plan_     = plan_execution(std::move(profile), options, device.multiprocessor_count());
    capacity_ = describe_execution_capacity(plan_);
}

RankPreparation::~RankPreparation() = default;

runtime::RankCapacityOffer RankPreparation::offer(std::size_t available,
                                                  std::size_t available_host) const {
    return {capacity_, available, available_host, NINFER_SHARED_PHYSICAL_MEMORY != 0};
}

void RankPreparation::materialize(const StartupObserver& observer) {
    if (!load_) throw std::logic_error("GLM5Next rank weights are already materialized");
    model_ = materialize_model(std::move(*load_), device_, &observer);
    load_.reset();
    parameters_ = std::make_unique<const execution::Parameters>(*model_, plan_.profile);
}

void RankPreparation::allocate(runtime::ContextCapacityResolution capacity,
                               ContextCommunication peer) {
    if (!parameters_ || resources_)
        throw std::logic_error(
            "GLM5Next resource allocation requires weights and an empty resource slot");
    if (capacity.device_bytes < capacity_.minimum_capacity().device_bytes ||
        capacity.runtime_reservation_bytes != capacity_.reservation_bytes(capacity.capacity()))
        throw std::invalid_argument(
            "GLM5Next resolved capacity differs from its preparation offer");
    auto plan     = plan_;
    plan.capacity = capacity;
    resources_    = std::make_unique<Resources>(std::move(plan), device_, peer);
}

void RankPreparation::release_resources() {
    if (resources_) resources_->synchronize();
    resources_.reset();
}

const execution::Parameters& RankPreparation::parameters() const {
    if (!parameters_) throw std::logic_error("GLM5Next parameters have not been materialized");
    return *parameters_;
}

Resources& RankPreparation::resources() {
    if (!resources_) throw std::logic_error("GLM5Next execution resources have not been allocated");
    return *resources_;
}

RankResourceDescription RankPreparation::describe() const {
    if (!resources_) throw std::logic_error("GLM5Next resources have not been prepared");
    const auto& p = resources_->plan;
    const auto& c = p.capacity;
    RankResourceDescription out{p.profile->local.rank, p.shape, p.state, describe_context(p),
                                resources_->observe(), {}};
    const auto& weights = model_->storage_stats();
    out.memory.allocations.device_weights_bytes =
        weights.device_capacity_bytes + model_->metadata_bytes();
    out.memory.allocations.host_weights_bytes = weights.host_capacity_bytes;
    out.budget = {.runtime_reservation_bytes         = c.runtime_reservation_bytes,
                  .minimum_runtime_reservation_bytes = c.minimum_runtime_reservation_bytes,
                  .context_capacity_bytes            = c.device_bytes,
                  .minimum_context_capacity_bytes    = c.minimum_device_bytes,
                  .context_control_budget_bytes      = c.control_reservation_bytes,
                  .available_after_weights_bytes     = c.available_after_weights_bytes,
                  .headroom_bytes                    = c.automatic_headroom_bytes,
                  .planned_slack_bytes               = c.planned_slack_bytes};
    return out;
}

Frontend create_frontend(const ModelDefinition& definition, const EngineOptions& o) {
    return make_frontend(definition.resources(),
                         {o.chat_template_path, o.grammar_cache_bytes,
                          definition.config().vision.has_value(), o.max_context,
                          o.media_cache_bytes, o.media_live_bytes, o.media_preprocess_threads});
}

} // namespace ninfer::models::glm5next
