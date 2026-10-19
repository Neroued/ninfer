#include "models/qwen4_exp/parallel/preparation.h"

namespace ninfer::models::qwen4_exp {
RankPreparation::RankPreparation(ResolvedLoadPlan load, EngineOptions options,
                                 DeviceContext& device)
    : options_(std::move(options)), device_(device),
      profile_(execution::describe_model(load.definition(), load.weight_data(), load.rank_plan())),
      curve_(describe_execution_capacity(profile_, options_, device)) {
    if (options_.use_cuda_graph)
        throw std::invalid_argument(
            "Qwen4Exp distributed preparation currently uses eager execution");
    model_      = materialize_model(std::move(load), device_, &options_.startup_observer);
    parameters_ = std::make_unique<execution::Parameters>(*model_, profile_);
}

void RankPreparation::prepare(std::uint32_t pages, std::size_t available) {
    if (resources_) throw std::logic_error("rank resources are already prepared");
    auto options        = options_;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(curve_.resolved_tokens(pages));
    auto plan           = plan_execution(profile_, {}, options, device_, available);
    if (!plan) throw DeviceAllocationError("rank runtime capacity exceeds available memory");
    resources_ = std::make_unique<Resources>(*parameters_, std::move(*plan), device_,
                                             options_.startup_observer);
}

void RankPreparation::release_resources() {
    resources_.reset();
    device_.synchronize();
}

RankResourceLayout RankPreparation::layout() const {
    const auto& p = resources().plan;
    const auto& g = p.profile->local;
    RankResourceLayout r{
        p.persistent.state.images.image_stride,
        p.workspace.bytes,
        p.device_bytes(),
        p.device_reservation_bytes(),
        static_cast<std::uint32_t>(p.persistent.state.images.slot_count),
        p.persistent.main.pages.spec.page_group_count,
        p.persistent.mtp ? p.persistent.mtp->pages.spec.page_group_count : 0,
        g.gdn.key_heads,
        g.gdn.value_heads,
        g.qsa.query_heads,
        g.qsa.kv_heads,
        g.vocabulary_begin,
        g.vocabulary,
        static_cast<std::uint32_t>(std::get<DevicePleTable>(model().ple_table()).shards.size())};
    return r;
}

} // namespace ninfer::models::qwen4_exp
