#include "models/glm5next/startup.h"

#include "models/glm5next/parallel/codec.h"
#include "models/glm5next/parallel/wire.h"
#include "models/common/vocabulary.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace {

RankLoadRequest request(std::span<const std::byte> payload,
                        const distributed::RankIdentity& identity,
                        const StartupObserver& observer) {
    distributed::wire::Decoder in(payload);
    auto r = decode_rank_load(in);
    in.finish();
    if (r.rank.rank != identity.rank || r.rank.size != identity.tp.size ||
        identity.tp.rank != identity.rank || identity.ep.size != r.rank.ep_size ||
        identity.ep.rank != (r.rank.ep_size == 1 ? 0U : r.rank.rank) ||
        identity.adp.size != r.rank.size || identity.adp.rank != identity.rank ||
        identity.dp.size != 1 || identity.pp.size != 1)
        throw std::invalid_argument("GLM5Next rank plan differs from worker identity");
    r.options.startup_observer = observer;
    return r;
}

ContextCommunication context_group(const distributed::ExecutionTopology& topology,
                                   distributed::CommunicationContext& communication) {
    if (topology.ranks.size() == 1) return {};
    const auto role = std::find_if(topology.roles.begin(), topology.roles.end(),
                                   [](const auto& r) { return r.role == "context"; });
    if (role == topology.roles.end())
        throw std::invalid_argument("GLM5Next context group is absent");
    const auto& group = topology.group(role->group);
    return {&communication.group(group.id), group.members};
}

} // namespace

RankPreparation::RankPreparation(const artifact::Reader& reader, std::span<const std::byte> payload,
                                 const distributed::RankIdentity& identity, DeviceContext& device,
                                 const distributed::ExecutionTopology& topology,
                                 distributed::CommunicationContext& communication,
                                 const StartupObserver& observer)
    : RankPreparation(reader, request(payload, identity, observer), device,
                      context_group(topology, communication)) {
    const WorkContract contract{plan_.shape,
                                plan_.profile->config.text.vocabulary,
                                plan_.profile->public_tokens,
                                plan_.profile->local.size,
                                0,
                                plan_.profile->control_token_limit};
    memory_.host_reservation_bytes +=
        runtime::protocol::work_payload_memory(contract.payload_capacity(topology.ranks.size()),
                                               {1, 1}, topology.ranks.size(), control_owner_);
    const auto role = std::find_if(topology.roles.begin(), topology.roles.end(),
                                   [](const auto& r) { return r.role == "tensor"; });
    if (role == topology.roles.end())
        throw std::invalid_argument("GLM5Next tensor group is absent");
    const auto& group = topology.group(role->group);
    binding_          = {&communication.group(group.id), group.index(topology.output_rank)};
}

ExecutionCodec RankPreparation::codec(const distributed::ExecutionTopology& topology) const {
    return ExecutionCodec(instance_->model->definition(), describe(), topology);
}

std::vector<distributed::GroupRequirements> RankPreparation::communication_requirements() const {
    if (!binding_.group) return {};
    using distributed::Collective;
    using distributed::Reduction;
    const auto& shape = plan_.shape;
    const auto hidden = std::size_t(plan_.profile->config.text.hidden);
    const auto batch  = std::size_t(shape.max_concurrency);
    const auto decode = batch * (shape.draft_tokens + 1);
    distributed::GroupRequirements tensor{binding_.group->id(), {}, {}};
    auto& work         = tensor.collectives;
    const auto columns = std::max({batch, decode, std::size_t(shape.prefill_chunk)});
    work.push_back({Collective::AllReduce, DType::BF16, hidden, hidden * columns, hidden});
    for (std::uint32_t root = 0; root < binding_.group->size(); ++root) {
        work.push_back({Collective::Broadcast, DType::BF16, hidden,
                        hidden * std::size_t(shape.prefill_chunk), hidden, Reduction::Sum, root});
        if (plan_.workspace.vision)
            work.push_back({Collective::Broadcast, DType::BF16, hidden,
                            plan_.workspace.vision->output_bytes / dtype_size(DType::BF16), hidden,
                            Reduction::Sum, root});
    }
    if (shape.purpose == EnginePurpose::Generation) {
        work.push_back({Collective::AllGather, DType::BF16, hidden, hidden * decode, hidden});
        constexpr auto candidates = std::size_t(DistributedVocabulary::candidates);
        work.push_back(
            {Collective::AllGather, DType::I64, candidates, candidates * decode, candidates});
        if (plan_.profile->draft) {
            const auto& p              = *plan_.profile->draft;
            const auto context_columns = std::max(
                std::min(std::size_t(shape.prefill_chunk), std::size_t(p.context.capacity)),
                decode);
            work.push_back({Collective::AllGather, DType::BF16, std::size_t(p.feature_projection.n),
                            p.feature_projection.n * context_columns,
                            std::size_t(p.feature_projection.n)});
            const auto k = std::size_t(plan_.profile->config.draft->dflash2->selector_top_k);
            work.push_back(
                {Collective::AllGather, DType::I64, k, k * batch * shape.draft_tokens, k});
        }
        if (plan_.profile->mtp)
            work.push_back({Collective::AllReduce, DType::U64, 1, batch, 1, Reduction::Max});
    } else {
        work.push_back({Collective::AllGather, DType::FP32, 3,
                        3 * std::size_t(std::min(shape.prefill_chunk, 1024U)), 3});
    }
    std::vector<distributed::GroupRequirements> result;
    result.push_back(std::move(tensor));
    if (peer_.group)
        result.push_back(context::peer_communication_requirements(
            *peer_.group, plan_.peer_staging.front().bytes));
    return result;
}

} // namespace ninfer::models::glm5next
