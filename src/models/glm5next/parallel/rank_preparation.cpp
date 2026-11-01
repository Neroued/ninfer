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

void RankPreparation::prepare_communication() {
    if (!binding_.group) return;
    using distributed::Collective;
    using distributed::Reduction;
    const auto& shape = plan_.shape;
    const auto hidden = std::size_t(plan_.profile->config.text.hidden);
    const auto batch  = std::size_t(shape.max_concurrency);
    const auto decode = batch * (shape.draft_tokens + 1);
    std::vector<distributed::CollectivePreparation> work;
    for (const auto columns : {std::size_t(1), batch, decode, std::size_t(shape.prefill_chunk)})
        work.push_back({Collective::AllReduce, DType::BF16, hidden * columns});
    for (std::uint32_t root = 0; root < binding_.group->size(); ++root) {
        for (const auto columns : {std::size_t(1), std::size_t(shape.prefill_chunk)})
            work.push_back(
                {Collective::Broadcast, DType::BF16, hidden * columns, Reduction::Sum, root});
        if (plan_.workspace.vision)
            work.push_back({Collective::Broadcast, DType::BF16,
                            plan_.workspace.vision->output_bytes / dtype_size(DType::BF16),
                            Reduction::Sum, root});
    }
    if (shape.purpose == EnginePurpose::Generation) {
        for (const auto columns : {std::size_t(1), batch, decode}) {
            work.push_back({Collective::AllGather, DType::BF16, hidden * columns});
            work.push_back(
                {Collective::AllGather, DType::I64, DistributedVocabulary::candidates * columns});
        }
        if (plan_.profile->mtp)
            work.push_back({Collective::AllReduce, DType::U64, batch, Reduction::Max});
    } else {
        work.push_back({Collective::AllGather, DType::FP32,
                        3 * std::size_t(std::min(shape.prefill_chunk, 1024U))});
    }
    binding_.group->prepare(work, {}, device_.execution_view());
    if (peer_.group)
        context::prepare_peer_communication(*peer_.group, plan_.peer_staging.front().bytes,
                                            device_.transfer_view());
}

} // namespace ninfer::models::glm5next
