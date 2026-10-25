#include "models/glm5next/startup.h"
#include "models/glm5next/parallel/wire.h"

#include <algorithm>

namespace ninfer::models::glm5next {
namespace {

RankLoadRequest request(std::span<const std::byte> payload,
                        const distributed::RankIdentity& identity) {
    distributed::wire::Decoder in(payload);
    auto r = decode_rank_load(in);
    in.finish();
    if (r.rank.rank != identity.rank || r.rank.size != identity.tp.size ||
        identity.tp.rank != identity.rank || identity.ep.size != r.rank.size ||
        identity.ep.rank != identity.rank || identity.adp.size != r.rank.size ||
        identity.adp.rank != identity.rank || identity.dp.size != 1 || identity.pp.size != 1)
        throw std::invalid_argument("GLM5Next rank plan differs from worker identity");
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
                                 distributed::CommunicationContext& communication)
    : RankPreparation(reader, request(payload, identity), device,
                      context_group(topology, communication)) {}

} // namespace ninfer::models::glm5next
