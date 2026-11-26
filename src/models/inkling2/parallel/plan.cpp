#include "models/inkling2/parallel/plan.h"

#include <numeric>
#include <stdexcept>

namespace ninfer::models::inkling2 {

RankPlan plan_single_rank(const ModelDefinition& definition) {
    const auto& c = definition.config().text;
    RankPlan p;
    p.rank       = 0;
    p.size       = 1;
    p.ep_size    = 1;
    p.vocabulary = {0, c.vocabulary};
    p.experts.resize(definition.expert_banks().size());
    for (auto& bank : p.experts) {
        bank.ids.resize(c.moe.experts);
        std::iota(bank.ids.begin(), bank.ids.end(), 0U);
        bank.intermediate = {0, c.moe.intermediate};
    }
    return p;
}

ParallelPlan plan_parallel(const ModelDefinition& definition, const DeploymentOptions& deployment) {
    const auto& o   = deployment.parallel;
    const auto size = static_cast<std::uint32_t>(deployment.ranks.size());
    if (size != 1 || o.tp != 1 || o.ep != 1 || o.adp != 1 || o.dp != 1 || o.pp != 1)
        throw std::invalid_argument("Inkling2 currently supports single-rank execution only");

    ParallelPlan out;
    const auto& r = deployment.ranks.front();
    out.ranks.push_back(plan_single_rank(definition));
    out.topology.ranks.push_back(
        {r.rank, r.node, 0, r.device, {0, 1}, {0, 1}, {0, 1}, {0, 1}, {0, 1}});

    std::uint64_t identity = 14695981039346656037ULL;
    const auto mix         = [&](std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) {
            identity ^= (value >> (i * 8)) & 255;
            identity *= 1099511628211ULL;
        }
    };
    for (const auto byte : definition.artifact_id()) mix(std::to_integer<unsigned>(byte));
    mix(definition.options().vision);
    mix(static_cast<std::uint64_t>(definition.options().purpose));
    mix(static_cast<std::uint64_t>(definition.options().speculative));
    mix(static_cast<std::uint64_t>(definition.options().proposal_head));
    out.topology.plan = identity ? identity : 1;
    out.topology.validate(deployment);
    return out;
}

} // namespace ninfer::models::inkling2
