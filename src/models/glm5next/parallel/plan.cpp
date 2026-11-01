#include "models/glm5next/parallel/plan.h"

#include <algorithm>
#include <numeric>

namespace ninfer::models::glm5next {
namespace {

std::uint32_t dimension(std::uint64_t value) {
    if (value > UINT32_MAX) throw artifact::ArtifactError("GLM parallel dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

RankPlan rank_plan(const ModelDefinition& definition, std::uint32_t size, std::uint32_t rank,
                   std::uint32_t ep_size) {
    const auto& c = definition.config().text;
    const auto& w = definition.weights();
    RankPlan p;
    p.rank                = rank;
    p.size                = size;
    p.ep_size             = ep_size;
    p.kda_heads           = partition_range(c.kda.heads, size, rank);
    p.vocabulary          = partition_range(c.vocabulary, size, rank);
    p.dense_intermediate  = partition_range(c.dense_intermediate, size, rank);
    p.shared_intermediate = partition_range(dimension(c.moe.shared_intermediate()), size, rank);
    const auto experts    = partition_range(c.moe.experts, ep_size, ep_size == 1 ? 0 : rank);
    p.experts.resize(definition.expert_banks().size());
    for (auto& bank : p.experts) {
        bank.ids.resize(experts.size);
        std::iota(bank.ids.begin(), bank.ids.end(), experts.begin);
        bank.intermediate =
            partition_range(c.moe.intermediate, size / ep_size, ep_size == 1 ? rank : 0);
    }
    if (size == 1) return p;

    const auto rows = [&](std::initializer_list<std::pair<WeightId, bool>> weights) {
        ParameterAssembly a;
        std::uint64_t n = 0, k = 0;
        for (const auto& [id, split] : weights) {
            const auto& shape = definition.parameter(id).shape;
            k                 = shape.at(1);
            const auto r =
                partition_range(dimension(shape.at(0)), split ? size : 1, split ? rank : 0);
            a.slices.push_back({id, r.begin, 0, r.size, k, n, 0});
            a.bindings.push_back({id, {r.size, k}, n * k, (n + r.size) * k});
            n += r.size;
        }
        a.shape = {n, k};
        p.assemblies.push_back(std::move(a));
    };
    const auto columns = [&](WeightId id) {
        const auto& shape = definition.parameter(id).shape;
        const auto n      = shape.size() == 1 ? 1 : shape.at(0);
        const auto r      = partition_range(dimension(shape.back()), size, rank);
        ParameterAssembly a;
        a.shape = shape.size() == 1 ? artifact::Shape{r.size} : artifact::Shape{n, r.size};
        a.slices.push_back({id, 0, r.begin, n, r.size});
        a.bindings.push_back({id, a.shape, 0, n * r.size});
        p.assemblies.push_back(std::move(a));
    };
    const auto dense = [&](const DenseWeights& d) {
        rows({{d.gate, true}, {d.up, true}});
        columns(d.down);
    };
    for (const auto& layer : w.text.layers) {
        if (const auto* k = std::get_if<KdaWeights>(&layer.mixer)) {
            rows({{k->query, true}, {k->key, true}, {k->value, true}});
            rows({{k->beta, true}, {k->forget_a, false}, {k->gate_a, false}});
            rows({{k->forget_b, true}});
            rows({{k->gate_b, true}});
            columns(k->output);
            columns(k->a_log);
            columns(k->dt_bias);
            ParameterAssembly conv;
            const auto width = c.kda.width();
            const auto local = width / size;
            conv.shape       = {c.kda.conv_width, 3 * local};
            for (std::uint64_t field = 0; field < 3; ++field)
                conv.slices.push_back({k->convolution, 0, field * width + rank * local,
                                       c.kda.conv_width, local, 0, field * local});
            conv.bindings.push_back(
                {k->convolution, conv.shape, 0, std::uint64_t(c.kda.conv_width) * 3 * local});
            p.assemblies.push_back(std::move(conv));
        }
        if (const auto* d = std::get_if<DenseWeights>(&layer.ffn))
            dense(*d);
        else
            dense(std::get<MoeWeights>(layer.ffn).shared);
    }
    rows({{w.text.head, true}});
    if (w.mtp) {
        dense(w.mtp->moe.shared);
        rows({{w.mtp->head, true}});
    }
    return p;
}

} // namespace

RankPlan plan_single_rank(const ModelDefinition& definition) {
    return rank_plan(definition, 1, 0, 1);
}

ParallelPlan plan_parallel(const ModelDefinition& definition, const DeploymentOptions& deployment) {
    const auto& o   = deployment.parallel;
    const auto size = static_cast<std::uint32_t>(deployment.ranks.size());
    if ((size != 1 && size != 2) || o.tp != size || (o.ep != 1 && o.ep != size) || o.adp != size ||
        o.dp != 1 || o.pp != 1)
        throw std::invalid_argument(
            "GLM5Next weight placement supports one rank or TP2/ADP2 with EP1 or EP2 and DP1/PP1");
    ParallelPlan out;
    for (const auto& r : deployment.ranks) {
        out.ranks.push_back(rank_plan(definition, size, r.rank, o.ep));
        std::uint32_t local = 0;
        for (const auto& other : deployment.ranks)
            if (other.node == r.node && other.rank < r.rank) ++local;
        out.topology.ranks.push_back({r.rank,
                                      r.node,
                                      local,
                                      r.device,
                                      {r.rank, size},
                                      {o.ep == 1 ? 0U : r.rank, o.ep},
                                      {0, 1},
                                      {0, 1},
                                      {r.rank, size}});
    }
    std::sort(out.ranks.begin(), out.ranks.end(),
              [](const auto& a, const auto& b) { return a.rank < b.rank; });
    std::sort(out.topology.ranks.begin(), out.topology.ranks.end(),
              [](const auto& a, const auto& b) { return a.rank < b.rank; });
    std::uint64_t identity = 14695981039346656037ULL;
    const auto mix         = [&](std::uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) {
            identity ^= (value >> (i * 8)) & 255;
            identity *= 1099511628211ULL;
        }
    };
    for (const auto byte : definition.artifact_id()) mix(std::to_integer<unsigned>(byte));
    mix(size);
    mix(o.ep);
    mix(definition.options().vision);
    mix(static_cast<std::uint64_t>(definition.options().purpose));
    mix(static_cast<std::uint64_t>(definition.options().speculative));
    mix(static_cast<std::uint64_t>(definition.options().proposal_head));
    for (const auto& rank : out.ranks) {
        mix(rank.rank);
        for (const auto& bank : rank.experts) {
            mix(bank.intermediate.begin);
            mix(bank.intermediate.size);
            for (auto expert : bank.ids) mix(expert);
        }
    }
    out.topology.plan = identity ? identity : 1;
    if (size > 1) {
        out.topology.groups.push_back({0, {0, 1}});
        out.topology.groups.push_back({1, {0, 1}, distributed::GroupUsage::PeerExchange});
        out.topology.roles = {{"tensor", 0},     {"expert", 0}, {"attention", 0},
                              {"vocabulary", 0}, {"vision", 0}, {"context", 1}};
    }
    out.topology.validate(deployment);
    return out;
}

} // namespace ninfer::models::glm5next
