#include "models/qwen4_exp/parallel/plan.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
RankPlan rank_plan(const ModelDefinition& definition, std::uint32_t size, std::uint32_t rank) {
    const auto& c = definition.config().text;
    const auto& w = definition.weights();
    RankPlan p;
    p.rank                = rank;
    p.size                = size;
    p.ple_placement       = size == 1 ? ple::Placement::Disk : ple::Placement::Device;
    p.gdn_keys            = partition_range(c.gdn.key_heads, size, rank);
    p.gdn_values          = partition_range(c.gdn.value_heads, size, rank);
    p.qsa_queries         = partition_range(c.qsa.query_heads, size, rank);
    p.qsa_kv              = partition_range(c.qsa.kv_heads, size, rank);
    p.vocabulary          = partition_range(c.vocabulary, size, rank);
    p.ple_shards          = partition_range(c.ple.shards, size, rank);
    p.shared_intermediate = partition_range(c.moe.shared_intermediate, size, rank).size;
    (void)partition_range(c.moe.experts, size, rank);
    p.experts.resize(definition.expert_banks().size());
    for (std::size_t layer = 0; layer < p.experts.size(); ++layer) {
        const auto priority = layer < w.text.layers.size() ? default_expert_priority(c, layer)
                                                           : std::span<const std::uint32_t>{};
        for (std::uint32_t index = rank; index < c.moe.experts; index += size)
            p.experts[layer].push_back(priority.empty() ? index : priority[index]);
        std::sort(p.experts[layer].begin(), p.experts[layer].end());
    }
    if (size == 1) return p;

    const auto rows = [&](std::initializer_list<std::pair<WeightId, bool>> parameters) {
        ParameterAssembly a;
        std::uint64_t n = 0, k = 0;
        for (const auto& [id, split] : parameters) {
            const auto& shape = definition.parameter(id).shape;
            k                 = shape.at(1);
            const auto r = partition_range(dimension(shape[0]), split ? size : 1, split ? rank : 0);
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
    const auto block = [&](const BlockWeights& b, bool mtp) {
        if (const auto* g = std::get_if<GdnWeights>(&b.mixer)) {
            rows({{g->query, true}, {g->key, true}, {g->value, true}, {g->z, true}});
            rows({{g->a, true}, {g->b, true}});
            columns(g->a_log);
            columns(g->dt_bias);
            columns(g->output);
            ParameterAssembly conv;
            const auto k         = c.gdn.conv_channels() / size;
            conv.shape           = {c.gdn.conv_width, k};
            std::uint64_t source = 0, destination = 0;
            for (const auto width : {c.gdn.key_width(), c.gdn.key_width(), c.gdn.value_width()}) {
                conv.slices.push_back({g->convolution, 0, source + rank * (width / size),
                                       c.gdn.conv_width, width / size, 0, destination});
                source += width;
                destination += width / size;
            }
            conv.bindings.push_back({g->convolution, conv.shape, 0, c.gdn.conv_width * k});
            p.assemblies.push_back(std::move(conv));
        } else {
            const auto& q = std::get<QsaWeights>(b.mixer);
            if (mtp)
                rows({{q.key, true},
                      {q.value, true},
                      {q.index_key, false},
                      {q.query, true},
                      {q.gate, true},
                      {q.index_query, false}});
            else {
                rows({{q.query, true}, {q.gate, true}});
                rows(
                    {{q.key, true}, {q.value, true}, {q.index_query, false}, {q.index_key, false}});
            }
            columns(q.output);
        }
        const auto& m = b.moe;
        rows({{m.router, false}, {m.shared_score, false}});
        rows({{m.shared_gate, true}, {m.shared_up, true}});
        columns(m.shared_down);
    };
    for (const auto& b : w.text.layers) block(b, false);
    rows({{w.text.head, true}});
    if (w.mtp) {
        block(w.mtp->layer, true);
        rows({{w.mtp->head, true}});
    }
    return p;
}
} // namespace

RankPlan plan_single_rank(const ModelDefinition& definition) { return rank_plan(definition, 1, 0); }

ParallelPlan plan_parallel(const ModelDefinition& definition, const DeploymentOptions& deployment) {
    const auto& o   = deployment.parallel;
    const auto size = static_cast<std::uint32_t>(deployment.ranks.size());
    if ((size != 1 && size != 2) || o.tp != size || o.ep != size || o.dp != 1 || o.pp != 1 ||
        o.adp != 1)
        throw std::invalid_argument("Qwen4Exp supports one rank or TP2/EP2 with DP1/ADP1/PP1");
    ParallelPlan out;
    for (const auto& r : deployment.ranks) {
        out.ranks.push_back(rank_plan(definition, size, r.rank));
        std::uint32_t local = 0;
        for (const auto& other : deployment.ranks)
            if (other.node == r.node && other.rank < r.rank) ++local;
        out.topology.ranks.push_back(
            {r.rank, r.node, local, r.device, {r.rank, size}, {r.rank, size}, {0, 1}, {0, 1}});
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
    mix(definition.options().vision);
    mix(static_cast<std::uint64_t>(definition.options().purpose));
    mix(static_cast<std::uint64_t>(definition.options().speculative));
    mix(static_cast<std::uint64_t>(definition.options().proposal_head));
    for (const auto& rank : out.ranks) {
        mix(rank.rank);
        mix(static_cast<std::uint64_t>(rank.ple_placement));
        for (const auto& bank : rank.experts)
            for (auto expert : bank) mix(expert);
        for (const auto& a : rank.assemblies) {
            for (auto dimension : a.shape) mix(dimension);
            for (const auto& s : a.slices) {
                mix(s.parameter.index);
                mix(s.row);
                mix(s.column);
                mix(s.rows);
                mix(s.columns);
                mix(s.destination_row);
                mix(s.destination_column);
            }
        }
    }
    out.topology.plan = identity ? identity : 1;
    if (size > 1) {
        out.topology.groups.push_back({0, {0, 1}});
        out.topology.roles = {{"tensor", 0}, {"expert", 0}, {"ple", 0}, {"vocabulary", 0}};
    }
    out.topology.validate(deployment);
    return out;
}
} // namespace ninfer::models::qwen4_exp
