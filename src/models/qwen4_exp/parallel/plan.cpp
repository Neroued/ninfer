#include "models/qwen4_exp/parallel/plan.h"

#include "models/common/parallel_weights.h"

#include <algorithm>
#include <bit>
#include <numeric>

namespace ninfer::models::qwen4_exp {
namespace {
RankPlan rank_plan(const ModelDefinition& definition, std::uint32_t size, std::uint32_t rank,
                   std::uint32_t ep_size) {
    const auto& c = definition.config().text;
    const auto& w = definition.weights();
    RankPlan p;
    p.rank                = rank;
    p.size                = size;
    p.ep_size             = ep_size;
    p.ple_placement       = size == 1 ? ple::Placement::Disk : ple::Placement::Device;
    p.gdn_keys            = partition_range(c.gdn.key_heads, size, rank);
    p.gdn_values          = partition_range(c.gdn.value_heads, size, rank);
    p.qsa_queries         = partition_range(c.qsa.query_heads, size, rank);
    p.qsa_kv              = partition_range(c.qsa.kv_heads, size, rank);
    p.vocabulary          = partition_vocabulary_range(c.vocabulary, size, rank);
    p.ple_shards          = partition_range(c.ple.shards, size, rank);
    p.shared_intermediate = partition_range(c.moe.shared_intermediate, size, rank).size;
    (void)partition_range(c.moe.experts, ep_size, ep_size == 1 ? 0 : rank);
    p.experts.resize(definition.expert_banks().size());
    for (std::size_t layer = 0; layer < p.experts.size(); ++layer) {
        auto& bank = p.experts[layer];
        bank.intermediate =
            partition_range(c.moe.intermediate, size / ep_size, ep_size == 1 ? rank : 0);
        if (ep_size == 1) {
            bank.ids.resize(c.moe.experts);
            std::iota(bank.ids.begin(), bank.ids.end(), 0);
        } else {
            const auto priority = layer < w.text.layers.size() ? default_expert_priority(c, layer)
                                                               : std::span<const std::uint32_t>{};
            for (std::uint32_t index = rank; index < c.moe.experts; index += ep_size)
                bank.ids.push_back(priority.empty() ? index : priority[index]);
            std::sort(bank.ids.begin(), bank.ids.end());
        }
    }

    const auto whole = [&](std::initializer_list<WeightId> ids) {
        p.whole.insert(p.whole.end(), ids.begin(), ids.end());
    };
    const auto whole_hc = [&](const HcWeights& h) {
        whole({h.norm, h.down, h.up});
        if (h.injection) whole({*h.injection});
    };
    const auto whole_vision = [&](const qwen::VisionWeights& v) {
        whole({v.patch_embedding, v.patch_embedding_bias, v.position_embedding});
        for (const auto& layer : v.layers) {
            whole({layer.norm1.weight, layer.norm1.bias, layer.norm2.weight, layer.norm2.bias,
                   layer.query, layer.key, layer.value, layer.query_bias, layer.key_bias,
                   layer.value_bias, layer.output, layer.output_bias, layer.fc1, layer.fc1_bias,
                   layer.fc2, layer.fc2_bias});
        }
        whole({v.merger_norm.weight, v.merger_norm.bias, v.merger_fc1, v.merger_fc1_bias,
               v.merger_fc2, v.merger_fc2_bias});
    };
    const auto whole_block_constants = [&](const BlockWeights& b) {
        whole_hc(b.attention_hc);
        whole_hc(b.moe_hc);
        if (const auto* g = std::get_if<GdnWeights>(&b.mixer))
            whole({g->norm});
        else {
            const auto& q = std::get<QsaWeights>(b.mixer);
            whole({q.query_norm, q.key_norm, q.index_query_norm, q.index_key_norm});
        }
    };

    whole({w.text.embedding, w.text.ple.key, w.text.ple.value, w.text.ple.norm_key,
           w.text.ple.norm_query, w.text.ple.norm_conv, w.text.ple.convolution});
    whole_hc(w.text.final_hc);
    for (const auto& b : w.text.layers) whole_block_constants(b);
    if (w.mtp) {
        whole({w.mtp->embedding, w.mtp->embedding_norm, w.mtp->hidden_norm, w.mtp->fc_embedding,
               w.mtp->fc_hidden});
        whole_block_constants(w.mtp->layer);
        whole_hc(w.mtp->final_hc);
    }
    if (w.vision) whole_vision(*w.vision);

    if (size == 1) {
        whole({w.text.head});
        const auto whole_block_dynamic = [&](const BlockWeights& b) {
            if (const auto* g = std::get_if<GdnWeights>(&b.mixer)) {
                whole({g->query, g->key, g->value, g->z, g->a, g->b, g->a_log, g->dt_bias,
                       g->convolution, g->output});
            } else {
                const auto& q = std::get<QsaWeights>(b.mixer);
                whole({q.query, q.key, q.gate, q.value, q.index_query, q.index_key, q.output});
            }
            const auto& m = b.moe;
            whole({m.router, m.shared_score, m.shared_gate, m.shared_up, m.shared_down});
        };
        for (const auto& b : w.text.layers) whole_block_dynamic(b);
        if (w.mtp) {
            whole({w.mtp->head});
            whole_block_dynamic(w.mtp->layer);
        }
        return p;
    }

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
    const auto vocabulary_head = [&](WeightId id) {
        validate_vocabulary_partition(definition.weight_data()[id.index], p.vocabulary);
        const auto& shape = definition.parameter(id).shape;
        const auto k      = shape.at(1);
        const auto r      = p.vocabulary;
        ParameterAssembly a;
        a.shape = {r.size, k};
        a.slices.push_back({id, r.begin, 0, r.size, k, 0, 0});
        a.bindings.push_back({id, {r.size, k}, 0, std::uint64_t(r.size) * k});
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
    vocabulary_head(w.text.head);
    if (w.mtp) {
        block(w.mtp->layer, true);
        vocabulary_head(w.mtp->head);
    }
    return p;
}
} // namespace

std::uint64_t compute_rank_shard_digest(const artifact::ArtifactId& artifact_id,
                                        LoadOptions options, const RankPlan& r) {
    std::vector<std::uint64_t> geometry{r.rank, r.size, r.ep_size,
                                        static_cast<std::uint64_t>(r.ple_placement)};
    for (const auto range :
         {r.gdn_keys, r.gdn_values, r.qsa_queries, r.qsa_kv, r.vocabulary, r.ple_shards}) {
        geometry.push_back(range.begin);
        geometry.push_back(range.size);
    }
    geometry.push_back(r.shared_intermediate);
    return rank_weight_digest(artifact_id, options, geometry, r.whole, r.omitted, r.assemblies,
                              r.experts);
}

RankPlan plan_single_rank(const ModelDefinition& definition) {
    return rank_plan(definition, 1, 0, 1);
}

ParallelPlan plan_parallel(const ModelDefinition& definition, const DeploymentOptions& deployment) {
    const auto normalized = normalize_deployment(deployment);
    const auto& o         = normalized.parallel;
    const auto size       = static_cast<std::uint32_t>(normalized.ranks.size());
    if ((size != 1 && size != 2) || o.tp != size || (o.ep != 1 && o.ep != size) || o.dp != 1 ||
        o.pp != 1 || o.adp != 1)
        throw std::invalid_argument(
            "Qwen4Exp supports one rank or TP2 with EP1 or EP2 and DP1/ADP1/PP1");
    ParallelPlan out;
    out.topology = distributed::build_cluster_topology(normalized);
    out.ranks.reserve(size);
    for (const auto& r : normalized.ranks) {
        auto shard = rank_plan(definition, size, r.rank, o.ep);
        out.topology.ranks[r.rank].shard_digest =
            compute_rank_shard_digest(definition.artifact_id(), definition.options(), shard);
        out.ranks.push_back(std::move(shard));
    }
    if (size > 1) {
        out.topology.groups.push_back({0, {0, 1}});
        out.topology.roles = {{"tensor", 0}, {"expert", 0}, {"ple", 0}, {"vocabulary", 0}};
    }
    out.topology.plan =
        distributed::compute_topology_plan_digest(definition.artifact_id(), out.topology);
    out.topology.validate(normalized);
    return out;
}
} // namespace ninfer::models::qwen4_exp
