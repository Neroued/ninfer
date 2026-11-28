#include "models/qwen4_exp/parallel/plan.h"

#include "models/common/parallel_weights.h"
#include "models/qwen4_exp/ple/placement.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>

namespace ninfer::models::qwen4_exp {
RankPlan plan_rank_layers(const ModelDefinition& definition, const distributed::RankIdentity& id,
                          ple::Placement table, IndexRange layers) {
    const auto& c = definition.config().text;
    const auto& w = definition.weights();
    RankPlan p;
    p.placement      = RankPlacement(id);
    p.layers         = layers;
    p.owns_ple       = c.ple.layer >= layers.begin && c.ple.layer - layers.begin < layers.size;
    const auto local = std::span(w.text.layers).subspan(layers.begin, layers.size);
    const bool first = id.pp.rank == 0, last = id.pp.rank + 1 == id.pp.size;
    const auto [rank, size] = id.tp;
    p.ple_candidates        = {table};
    p.ple_placement         = table;
    p.gdn_keys              = partition_range(c.gdn.key_heads, size, rank);
    p.gdn_values            = partition_range(c.gdn.value_heads, size, rank);
    p.qsa_queries           = partition_range(c.qsa.query_heads, size, rank);
    p.qsa_kv                = partition_range(c.qsa.kv_heads, size, rank);
    p.vocabulary            = partition_vocabulary_range(c.vocabulary, size, rank);
    if (p.owns_ple) p.ple_shards = ple::shard_range(table, c.ple.shards, id.tp);
    p.shared_intermediate = partition_range(c.moe.shared_intermediate, size, rank).size;
    const auto per_owner  = partition_range(c.moe.experts, id.ep.size, id.ep.rank);
    const auto slice      = p.placement.expert_slice();
    p.experts.resize(layers.size + bool(w.mtp));
    for (std::size_t layer = 0; layer < p.experts.size(); ++layer) {
        auto& bank        = p.experts[layer];
        bank.intermediate = partition_range(c.moe.intermediate, slice.size, slice.rank);
        if (id.ep.size == 1) {
            bank.ids.resize(c.moe.experts);
            std::iota(bank.ids.begin(), bank.ids.end(), 0);
        } else {
            const auto priority = layer < layers.size
                                      ? default_expert_priority(c, layers.begin + layer)
                                      : std::span<const std::uint32_t>{};
            bank.ids.reserve(per_owner.size);
            for (std::uint32_t index = id.ep.rank; index < c.moe.experts; index += id.ep.size)
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

    if (first) whole({w.text.embedding});
    if (p.owns_ple)
        whole({w.text.ple.key, w.text.ple.value, w.text.ple.norm_key, w.text.ple.norm_query,
               w.text.ple.norm_conv, w.text.ple.convolution});
    if (last) whole_hc(w.text.final_hc);
    for (const auto& b : local) whole_block_constants(b);
    if (w.mtp) {
        whole({w.mtp->embedding, w.mtp->embedding_norm, w.mtp->hidden_norm, w.mtp->fc_embedding,
               w.mtp->fc_hidden});
        whole_block_constants(w.mtp->layer);
        whole_hc(w.mtp->final_hc);
    }
    if (w.vision && first) whole_vision(*w.vision);

    const auto omit_foreign = [&] {
        if (id.pp.size == 1) return;
        std::vector<bool> bound(definition.parameters().size());
        for (auto weight : p.whole) bound.at(weight.index) = true;
        for (const auto& a : p.assemblies)
            for (const auto& binding : a.bindings) bound.at(binding.parameter.index) = true;
        for (const auto& layer : local)
            for (auto weight : {layer.moe.gate, layer.moe.up, layer.moe.down})
                bound.at(weight.index) = true;
        for (std::size_t i = 0; i < bound.size(); ++i)
            if (!bound[i]) p.omitted.push_back({i});
    };
    if (size == 1) {
        if (last) whole({w.text.head});
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
        for (const auto& b : local) whole_block_dynamic(b);
        if (w.mtp) {
            whole({w.mtp->head});
            whole_block_dynamic(w.mtp->layer);
        }
        omit_foreign();
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
    for (const auto& b : local) block(b, false);
    if (last) vocabulary_head(w.text.head);
    if (w.mtp) {
        block(w.mtp->layer, true);
        vocabulary_head(w.mtp->head);
    }
    omit_foreign();
    return p;
}

std::uint64_t compute_rank_shard_digest(const artifact::ArtifactId& artifact_id,
                                        LoadOptions options, const RankPlan& r) {
    std::vector<std::uint64_t> geometry{
        r.placement.pp.rank, r.placement.pp.size,
        r.layers.begin,      r.layers.size,
        r.owns_ple,          r.placement.tp.rank,
        r.placement.tp.size, r.placement.ep.rank,
        r.placement.ep.size, static_cast<std::uint64_t>(r.ple_placement)};
    for (const auto range :
         {r.gdn_keys, r.gdn_values, r.qsa_queries, r.qsa_kv, r.vocabulary, r.ple_shards}) {
        geometry.push_back(range.begin);
        geometry.push_back(range.size);
    }
    geometry.push_back(r.ple_candidates.size());
    for (const auto candidate : r.ple_candidates)
        geometry.push_back(static_cast<std::uint64_t>(candidate));
    geometry.push_back(r.shared_intermediate);
    return rank_weight_digest(artifact_id, options, geometry, r.whole, r.omitted, r.assemblies,
                              r.experts, r.expert_domain.get());
}

std::vector<PipelineLayerDemand> pipeline_layer_demands(const ModelDefinition& definition) {
    const auto& config = definition.config().text;
    std::vector<PipelineLayerDemand> result;
    for (const auto& layer : definition.weights().text.layers) {
        PipelineLayerDemand demand{};
        const auto add = [&](std::initializer_list<WeightId> ids, bool experts = false) {
            for (const auto id : ids) {
                const auto& view = definition.weight_data()[id.index].view;
                const auto count = weight_element_count(definition.parameter(id).shape);
                demand.work += experts ? count / config.moe.experts * config.moe.top_k : count;
                for (const auto& part : view.parts) {
                    const auto& geometry = part.parent->geometry;
                    demand.resident_bytes += static_cast<std::uint64_t>(
                        std::ceil(static_cast<long double>(geometry.bytes) *
                                  (part.end - part.begin) / geometry.elements));
                }
            }
        };
        for (const auto* hc : {&layer.attention_hc, &layer.moe_hc}) {
            add({hc->norm, hc->down, hc->up});
            if (hc->injection) add({*hc->injection});
        }
        if (const auto* g = std::get_if<GdnWeights>(&layer.mixer))
            add({g->query, g->key, g->value, g->z, g->a, g->b, g->a_log, g->dt_bias, g->convolution,
                 g->norm, g->output});
        else {
            const auto& q = std::get<QsaWeights>(layer.mixer);
            add({q.query, q.key, q.gate, q.value, q.index_query, q.index_key, q.query_norm,
                 q.key_norm, q.index_query_norm, q.index_key_norm, q.output});
        }
        const auto& m = layer.moe;
        add({m.router, m.shared_score, m.shared_gate, m.shared_up, m.shared_down});
        add({m.gate, m.up, m.down}, true);
        result.push_back(demand);
    }
    return result;
}

RankPlan plan_single_rank(const ModelDefinition& definition, ple::Placement table) {
    return plan_rank_layers(
        definition, {}, table,
        {0, static_cast<std::uint32_t>(definition.config().text.layers.size())});
}

ParallelPlan plan_parallel(const ModelDefinition& definition, const DeploymentOptions& deployment,
                           ple::Placement table) {
    return plan_parallel(definition, deployment, std::span(&table, 1));
}

ParallelPlan plan_parallel(const ModelDefinition& definition, const DeploymentOptions& deployment,
                           std::span<const ple::Placement> candidates) {
    if (candidates.empty()) throw std::invalid_argument("Qwen4Exp requires a PLE placement");
    const auto normalized = normalize_deployment(deployment);
    const auto& o         = normalized.parallel;
    if (o.adp != 1) throw std::invalid_argument("Qwen4Exp requires a single QSA owner per stage");
    if (o.pp > 1 && definition.options().mtp())
        throw std::invalid_argument("Qwen4Exp pipeline stages support target width one");
    const auto ranges =
        o.pp == 1 && o.stage_layers.empty()
            ? std::vector<IndexRange>{{0, static_cast<std::uint32_t>(
                                              definition.config().text.layers.size())}}
            : partition_pipeline_layers(pipeline_layer_demands(definition), o);
    ParallelPlan out;
    out.topology = distributed::build_cluster_topology(normalized);
    out.ranks.reserve(out.topology.ranks.size());
    for (const auto& id : out.topology.ranks) {
        auto& rank = out.ranks.emplace_back(
            plan_rank_layers(definition, id, candidates.front(), ranges[id.pp.rank]));
        rank.ple_candidates.assign(candidates.begin(), candidates.end());
    }
    for (const auto& replica : out.topology.replicas) {
        for (const auto& stage : replica.stage_ranks) {
            out.topology.add_role("tensor", stage);
            if (stage == replica.stage_ranks.back()) out.topology.add_role("vocabulary", stage);
            if (std::ranges::find(candidates, ple::Placement::Sharded) != candidates.end())
                out.topology.add_role("ple", stage);
        }
    }
    add_expert_roles(out.topology);
    const auto& moe = definition.config().text.moe;
    moe::bind_expert_domains(out.topology, out.ranks, moe.experts, moe.intermediate);
    for (auto& id : out.topology.ranks)
        id.shard_digest = compute_rank_shard_digest(definition.artifact_id(), definition.options(),
                                                    out.ranks.at(id.rank));
    out.topology.plan =
        distributed::compute_topology_plan_digest(definition.artifact_id(), out.topology);
    out.topology.validate(normalized);
    return out;
}
} // namespace ninfer::models::qwen4_exp
