#include "models/glm5next/parallel/plan.h"

#include "models/common/parallel_weights.h"

#include <algorithm>
#include <bit>
#include <map>
#include <numeric>

namespace ninfer::models::glm5next {
namespace {

std::uint32_t dimension(std::uint64_t value) {
    if (value > UINT32_MAX) throw artifact::ArtifactError("GLM parallel dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}

RankPlan rank_plan(const ModelDefinition& definition, const distributed::RankIdentity& id) {
    const auto [rank, size] = id.tp;
    const auto& c = definition.config().text;
    const auto& w = definition.weights();
    RankPlan p;
    p.placement                = RankPlacement(id);
    const auto attention_slice = p.placement.attention_slice();
    p.mla_heads          = partition_range(c.mla.heads, attention_slice.size, attention_slice.rank);
    p.kda_heads          = partition_range(c.kda.heads, size, rank);
    p.vocabulary         = partition_vocabulary_range(c.vocabulary, size, rank);
    p.dense_intermediate = partition_range(c.dense_intermediate, size, rank);
    p.shared_intermediate = partition_range(dimension(c.moe.shared_intermediate()), size, rank);
    if (const auto& draft = definition.config().draft) {
        p.draft = {partition_range(c.hidden, size, rank),
                   partition_range(draft->attention.num_attention_heads, size, rank),
                   partition_range(draft->attention.num_key_value_heads, size, rank),
                   partition_range(draft->intermediate_size, size, rank)};
    }
    const auto experts = partition_range(c.moe.experts, id.ep.size, id.ep.rank);
    const auto slice   = p.placement.expert_slice();
    p.experts.resize(definition.expert_banks().size());
    for (auto& bank : p.experts) {
        bank.ids.resize(experts.size);
        std::iota(bank.ids.begin(), bank.ids.end(), experts.begin);
        bank.intermediate = partition_range(c.moe.intermediate, slice.size, slice.rank);
    }

    const auto whole = [&](std::initializer_list<WeightId> ids) {
        p.whole.insert(p.whole.end(), ids.begin(), ids.end());
    };
    const auto whole_mhc = [&](const MhcWeights& m) { whole({m.projection, m.base, m.scale}); };
    const auto mla       = [&](const MlaWeights& m) {
        whole({m.query_norm, m.latent_norm, m.index_norm, m.index_bias, m.pool_ape});
        if (attention_slice.size == 1) {
            whole({m.latent, m.index_key, m.pool_gate, m.query_a, m.index_weight, m.query_b,
                   m.index_query, m.key_up, m.value_up, m.output});
            return;
        }
        // Preserve the artifact's fused projection parents, but include only the rows read
        // by this member. In particular index_weight can share storage with index_key.
        std::map<std::pair<std::size_t, std::uint64_t>, std::size_t> projections;
        const auto projection = [&](WeightId weight, std::uint64_t first_row, std::uint64_t rows) {
            const auto& ref    = definition.parameter(weight);
            const auto columns = ref.shape.back();
            auto assembly      = p.assemblies.size();
            if (ref.binding.parts.size() == 1) {
                const auto key = std::pair{ref.binding.parts.front().object.index, columns};
                assembly       = projections.try_emplace(key, assembly).first->second;
            }
            if (assembly == p.assemblies.size()) p.assemblies.push_back({{0, columns}, {}, {}});
            auto& dest       = p.assemblies[assembly];
            const auto begin = dest.shape[0];
            dest.slices.push_back({weight, first_row, 0, rows, columns, begin, 0});
            dest.bindings.push_back(
                {weight, {rows, columns}, begin * columns, (begin + rows) * columns});
            dest.shape[0] += rows;
        };
        for (const auto weight : {m.latent, m.index_key, m.pool_gate, m.query_a})
            projection(weight, 0, definition.parameter(weight).shape.front());
        const auto& a             = c.mla;
        const std::uint64_t first = p.mla_heads.begin, heads = p.mla_heads.size;
        if (attention_slice.rank == 0)
            projection(m.index_weight, 0, definition.parameter(m.index_weight).shape.front());
        projection(m.query_b, first * a.query_dim, heads * a.query_dim);
        if (attention_slice.rank == 0)
            projection(m.index_query, 0, definition.parameter(m.index_query).shape.front());
        else
            p.omitted.insert(p.omitted.end(), {m.index_weight, m.index_query});
        const auto select = [&](WeightId weight, artifact::Shape shape, std::uint64_t row,
                                std::uint64_t column) {
            const auto elements = weight_element_count(shape);
            const auto columns  = shape.back();
            p.assemblies.push_back({shape,
                                    {{weight, row, column, elements / columns, columns}},
                                    {{weight, shape, 0, elements}}});
        };
        select(m.key_up, {heads, a.query_dim, a.latent_dim}, first * a.query_dim, 0);
        select(m.value_up, {heads, a.value_dim, a.latent_dim}, first * a.value_dim, 0);
        select(m.output, {c.hidden, heads * a.value_dim}, 0, first * a.value_dim);
    };
    const auto whole_vision = [&](const VisionWeights& v) {
        whole({v.patch_embedding, v.patch_embedding_bias, v.post_norm});
        for (const auto& layer : v.layers) {
            whole({layer.input_norm, layer.post_attention_norm, layer.query, layer.key, layer.value,
                   layer.query_bias, layer.key_bias, layer.value_bias, layer.query_norm,
                   layer.key_norm, layer.output, layer.output_bias, layer.mlp.gate, layer.mlp.up,
                   layer.mlp.down, layer.gate_bias, layer.up_bias, layer.down_bias});
        }
        whole({v.downsample, v.downsample_bias, v.merger_projection, v.merger_norm_weight,
               v.merger_norm_bias, v.merger.gate, v.merger.up, v.merger.down});
    };

    whole({w.text.embedding, w.text.final_norm});
    for (const auto& layer : w.text.layers) {
        whole({layer.input_norm, layer.post_attention_norm});
        whole_mhc(layer.attention_hc);
        whole_mhc(layer.mlp_hc);
        if (const auto* k = std::get_if<KdaWeights>(&layer.mixer))
            whole({k->norm});
        else
            mla(std::get<MlaWeights>(layer.mixer));
        if (const auto* m = std::get_if<MoeWeights>(&layer.ffn))
            whole({m->router, m->correction_bias});
    }
    if (w.mtp) {
        whole({w.mtp->embedding, w.mtp->embedding_norm, w.mtp->hidden_norm, w.mtp->fc,
               w.mtp->final_norm, w.mtp->input_norm, w.mtp->post_attention_norm});
        mla(w.mtp->attention);
        whole({w.mtp->moe.router, w.mtp->moe.correction_bias});
    }
    if (w.draft) {
        whole({w.draft->token_embedding, w.draft->context_norm, w.draft->final_norm});
        if (w.draft->selector)
            whole({w.draft->selector->hidden_projection, w.draft->selector->predecessor_codebook,
                   w.draft->selector->successor_codebook});
        for (const auto& layer : w.draft->layers) {
            whole({layer.input_norm, layer.post_attention_norm, layer.attention.query_norm,
                   layer.attention.key_norm});
            if (layer.attention_conv)
                whole({layer.attention_conv->base_kernel, layer.attention_conv->kernel_projection});
            if (layer.mlp_conv)
                whole({layer.mlp_conv->base_kernel, layer.mlp_conv->kernel_projection});
        }
    }
    if (w.vision) whole_vision(*w.vision);

    if (size == 1) {
        whole({w.text.head});
        for (const auto& layer : w.text.layers) {
            if (const auto* k = std::get_if<KdaWeights>(&layer.mixer))
                whole({k->query, k->key, k->value, k->output, k->beta, k->forget_a, k->forget_b,
                       k->gate_a, k->gate_b, k->convolution, k->a_log, k->dt_bias});
            if (const auto* d = std::get_if<DenseWeights>(&layer.ffn))
                whole({d->gate, d->up, d->down});
            else {
                const auto& s = std::get<MoeWeights>(layer.ffn).shared;
                whole({s.gate, s.up, s.down});
            }
        }
        if (w.mtp) {
            whole({w.mtp->head, w.mtp->moe.shared.gate, w.mtp->moe.shared.up,
                   w.mtp->moe.shared.down});
        }
        if (w.draft) {
            whole({w.draft->feature_projection, w.draft->output_head});
            for (const auto& layer : w.draft->layers) {
                const auto& a = layer.attention;
                whole({a.query, a.key, a.value, a.context_key, a.context_value, a.output,
                       layer.mlp.gate, layer.mlp.up, layer.mlp.down});
            }
        }
        return p;
    }

    const auto rows =
        [&](std::initializer_list<std::pair<WeightId, bool>> weights) -> ParameterAssembly& {
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
        return p.assemblies.back();
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
    vocabulary_head(w.text.head);
    if (w.mtp) {
        dense(w.mtp->moe.shared);
        vocabulary_head(w.mtp->head);
    }
    if (w.draft) {
        rows({{w.draft->feature_projection, true}});
        for (const auto& layer : w.draft->layers) {
            const auto& a    = layer.attention;
            auto& projection = rows({{a.key, true}, {a.value, true}, {a.query, true}});
            auto key         = projection.bindings[0];
            auto value       = projection.bindings[1];
            key.parameter    = a.context_key;
            value.parameter  = a.context_value;
            projection.bindings.push_back(std::move(key));
            projection.bindings.push_back(std::move(value));
            columns(a.output);
            dense(layer.mlp);
        }
        vocabulary_head(w.draft->output_head);
    }
    return p;
}

} // namespace

std::uint64_t compute_rank_shard_digest(const artifact::ArtifactId& artifact_id,
                                        LoadOptions options, const RankPlan& r) {
    std::vector<std::uint64_t> geometry{
        r.placement.tp.rank,  r.placement.tp.size,  r.placement.ep.rank, r.placement.ep.size,
        r.placement.adp.rank, r.placement.adp.size, r.mla_heads.begin,   r.mla_heads.size};
    for (const auto range :
         {r.kda_heads, r.vocabulary, r.dense_intermediate, r.shared_intermediate}) {
        geometry.push_back(range.begin);
        geometry.push_back(range.size);
    }
    geometry.push_back(r.draft.has_value());
    if (r.draft) {
        for (const auto range : {r.draft->features, r.draft->query_heads, r.draft->key_value_heads,
                                 r.draft->intermediate}) {
            geometry.push_back(range.begin);
            geometry.push_back(range.size);
        }
    }
    return rank_weight_digest(artifact_id, options, geometry, r.whole, r.omitted, r.assemblies,
                              r.experts);
}

RankPlan plan_single_rank(const ModelDefinition& definition) { return rank_plan(definition, {}); }

ParallelPlan plan_parallel(const ModelDefinition& definition, const DeploymentOptions& deployment) {
    const auto normalized = normalize_deployment(deployment);
    const auto& o         = normalized.parallel;
    if (o.pp != 1) throw std::invalid_argument("GLM5Next weight placement requires PP1");
    ParallelPlan out;
    out.topology = distributed::build_cluster_topology(normalized);
    out.ranks.reserve(out.topology.ranks.size());
    for (auto& id : out.topology.ranks) {
        auto shard = rank_plan(definition, id);
        id.shard_digest =
            compute_rank_shard_digest(definition.artifact_id(), definition.options(), shard);
        out.ranks.push_back(std::move(shard));
    }
    for (const auto& rep : out.topology.replicas) {
        for (const char* role : {"tensor", "attention", "vocabulary", "vision"})
            out.topology.add_role(role, rep.ranks);
        out.topology.add_role("context", rep.ranks, distributed::GroupUsage::PeerExchange);
        const auto lanes = out.ranks[rep.ranks.front()].placement.attention_slice().size;
        for (std::size_t i = 0; i < rep.ranks.size(); i += lanes)
            out.topology.add_role("attention_owner", std::span(rep.ranks).subspan(i, lanes));
    }
    add_expert_roles(out.topology);
    out.topology.plan =
        distributed::compute_topology_plan_digest(definition.artifact_id(), out.topology);
    out.topology.validate(normalized);
    return out;
}

} // namespace ninfer::models::glm5next
