#include "models/qwen3_5/parallel/plan.h"

#include "models/common/parallel_weights.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint32_t dimension_u32(std::uint64_t value) {
    if (value > UINT32_MAX) {
        throw artifact::ArtifactError("Qwen3.5 parallel dimension exceeds uint32");
    }
    return static_cast<std::uint32_t>(value);
}

RankPlan classify_single_tp_rank(const ModelDefinition& definition) {
    const auto& c = definition.config().text;
    const auto& w = definition.weights();
    const std::size_t total_weights = definition.weight_data().size();

    enum class State : std::uint8_t { Unclassified = 0, Whole = 1, Expert = 2, Omitted = 3 };
    std::vector<State> states(total_weights, State::Unclassified);

    RankPlan p;
    p.rank    = 0;
    p.size    = 1;
    p.ep_size = 1;
    p.experts.resize(definition.expert_banks().size());

    const auto mark_whole = [&](WeightId id) {
        if (id.index >= total_weights) {
            throw std::logic_error("Qwen3.5 weight classification index out of bounds");
        }
        auto& st = states[id.index];
        if (st == State::Whole) { return; } // Legitimate shared alias (e.g. tied head/embedding)
        if (st != State::Unclassified) {
            throw std::logic_error("Qwen3.5 weight classified in conflicting categories");
        }
        st = State::Whole;
        p.whole.push_back(id);
    };

    const auto mark_dense = [&](const DenseWeights& d) {
        mark_whole(d.gate);
        mark_whole(d.up);
        mark_whole(d.down);
    };

    const auto mark_block = [&](const BlockWeights& block) {
        mark_whole(block.input_norm);
        mark_whole(block.post_attention_norm);
        if (const auto* a = std::get_if<AttentionWeights>(&block.mixer)) {
            for (const auto id :
                 {a->query, a->key, a->gate, a->value, a->query_norm, a->key_norm, a->output}) {
                mark_whole(id);
            }
        } else {
            const auto& g = std::get<GdnWeights>(block.mixer);
            for (const auto id : {g.query, g.key, g.value, g.z, g.a_projection, g.b_projection,
                                  g.a_log, g.dt_bias, g.convolution, g.norm, g.output}) {
                mark_whole(id);
            }
        }
        if (const auto* d = std::get_if<DenseWeights>(&block.ffn)) {
            mark_dense(*d);
        } else {
            const auto& m = std::get<MoeWeights>(block.ffn);
            mark_whole(m.router);
            mark_whole(m.shared_score);
            mark_dense(m.shared);
            const auto& moe_cfg = std::get<MoeConfig>(c.ffn);
            auto& part          = p.experts.at(m.bank.index);
            part.intermediate   = {0, dimension_u32(moe_cfg.moe_intermediate_size)};
            part.ids.resize(m.experts.size());
            std::iota(part.ids.begin(), part.ids.end(), 0U);
            for (const auto& exp : m.experts) {
                for (const auto id : {exp.gate, exp.up, exp.down}) {
                    if (id.index >= total_weights || states[id.index] != State::Unclassified) {
                        throw std::logic_error("Qwen3.5 expert weight duplicate or invalid");
                    }
                    states[id.index] = State::Expert;
                }
            }
        }
    };

    mark_whole(w.text.token_embedding);
    mark_whole(w.text.output_head);
    mark_whole(w.text.final_norm);
    for (const auto& layer : w.text.layers) { mark_block(layer); }

    if (w.vision) {
        const auto& v = *w.vision;
        mark_whole(v.patch_embedding);
        mark_whole(v.patch_embedding_bias);
        mark_whole(v.position_embedding);
        for (const auto& layer : v.layers) {
            for (const auto id :
                 {layer.norm1.weight, layer.norm1.bias, layer.norm2.weight, layer.norm2.bias,
                  layer.query, layer.key, layer.value, layer.query_bias, layer.key_bias,
                  layer.value_bias, layer.output, layer.output_bias, layer.fc1, layer.fc1_bias,
                  layer.fc2, layer.fc2_bias}) {
                mark_whole(id);
            }
        }
        for (const auto id : {v.merger_norm.weight, v.merger_norm.bias, v.merger_fc1,
                              v.merger_fc1_bias, v.merger_fc2, v.merger_fc2_bias}) {
            mark_whole(id);
        }
    }

    if (w.mtp) {
        for (const auto id : {w.mtp->input_projection, w.mtp->embedding_norm, w.mtp->hidden_norm,
                              w.mtp->final_norm, w.mtp->token_embedding, w.mtp->output_head}) {
            mark_whole(id);
        }
        mark_block(w.mtp->layer);
    }

    if (w.draft) {
        const auto& d = *w.draft;
        for (const auto id :
             {d.feature_projection, d.context_norm, d.final_norm, d.token_embedding, d.output_head}) {
            mark_whole(id);
        }
        for (const auto& layer : d.layers) {
            mark_whole(layer.input_norm);
            mark_whole(layer.post_attention_norm);
            const auto& a = layer.attention;
            for (const auto id : {a.query, a.key, a.value, a.context_key, a.context_value,
                                  a.query_norm, a.key_norm, a.output}) {
                mark_whole(id);
            }
            mark_dense(layer.mlp);
            if (layer.attention_conv) {
                mark_whole(layer.attention_conv->base_kernel);
                mark_whole(layer.attention_conv->kernel_projection);
            }
            if (layer.mlp_conv) {
                mark_whole(layer.mlp_conv->base_kernel);
                mark_whole(layer.mlp_conv->kernel_projection);
            }
        }
        if (d.selector) {
            mark_whole(d.selector->hidden_projection);
            mark_whole(d.selector->predecessor_codebook);
            mark_whole(d.selector->successor_codebook);
        }
    }

    if (w.proposal) {
        mark_whole(w.proposal->head);
        if (w.proposal->token_ids) { mark_whole(*w.proposal->token_ids); }
    }

    for (std::size_t i = 0; i < total_weights; ++i) {
        if (states[i] == State::Unclassified) {
            throw std::logic_error("Qwen3.5 parallel plan left parameter unclassified: " +
                                   definition.weight_data()[i].name);
        }
    }
    std::sort(p.whole.begin(), p.whole.end(),
              [](WeightId a, WeightId b) { return a.index < b.index; });
    return p;
}

} // namespace

std::uint64_t compute_rank_shard_digest(const artifact::ArtifactId& artifact_id,
                                        LoadOptions options, const RankPlan& r) {
    const std::uint64_t geometry[]{r.rank, r.size, r.ep_size};
    return rank_weight_digest(artifact_id, options, geometry, r.whole, r.omitted, r.assemblies,
                              r.experts);
}

RankPlan plan_single_rank(const ModelDefinition& definition) {
    return classify_single_tp_rank(definition);
}

ParallelPlan plan_parallel(const ModelDefinition& definition, const DeploymentOptions& deployment) {
    const auto normalized = normalize_deployment(deployment);
    const auto& o         = normalized.parallel;
    if (o.tp != 1 || o.ep != 1 || o.pp != 1 || o.adp != 1 || o.dp < 1 ||
        normalized.ranks.size() != o.dp) {
        throw std::invalid_argument(
            "Qwen3_5 Stage 1 distributed preparation supports pure DP (TP=EP=PP=ADP=1, DP>=1)");
    }
    ParallelPlan out;
    out.topology      = distributed::build_cluster_topology(normalized);
    const auto single = classify_single_tp_rank(definition);
    const auto digest =
        compute_rank_shard_digest(definition.artifact_id(), definition.options(), single);
    out.ranks.reserve(normalized.ranks.size());
    for (const auto& r : normalized.ranks) {
        out.topology.ranks[r.rank].shard_digest = digest;
        out.ranks.push_back(single);
    }
    out.topology.plan =
        distributed::compute_topology_plan_digest(definition.artifact_id(), out.topology);
    out.topology.validate(normalized);
    return out;
}

} // namespace ninfer::models::qwen3_5
