#include "models/qwen4_exp/execution/profile.h"
#include "models/qwen4_exp/definition.h"
#include "models/qwen4_exp/parallel/plan.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/linear_split.h"

#include <array>

namespace ninfer::models::qwen4_exp::execution {
namespace {
class Describe {
public:
    Describe(const ModelDefinition& source, std::span<const BoundWeight> weights,
             std::span<const moe::ExpertDescription> experts)
        : source_(source), weights_(weights), experts_(experts) {}

    ops::WeightInput input(WeightId id) const { return weight_input(weights_, id); }

    ops::LinearProfile linear(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(input(id));
        return ops::describe_linear_weight(inputs);
    }

    ops::HcMixProfile hc(const HcWeights& w) const {
        return {linear({w.down, w.injection.value()}), linear({w.up})};
    }

    ops::HcFinalMixProfile final_hc(const HcWeights& w) const {
        return {linear({w.down}), linear({w.up})};
    }

    ops::ProjectionProfile projection(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(input(id));
        return ops::describe_projection_weights(inputs);
    }

    std::int32_t rows(WeightId id) const { return dimension(input(id).weight.shape[0]); }

    MtpHistoryProfile history(const BlockWeights& w) const {
        const auto& q = std::get<QsaWeights>(w.mixer);
        return {final_hc(w.attention_hc),
                linear({q.key, q.value, q.index_key}),
                {rows(q.key), rows(q.value), rows(q.index_key)}};
    }

    BlockProfile block(const BlockWeights& w, bool mtp) const {
        BlockProfile out;
        out.attention_hc = hc(w.attention_hc);
        out.moe_hc       = hc(w.moe_hc);
        if (const auto* g = std::get_if<GdnWeights>(&w.mixer)) {
            out.mixer = GdnProfile{
                projection({g->query, g->key, g->value, g->z}),
                projection({g->a, g->b}),
                {dimension(std::uint64_t(rows(g->query)) + rows(g->key) + rows(g->value)),
                 rows(g->z)},
                linear({g->output})};
        } else {
            const auto& q = std::get<QsaWeights>(w.mixer);
            QsaProfile profile;
            profile.output_rows = {rows(q.key),   rows(q.value), rows(q.index_key),
                                   rows(q.query), rows(q.gate),  rows(q.index_query)};
            if (mtp)
                profile.input =
                    linear({q.key, q.value, q.index_key, q.query, q.gate, q.index_query});
            else
                profile.input = MainQsaProjectionProfile{
                    linear({q.query, q.gate}),
                    linear({q.key, q.value, q.index_query, q.index_key}),
                    {rows(q.query), rows(q.gate)},
                    {rows(q.key), rows(q.value), rows(q.index_query), rows(q.index_key)}};
            profile.output = linear({q.output});
            (void)qsa_projection_workspace_bytes(profile, 1, 1);
            out.mixer = profile;
        }
        const auto& m    = w.moe;
        const auto& bank = source_.expert_banks()[m.bank.index];
        ops::ExpertInputUsage gu, down;
        gu.include(input(m.gate));
        gu.include(input(m.up));
        down.include(input(m.down));
        const auto routed =
            experts_.empty()
                ? ops::routed_swiglu_profile(bank.gate_up.layout, bank.down.layout,
                                             source_.config().text.moe.top_k, gu.policy(),
                                             down.policy())
                : moe::describe_experts(experts_[m.bank.index], source_.config().text.moe.top_k,
                                        gu.policy(), down.policy());
        out.moe = {{routed, ops::swiglu_profile(linear({m.shared_gate, m.shared_up}),
                                                linear({m.shared_down}))},
                   m.bank,
                   {gu.activation_divisor(routed.ffn.gate_up_type),
                    down.activation_divisor(routed.ffn.down_type)},
                   experts_.empty() ? ops::ExpertSubsetKind::All : experts_[m.bank.index].subset};
        return out;
    }
private:
    const ModelDefinition& source_;
    std::span<const BoundWeight> weights_;
    std::span<const moe::ExpertDescription> experts_;
};

std::shared_ptr<const ModelProfile> describe(const ModelDefinition& source,
                                             std::span<const BoundWeight> weights,
                                             std::span<const moe::ExpertDescription> experts,
                                             const RankPlan& rank) {
    if (rank.size > 1 && experts.size() != source.expert_banks().size())
        throw std::invalid_argument("Qwen4Exp local expert descriptions are incomplete");
    auto out                   = std::make_shared<ModelProfile>();
    out->config                = source.config();
    out->options               = source.options();
    out->ple_placement         = rank.ple_placement;
    out->local                 = {out->config.text.gdn,
                                  out->config.text.qsa,
                                  rank.vocabulary.begin,
                                  rank.vocabulary.size,
                                  rank.rank,
                                  rank.size};
    out->local.gdn.key_heads   = rank.gdn_keys.size;
    out->local.gdn.value_heads = rank.gdn_values.size;
    out->local.qsa.query_heads = rank.qsa_queries.size;
    out->local.qsa.kv_heads    = rank.qsa_kv.size;
    out->public_tokens         = source.resources().text.public_token_count;
    out->control_token_limit   = source.resources().text.control_tokens->size();
    const Describe d(source, weights, experts);
    const auto& w = source.weights();
    for (const auto& layer : w.text.layers) out->text.layers.push_back(d.block(layer, false));
    out->text.final_hc = d.final_hc(w.text.final_hc);
    out->text.ple      = d.linear({w.text.ple.key, w.text.ple.value});
    out->text.head     = d.linear({w.text.head});
    if (w.mtp) {
        const auto& m = *w.mtp;
        out->mtp      = {d.linear({m.fc_embedding}), d.linear({m.fc_hidden}), d.linear({m.head}),
                         d.block(m.layer, true),     d.final_hc(m.final_hc),  d.history(m.layer)};
    }
    if (w.vision)
        out->vision = qwen::describe_vision(*w.vision,
                                            [&](WeightId id) { return weight_input(weights, id); });
    if (rank.size == 1)
        for (const auto& b : source.expert_banks())
            out->banks.push_back({b.gate_up.layout, b.down.layout});
    return out;
}

} // namespace

std::shared_ptr<const ModelProfile> describe_model(const ModelDefinition& source) {
    return describe(source, source.weight_data(), {}, plan_single_rank(source));
}

std::shared_ptr<const ModelProfile> describe_model(const ModelDefinition& source,
                                                   std::span<const BoundWeight> weights,
                                                   std::span<const moe::ExpertDescription> experts,
                                                   const RankPlan& rank) {
    return describe(source, weights, experts, rank);
}

std::size_t qsa_projection_workspace_bytes(const QsaProfile& p, std::int32_t first,
                                           std::int32_t last) {
    if (const auto* main = std::get_if<MainQsaProjectionProfile>(&p.input)) {
        return std::max(ops::linear_split_workspace_capacity_bytes(
                            {main->query_gate, {}}, main->query_gate_rows, first, last),
                        ops::linear_split_workspace_capacity_bytes(
                            {main->kv_index, {}}, main->kv_index_rows, first, last));
    }
    return ops::linear_split_workspace_capacity_bytes({std::get<ops::LinearProfile>(p.input), {}},
                                                      p.output_rows, first, last);
}
} // namespace ninfer::models::qwen4_exp::execution
