#include "models/glm5next/execution/profile.h"

#include <algorithm>

namespace ninfer::models::glm5next::execution {

MlaProjectionInputs mla_projection_inputs(std::span<const BoundWeight> weights, const MlaWeights& w,
                                          ops::KpoolMlaPrepareProjection projection) {
    MlaProjectionInputs out;
    const std::array ids{w.latent, w.index_key, w.pool_gate, w.query_a, w.index_weight};
    const std::array rows{&out.rows.latent, &out.rows.index_key, &out.rows.pool_gate,
                          &out.rows.query, &out.rows.coefficients};
    const std::size_t count = projection == ops::KpoolMlaPrepareProjection::History           ? 3
                              : projection == ops::KpoolMlaPrepareProjection::QueryAndHistory ? 4
                                                                                              : 5;

    struct Role {
        WeightId id;
        WeightRegion region;
        ops::KpoolProjectionRows* rows;
        std::size_t parent;
    };

    std::vector<Role> roles;
    std::vector<const WeightParent*> parents;
    for (std::size_t i = 0; i < count; ++i) {
        const auto input  = weight_input(weights, ids[i]);
        const auto region = contiguous_weight_region(input.weight);
        auto found        = std::find(parents.begin(), parents.end(), region.parent);
        if (found == parents.end()) {
            parents.push_back(region.parent);
            found = std::prev(parents.end());
        }
        roles.push_back({ids[i], region, rows[i], std::size_t(found - parents.begin())});
    }
    std::sort(roles.begin(), roles.end(), [](const Role& a, const Role& b) {
        return a.parent != b.parent ? a.parent < b.parent : a.region.begin < b.region.begin;
    });
    const WeightParent* parent = nullptr;
    std::uint64_t end = 0, row = 0;
    int group = -1;
    for (const auto& role : roles) {
        if (role.region.parent != parent || role.region.begin != end) {
            ++group;
            row = 0;
        }
        const auto input   = weight_input(weights, role.id);
        const auto n       = dimension(input.weight.shape.at(0));
        const auto row_end = dimension(row + n);
        *role.rows         = {group, row_end - n, n};
        row                = row_end;
        parent             = role.region.parent;
        end                = role.region.end;
        out.weights.push_back(input);
    }
    return out;
}

namespace {
class Describe {
public:
    Describe(const ModelDefinition& definition, std::span<const BoundWeight> weights,
             std::span<const moe::ExpertDescription> experts)
        : definition_(definition), weights_(weights), experts_(experts) {}

    ops::LinearProfile linear(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(weight_input(weights_, id));
        return ops::describe_linear_weight(inputs);
    }

    ops::ProjectionProfile projection(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(weight_input(weights_, id));
        return ops::describe_projection_weights(inputs);
    }

    KdaProfile kda(const KdaWeights& w) const {
        return {linear({w.query, w.key, w.value}),
                linear({w.beta, w.forget_a, w.gate_a}),
                linear({w.forget_b}),
                linear({w.gate_b}),
                linear({w.output}),
                {linear({w.beta}).n, linear({w.forget_a}).n, linear({w.gate_a}).n}};
    }

    MlaProfile mla(const MlaWeights& w) const {
        const auto& c = definition_.config().text;
        const auto& m = c.mla;
        MlaProfile out;
        for (std::size_t i = 0; i < out.prepare.size(); ++i) {
            const auto kind   = static_cast<ops::KpoolMlaPrepareProjection>(i);
            const auto inputs = mla_projection_inputs(weights_, w, kind);
            out.prepare[i] = {kind, ops::describe_projection_weights(inputs.weights), inputs.rows,
                              dimension(m.pool_size)};
        }
        out.query_index = projection({w.query_b, w.index_query});
        out.query       = linear({w.query_b});
        out.output      = linear({w.output});
        out.query_rows  = {dimension(m.query_width()), dimension(m.index_query_width())};
        out.indexer     = {dimension(m.index_heads), dimension(m.index_dim), dimension(m.pool_size),
                           dimension(m.pool_budget)};
        out.attention   = {
            dimension(m.heads),
            dimension(m.query_dim),
            dimension(m.value_dim),
            dimension(m.latent_dim),
            dimension(m.pool_size),
            dimension(m.pool_budget),
            contiguous_weight_region(weights_[w.key_up.index].view).parent->geometry.format,
            contiguous_weight_region(weights_[w.value_up.index].view).parent->geometry.format};
        return out;
    }

    DenseProfile dense(const DenseWeights& w, float clamp) const {
        auto math        = ops::swiglu_profile(linear({w.gate, w.up}), linear({w.down}));
        math.clamp_limit = clamp;
        return {math};
    }

    MoeProfile moe(const MoeWeights& w) const {
        const auto& c    = definition_.config().text;
        const auto& bank = experts_[w.bank.index];
        ops::ExpertInputUsage gate_up, down;
        gate_up.include(weight_input(weights_, w.gate));
        gate_up.include(weight_input(weights_, w.up));
        down.include(weight_input(weights_, w.down));
        auto routed =
            moe::describe_experts(bank, dimension(c.moe.top_k), gate_up.policy(), down.policy());
        routed.ffn.clamp_limit = c.swiglu_limit;
        // GLM's current routing semantics select globally, without a preliminary group filter.
        if (c.moe.groups != 1 || c.moe.selected_groups != 1)
            throw std::invalid_argument("GLM5Next grouped expert routing is not implemented");
        return {{{{dimension(c.hidden), dimension(c.moe.experts), dimension(c.moe.top_k)},
                  c.moe.routed_scaling_factor,
                  1e-20F},
                 routed,
                 dense(w.shared, c.swiglu_limit).math},
                w.bank,
                bank.subset};
    }

    BlockProfile block(const BlockWeights& w) const {
        BlockProfile out;
        if (const auto* k = std::get_if<KdaWeights>(&w.mixer))
            out.mixer = kda(*k);
        else
            out.mixer = mla(std::get<MlaWeights>(w.mixer));
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn))
            out.ffn = dense(*d, definition_.config().text.swiglu_limit);
        else
            out.ffn = moe(std::get<MoeWeights>(w.ffn));
        return out;
    }

    VisionProfile vision(const VisionWeights& w) const {
        VisionProfile out{linear({w.patch_embedding}),   linear({w.downsample}),
                          linear({w.merger_projection}), linear({w.merger.gate, w.merger.up}),
                          linear({w.merger.down}),       {}};
        for (const auto& layer : w.layers)
            out.layers.push_back({linear({layer.query, layer.key, layer.value}),
                                  linear({layer.output}), linear({layer.mlp.gate, layer.mlp.up}),
                                  linear({layer.mlp.down})});
        return out;
    }
private:
    const ModelDefinition& definition_;
    std::span<const BoundWeight> weights_;
    std::span<const moe::ExpertDescription> experts_;
};
} // namespace

std::shared_ptr<const ModelProfile> describe_model(const ModelDefinition& definition,
                                                   std::span<const BoundWeight> weights,
                                                   std::span<const moe::ExpertDescription> experts,
                                                   const RankPlan& rank) {
    if (experts.size() != definition.expert_banks().size())
        throw std::invalid_argument("GLM5Next local expert descriptions are incomplete");
    auto out                 = std::make_shared<ModelProfile>();
    out->config              = definition.config();
    out->options             = definition.options();
    out->local               = {rank.rank,
                                rank.size,
                                rank.kda_heads,
                                rank.vocabulary,
                                rank.dense_intermediate,
                                rank.shared_intermediate};
    out->public_tokens       = definition.resources().text.public_token_count;
    out->control_token_limit = definition.resources().text.control_tokens->size();
    const auto& c            = out->config.text;
    const auto& w            = definition.weights();
    const Describe d(definition, weights, experts);
    out->text.mhc            = {dimension(c.hidden), dimension(c.mhc.streams)};
    out->text.mhc_parameters = {c.norm_epsilon, c.norm_epsilon,
                                c.mhc.epsilon,  c.mhc.epsilon,
                                2.0F,           dimension(c.mhc.sinkhorn_iterations)};
    out->text.head           = d.linear({w.text.head});
    for (const auto& layer : w.text.layers) out->text.layers.push_back(d.block(layer));
    if (w.mtp) {
        const auto& m = *w.mtp;
        out->mtp      = {d.linear({m.fc}), d.linear({m.head}), d.mla(m.attention), d.moe(m.moe)};
    }
    if (w.draft) {
        const auto& weights = *w.draft;
        const auto& config  = *out->config.draft;
        if (!config.dflash2 || config.full_layer_count())
            throw std::invalid_argument("GLM5Next drafter requires local DFlash2 layers");
        out->text.feature_slots.assign(c.layers.size(), -1);
        for (std::size_t i = 0; i < config.target_layer_ids.size(); ++i) {
            const auto layer = config.target_layer_ids[i];
            if (layer + 1 == c.layers.size())
                throw std::invalid_argument(
                    "GLM5Next draft feature capture requires a post/pre boundary");
            out->text.feature_slots.at(layer) = i;
        }
        auto& draft              = out->draft.emplace();
        draft.feature_projection = d.linear({weights.feature_projection});
        draft.head               = d.linear({weights.output_head});
        draft.selector           = d.linear({weights.selector->hidden_projection});
        draft.attention          = {dimension(config.attention.head_dim),
                                    dimension(rank.draft->query_heads.size),
                                    dimension(rank.draft->key_value_heads.size)};
        draft.context            = {dimension(c.hidden),
                                    draft.attention.head_dim,
                                    draft.attention.kv_heads,
                                    config.sliding_window.value(),
                                    d.linear({weights.layers.front().attention.context_key}).qtype,
                                    config.rms_norm_eps,
                                    config.rope_theta};
        for (const auto& layer : weights.layers)
            draft.layers.push_back(
                {d.linear({layer.attention.key, layer.attention.value, layer.attention.query}),
                 d.linear({layer.mlp.gate, layer.mlp.up}), d.linear({layer.mlp.down})});
    }
    if (w.vision) out->vision = d.vision(*w.vision);
    return out;
}

} // namespace ninfer::models::glm5next::execution
