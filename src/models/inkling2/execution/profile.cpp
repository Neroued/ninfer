#include "models/inkling2/execution/profile.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::models::inkling2::execution {
namespace {

class Describe {
public:
    Describe(const ModelDefinition& definition, std::span<const BoundWeight> weights,
             std::span<const moe::ExpertDescription> experts)
        : definition_(definition), weights_(weights), experts_(experts) {}

    ops::LinearProfile linear(WeightId id, std::uint32_t use_index = 0) const {
        const auto input = weight_input(weights_, WeightUseId{id, use_index});
        return ops::describe_linear_weight({&input, 1});
    }

    ops::ProjectionProfile projection(std::initializer_list<WeightUseId> ids) const {
        std::vector<ops::WeightInput> inputs;
        inputs.reserve(ids.size());
        for (const auto id : ids) inputs.push_back(weight_input(weights_, id));
        return ops::describe_projection_weights(inputs);
    }

    PhysicalLayerProfile layer(const PhysicalLayerConfig& cfg,
                               const PhysicalLayerWeights& w) const {
        const auto& c    = definition_.config().text;
        const auto& attn = c.attention(cfg.is_swa);
        const auto q_dim = dimension(attn.query_width());
        const auto r_dim = dimension(c.rel_width(cfg.is_swa));
        const auto kv    = dimension(attn.kv_width());

        PhysicalLayerProfile out;
        out.is_swa             = cfg.is_swa;
        out.in_core            = cfg.in_core;
        out.attention_geometry = {dimension(attn.head_dim), dimension(attn.heads),
                                  dimension(attn.kv_heads)};

        out.attention_passes.reserve(cfg.num_passes);
        out.moe_passes.reserve(cfg.num_passes);

        const auto& bank = experts_[w.moe.bank.index];
        for (std::uint32_t pass = 0; pass < cfg.num_passes; ++pass) {
            AttentionPassProfile ap;
            if (cfg.is_swa && pass == 0) {
                ap.qkv_split = projection({{w.attention.key, pass},
                                           {w.attention.value, pass},
                                           {w.attention.query, pass},
                                           {w.attention.output_gate, pass},
                                           {w.attention.relative, pass}});
                ap.qkv_rows  = {kv, kv, q_dim, q_dim, r_dim};
            }
            ap.qrg_split = projection({{w.attention.query, pass},
                                       {w.attention.output_gate, pass},
                                       {w.attention.relative, pass}});
            ap.qrg_rows  = {q_dim, q_dim, r_dim};
            ap.key       = linear(w.attention.key, pass);
            ap.value     = linear(w.attention.value, pass);
            if (!cfg.is_swa) {
                ap.k_conv_proj = linear(w.attention.k_conv_proj.value(), pass);
                ap.v_conv_proj = linear(w.attention.v_conv_proj.value(), pass);
            }
            ap.output = linear(w.attention.output, pass);
            out.attention_passes.push_back(std::move(ap));

            ops::ExpertInputUsage gate_up, down;
            gate_up.include(weight_input(weights_, WeightUseId{w.moe.gate, pass}));
            gate_up.include(weight_input(weights_, WeightUseId{w.moe.up, pass}));
            down.include(weight_input(weights_, WeightUseId{w.moe.down, pass}));
            auto routed = moe::describe_experts(bank, dimension(c.moe.top_k), gate_up.policy(),
                                                down.policy());
            routed.ffn.softcap = ops::SwiGLUSoftcap{c.moe.gate_softcap, c.moe.up_softcap};

            MoePassProfile mp;
            mp.routing = {{dimension(c.hidden), dimension(c.moe.experts), dimension(c.moe.top_k)},
                          c.moe.route_scale,
                          1e-20F};
            mp.routed  = routed;
            mp.bank    = w.moe.bank;
            mp.subset  = bank.subset;
            out.moe_passes.push_back(std::move(mp));
        }
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
    if (rank.size != 1) throw std::invalid_argument("Inkling2 currently supports single-rank only");
    if (experts.size() != definition.expert_banks().size())
        throw std::invalid_argument("Inkling2 local expert descriptions are incomplete");
    auto out                 = std::make_shared<ModelProfile>();
    out->config              = definition.config();
    out->options             = definition.options();
    out->local               = {rank.rank, rank.size, rank.vocabulary};
    const auto& res          = definition.resources();
    out->public_tokens       = res.text.public_token_count;
    out->control_token_limit = static_cast<std::uint32_t>(std::max({
        res.text.control_tokens ? res.text.control_tokens->size() : std::size_t{0},
        res.control_to_text ? res.control_to_text->size() : std::size_t{0},
        res.control_to_header ? res.control_to_header->size() : std::size_t{0},
    }));

    const auto& c = out->config.text;
    const auto& w = definition.weights();
    const Describe d(definition, weights, experts);

    out->text.attn_res = {
        .query_scale      = 1.0F / std::sqrt(static_cast<float>(c.hidden)),
        .rms_eps          = 1e-6F,
        .norm_eps         = c.norm_epsilon,
        .normalize_values = true,
    };
    out->text.head       = d.linear(w.text.head);
    out->text.max_blocks = (c.num_layers + c.attn_res_block_size - 1U) / c.attn_res_block_size;
    if (c.loop) { out->text.bank_blocks = c.loop->core_layers() / c.attn_res_block_size; }
    out->text.layers.reserve(c.physical_layers.size());
    for (std::size_t i = 0; i < c.physical_layers.size(); ++i) {
        out->text.layers.push_back(d.layer(c.physical_layers[i], w.text.layers[i]));
    }
    return out;
}

} // namespace ninfer::models::inkling2::execution
