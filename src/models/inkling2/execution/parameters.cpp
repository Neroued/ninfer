#include "models/inkling2/execution/parameters.h"

#include <stdexcept>

namespace ninfer::models::inkling2::execution {
namespace {

class Prepare {
public:
    explicit Prepare(const ModelShard& model) : model_(model) {}

    LinearParameters linear(WeightId id, std::uint32_t use_index = 0) const {
        const auto input = model_.input(WeightUseId{id, use_index});
        return ops::prepare_linear_weight({&input, 1});
    }

    LinearParameters linear(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        inputs.reserve(ids.size());
        for (const auto id : ids) inputs.push_back(model_.input(id));
        return ops::prepare_linear_weight(inputs);
    }

    Tensor bias(std::initializer_list<WeightId> ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            count += weight_element_count(input.shape);
            view.parts.insert(view.parts.end(), input.parts.begin(), input.parts.end());
        }
        view.shape = {count};
        return weight_tensor(view, {dimension(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out{
            .patch = linear(w.patch_embedding),
            .projector_in = linear(w.projector_in),
            .projector_out = linear(w.projector_out),
            .position_embedding = tensor(w.position_embedding),
            .class_embedding = tensor(w.class_embedding),
            .cls_position_embedding = tensor(w.cls_position_embedding),
            .pre_norm_weight = tensor(w.pre_norm_weight),
            .pre_norm_bias = tensor(w.pre_norm_bias),
        };
        if (w.projector_in_bias) out.projector_in_bias = tensor(*w.projector_in_bias);
        if (w.projector_out_bias) out.projector_out_bias = tensor(*w.projector_out_bias);
        out.layers.reserve(w.layers.size());
        for (const auto& l : w.layers)
            out.layers.push_back({tensor(l.ln_1_weight), tensor(l.ln_1_bias),
                                  tensor(l.ln_2_weight), tensor(l.ln_2_bias),
                                  tensor(l.ls_1_gamma), tensor(l.ls_2_gamma),
                                  linear({l.query, l.key, l.value}), linear(l.output),
                                  linear(l.fc1), linear(l.fc2),
                                  bias({l.query_bias, l.key_bias, l.value_bias}),
                                  tensor(l.output_bias), tensor(l.fc1_bias), tensor(l.fc2_bias)});
        return out;
    }

    ops::ProjectionWeights projection(std::initializer_list<WeightUseId> ids) const {
        std::vector<ops::WeightInput> inputs;
        inputs.reserve(ids.size());
        for (const auto id : ids) inputs.push_back(model_.input(id));
        return ops::prepare_projection_weights(inputs);
    }

    Tensor tensor(WeightId id) const { return parameter_tensor(model_.weight(id)); }

    PhysicalLayerParameters layer(const PhysicalLayerConfig& cfg,
                                  const PhysicalLayerWeights& w) const {
        PhysicalLayerParameters out;
        out.is_swa  = cfg.is_swa;
        out.in_core = cfg.in_core;
        if (w.attention.attn_res_query) out.attn_res_query = tensor(*w.attention.attn_res_query);
        out.attn_input_norm = tensor(w.attention.input_norm);
        out.sink            = tensor(w.attention.sink);
        out.q_scale         = tensor(w.attention.q_scale);
        out.r_scale         = tensor(w.attention.r_scale);
        out.rel_logits_proj = tensor(w.attention.rel_logits_proj);
        out.k_conv          = tensor(w.attention.k_conv);
        out.v_conv          = tensor(w.attention.v_conv);
        out.attn_short_conv = tensor(w.attention.short_conv);

        if (w.loop) {
            out.loop = LoopParameters{
                .message_queries = tensor(w.loop->message_queries),
                .attn_kv         = {tensor(w.loop->attn_kv_down), tensor(w.loop->attn_kv_up)},
                .mlp             = {tensor(w.loop->mlp_down), tensor(w.loop->mlp_up)},
            };
        }

        out.mlp_attn_res_query = tensor(w.moe.attn_res_query);
        out.mlp_input_norm     = tensor(w.moe.input_norm);
        out.mlp_short_conv     = tensor(w.moe.short_conv);
        out.correction_bias    = tensor(w.moe.correction_bias);

        const auto& bank = model_.experts(w.moe.bank);
        const auto view  = bank.view();

        out.attention_passes.reserve(cfg.num_passes);
        out.moe_passes.reserve(cfg.num_passes);
        for (std::uint32_t pass = 0; pass < cfg.num_passes; ++pass) {
            AttentionPassParameters ap;
            if (cfg.is_swa && pass == 0) {
                ap.qkv_split = projection({{w.attention.key, pass},
                                           {w.attention.value, pass},
                                           {w.attention.query, pass},
                                           {w.attention.output_gate, pass},
                                           {w.attention.relative, pass}});
            }
            ap.qrg_split = projection({{w.attention.query, pass},
                                       {w.attention.output_gate, pass},
                                       {w.attention.relative, pass}});
            ap.key   = linear(w.attention.key, pass);
            ap.value = linear(w.attention.value, pass);
            if (!cfg.is_swa) {
                ap.k_conv_proj = linear(w.attention.k_conv_proj.value(), pass);
                ap.v_conv_proj = linear(w.attention.v_conv_proj.value(), pass);
            }
            ap.output = linear(w.attention.output, pass);
            out.attention_passes.push_back(std::move(ap));

            ops::ExpertInputUsage gate_up, down;
            gate_up.include(model_.input(WeightUseId{w.moe.gate, pass}));
            gate_up.include(model_.input(WeightUseId{w.moe.up, pass}));
            down.include(model_.input(WeightUseId{w.moe.down, pass}));
            out.moe_passes.push_back(MoePassParameters{
                .router = linear(w.moe.router, pass).weight,
                .routed = {view,
                           {gate_up.calibration(view.gate_up.format),
                            down.calibration(view.down.format)}},
                .subset = bank.subset(),
            });
        }
        return out;
    }

private:
    const ModelShard& model_;
};

} // namespace

Parameters::Parameters(const ModelShard& model, std::shared_ptr<const ModelProfile> profile)
    : model(model), profile(std::move(profile)) {
    if (!this->profile || this->profile->local.rank != model.rank_plan().rank ||
        this->profile->local.size != model.rank_plan().size ||
        this->profile->options != model.definition().options()) {
        throw std::invalid_argument("Inkling2 Parameters require their selected rank profile");
    }
    const Prepare p(model);
    const auto& c     = model.definition().config().text;
    const auto& w     = model.definition().weights();
    text.embedding    = native_weight(model.weight(w.text.embedding).view);
    text.embed_norm   = p.tensor(w.text.embed_norm);
    text.final_norm   = p.tensor(w.text.final_norm);
    text.output_query = p.tensor(w.text.output_query);
    text.head         = p.linear(w.text.head);
    text.layers.reserve(c.physical_layers.size());
    for (std::size_t i = 0; i < c.physical_layers.size(); ++i) {
        text.layers.push_back(p.layer(c.physical_layers[i], w.text.layers[i]));
    }
    if (w.vision) vision = p.vision(*w.vision);
}

} // namespace ninfer::models::inkling2::execution
