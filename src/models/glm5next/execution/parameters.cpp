#include "models/glm5next/execution/parameters.h"

namespace ninfer::models::glm5next::execution {
namespace {
class Prepare {
public:
    explicit Prepare(const ModelShard& model) : model_(model) {}

    LinearParameters linear(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(model_.input(id));
        return ops::prepare_linear_weight(inputs);
    }

    ops::ProjectionWeights projection(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(model_.input(id));
        return ops::prepare_projection_weights(inputs);
    }

    Weight factor(WeightId id) const {
        auto view    = model_.weight(id).view;
        const auto k = view.shape.back();
        view.shape   = {weight_element_count(view.shape) / k, k};
        return native_weight(view);
    }

    Tensor tensor(WeightId id) const { return parameter_tensor(model_.weight(id)); }

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

    ops::MhcPreWeights mhc(const MhcWeights& w, WeightId gamma) const {
        return {native_weight(model_.weight(w.projection).view), tensor(w.base), tensor(w.scale),
                tensor(gamma)};
    }

    KdaParameters kda(const KdaWeights& w) const {
        return {linear({w.query, w.key, w.value}),
                linear({w.beta, w.forget_a, w.gate_a}),
                linear({w.forget_b}),
                linear({w.gate_b}),
                linear({w.output}),
                tensor(w.convolution),
                tensor(w.a_log),
                tensor(w.dt_bias).view({dimension(model_.definition().config().text.kda.head_dim),
                                        dimension(model_.rank_plan().kda_heads.size)}),
                tensor(w.norm)};
    }

    MlaParameters mla(const MlaWeights& w) const {
        const auto& c = model_.definition().config().text;
        MlaParameters out;
        for (std::size_t i = 0; i < out.prepare.size(); ++i) {
            const auto kind   = static_cast<ops::KpoolMlaPrepareProjection>(i);
            const auto inputs = mla_projection_inputs(model_.weight_data(), w, kind);
            out.prepare[i]    = {
                kind,
                ops::prepare_projection_weights(inputs.weights),
                inputs.rows,
                kind == ops::KpoolMlaPrepareProjection::History ? Tensor{} : tensor(w.query_norm),
                tensor(w.latent_norm),
                tensor(w.index_norm),
                tensor(w.index_bias),
                tensor(w.pool_ape),
                c.norm_epsilon,
                1e-6F};
        }
        out.query_index = projection({w.query_b, w.index_query});
        out.query       = linear({w.query_b});
        out.output      = linear({w.output});
        out.factors     = {factor(w.key_up), factor(w.value_up)};
        return out;
    }

    ops::SwiGLUWeights dense(const DenseWeights& w) const {
        return {linear({w.gate, w.up}), linear({w.down})};
    }

    MoeParameters moe(const MoeWeights& w) const {
        const auto& bank = model_.experts(w.bank);
        const auto view  = bank.view();
        ops::ExpertInputUsage gate_up, down;
        gate_up.include(model_.input(w.gate));
        gate_up.include(model_.input(w.up));
        down.include(model_.input(w.down));
        return {
            {linear({w.router}).weight,
             tensor(w.correction_bias),
             {view, {gate_up.calibration(view.gate_up.format), down.calibration(view.down.format)}},
             dense(w.shared)},
            bank.subset()};
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.attention_hc = mhc(w.attention_hc, w.input_norm);
        out.mlp_hc       = mhc(w.mlp_hc, w.post_attention_norm);
        if (const auto* k = std::get_if<KdaWeights>(&w.mixer))
            out.mixer = kda(*k);
        else
            out.mixer = mla(std::get<MlaWeights>(w.mixer));
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn))
            out.ffn = dense(*d);
        else
            out.ffn = moe(std::get<MoeWeights>(w.ffn));
        return out;
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out{linear({w.patch_embedding}),
                             linear({w.downsample}),
                             linear({w.merger_projection}),
                             linear({w.merger.gate, w.merger.up}),
                             linear({w.merger.down}),
                             tensor(w.patch_embedding_bias),
                             tensor(w.downsample_bias),
                             tensor(w.post_norm),
                             tensor(w.merger_norm_weight),
                             tensor(w.merger_norm_bias),
                             {}};
        for (const auto& l : w.layers)
            out.layers.push_back(
                {tensor(l.input_norm), tensor(l.post_attention_norm), tensor(l.query_norm),
                 tensor(l.key_norm), linear({l.query, l.key, l.value}), linear({l.output}),
                 linear({l.mlp.gate, l.mlp.up}), linear({l.mlp.down}),
                 bias({l.query_bias, l.key_bias, l.value_bias}), tensor(l.output_bias),
                 bias({l.gate_bias, l.up_bias}), tensor(l.down_bias)});
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
        this->profile->options != model.definition().options())
        throw std::invalid_argument("GLM5Next Parameters require their selected rank profile");
    const Prepare p(model);
    const auto& w   = model.definition().weights();
    text.embedding  = native_weight(model.weight(w.text.embedding).view);
    text.head       = p.linear({w.text.head});
    text.final_norm = p.tensor(w.text.final_norm);
    for (std::size_t i = 0; i < w.text.layers.size(); ++i)
        text.layers.push_back(p.block(w.text.layers[i]));
    if (w.mtp) {
        const auto& m = *w.mtp;
        mtp           = {native_weight(model.weight(m.embedding).view),
                         p.linear({m.fc}),
                         p.linear({m.head}),
                         p.tensor(m.embedding_norm),
                         p.tensor(m.hidden_norm),
                         p.tensor(m.input_norm),
                         p.tensor(m.post_attention_norm),
                         p.tensor(m.final_norm),
                         p.mla(m.attention),
                         p.moe(m.moe)};
    }
    if (w.draft) {
        const auto& source       = *w.draft;
        auto& out                = draft.emplace();
        out.embedding            = native_weight(model.weight(source.token_embedding).view);
        out.feature_projection   = p.linear({source.feature_projection});
        out.head                 = p.linear({source.output_head});
        out.selector             = p.linear({source.selector->hidden_projection});
        out.context_norm         = p.tensor(source.context_norm);
        out.final_norm           = p.tensor(source.final_norm);
        out.predecessor_codebook = p.tensor(source.selector->predecessor_codebook);
        out.successor_codebook   = p.tensor(source.selector->successor_codebook);
        for (const auto& l : source.layers) {
            const auto& a = l.attention;
            out.layers.push_back(
                {p.tensor(l.input_norm),
                 p.tensor(l.post_attention_norm),
                 p.tensor(a.query_norm),
                 p.tensor(a.key_norm),
                 p.linear({a.key, a.value, a.query}),
                 p.linear({a.output}),
                 p.linear({l.mlp.gate, l.mlp.up}),
                 p.linear({l.mlp.down}),
                 p.linear({a.context_key}).weight,
                 p.linear({a.context_value}).weight,
                 {p.tensor(l.attention_conv->base_kernel),
                  p.linear({l.attention_conv->kernel_projection})},
                 {p.tensor(l.mlp_conv->base_kernel), p.linear({l.mlp_conv->kernel_projection})}});
        }
    }
    if (w.vision) vision = p.vision(*w.vision);
}

} // namespace ninfer::models::glm5next::execution
