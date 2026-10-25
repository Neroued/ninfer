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
                tensor(w.dt_bias),
                tensor(w.norm)};
    }

    MlaParameters mla(const MlaWeights& w) const {
        const auto& c = model_.definition().config().text;
        MlaParameters out;
        const std::array projections{
            linear({w.latent, w.index_key, w.pool_gate, w.query_a, w.index_weight}),
            linear({w.latent, w.index_key, w.pool_gate, w.query_a}),
            linear({w.latent, w.index_key, w.pool_gate})};
        for (std::size_t i = 0; i < out.prepare.size(); ++i) {
            const auto kind = static_cast<ops::KpoolMlaPrepareProjection>(i);
            out.prepare[i]  = {
                kind,
                projections[i].weight,
                kind == ops::KpoolMlaPrepareProjection::History ? Tensor{} : tensor(w.query_norm),
                tensor(w.latent_norm),
                tensor(w.index_norm),
                tensor(w.index_bias),
                tensor(w.pool_ape),
                c.norm_epsilon,
                1e-6F};
        }
        out.query_index = linear({w.query_b, w.index_query});
        out.query       = linear({w.query_b});
        out.output      = linear({w.output});
        out.key_up      = tensor(w.key_up);
        out.value_up    = tensor(w.value_up);
        return out;
    }

    ops::SwiGLUWeights dense(const DenseWeights& w) const {
        return {linear({w.gate, w.up}), linear({w.down})};
    }

    MoeParameters moe(const MoeWeights& w, const MoeProfile& profile) const {
        const auto& bank = model_.experts(w.bank);
        return {{linear({w.router}).weight,
                 tensor(w.correction_bias),
                 {bank.view(), profile.activation_divisors},
                 dense(w.shared)},
                {ops::ExpertSubsetKind::MaskOnes, bank.mask()}};
    }

    BlockParameters block(const BlockWeights& w, const BlockProfile& profile) const {
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
            out.ffn = moe(std::get<MoeWeights>(w.ffn), std::get<MoeProfile>(profile.ffn));
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
        text.layers.push_back(p.block(w.text.layers[i], this->profile->text.layers.at(i)));
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
                         p.moe(m.moe, this->profile->mtp->moe)};
    }
    if (w.vision) vision = p.vision(*w.vision);
}

} // namespace ninfer::models::glm5next::execution
