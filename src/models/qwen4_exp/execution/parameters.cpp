#include "models/qwen4_exp/execution/parameters.h"

namespace ninfer::models::qwen4_exp::execution {
namespace {
class Prepare {
public:
    Prepare(const ModelShard& model, const ModelProfile& profile)
        : model_(model), profile_(profile) {}

    LinearParameters linear(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(model_.input(id));
        return ops::prepare_linear_weight(inputs);
    }

    Tensor tensor(WeightId id) const { return parameter_tensor(model_.weight(id)); }

    Tensor hc_norm(WeightId id) const {
        const auto& c = profile_.config.text;
        return tensor(id).view({dimension(c.hidden), dimension(c.hc.streams)});
    }

    ops::ProjectionWeights projection(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(model_.input(id));
        return ops::prepare_projection_weights(inputs);
    }

    HcParameters hc(const HcWeights& w) const {
        return {hc_norm(w.norm),
                {linear({w.down, w.injection.value()}).weight, linear({w.up}).weight}};
    }

    FinalHcParameters final_hc(const HcWeights& w) const {
        return {hc_norm(w.norm), {linear({w.down}).weight, linear({w.up}).weight}};
    }

    ops::QsaKvPrepareParameters kv_prepare(const QsaWeights& q) const {
        const auto& c = profile_.config.text;
        return {tensor(q.key_norm),    tensor(q.index_key_norm), c.norm_epsilon,
                c.qsa.rope_theta,      c.qsa.rotary_dim,         c.qsa.rope_sections,
                c.qsa.rope_interleaved};
    }

    MtpHistoryParameters history(const BlockWeights& w) const {
        const auto& q = std::get<QsaWeights>(w.mixer);
        return {{linear({w.attention_hc.down}).weight, linear({w.attention_hc.up}).weight},
                {linear({q.key, q.value, q.index_key}), kv_prepare(q)}};
    }

    BlockParameters block(const BlockWeights& w, const BlockProfile& profile, bool mtp) const {
        BlockParameters out;
        out.attention_hc = hc(w.attention_hc);
        out.moe_hc       = hc(w.moe_hc);
        if (const auto* g = std::get_if<GdnWeights>(&w.mixer)) {
            out.mixer = GdnParameters{
                projection({g->query, g->key, g->value, g->z}),
                ops::prepare_gdn_control_proj_weights(model_.input(g->a), model_.input(g->b)),
                tensor(g->a_log),
                tensor(g->dt_bias),
                tensor(g->convolution),
                tensor(g->norm),
                linear({g->output})};
        } else {
            const auto& q = std::get<QsaWeights>(w.mixer);
            QsaParameters p;
            if (mtp)
                p.input = linear({q.key, q.value, q.index_key, q.query, q.gate, q.index_query});
            else
                p.input = MainQsaProjection{linear({q.query, q.gate}),
                                            linear({q.key, q.value, q.index_query, q.index_key})};
            p.prepare = {kv_prepare(q), tensor(q.query_norm), tensor(q.index_query_norm)};
            p.output  = linear({q.output});
            out.mixer = std::move(p);
        }
        const auto& m               = w.moe;
        out.moe.profile             = profile.moe.math;
        out.moe.bank                = m.bank;
        out.moe.activation_divisors = profile.moe.activation_divisors;
        out.moe.control             = linear({m.router, m.shared_score}).weight;
        out.moe.shared = {linear({m.shared_gate, m.shared_up}), linear({m.shared_down})};
        if (const auto* bank = model_.resident_experts(m.bank)) {
            out.moe.resident = bank->view();
            out.moe.subset   = {ops::ExpertSubsetKind::MaskOnes, bank->mask()};
        } else {
            const auto& source = model_.expert_banks()[m.bank.index];
            if (source.gate_up.parent->device.type() == DeviceType::CUDA)
                out.moe.resident = device_expert_bank(source.gate_up, source.down);
        }
        return out;
    }
private:
    const ModelShard& model_;
    const ModelProfile& profile_;
};
} // namespace

Parameters::Parameters(const ModelShard& model, std::shared_ptr<const ModelProfile> profile)
    : model(model), profile(std::move(profile)) {
    if (!this->profile) throw std::invalid_argument("Parameters require an execution profile");
    const Prepare p(model, *this->profile);
    const auto& w  = model.definition().weights();
    text.embedding = native_weight(model.weight(w.text.embedding).view);
    text.head      = p.linear({w.text.head});
    text.final_hc  = p.final_hc(w.text.final_hc);
    for (std::size_t i = 0; i < w.text.layers.size(); ++i)
        text.layers.push_back(p.block(w.text.layers[i], this->profile->text.layers.at(i), false));
    const auto& ple = w.text.ple;
    text.ple        = {p.linear({ple.key, ple.value}),
                       {p.hc_norm(ple.norm_key), p.hc_norm(ple.norm_query), p.hc_norm(ple.norm_conv),
                        this->profile->config.text.norm_epsilon},
                       p.tensor(ple.convolution)};
    if (w.mtp) {
        const auto& m = *w.mtp;
        mtp           = {native_weight(model.weight(m.embedding).view),
                         p.linear({m.head}),
                         p.linear({m.fc_embedding}),
                         p.linear({m.fc_hidden}),
                         p.tensor(m.embedding_norm),
                         p.tensor(m.hidden_norm),
                         p.block(m.layer, this->profile->mtp.value().layer, true),
                         p.final_hc(m.final_hc),
                         p.history(m.layer)};
    }
    if (w.vision) vision = qwen::prepare_vision(model.weight_data(), *w.vision);
}
} // namespace ninfer::models::qwen4_exp::execution
