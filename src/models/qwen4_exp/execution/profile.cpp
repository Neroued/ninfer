#include "models/qwen4_exp/execution/profile.h"
#include "models/qwen4_exp/definition.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/linear_split.h"

#include <array>

namespace ninfer::models::qwen4_exp::execution {
namespace {
class Describe {
public:
    explicit Describe(const ModelDefinition& source) : source_(source) {}

    ops::LinearProfile linear(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) inputs.push_back(source_.input(id));
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
        for (const auto id : ids) inputs.push_back(source_.input(id));
        return ops::describe_projection_weights(inputs);
    }

    std::int32_t rows(WeightId id) const { return dimension(source_.input(id).weight.shape[0]); }

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
            profile.output_rows = {rows(q.query), rows(q.key),         rows(q.gate),
                                   rows(q.value), rows(q.index_query), rows(q.index_key)};
            if (mtp)
                profile.input =
                    linear({q.query, q.key, q.gate, q.value, q.index_query, q.index_key});
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
        gu.include(source_.input(m.gate));
        gu.include(source_.input(m.up));
        down.include(source_.input(m.down));
        out.moe = {
            {ops::routed_swiglu_profile(bank.gate_up.layout, bank.down.layout,
                                        source_.config().text.moe.top_k, gu.policy(),
                                        down.policy()),
             ops::swiglu_profile(linear({m.shared_gate, m.shared_up}), linear({m.shared_down}))},
            m.bank,
            {gu.activation_divisor(bank.gate_up.layout.format),
             down.activation_divisor(bank.down.layout.format)}};
        return out;
    }
private:
    const ModelDefinition& source_;
};

std::shared_ptr<const ModelProfile> describe(const ModelDefinition& source) {
    auto out           = std::make_shared<ModelProfile>();
    out->config        = source.config();
    out->options       = source.options();
    out->public_tokens = source.resources().public_token_count;
    const Describe d(source);
    const auto& w = source.weights();
    for (const auto& layer : w.text.layers) out->text.layers.push_back(d.block(layer, false));
    out->text.final_hc = d.final_hc(w.text.final_hc);
    out->text.ple      = d.linear({w.text.ple.key, w.text.ple.value});
    out->text.head     = d.linear({w.text.head});
    if (w.mtp) {
        const auto& m = *w.mtp;
        out->mtp      = {d.linear({m.fc_embedding}), d.linear({m.fc_hidden}), d.linear({m.head}),
                         d.block(m.layer, true), d.final_hc(m.final_hc)};
    }
    if (w.vision)
        out->vision =
            qwen::describe_vision(*w.vision, [&](WeightId id) { return source.input(id); });
    for (const auto& b : source.expert_banks())
        out->banks.push_back({b.gate_up.layout, b.down.layout});
    return out;
}

} // namespace

std::shared_ptr<const ModelProfile> describe_model(const ModelDefinition& source) {
    return describe(source);
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
