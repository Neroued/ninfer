#include "models/qwen3_5/execution/profile.h"

#include "models/qwen3_5/definition.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::int32_t dimension(std::uint64_t n) {
    if (!n || n > INT32_MAX) throw std::invalid_argument("execution profile exceeds int32");
    return static_cast<std::int32_t>(n);
}

class Describe {
public:
    explicit Describe(const ModelDefinition& plan) : plan_(plan) {}

    template <class Id>
    ops::LinearProfile linear(Id id) const {
        const auto input = plan_.input(id);
        return ops::describe_linear_weight({&input, 1});
    }

    DenseProfile dense(const DenseWeights& w) const {
        const std::array inputs{plan_.input(w.gate), plan_.input(w.up)};
        return {ops::describe_linear_weight(inputs), linear(w.down)};
    }

    FfnProfile ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) return dense(*d);
        const auto& moe  = std::get<MoeWeights>(w.ffn);
        const auto& bank = plan_.expert_banks()[moe.bank.index];
        ops::ExpertInputUsage gu, dn;
        for (const auto& e : moe.experts) {
            gu.include(plan_.input(e.gate));
            gu.include(plan_.input(e.up));
            dn.include(plan_.input(e.down));
        }
        const auto shared = dense(moe.shared);
        return MoeLayerProfile{
            {ops::routed_swiglu_profile(
                 bank.gate_up.layout, bank.down.layout,
                 dimension(std::get<MoeConfig>(plan_.config().text.ffn).num_experts_per_tok),
                 gu.policy(), dn.policy()),
             ops::swiglu_profile(shared.gate_up, shared.down)},
            moe.bank};
    }

    BlockProfile block(const BlockWeights& w) const {
        BlockProfile out;
        out.ffn = ffn(w);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            const std::array inputs{plan_.input(a->query), plan_.input(a->key),
                                    plan_.input(a->gate), plan_.input(a->value)};
            const std::array rows{
                dimension(inputs[0].weight.shape[0]), dimension(inputs[1].weight.shape[0]),
                dimension(inputs[2].weight.shape[0]), dimension(inputs[3].weight.shape[0])};
            out.mixer =
                AttentionProfile{ops::describe_projection_weights(inputs), rows, linear(a->output)};
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            const std::array inputs{plan_.input(g.query), plan_.input(g.key), plan_.input(g.value),
                                    plan_.input(g.z)};
            const std::array rows{dimension(inputs[0].weight.shape[0] + inputs[1].weight.shape[0] +
                                            inputs[2].weight.shape[0]),
                                  dimension(inputs[3].weight.shape[0])};
            const std::array control{plan_.input(g.a_projection), plan_.input(g.b_projection)};
            out.mixer =
                GdnProfile{ops::describe_projection_weights(inputs),
                           ops::describe_projection_weights(control), rows, linear(g.output)};
        }
        return out;
    }

    MtpProfile mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{plan_.input(a.query), plan_.input(a.key), plan_.input(a.gate),
                                plan_.input(a.value)};
        MtpProfile out;
        out.input_projection       = linear(w.input_projection);
        out.output                 = linear(a.output);
        out.output_head            = linear(w.output_head_use);
        out.ffn                    = ffn(w.layer);
        out.projection.packed      = ops::describe_linear_weight(inputs);
        out.projection.output_rows = {
            dimension(inputs[0].weight.shape[0]), dimension(inputs[1].weight.shape[0]),
            dimension(inputs[2].weight.shape[0]), dimension(inputs[3].weight.shape[0])};
        if (plan_.config().text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        }
        return out;
    }

    qwen::VisionProfile vision(const qwen::VisionWeights& weights) const {
        return qwen::describe_vision(weights, [&](WeightId id) { return plan_.input(id); });
    }

    DraftProfile draft(const DraftWeights& w) const {
        DraftProfile out;
        out.feature_projection = linear(w.feature_projection);
        out.output_head        = linear(w.output_head_use);
        for (const auto& layer : w.layers) {
            out.layers.push_back({linear(layer.attention.output), dense(layer.mlp)});
        }
        if (w.selector) out.selector = SelectorProfile{linear(w.selector->hidden_projection)};
        return out;
    }

private:
    const ModelDefinition& plan_;
};

std::shared_ptr<const ModelProfile> describe_model_impl(const ModelDefinition& plan) {
    auto out           = std::make_shared<ModelProfile>();
    out->config        = plan.config();
    out->options       = plan.options();
    out->public_tokens = plan.resources().text.public_token_count;
    const Describe describe(plan);
    const auto& w         = plan.weights();
    out->text.output_head = describe.linear(w.text.output_head_use);
    for (const auto& layer : w.text.layers) out->text.layers.push_back(describe.block(layer));
    if (w.mtp) out->mtp = describe.mtp(*w.mtp);
    if (w.vision) out->vision = describe.vision(*w.vision);
    if (w.draft) out->draft = describe.draft(*w.draft);
    if (w.proposal)
        out->proposal = ProposalProfile{describe.linear(w.proposal->head), w.proposal->rows};
    return out;
}

} // namespace

std::shared_ptr<const ModelProfile> describe_model(const ModelDefinition& source) {
    return describe_model_impl(source);
}


} // namespace ninfer::models::qwen3_5::execution
