#include "models/qwen3_5/execution/parameters.h"

#include "core/weight_view.h"
#include "ninfer/ops/linear_split.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    Prepare(const ModelShard& model, const ModelProfile& profile)
        : model_(model), profile_(profile) {}

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] { return parameter_tensor(bound); });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(model_.weight(w.gate).name,
                             [&] {
                                 return ops::prepare_linear_swiglu_weight(model_.input(w.gate),
                                                                          model_.input(w.up));
                             }),
                linear(w.down)};
    }

    FfnParameters ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
        return moe(std::get<MoeWeights>(w.ffn));
    }

    moe::Parameters moe(const MoeWeights& w) const {
        return with_context(model_.weight(w.router).name, [&] {
            const auto& source = model_.expert_banks()[w.bank.index];
            ops::ExpertInputUsage gate_up, down;
            for (const auto& expert : w.experts) {
                gate_up.include(model_.input(expert.gate));
                gate_up.include(model_.input(expert.up));
                down.include(model_.input(expert.down));
            }
            moe::Parameters out;
            const std::array control{model_.input(w.router), model_.input(w.shared_score)};
            out.control = ops::prepare_linear_weight(control).weight;
            out.shared  = {ops::prepare_linear_swiglu_weight(model_.input(w.shared.gate),
                                                             model_.input(w.shared.up)),
                           linear(w.shared.down)};
            out.profile = {
                ops::routed_swiglu_profile(
                    source.gate_up, source.down,
                    dimension(std::get<MoeConfig>(profile_.config.text.ffn).num_experts_per_tok),
                    gate_up.policy(), down.policy()),
                ops::swiglu_profile(out.shared)};
            out.bank                = w.bank;
            out.activation_divisors = {gate_up.activation_divisor(source.gate_up.layout.format),
                                       down.activation_divisor(source.down.layout.format)};
            if (source.gate_up.parent->device.type() == DeviceType::CUDA) {
                out.resident = device_expert_bank(source.gate_up, source.down);
            }
            return out;
        });
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            const std::array inputs{model_.input(a->query), model_.input(a->key),
                                    model_.input(a->gate), model_.input(a->value)};
            const std::array rows{
                dimension(inputs[0].weight.shape[0]), dimension(inputs[1].weight.shape[0]),
                dimension(inputs[2].weight.shape[0]), dimension(inputs[3].weight.shape[0])};
            auto projection = ops::prepare_projection_weights(inputs);
            (void)ops::linear_split_workspace_capacity_bytes(projection, rows, 1, 1);
            out.mixer = AttentionParameters{std::move(projection), rows, tensor(a->query_norm),
                                            tensor(a->key_norm), linear(a->output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            const std::array inputs{model_.input(g.query), model_.input(g.key),
                                    model_.input(g.value), model_.input(g.z)};
            const std::array rows{dimension(inputs[0].weight.shape[0] + inputs[1].weight.shape[0] +
                                            inputs[2].weight.shape[0]),
                                  dimension(inputs[3].weight.shape[0])};
            auto projection = ops::prepare_projection_weights(inputs);
            (void)ops::linear_split_workspace_capacity_bytes(projection, rows, 1, 1);
            out.mixer =
                GdnParameters{std::move(projection),
                              rows,
                              ops::prepare_gdn_control_proj_weights(model_.input(g.a_projection),
                                                                    model_.input(g.b_projection)),
                              tensor(g.a_log),
                              tensor(g.dt_bias),
                              tensor(g.convolution),
                              tensor(g.norm),
                              linear(g.output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    ops::MoeExecutionHints prefetch(const ops::ProjectionWeights& projection,
                                    WeightId query) const {
        const auto* single = std::get_if<LinearParameters>(&projection);
        const auto& weight =
            single ? single->weight : std::get<ops::PairedProjectionWeights>(projection).first;
        const auto& geometry = model_.weight(query).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{model_.input(a.query), model_.input(a.key), model_.input(a.gate),
                                model_.input(a.value)};
        MtpParameters out;
        out.input_projection       = linear(w.input_projection);
        out.embedding_norm         = tensor(w.embedding_norm);
        out.hidden_norm            = tensor(w.hidden_norm);
        out.input_norm             = tensor(w.layer.input_norm);
        out.post_attention_norm    = tensor(w.layer.post_attention_norm);
        out.final_norm             = tensor(w.final_norm);
        out.projection.packed      = ops::prepare_linear_weight(inputs);
        out.projection.output_rows = {
            dimension(inputs[0].weight.shape[0]), dimension(inputs[1].weight.shape[0]),
            dimension(inputs[2].weight.shape[0]), dimension(inputs[3].weight.shape[0])};
        if (profile_.config.text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        } else {
            (void)ops::linear_split_workspace_capacity_bytes(out.projection.packed,
                                                             out.projection.output_rows, 1, 1);
        }
        out.query_norm  = tensor(a.query_norm);
        out.key_norm    = tensor(a.key_norm);
        out.output      = linear(a.output);
        out.ffn         = ffn(w.layer);
        out.output_head = linear(w.output_head_use);
        return out;
    }

    qwen::VisionParameters vision(const qwen::VisionWeights& weights) const {
        return qwen::prepare_vision(model_.weight_data(), weights);
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(
                with_context(std::string(profile_.options.speculative_component()) + "/layers/" +
                                 std::to_string(i),
                             [&] {
                                 const auto& layer = w.layers[i];
                                 const auto& a     = layer.attention;
                                 DraftBlockParameters result;
                                 result.input_norm          = tensor(layer.input_norm);
                                 result.post_attention_norm = tensor(layer.post_attention_norm);
                                 const std::array inputs{model_.input(a.query), model_.input(a.key),
                                                         model_.input(a.value)};
                                 result.query_key_value = ops::prepare_linear_weight(inputs);
                                 const std::array rows{dimension(inputs[0].weight.shape[0]),
                                                       dimension(inputs[1].weight.shape[0]),
                                                       dimension(inputs[2].weight.shape[0])};
                                 (void)ops::linear_split_workspace_capacity_bytes(
                                     result.query_key_value, rows, 1, 1);
                                 result.context_key   = linear(a.context_key);
                                 result.context_value = linear(a.context_value);
                                 result.query_norm    = tensor(a.query_norm);
                                 result.key_norm      = tensor(a.key_norm);
                                 result.output        = linear(a.output);
                                 result.mlp           = dense(layer.mlp);
                                 if (layer.attention_conv) {
                                     result.attention_conv = convolution(*layer.attention_conv);
                                 }
                                 if (layer.mlp_conv) {
                                     result.mlp_conv = convolution(*layer.mlp_conv);
                                 }
                                 return result;
                             }));
        }
        if (w.selector) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

private:
    const ModelShard& model_;
    const ModelProfile& profile_;
};

} // namespace

Parameters::Parameters(const ModelShard& source, std::shared_ptr<const ModelProfile> profile)
    : model(source), profile(std::move(profile)) {
    if (!this->profile) throw std::invalid_argument("Parameters require an execution profile");
    const Prepare prepare(model, *this->profile);
    const auto& w        = model.definition().weights();
    text.token_embedding = native_weight(model.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    text.final_norm      = prepare.tensor(w.text.final_norm);
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i]); }));
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft) {
        draft = with_context(std::string(this->profile->options.speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids) { proposal->token_ids = prepare.tensor(*w.proposal->token_ids); }
    }
}

} // namespace ninfer::models::qwen3_5::execution
