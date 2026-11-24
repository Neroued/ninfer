#include "models/common/qwen/vision/parameters.h"
#include "models/common/bindings.h"

#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen {

std::uint64_t VisionConfig::patch_width() const {
    return artifact::checked_mul(artifact::checked_mul(3, temporal_patch_size, "patch channels"),
                                 std::uint64_t(patch_size) * patch_size, "patch width");
}

std::uint64_t VisionConfig::merger_width() const {
    return artifact::checked_mul(std::uint64_t(spatial_merge_size) * spatial_merge_size,
                                 hidden_size, "merger width");
}

VisionWeights bind_vision(Bindings& b, const VisionConfig& config, std::uint32_t output_hidden) {
    const auto h            = config.hidden_size;
    const auto intermediate = config.intermediate_size;
    VisionWeights out;
    out.patch_embedding =
        b.parameter("vision/patch_embedding", {h, config.patch_width()}, {"vision/patch_input"});
    out.patch_embedding_bias = b.direct("vision/patch_embedding_bias", {h});
    out.position_embedding =
        b.direct("vision/position_embedding", {config.num_position_embeddings, h});
    out.layers.reserve(config.depth);
    for (std::uint32_t i = 0; i < config.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.norm1       = {b.direct(p + "norm1_weight", {h}), b.direct(p + "norm1_bias", {h})};
        layer.norm2       = {b.direct(p + "norm2_weight", {h}), b.direct(p + "norm2_bias", {h})};
        layer.query       = b.parameter(p + "attention/query", {h, h}, {p + "attention_input"});
        layer.key         = b.parameter(p + "attention/key", {h, h}, {p + "attention_input"});
        layer.value       = b.parameter(p + "attention/value", {h, h}, {p + "attention_input"});
        layer.query_bias  = b.direct(p + "attention/query_bias", {h});
        layer.key_bias    = b.direct(p + "attention/key_bias", {h});
        layer.value_bias  = b.direct(p + "attention/value_bias", {h});
        layer.output      = b.parameter(p + "attention/output", {h, h}, {p + "attention_output"});
        layer.output_bias = b.direct(p + "attention/output_bias", {h});
        layer.fc1         = b.parameter(p + "mlp/fc1", {intermediate, h}, {p + "mlp_input"});
        layer.fc1_bias    = b.direct(p + "mlp/fc1_bias", {intermediate});
        layer.fc2         = b.parameter(p + "mlp/fc2", {h, intermediate}, {p + "mlp_activation"});
        layer.fc2_bias    = b.direct(p + "mlp/fc2_bias", {h});
        out.layers.push_back(layer);
    }
    const auto merger = config.merger_width();
    out.merger_norm   = {b.direct("vision/merger/norm_weight", {h}),
                         b.direct("vision/merger/norm_bias", {h})};
    out.merger_fc1    = b.parameter("vision/merger/fc1", {merger, merger}, {"vision/merger/input"});
    out.merger_fc1_bias = b.direct("vision/merger/fc1_bias", {merger});
    out.merger_fc2 =
        b.parameter("vision/merger/fc2", {output_hidden, merger}, {"vision/merger/activation"});
    out.merger_fc2_bias = b.direct("vision/merger/fc2_bias", {output_hidden});
    return out;
}

namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class PrepareVision {
public:
    explicit PrepareVision(std::span<const BoundWeight> weights) : weights_(weights) {}

    const BoundWeight& weight(WeightId id) const {
        if (id.index >= weights_.size()) throw std::out_of_range("Vision parameter index");
        return weights_[id.index];
    }

    ops::SingleProjectionWeight linear(WeightId id) const {
        return with_context(weight(id).name,
                            [&] { return ops::prepare_linear_weight(weight_input(weights_, id)); });
    }

    Tensor tensor(WeightId id) const {
        return with_context(weight(id).name, [&] { return parameter_tensor(weight(id)); });
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters prepare(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{weight_input(weights_, layer.query),
                                     weight_input(weights_, layer.key),
                                     weight_input(weights_, layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

private:
    std::span<const BoundWeight> weights_;
};

} // namespace

VisionParameters prepare_vision(std::span<const BoundWeight> weights, const VisionWeights& names) {
    return PrepareVision(weights).prepare(names);
}

} // namespace ninfer::models::qwen
