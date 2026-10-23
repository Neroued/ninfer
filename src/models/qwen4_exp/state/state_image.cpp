#include "models/qwen4_exp/state/state_image.h"

namespace ninfer::models::qwen4_exp {
StateLayout plan_state_image(const Config& config, const GdnConfig& gdn) {
    const auto& c = config.text;
    StateImageBuilder b;
    StateLayout out;
    out.gdn        = plan_linear_attention_state(b, {c.gdn_layers, dimension(gdn.conv_channels()),
                                                     dimension(gdn.conv_width - 1),
                                                     dimension(gdn.value_heads), dimension(gdn.value_dim),
                                                     dimension(gdn.key_dim), DType::BF16});
    const auto qsa = [&](std::uint32_t layers) {
        return QsaTailLayout{
            b.add(DType::BF16,
                  {dimension(c.qsa.index_key_width()), dimension(c.qsa.compression - 1)}, layers,
                  "QSA raw tail"),
            b.add(DType::I32, {3}, layers, "QSA block coordinates")};
    };
    out.main_qsa = qsa(c.qsa_layers);
    out.ple_history =
        b.add(DType::BF16, {dimension(c.residual_width()), dimension(c.ple.history_columns())}, 1,
              "PLE history");
    out.sample_hidden = b.add(DType::BF16, {dimension(c.hidden)}, 1, "Main sample hidden");
    if (config.mtp) {
        out.conditioning =
            b.add(DType::BF16, {dimension(c.residual_width())}, 1, "Main MTP conditioning");
        out.mtp_qsa = qsa(1);
    }
    out.images = b.finish();
    return out;
}

StateViews::StateViews(MemorySpan backing, const StateLayout& layout)
    : layout_(layout), storage_(backing, layout.images), gdn_(storage_, layout.gdn) {}

ops::QsaTailStateView StateViews::main_tail(std::uint32_t layer, Tensor offsets) const {
    return {storage_.field(layout_.main_qsa.raw_keys, offsets, layer),
            storage_.field(layout_.main_qsa.first_rope, offsets, layer)};
}

ops::QsaTailStateView StateViews::mtp_tail(Tensor offsets) const {
    const auto& tail = layout_.mtp_qsa.value();
    return {storage_.field(tail.raw_keys, offsets), storage_.field(tail.first_rope, offsets)};
}

ops::PleConvStateView StateViews::ple(Tensor offsets) const {
    return {storage_.field(layout_.ple_history, offsets)};
}

StateFieldView StateViews::sample_hidden(Tensor offsets) const {
    return storage_.field(layout_.sample_hidden, offsets);
}

Tensor StateViews::sample_hidden(std::int64_t offset) const {
    return storage_.tensor(layout_.sample_hidden, offset);
}

StateFieldView StateViews::conditioning(Tensor offsets) const {
    return storage_.field(layout_.conditioning.value(), offsets);
}

Tensor StateViews::conditioning(std::int64_t offset) const {
    return storage_.tensor(layout_.conditioning.value(), offset);
}

} // namespace ninfer::models::qwen4_exp
