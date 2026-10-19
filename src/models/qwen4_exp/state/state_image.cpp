#include "models/qwen4_exp/state/state_image.h"

namespace ninfer::models::qwen4_exp {
StateLayout plan_state_image(LayoutBuilder& backing, const Config& config, const GdnConfig& gdn,
                             std::int32_t slots, bool device_ple) {
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
    if (device_ple)
        out.ple_tokens = b.add(DType::I32, {dimension(c.ple.ngram_size - 1)}, 1, "PLE token tail");
    out.sample_hidden = b.add(DType::BF16, {dimension(c.hidden)}, 1, "Main sample hidden");
    if (config.mtp) {
        out.conditioning =
            b.add(DType::BF16, {dimension(c.residual_width())}, 1, "Main MTP conditioning");
        out.mtp_qsa = qsa(1);
    }
    out.images = b.finish(backing, slots);
    return out;
}

StateViews::StateViews(MemorySpan backing, const StateLayout& layout)
    : layout_(layout), storage_(backing, layout.images), gdn_(storage_, layout.gdn) {}

ops::QsaTailStateView StateViews::main_tail(std::uint32_t layer, Tensor slots) const {
    return {storage_.field(layout_.main_qsa.raw_keys, layer),
            storage_.field(layout_.main_qsa.first_rope, layer), slots};
}

ops::QsaTailStateView StateViews::mtp_tail(Tensor slots) const {
    const auto& tail = layout_.mtp_qsa.value();
    return {storage_.field(tail.raw_keys), storage_.field(tail.first_rope), slots};
}

ops::PleConvStateView StateViews::ple(Tensor slots) const {
    return {storage_.field(layout_.ple_history), slots};
}

Tensor StateViews::sample_hidden() const { return storage_.field(layout_.sample_hidden); }

Tensor StateViews::conditioning() const { return storage_.field(layout_.conditioning.value()); }

} // namespace ninfer::models::qwen4_exp
