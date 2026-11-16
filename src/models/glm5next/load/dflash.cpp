#include "models/glm5next/load/bindings.h"

namespace ninfer::models::glm5next::loading {

dflash::Weights bind_dflash2(Bindings& b, const Config& config, const TextWeights& main,
                             ProposalHead head) {
    const auto& c = config.text;
    auto out      = dflash::bind_weights(
        b, config.draft.value(),
        {c.hidden, c.vocabulary, static_cast<std::uint32_t>(c.layers.size())}, "dflash2");
    out.token_embedding = b.direct("dflash2/token_embedding", {c.vocabulary, c.hidden});
    require_shared_binding(b.at(out.token_embedding).reference, b.at(main.embedding).reference);
    out.output_head     = bind_speculative_head(b, c, main.head, "dflash2", head);
    out.output_head_use = b.use(out.output_head, "dflash2/final_hidden");
    for (const auto& layer : out.layers) {
        const auto& a = layer.attention;
        require_shared_binding(b.at(a.context_key).reference, b.at(a.key).reference);
        require_shared_binding(b.at(a.context_value).reference, b.at(a.value).reference);
    }
    return out;
}

} // namespace ninfer::models::glm5next::loading
