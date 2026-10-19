#pragma once

#include "core/linear_attention_state.h"
#include "models/qwen4_exp/state/continuation.h"
#include "models/qwen4_exp/config.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/ple.h"

namespace ninfer::models::qwen4_exp {
struct QsaTailLayout {
    StateField raw_keys, first_rope;
};

struct StateLayout {
    StateImageLayout images;
    LinearAttentionStateLayout gdn;
    QsaTailLayout main_qsa;
    StateField ple_history, sample_hidden;
    std::optional<StateField> conditioning;
    std::optional<QsaTailLayout> mtp_qsa;
};

[[nodiscard]] StateLayout plan_state_image(LayoutBuilder&, const Config&, const GdnConfig&,
                                           std::int32_t slots);

class StateViews {
public:
    StateViews(MemorySpan, const StateLayout&);

    [[nodiscard]] StateImageStorage& storage() { return storage_; }

    [[nodiscard]] const StateImageStorage& storage() const { return storage_; }

    [[nodiscard]] const LinearAttentionStateView& gdn() const { return gdn_; }

    [[nodiscard]] ops::QsaTailStateView main_tail(std::uint32_t layer, Tensor slots) const;
    [[nodiscard]] ops::QsaTailStateView mtp_tail(Tensor slots) const;
    [[nodiscard]] ops::PleConvStateView ple(Tensor slots) const;
    [[nodiscard]] Tensor sample_hidden() const;
    [[nodiscard]] Tensor conditioning() const;
    // Initializes a reserved physical image; logical publication belongs to the caller.

private:
    StateLayout layout_;
    StateImageStorage storage_;
    LinearAttentionStateView gdn_;
};
} // namespace ninfer::models::qwen4_exp
