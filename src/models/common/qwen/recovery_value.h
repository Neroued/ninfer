#pragma once

#include "models/common/context/recovery_value.h"
#include "models/common/qwen/prefix_identity.h"
#include "models/common/vision/window.h"

namespace ninfer::models::qwen {

inline runtime::PrefillWork recovery_work(std::span<const models::frontend::VisionItem> items,
                                          std::uint32_t begin, std::uint32_t end,
                                          std::uint32_t chunk, std::uint32_t feature_capacity) {
    return vision::recovery_work(
        items, [](const auto& item) -> const auto& { return item; }, begin, end, chunk,
        feature_capacity);
}

class RecoveryIdentity final : public runtime::RecoveryIdentity {
public:
    RecoveryIdentity(context::MetadataBudget& budget, const void* scope,
                     std::shared_ptr<const void> backing, RecoveryIdentityView view,
                     std::uint32_t chunk, std::uint32_t feature_capacity)
        : runtime::RecoveryIdentity(scope, view.frontier, view.domain), account_(budget),
          backing_(std::move(backing)), view_(view), chunk_(chunk), capacity_(feature_capacity) {
        account_.acquire(sizeof(*this));
    }

    std::array<std::uint64_t, 2> digest(std::uint32_t f) const override {
        return view_.digests.at(f);
    }

    runtime::PrefillWork rebuild(std::uint32_t begin) const override {
        return recovery_work(view_.positions.vision_items(), begin, frontier, chunk_, capacity_);
    }
private:
    bool exact_prefix(const runtime::RecoveryIdentity& other) const override {
        const auto* value = dynamic_cast<const RecoveryIdentity*>(&other);
        return value && recovery_prefix_matches(view_, value->view_);
    }

    context::MetadataAccount account_;
    std::shared_ptr<const void> backing_;
    RecoveryIdentityView view_;
    std::uint32_t chunk_, capacity_;
};

static_assert(sizeof(RecoveryIdentity) <= runtime::kRecoveryIdentityMetadataLimit);

} // namespace ninfer::models::qwen
