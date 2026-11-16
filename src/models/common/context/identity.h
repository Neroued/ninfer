#pragma once

#include "models/common/context/directory/metadata.h"

namespace ninfer::models::context {

// Immutable input backing can be shared by several checkpoints and by decisions that
// outlive their retirement. Its cache lease owns one charge, independent of alias count.
// Request/active copies have a separate lifetime; copying an identity never copies a lease.
class IdentityMetadata {
public:
    IdentityMetadata() = default;

    IdentityMetadata(const IdentityMetadata&) noexcept {}

    IdentityMetadata& operator=(const IdentityMetadata&) noexcept {
        retained_.reset();
        return *this;
    }

    template <class Identity>
    [[nodiscard]] std::shared_ptr<const Identity>
    retain(MetadataBudget& budget, const std::shared_ptr<const Identity>& identity) const {
        if (const auto existing = retained_.lock())
            return std::static_pointer_cast<const Identity>(existing);
        const auto bytes = sizeof(Owner<Identity>) + identity->metadata_bytes();
        if (bytes > budget.available()) return {};
        // A separate object allocation lets the charge retire with the last strong lease.
        // The backing's weak cache may outlive the lease while its request remains active.
        auto owner = std::shared_ptr<Owner<Identity>>(new Owner<Identity>(budget, identity, bytes));
        std::shared_ptr<const Identity> result(std::move(owner), identity.get());
        retained_ = result;
        return result;
    }

    template <class Identity>
    [[nodiscard]] static bool can_copy(const MetadataBudget& budget, const Identity& identity) {
        return can_retain<Identity>(budget, identity.metadata_bytes());
    }

    template <class Identity>
    [[nodiscard]] static bool can_retain(const MetadataBudget& budget, std::size_t bytes) {
        return sizeof(Owner<Identity>) + bytes <= budget.available();
    }

    template <class Identity>
    [[nodiscard]] static std::size_t cache_bytes(std::size_t tokens, std::size_t concurrency,
                                                 std::size_t text_bytes_per_token) {
        // This finite identity-cache capacity covers one retained text identity and its
        // replacement/continuation preparation per resident lane. It does not promise to
        // retain every checkpoint or every in-flight decision generation at maximum length.
        // Actual capacities (including media metadata) compete within this same byte pool;
        // a full pool rejects optional saves while required recovery borrows paid leases.
        return 2 * concurrency *
               (sizeof(Identity) + sizeof(Owner<Identity>) + (tokens + 1) * text_bytes_per_token);
    }

private:
    template <class Identity>
    struct Owner {
        MetadataAccount account;
        std::shared_ptr<const Identity> identity;

        Owner(MetadataBudget& budget, std::shared_ptr<const Identity> value, std::size_t bytes)
            : account(budget), identity(std::move(value)) {
            account.acquire(bytes);
        }
    };

    mutable std::weak_ptr<const void> retained_;
};

} // namespace ninfer::models::context
