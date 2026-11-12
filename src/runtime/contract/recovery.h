#pragma once

#include "runtime/contract/resources.h"
#include "runtime/contract/placement.h"

#include <memory>

namespace ninfer::runtime {

// Besides per-lane captures and execution: admission, service/relocation, and synchronous
// current/projected views. Startup and Native use this same control-lifetime bound.
inline constexpr std::size_t kRecoveryAuxiliaryViews        = 5;
inline constexpr std::size_t kRecoveryIdentityMetadataLimit = 128;
inline constexpr std::size_t kRecoveryEvaluationEntryLimit  = 256;
inline constexpr std::size_t kRecoveryPricedDescriptorLimit = 64;

struct RecoveryRestoreCost {
    PlacementId placement;
    std::uint64_t nanoseconds;
    std::uint64_t replay_nanoseconds;
    friend bool operator==(const RecoveryRestoreCost&, const RecoveryRestoreCost&) = default;
};

// Owning semantic metadata only: no State/KV lease, execution address, or media payload.
// Native supplies exact identity proofs and its primary reconstruction work.
class RecoveryIdentity {
public:
    RecoveryIdentity(const void* owner, std::uint32_t frontier, std::uint32_t domain)
        : owner(owner), frontier(frontier), domain(domain) {}

    virtual ~RecoveryIdentity() = default;
    const void* const owner;
    const std::uint32_t frontier, domain;

    [[nodiscard]] bool prefix_of(const RecoveryIdentity& other) const {
        return owner == other.owner && domain == other.domain && frontier <= other.frontier &&
               digest(frontier) == other.digest(frontier) && exact_prefix(other);
    }

    [[nodiscard]] virtual std::array<std::uint64_t, 2> digest(std::uint32_t) const = 0;
    [[nodiscard]] virtual PrefillWork rebuild(std::uint32_t begin) const           = 0;
private:
    [[nodiscard]] virtual bool exact_prefix(const RecoveryIdentity&) const = 0;
};

struct RecoveryRestore {
    PlacementId placement;
    std::vector<ContextTransferRequirement> transfers;
    // Additional preparation when this source is replayed to a later frontier. Same-frontier
    // recovery does not write its KV tail. This quotes the current directory references;
    // retiring their last deeper owner can make a later, refreshed quote cheaper.
    std::vector<ContextTransferRequirement> replay_transfers;
};

struct RecoveryDescription {
    std::shared_ptr<const RecoveryIdentity> identity;
    PlacementId placement;
    std::vector<RecoveryRestore> restores;
};

struct RecoveryRouteWork {
    PrefillWork rebuild;
    std::vector<ContextTransferRequirement> transfers;
};

struct PauseCostQuote {
    std::vector<ContextTransferRequirement> work;
    std::vector<RecoveryRouteWork> recovery;
    // An operation without a matching static coefficient cannot be priced as free.
    bool unpriced_work = false;
};

struct CaptureCostQuote {
    std::shared_ptr<const RecoveryDescription> retained;
    std::vector<ContextTransferRequirement> work;
};

template <class Handle>
struct RecoveryPoint {
    Handle handle;
    std::shared_ptr<const RecoveryDescription> description;
};

// Evaluation-local IDs never alias a recycled native checkpoint slot.
struct RecoveryHandle {
    std::uint32_t index                                    = 0;
    friend bool operator==(RecoveryHandle, RecoveryHandle) = default;
};

} // namespace ninfer::runtime
