#pragma once

#include "runtime/contract/resources.h"
#include "runtime/contract/placement.h"

#include <memory>

namespace ninfer::runtime {

// Session-scoped exact semantic endpoint. A handle is borrowed; Native owns its lifetime.
struct PrefixId {
    const void* owner                          = nullptr;
    std::uint32_t index                        = 0;
    std::uint64_t generation                   = 0;
    friend bool operator==(PrefixId, PrefixId) = default;
};

// Non-owning demand evidence survives eviction without retaining token contents.
struct PrefixKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier = 0, domain = 0;
    friend auto operator<=>(PrefixKey, PrefixKey) = default;
};

// Besides per-lane captures and execution: admission, service/relocation, and synchronous
// current/projected views. Startup and Native use this same control-lifetime bound.
inline constexpr std::size_t kRecoveryAuxiliaryViews        = 5;
inline constexpr std::size_t kRecoveryWorkMetadataLimit     = 128;
inline constexpr std::size_t kRecoveryEvaluationEntryLimit  = 256;
inline constexpr std::size_t kRecoveryPricedDescriptorLimit = 112;

struct RecoveryRestoreCost {
    PlacementId placement;
    std::uint64_t nanoseconds;
    std::uint64_t replay_nanoseconds;
    friend bool operator==(const RecoveryRestoreCost&, const RecoveryRestoreCost&) = default;
};

// A synchronous relation projection. Equal endpoints use the same representative;
// parent identifies the nearest strict ancestor in the supplied set.
struct PrefixRelation {
    std::uint32_t representative = UINT32_MAX;
    std::uint32_t parent         = UINT32_MAX;
};

// Reconstruction geometry only; no token/coordinate storage or physical leases.
// Request-target quotes borrow RequestPlan geometry; the caller keeps that plan alive.
// Catalog quotes own their compact geometry and may outlive checkpoint retirement.
class RecoveryWork {
public:
    virtual ~RecoveryWork()                                              = default;
    [[nodiscard]] virtual PrefillWork rebuild(std::uint32_t begin) const = 0;
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
    PrefixId prefix;
    std::uint32_t frontier = 0;
    std::shared_ptr<const RecoveryWork> work;
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
