#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

enum class BackfillClass : std::uint8_t {
    None,
    Persistent,
};

struct ActiveAdmissionSnapshot {
    std::uint64_t request_id     = 0;
    std::uint64_t backfill_epoch = 0;
    BackfillClass backfill_class = BackfillClass::None;
};

// Logical half of a blocked-head reservation. Program owns every physical quantity and mints a
// fresh proof for each borrower; Scheduler freezes only the incumbent identities whose eventual
// release is allowed to make the FIFO head runnable.
struct AdmissionProtection {
    std::uint64_t epoch_id        = 0;
    std::uint64_t head_request_id = 0;
    ProgramResourceRevision resource_revision;
    std::array<std::uint64_t, kMaximumConcurrency> donor_ids{};
    std::size_t donor_count = 0;
};

[[nodiscard]] AdmissionProtection
make_admission_protection(std::uint64_t epoch_id, std::uint64_t head_request_id,
                          ProgramResourceRevision resource_revision,
                          std::span<const ActiveAdmissionSnapshot> active);

// Revalidates the frozen donor partition after Program topology changed. Current-epoch persistent
// borrowers are never promoted to donors, so repeated backfill cannot move the head's frontier.
void rebind_admission_protection(AdmissionProtection& protection,
                                 std::span<const ActiveAdmissionSnapshot> active,
                                 ProgramResourceRevision resource_revision);

[[nodiscard]] bool
protection_has_live_donor(const AdmissionProtection& protection,
                          std::span<const ActiveAdmissionSnapshot> active) noexcept;

[[nodiscard]] bool
persistent_backfill_is_authorized(const AdmissionProtection& protection,
                                  std::uint64_t candidate_request_id,
                                  std::span<const ActiveAdmissionSnapshot> active,
                                  ProgramResourceRevision program_proof_revision) noexcept;

} // namespace ninfer::runtime
