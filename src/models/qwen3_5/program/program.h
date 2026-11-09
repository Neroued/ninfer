#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
struct ExecutionContext;
} // namespace ninfer

namespace ninfer::models::qwen3_5 {

struct Config;

namespace execution {
class Parameters;
}

namespace detail {
struct CaptureAssessmentImpl;

} // namespace detail

// Read-only diagnostics sampled from the real Program stores.  This is not an accounting input.
// The `_lease_pages` fields split the un-written growth reservation of active requests out of the
// corresponding `device_*_kv_pages` totals: KV that exists versus KV a request may still write.
struct PhysicalUsageSnapshot {
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t device_state_slots            = 0;
    std::uint32_t host_state_slots              = 0;
    std::uint32_t device_main_kv_pages          = 0;
    std::uint32_t device_backend_kv_pages       = 0;
    std::uint32_t device_main_kv_lease_pages    = 0;
    std::uint32_t device_backend_kv_lease_pages = 0;
    std::size_t host_kv_bytes                   = 0;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalUsageSnapshot&,
                                                   const PhysicalUsageSnapshot&) noexcept = default;
};

// Device KV pages a settled lease still needs beyond what its pools can currently reserve.
struct DeviceKVLeaseShortfall {
    std::uint32_t main_pages            = 0; // the full growth step
    std::uint32_t backend_pages         = 0;
    std::uint32_t minimum_main_pages    = 0; // the smallest growth step
    std::uint32_t minimum_backend_pages = 0;
};

struct DeviceKVPages {
    std::uint32_t main    = 0;
    std::uint32_t backend = 0;
};

// Program::disk_cache_save_one (v0.7.2, disk tier of the context cache).
enum class DiskCacheSave : std::uint8_t {
    NothingToSave,
    Saved,
    Aborted,
    Failed,
};

// Program::compact_device_kv (v0.6, idle KV to the Host).
struct DeviceKVCompaction {
    bool ran                    = false;
    std::uint32_t moved_pages   = 0; // Device replicas packed into lower physical pages
    std::int32_t columns_before = 0; // active elastic expert columns
    std::int32_t columns_after  = 0;
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

// Program-minted shortlist metadata. It only narrows catalog inspection; Program still performs
// exact token, position, media and runtime-mode verification before a checkpoint can be selected.
struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefixShortlistKey,
                                                   PrefixShortlistKey) noexcept = default;
};

struct TargetKVRequirement {
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
    std::uint32_t main_pages       = 0;
    std::uint32_t backend_pages    = 0;

    [[nodiscard]] friend constexpr bool operator==(TargetKVRequirement,
                                                   TargetKVRequirement) noexcept = default;
};

struct CheckpointSummary {
    runtime::CheckpointRef ref;
    runtime::CheckpointScope scope = runtime::CheckpointScope::Private;
    PrefixShortlistKey shortlist_key;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
    TargetKVRequirement required_kv;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] friend bool operator==(const CheckpointSummary&,
                                         const CheckpointSummary&) noexcept = default;
};

struct ContinuationSummary {
    std::optional<CheckpointSummary> endpoint;
    std::optional<CheckpointSummary> rewrite;
    std::vector<CheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const ContinuationSummary&,
                                         const ContinuationSummary&) noexcept = default;
};

struct SharedPrefixSummary {
    CheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const SharedPrefixSummary&,
                                         const SharedPrefixSummary&) noexcept = default;
};

namespace detail {

struct SequencePlanImpl;

struct SequencePlannerImpl;

struct AdmissionCandidateImpl;

struct CapturePressureCandidateImpl;

struct RequestBasePlanImpl;

struct PressurePlanningSessionImpl;

class ProgramImpl;

struct RuntimeContractAccess;
} // namespace detail

class SequencePlanner;

class Program;

class PressurePlanningSession;

class CapturePressurePlanningSession;

class CapturePressurePlan;

class CapturePressureCandidate;

// Concrete Qwen execution and resource contracts; model instances supply their own data.
// target selection remains outside this layer and happens once in the closed Engine registry.

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;

    friend class detail::ProgramImpl;
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    // Bytes of the curve's per-rank reservation that `rank` does not allocate, whatever the page
    // count: at tp 2 with Vision, the encode workspace of the rank without the tower.
    [[nodiscard]] std::size_t unallocated_reservation_bytes(int rank) const noexcept;
    // T64: one rank's KV page-pool device bytes (Text + MTP pools) at `main_page_groups` Text
    // pages (the MTP pool has its lanes' draft pages more), the curve's minimum page count, and
    // the number of KV planes (each elastic plane may cost one granule more).
    [[nodiscard]] std::size_t kv_pool_bytes(std::uint32_t main_page_groups) const noexcept;
    [[nodiscard]] std::uint32_t kv_pool_pages() const noexcept;
    [[nodiscard]] std::size_t kv_plane_count() const noexcept;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                 const EngineOptions&);
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;
    [[nodiscard]] std::optional<runtime::PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl> impl_;
};

class AdmissionCandidate {
public:
    AdmissionCandidate(AdmissionCandidate&&) noexcept;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept;
    ~AdmissionCandidate();

    AdmissionCandidate(const AdmissionCandidate&)            = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept;

public:
    // Family-private construction/storage seam. Exact packages expose only the completed alias;
    // Engine code can inspect summary() but not target planning state.
    explicit AdmissionCandidate(std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept;
    std::unique_ptr<detail::AdmissionCandidateImpl> impl_;

    friend class Program;
    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

// Family-private owning wrapper for the physical capture-pressure candidate. It intentionally has
// no request summary or admission API.

class CapturePressureCandidate {
public:
    CapturePressureCandidate(CapturePressureCandidate&&) noexcept;
    CapturePressureCandidate& operator=(CapturePressureCandidate&&) noexcept;
    ~CapturePressureCandidate();

    CapturePressureCandidate(const CapturePressureCandidate&)            = delete;
    CapturePressureCandidate& operator=(const CapturePressureCandidate&) = delete;

public:
    explicit CapturePressureCandidate(
        std::unique_ptr<detail::CapturePressureCandidateImpl> impl) noexcept;
    std::unique_ptr<detail::CapturePressureCandidateImpl> impl_;

    friend class Program;
    friend class PressurePlanningSession;
    friend class CapturePressurePlan;
    friend struct detail::PressurePlanningSessionImpl;
};

// A sealed pressure-only post-state for one active capture. Its payload is Program-private and
// cannot be inspected or executed through the request-admission interface.

class CapturePressurePlan {
public:
    CapturePressurePlan(CapturePressurePlan&&) noexcept            = default;
    CapturePressurePlan& operator=(CapturePressurePlan&&) noexcept = default;
    ~CapturePressurePlan()                                         = default;

    CapturePressurePlan(const CapturePressurePlan&)            = delete;
    CapturePressurePlan& operator=(const CapturePressurePlan&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    CapturePressurePlan(CapturePressureCandidate&& pressure,
                        runtime::ProgramResourceRevision revision) noexcept
        : pressure_(std::move(pressure)), revision_(revision) {}

    CapturePressureCandidate pressure_;
    runtime::ProgramResourceRevision revision_;

    friend class Program;
    friend class PressurePlanningSession;
};

// A sealed Program-owned physical decision.  ResourceManager may retain it and inspect the
// request-level summary, but cannot see allocator quantities, references, reservations, or stage
// deltas.  Start validates the bound Program revision before performing any mutation.

class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept            = default;
    ResourcePlan& operator=(ResourcePlan&&) noexcept = default;
    ~ResourcePlan()                                  = default;

    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return admission_.summary();
    }

    [[nodiscard]] bool needs_transfer() const noexcept { return needs_transfer_; }

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    ResourcePlan(AdmissionCandidate&& admission, runtime::ProgramResourceRevision revision,
                 bool needs_transfer) noexcept
        : admission_(std::move(admission)), revision_(revision), needs_transfer_(needs_transfer) {}

    AdmissionCandidate admission_;
    runtime::ProgramResourceRevision revision_;
    bool needs_transfer_ = false;

    friend class Program;
    friend class PressurePlanningSession;
};

// A Program-minted proof that one FIFO borrower cannot consume the maximum physical entitlement
// reserved for the blocked head.  Common scheduling binds the opaque proof to logical identities
// and a revision; it cannot inspect or reproduce the resource arithmetic.

class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;

    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    explicit PersistentBackfillProof(runtime::ProgramResourceRevision revision) noexcept
        : revision_(revision) {}

    runtime::ProgramResourceRevision revision_;

    friend class Program;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::RuntimeContractAccess;
};

class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    ContinuationHandle& operator=(ContinuationHandle&&)      = delete;
    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess;
};

class SharedPrefixHandle {
public:
    SharedPrefixHandle() noexcept = default;
    ~SharedPrefixHandle()         = default;

    SharedPrefixHandle(SharedPrefixHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    SharedPrefixHandle& operator=(SharedPrefixHandle&&)      = delete;
    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend struct detail::RuntimeContractAccess;
};

class PressureTargetHandle {
public:
    PressureTargetHandle() noexcept = default;

    [[nodiscard]] friend constexpr bool operator==(PressureTargetHandle,
                                                   PressureTargetHandle) noexcept = default;

private:
    const void* session_      = nullptr;
    std::uint32_t generation_ = 0;
    std::uint32_t index_      = 0;

    friend struct detail::PressurePlanningSessionImpl;

    friend class PressurePlanningSession;
};

class PressureConstructionCursor {
public:
    PressureConstructionCursor(PressureConstructionCursor&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)), slot_(other.slot_),
          generation_(other.generation_), release_(other.release_) {}

    PressureConstructionCursor& operator=(PressureConstructionCursor&&)      = delete;
    PressureConstructionCursor(const PressureConstructionCursor&)            = delete;
    PressureConstructionCursor& operator=(const PressureConstructionCursor&) = delete;

    ~PressureConstructionCursor() {
        if (session_) { release_(session_, slot_, generation_); }
    }

private:
    PressureConstructionCursor(const void* session, std::uint32_t slot, std::uint32_t generation,
                               void (*release)(const void*, std::uint32_t, std::uint32_t) noexcept)
        : session_(session), slot_(slot), generation_(generation), release_(release) {}

    const void* session_;
    std::uint32_t slot_;
    std::uint32_t generation_;
    void (*release_)(const void*, std::uint32_t, std::uint32_t) noexcept;

    friend struct detail::PressurePlanningSessionImpl;
};

class AssessedPressureTarget {
public:
    AssessedPressureTarget(AssessedPressureTarget&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          target_index_(other.target_index_), assessment_(other.assessment_),
          assessment_slot_(std::exchange(other.assessment_slot_, 0)),
          assessment_slot_generation_(std::exchange(other.assessment_slot_generation_, 0)),
          release_slot_(std::exchange(other.release_slot_, nullptr)),
          executable_(std::move(other.executable_)),
          capture_executable_(std::move(other.capture_executable_)) {}

    ~AssessedPressureTarget() { reset(); }

    AssessedPressureTarget& operator=(AssessedPressureTarget&& other) noexcept {
        if (this == &other) { return *this; }
        reset();
        session_                    = std::exchange(other.session_, nullptr);
        session_generation_         = std::exchange(other.session_generation_, 0);
        target_index_               = other.target_index_;
        assessment_                 = other.assessment_;
        assessment_slot_            = std::exchange(other.assessment_slot_, 0);
        assessment_slot_generation_ = std::exchange(other.assessment_slot_generation_, 0);
        release_slot_               = std::exchange(other.release_slot_, nullptr);
        executable_                 = std::move(other.executable_);
        capture_executable_         = std::move(other.capture_executable_);
        return *this;
    }

    AssessedPressureTarget(const AssessedPressureTarget&)            = delete;
    AssessedPressureTarget& operator=(const AssessedPressureTarget&) = delete;

    [[nodiscard]] const runtime::PressureTargetAssessment& assessment() const noexcept {
        return assessment_;
    }

private:
    AssessedPressureTarget(const void* session, std::uint32_t session_generation,
                           std::uint32_t target_index, runtime::PressureTargetAssessment assessment,
                           std::uint32_t assessment_slot, std::uint32_t assessment_slot_generation,
                           void (*release_slot)(const void*, std::uint32_t, std::uint32_t) noexcept,
                           std::optional<AdmissionCandidate>&& executable,
                           std::optional<CapturePressureCandidate>&& capture_executable) noexcept
        : session_(session), session_generation_(session_generation), target_index_(target_index),
          assessment_(assessment), assessment_slot_(assessment_slot),
          assessment_slot_generation_(assessment_slot_generation), release_slot_(release_slot),
          executable_(std::move(executable)), capture_executable_(std::move(capture_executable)) {}

    void reset() noexcept {
        if (session_ != nullptr && release_slot_ != nullptr) {
            release_slot_(session_, assessment_slot_, assessment_slot_generation_);
        }
        session_                    = nullptr;
        session_generation_         = 0;
        assessment_slot_            = 0;
        assessment_slot_generation_ = 0;
        release_slot_               = nullptr;
    }

    const void* session_              = nullptr;
    std::uint32_t session_generation_ = 0;
    std::uint32_t target_index_       = 0;
    runtime::PressureTargetAssessment assessment_;
    std::uint32_t assessment_slot_                                            = 0;
    std::uint32_t assessment_slot_generation_                                 = 0;
    void (*release_slot_)(const void*, std::uint32_t, std::uint32_t) noexcept = nullptr;
    std::optional<AdmissionCandidate> executable_;
    std::optional<CapturePressureCandidate> capture_executable_;

    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

class PreparedPressureExpansion {
public:
    PreparedPressureExpansion(PreparedPressureExpansion&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          scratch_generation_(std::exchange(other.scratch_generation_, 0)),
          parent_index_(other.parent_index_), new_canonical_count_(other.new_canonical_count_) {}

    PreparedPressureExpansion& operator=(PreparedPressureExpansion&&)      = delete;
    PreparedPressureExpansion(const PreparedPressureExpansion&)            = delete;
    PreparedPressureExpansion& operator=(const PreparedPressureExpansion&) = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept {
        return new_canonical_count_;
    }

private:
    PreparedPressureExpansion(const void* session, std::uint32_t session_generation,
                              std::uint32_t scratch_generation, std::uint32_t parent_index,
                              std::uint32_t new_canonical_count) noexcept
        : session_(session), session_generation_(session_generation),
          scratch_generation_(scratch_generation), parent_index_(parent_index),
          new_canonical_count_(new_canonical_count) {}

    const void* session_               = nullptr;
    std::uint32_t session_generation_  = 0;
    std::uint32_t scratch_generation_  = 0;
    std::uint32_t parent_index_        = 0;
    std::uint32_t new_canonical_count_ = 0;

    friend class PressurePlanningSession;
    friend struct detail::PressurePlanningSessionImpl;
};

struct PressureExpansionView {
    std::span<const PressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
    bool complete                     = true;
};

class PressurePlanningSession {
public:
    PressurePlanningSession(PressurePlanningSession&&) noexcept;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept;
    ~PressurePlanningSession();

    PressurePlanningSession(const PressurePlanningSession&)            = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle
    identity_target(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] PressureTargetHandle
    root_maximal_target(runtime::PlanningCandidateId root_candidate);
    [[nodiscard]] PressureTargetHandle maximal_target(runtime::PlanningCandidateId candidate);
    [[nodiscard]] PressureConstructionCursor begin_construction(PressureTargetHandle target,
                                                                bool restore = false);
    [[nodiscard]] runtime::PressureConstructionStep
    next_construction_option(PressureConstructionCursor& cursor);
    void choose_construction(PressureConstructionCursor& cursor,
                             runtime::PressureConstructionOptionId option);
    [[nodiscard]] std::optional<PressureTargetHandle>
    construction_target(const PressureConstructionCursor& cursor);
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion
    prepare_expansion(PressureTargetHandle parent,
                      std::uint32_t maximum_owners = std::numeric_limits<std::uint32_t>::max());
    [[nodiscard]] PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AssessedPressureTarget& assessed,
                                      const PreparedPrompt& prompt,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<ResourcePlan> seal(AssessedPressureTarget&& assessed,
                                                   const PreparedPrompt& prompt,
                                                   runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<CapturePressurePlan>
    seal_capture(AssessedPressureTarget&& assessed);

private:
    explicit PressurePlanningSession(
        std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept;

    std::unique_ptr<detail::PressurePlanningSessionImpl> impl_;

    friend class Program;
};

// Typed pressure domain for one active capture. The capture candidate remains Program-owned and
// cannot be inspected, sealed, or executed as a request admission candidate.

class CapturePressurePlanningSession {
public:
    CapturePressurePlanningSession(CapturePressurePlanningSession&&) noexcept;
    CapturePressurePlanningSession& operator=(CapturePressurePlanningSession&&) noexcept;
    ~CapturePressurePlanningSession();

    CapturePressurePlanningSession(const CapturePressurePlanningSession&)            = delete;
    CapturePressurePlanningSession& operator=(const CapturePressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle identity_target() const;
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle parent);
    [[nodiscard]] PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] std::optional<CapturePressurePlan> seal(AssessedPressureTarget&& assessed);

    [[nodiscard]] static constexpr runtime::PlanningCandidateId candidate_id() noexcept {
        return runtime::PlanningCandidateId{.value = 0};
    }

private:
    CapturePressurePlanningSession(CapturePressureCandidate&& candidate,
                                   PressurePlanningSession&& session) noexcept
        : candidate_(std::move(candidate)), session_(std::move(session)) {}

    CapturePressureCandidate candidate_;
    PressurePlanningSession session_;

    friend class Program;
};

class CaptureOffer {
public:
    CaptureOffer() noexcept = default;
    ~CaptureOffer()         = default;

    CaptureOffer(CaptureOffer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_), epoch_(other.epoch_),
          id_(std::exchange(other.id_, 0)) {}

    CaptureOffer& operator=(CaptureOffer&&)      = delete;
    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_    = 0;

    friend struct detail::RuntimeContractAccess;
};

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_),
          constraint_failed_(other.constraint_failed_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

    // Structured output: execution reached a position of this row whose language had no legal
    // token (its produced span ends in a placeholder). The Engine fails only that request.
    [[nodiscard]] bool constraint_failed(std::size_t row) const noexcept {
        return row < row_count_ && constraint_failed_[row];
    }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;
    std::array<bool, kMaximumConcurrency> constraint_failed_{};

    friend struct detail::RuntimeContractAccess;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
};

enum class CaptureStatePlacement : std::uint8_t {
    DeviceFork,
    HostSnapshot,
};

struct CaptureAssessment {
    CaptureAssessment();

    // Program retains the physical assessment in this opaque package-private payload.
    std::shared_ptr<detail::CaptureAssessmentImpl> implementation;
    PrefixShortlistKey shortlist_key;
    SharedCandidateEvidence shared_evidence = SharedCandidateEvidence::None;
    runtime::PrefillWork protected_rebuild_work;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> projected_recovery_work;
    std::vector<runtime::CheckpointRef> private_replacement_candidates;
    std::uint32_t frontier                = 0;
    bool publishes_private                = false;
    bool publishes_shared                 = false;
    bool needs_transfer                   = false;
    bool physically_feasible              = false;
    bool recycles_private_state           = false;
    CaptureStatePlacement state_placement = CaptureStatePlacement::DeviceFork;
};

struct SharedPrefixPublication {
    SharedPrefixHandle handle;
    SharedPrefixSummary summary;
};

struct MaterializationVictimResult;
struct MaterializationSharedVictimResult;

struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication> shared;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationVictimResult {
    runtime::PlanningOwnerId owner;
    runtime::VictimDisposition disposition = runtime::VictimDisposition::Retained;
    bool pressure_committed                = false;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedVictimResult {
    runtime::PlanningOwnerId owner;
    runtime::VictimDisposition disposition = runtime::VictimDisposition::Retained;
    bool pressure_committed                = false;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationSourceResult {
    runtime::PrivateSourceMode mode = runtime::PrivateSourceMode::Retain;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedSourceResult {
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<MaterializationSourceResult> source;
    std::optional<MaterializationSharedSourceResult> shared_source;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult, ActiveCaptureResult>;

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    // A prompt-frontier capture becomes valid only after the generated Begin token is committed.
    // Keeping the move-only capability row-aligned avoids exposing provisional prompt state.
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

// The transport the captured tp 2 all-reduces use after startup; see LoadSummary::tp_transport.
struct TpTransportStatus {
    // "mailbox", "mailbox, MTP draft on copies", "mailbox, DFlash2 draft on copies" or "copies";
    // empty at tp 1.
    std::string transport;
    double probe_ms = 0.0; // startup probe round trip; 0 when no probe ran
    std::string fallback;  // why the mailbox was dropped or narrowed; empty otherwise
    // The mailbox's exchange kernel, "pipelined" or "legacy" (NINFER_TP_MAILBOX_LEGACY=1); empty
    // when the captured all-reduces run on the copies.
    std::string exchange_kernel;
    // The mailbox's slot, the widest captured all-reduce it carries (hidden x (K+1) x C BF16 with
    // the pipelined kernel); 0 when the captured all-reduces run on the copies.
    std::size_t slot_bytes = 0;
    // Under MTP or DFlash2 with the optimized proposal head: "split by vocabulary" or "rank 0"
    // (NINFER_TP_DRAFT_HEAD=primary; for DFlash2 also NINFER_TP_DRAFTER=primary); empty otherwise.
    std::string proposal_head;
    // Under DFlash2: "split (16/4 heads, MLP halves; selector on rank 1)" (on rank 0 with
    // NINFER_TP_SELECTOR=primary, NINFER_TP_DRAFT_HEAD=primary or Vision on rank 1) or "rank 0"
    // (NINFER_TP_DRAFTER=primary); empty otherwise.
    std::string drafter;
    // The eager (prefill) all-reduces' transport: "pinned host exchange", "direct peer copies"
    // or "driver-staged copies"; empty at tp 1.
    std::string eager_transport;
    // The text prefill overlap's split-phase sums (dense models): "pinned host lanes", "direct
    // peer lanes", "driver-staged peer lanes (test only)" (NINFER_TP_PEER_OVERLAP=force without
    // P2P), "off (one call per chunk)" or "off (NINFER_TP_PREFILL_OVERLAP=0)"; empty at tp 1
    // and for Qwen4Exp.
    std::string prefill_overlap;
};

class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine owns scheduling and logical residency policy. Program owns physical lanes, opaque
    // capabilities, model state and one immutable pending transaction at a time.
    [[nodiscard]] RequestBasePlan plan_request(const PreparedPrompt& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::vector<float> causal_score(PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);
    // Prompt tokens a root admission would resume from the disk tier (0 without one).
    [[nodiscard]] std::uint32_t disk_resume_tokens(const PreparedPrompt& prompt);
    [[nodiscard]] std::optional<AdmissionCandidate> inspect_admission(
        const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId destination,
        const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
        std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source);
    [[nodiscard]] std::optional<ResourcePlan> seal_identity(const AdmissionCandidate& candidate,
                                                            const PreparedPrompt& prompt,
                                                            runtime::FinalScheduleIntent intent);
    [[nodiscard]] PressurePlanningSession
    begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                            std::span<const runtime::PlanningCandidateId> candidate_ids,
                            std::span<const ContinuationHandle* const> private_owners,
                            std::span<const runtime::PlanningOwnerId> private_owner_ids,
                            std::span<const SharedPrefixHandle* const> shared_owners,
                            std::span<const runtime::PlanningOwnerId> shared_owner_ids);
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                      const PreparedPrompt& prompt,
                                      std::span<const std::uint32_t> frontiers);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& blocked_head, const ResourcePlan& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    // `masks` (structured output): the request's first-token mask; null for an unconstrained one.
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const SharedPrefixHandle& owner,
                             runtime::CheckpointRef checkpoint) const;
    [[nodiscard]] CapturePressurePlanningSession
    begin_capture_pressure_planning(const CaptureAssessment& assessment,
                                    std::span<const ContinuationHandle* const> private_owners,
                                    std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                    std::span<const SharedPrefixHandle* const> shared_owners,
                                    std::span<const runtime::PlanningOwnerId> shared_owner_ids);
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        CapturePressurePlan&& pressure, runtime::CancellationFlagView cancellation);
    // `masks` (structured output): token masks of the constrained rows, filled on the host before
    // the round's launch; null when no row is constrained (the round is then unchanged).
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
    // Development hooks that replace sampled tokens (NINFER_QWEN4_FORCE_TOKENS, replay, MTP
    // oracle, teacher-forced dump) are active; constrained requests are refused then.
    [[nodiscard]] bool sampling_bypass_hooks() const noexcept;
    // Advance each live sequence with its exact target-owned token row. This does not sample or
    // advance sampler RNG/occurrence state; callers own output publication and budget accounting.
    // Each optional execution split is relative to its row's forced-token span.
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    // Terminal settlement of a cancelled request. The sequence is catalogued at its committed
    // frontier like a finished turn (same result contract as finish); the Engine falls back to
    // abort when the Program cannot retain it.
    [[nodiscard]] FinishResult finish_cancelled(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    // Device KV lease (capy-flash, after ninfer-custom 9f108db7): an active sequence holds its
    // prompt plus a bounded window of its output, extended at a decode-round boundary up to its
    // own output ceiling. When the pools cannot extend it the lease settles: the sequence then
    // finishes at the frontier the lease covers with its generation limit reason, instead of
    // failing a launch on coverage. This reports how many further model tokens that finish
    // licenses (`forced_span_tokens`: the largest forced control span the caller may still
    // apply); absent while the lease can grow.
    [[nodiscard]] std::optional<std::uint32_t>
    device_kv_lease_settlement_tokens(SequenceHandle sequence,
                                      std::uint32_t forced_span_tokens) const noexcept;
    // A lease settled for lack of pool space (not its ceiling): the pages its next growth step
    // still misses. The caller may free retained cache and resume the lease first.
    [[nodiscard]] std::optional<DeviceKVLeaseShortfall>
    device_kv_lease_shortfall(SequenceHandle sequence) const noexcept;
    // Re-opens growth once the smallest step fits; false while it still does not.
    [[nodiscard]] bool resume_device_kv_lease(SequenceHandle sequence) noexcept;
    // Device KV pages that releasing this retained owner would return to the pools: the pages it
    // maps alone.
    [[nodiscard]] DeviceKVPages
    retained_device_kv_pages(const ContinuationHandle& continuation) const noexcept;
    [[nodiscard]] DeviceKVPages
    retained_device_kv_pages(const SharedPrefixHandle& shared) const noexcept;
    // Device KV pages this retained owner maps that no active sequence reads, alone or with other
    // retained owners: what moving its KV to the Host can free. A page that two idle owners share
    // (a conversation's continuation and the shared prefix it was built on) is in neither owner's
    // retained_device_kv_pages, and returns to the pool when one of them moves it to the Host or
    // when both are released. A stale handle yields no pages; an inconsistent KV store throws.
    [[nodiscard]] DeviceKVPages idle_device_kv_pages(const ContinuationHandle& continuation) const;
    [[nodiscard]] DeviceKVPages idle_device_kv_pages(const SharedPrefixHandle& shared) const;
    // For each listed retained owner (distinct), continuations first, then shared prefixes: the
    // Device KV pages it maps that releasing all the listed owners together returns to the pools,
    // i.e. pages no active sequence reads and no owner outside the list maps. A page the listed
    // owners share counts for each of them. Stale handles throw.
    [[nodiscard]] std::vector<DeviceKVPages>
    releasable_device_kv_pages(std::span<const ContinuationHandle* const> continuations,
                               std::span<const SharedPrefixHandle* const> shared) const;
    // Unreserved pages of the Device KV pools (Main and, with a backend, Backend).
    [[nodiscard]] DeviceKVPages device_kv_available_pages() const noexcept;
    // Moves `needed` Device KV pages of an idle retained continuation to the Host tier (the
    // admission pressure's DemoteToHost option, run synchronously), so a lease can take them
    // without losing the conversation. Returns its summary after the move; absent when there is
    // no Host tier, no room in it, or the owner cannot be moved now (nothing then changes).
    [[nodiscard]] std::optional<ContinuationSummary>
    demote_retained_kv_for_lease(const ContinuationHandle& continuation,
                                 DeviceKVPages needed) noexcept;
    // The same for an idle shared prefix: shared and private owners hold the same prefix pages,
    // so freeing them takes both moves.
    [[nodiscard]] std::optional<SharedPrefixSummary>
    demote_retained_shared_kv_for_lease(const SharedPrefixHandle& shared,
                                        DeviceKVPages needed) noexcept;
    // v0.6 (idle KV to the Host): with elastic KV and a Host KV tier the Device keeps only the
    // KV of active requests. The Engine moves the Device KV of idle retained owners to the Host
    // (demote_retained_*_for_lease with every page) and then calls compact_device_kv(), which
    // packs the remaining Device pages into the lowest physical pages and gives the KV backing no
    // longer needed back to the routed-expert cache at once. Between execution units only, with
    // no context transaction and no request prefilling; otherwise (or without elastic KV and a
    // Host KV tier) it does nothing and returns ran = false.
    [[nodiscard]] bool idle_device_kv_offload() const noexcept;
    // Host KV bytes that moving every movable Device KV page of this idle continuation would
    // take (absent when none can move now), and the Host KV tier's free bytes.
    [[nodiscard]] std::optional<std::size_t>
    idle_device_kv_host_bytes(const ContinuationHandle& continuation) const noexcept;
    [[nodiscard]] std::size_t host_kv_free_bytes() const noexcept;
    DeviceKVCompaction compact_device_kv();
    // Disk tier of the context cache (v0.7.2; ContextCacheOptions::disk_directory). Off: every
    // call is a no-op. Between execution units only, with no context transaction.
    [[nodiscard]] bool disk_cache_enabled() const noexcept;
    // ContextCacheOptions::disk_idle_ms (0: no idle export).
    [[nodiscard]] std::uint32_t disk_cache_idle_ms() const noexcept;
    // Whether some retained conversation is not on disk yet.
    [[nodiscard]] bool disk_cache_has_unsaved() const noexcept;
    // Writes one retained conversation that is not on disk yet (most tokens first): Saved when
    // it was handed to the writer thread, NothingToSave, Aborted when `abort` returned true (the
    // partial record is dropped; tried again later) or Failed (not tried again).
    [[nodiscard]] DiskCacheSave disk_cache_save_one(const std::function<bool()>& abort);
    [[nodiscard]] std::uint32_t disk_cache_flush_seconds() const noexcept;
    // Waits for the writer thread (pending records are dropped at the deadline).
    bool disk_cache_drain(std::chrono::steady_clock::time_point deadline);
    // One line for the log: records, bytes, hits, writes.
    [[nodiscard]] std::string disk_cache_status() const;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& continuation) noexcept;
    // Drops one long anchor of an idle catalogued continuation and keeps the rest of it. Returns
    // the continuation's new summary, or nothing (and no change) when the drop is not possible now.
    [[nodiscard]] std::optional<ContinuationSummary>
    drop_continuation_anchor(const ContinuationHandle& continuation,
                             runtime::CheckpointRef anchor) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    void fail_all_cleanup() noexcept;

    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;
    [[nodiscard]] const TpTransportStatus& tp_transport() const noexcept;
    // A captured tensor-parallel exchange timed out (the pinned-host mailbox's sticky hang word;
    // under WSL2 an exchange occasionally hangs while serving). The work in flight
    // diverged between the ranks, but the poller gave up after a bounded spin, so both devices
    // are idle and usable.
    [[nodiscard]] bool peer_exchange_timed_out() const noexcept;
    // After such a timeout, once fail_all_cleanup() has left no request, context transaction or
    // retained context: steps the transport down once, as a hang in a graph's first launch does
    // at startup (mailbox, then the draft phase on the staged copies, then the copies
    // everywhere), and captures the CUDA Graphs again. Returns the transport now in use; throws
    // when the graphs cannot be captured again.
    std::string recover_peer_transport();

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;

    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
    friend std::unique_ptr<Program> create_program(const execution::Parameters&,
                                                   const execution::Parameters&, SequencePlan&&,
                                                   ExecutionContext&, const StartupObserver&);
};

namespace detail {

struct RuntimeContractAccess {
    [[nodiscard]] static SequenceHandle make_sequence(const void* owner, runtime::LaneId lane,
                                                      std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    [[nodiscard]] static ContinuationHandle
    make_continuation(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        ContinuationHandle out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static SharedPrefixHandle
    make_shared_prefix(const void* owner, std::uint32_t index, std::uint64_t generation) noexcept {
        SharedPrefixHandle out;
        out.owner_      = owner;
        out.index_      = index;
        out.generation_ = generation;
        return out;
    }

    [[nodiscard]] static CaptureOffer make_capture_offer(const void* owner, runtime::LaneId lane,
                                                         std::uint64_t epoch,
                                                         std::uint64_t id) noexcept {
        CaptureOffer out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        out.id_    = id;
        return out;
    }

    [[nodiscard]] static const void* owner(const SequenceHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static const void* owner(const CaptureOffer& offer) noexcept {
        return offer.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const CaptureOffer& offer) noexcept {
        return offer.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const CaptureOffer& offer) noexcept {
        return offer.epoch_;
    }

    [[nodiscard]] static std::uint64_t id(const CaptureOffer& offer) noexcept { return offer.id_; }

    static void consume(CaptureOffer& offer) noexcept {
        offer.owner_ = nullptr;
        offer.id_    = 0;
    }

    [[nodiscard]] static runtime::LaneId lane(const SequenceHandle& handle) noexcept {
        return handle.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SequenceHandle& handle) noexcept {
        return handle.epoch_;
    }

    [[nodiscard]] static const void* owner(const ContinuationHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const ContinuationHandle& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const ContinuationHandle& handle) noexcept {
        return handle.generation_;
    }

    static void consume(ContinuationHandle& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static const void* owner(const SharedPrefixHandle& handle) noexcept {
        return handle.owner_;
    }

    [[nodiscard]] static std::uint32_t index(const SharedPrefixHandle& handle) noexcept {
        return handle.index_;
    }

    [[nodiscard]] static std::uint64_t epoch(const SharedPrefixHandle& handle) noexcept {
        return handle.generation_;
    }

    static void consume(SharedPrefixHandle& handle) noexcept {
        handle.owner_      = nullptr;
        handle.generation_ = 0;
    }

    [[nodiscard]] static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                 std::span<const TokenId> tokens, std::span<const std::int32_t> row_counts,
                 std::uint32_t row_stride, runtime::ExecutionTiming timing) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

    static void set_constraint_failed(PendingBatch& pending, std::size_t row, bool failed) {
        pending.constraint_failed_.at(row) = failed;
    }

    [[nodiscard]] static const void* owner(const PendingBatch& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle>
    rows(const PendingBatch& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }
};

} // namespace detail

[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                                    DeviceContext& device,
                                                    const EngineOptions& options);

// Device bytes one more offloaded-expert slot per layer costs (every layer's slot), or 0 when the
// model's routed experts are not offloaded (EngineOptions::routed_experts does not apply).
// True when the Qwen4Exp routed experts are ggml block banks (capy-flash v0.7).
[[nodiscard]] bool gguf_routed_experts(const execution::Parameters& parameters);
[[nodiscard]] std::int64_t offloaded_expert_slot_bytes(const execution::Parameters& parameters,
                                                       const EngineOptions& options);
// Device bytes of the offload backend's prefetch staging slots (T25; 0 when prefetch is off).
[[nodiscard]] std::int64_t offloaded_expert_staging_bytes(const execution::Parameters& parameters,
                                                          const EngineOptions& options);
// Device bytes of the staged miss split's scratch slots (T18), or 0 when not used.
[[nodiscard]] std::int64_t offloaded_expert_split_bytes(const execution::Parameters& parameters,
                                                        const EngineOptions& options);

// Page-locked Host bytes a Program of `options` allocates on all its ranks for active KV in RAM
// (EngineOptions::kv_ram_active: the Host planes of the text KV pool); 0 without it. They share the
// page-locked limit with the host-mapped weights, which are placed first (LoadPlan::
// reserve_host_pinned_bytes).
[[nodiscard]] std::uint64_t active_kv_host_pinned_bytes(const Config& config,
                                                        const EngineOptions& options);

// Tensor-parallel width 2, with Parameters(model, 0) and Parameters(model, 1) of one two-device
// Model: rank 1's Parameters supply the Vision tower when rank 1 holds it.
[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                                    const execution::Parameters& peer_parameters,
                                                    DeviceContext& device,
                                                    const EngineOptions& options);

[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                                      SequencePlan&& plan, DeviceContext& device,
                                                      const StartupObserver& startup_observer);

// Tensor-parallel width 2: `parameters` and `peer_parameters` are Parameters(model, 0) and
// Parameters(model, 1) of one two-device Model; rank r executes on execution.dev[r]. The plan
// must come from a planner configured with tp 2.
[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                                      const execution::Parameters& peer_parameters,
                                                      SequencePlan&& plan,
                                                      ExecutionContext& execution,
                                                      const StartupObserver& startup_observer);

} // namespace ninfer::models::qwen3_5
