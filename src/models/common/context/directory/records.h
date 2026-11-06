#pragma once

#include "core/context_pool.h"
#include "models/common/context/contract.h"
#include "models/common/context/directory/metadata.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::context {

struct ContextDomainId {
    std::uint32_t rank                                       = 0;
    ContextMemory memory                                     = ContextMemory::Device;
    friend bool operator==(ContextDomainId, ContextDomainId) = default;
};

struct ContextRankSpec {
    std::uint32_t rank                                       = 0;
    std::size_t device_bytes                                 = 0;
    std::size_t host_bytes                                   = 0;
    friend bool operator==(ContextRankSpec, ContextRankSpec) = default;
};

struct RecordPartSpec {
    std::uint32_t rank = 0;
    RecordSpec record;
    friend bool operator==(RecordPartSpec, RecordPartSpec) = default;
};

struct ContextRecordLayout {
    std::uint32_t id = 0;
    std::vector<RecordPartSpec> parts;
};

struct ContextRecordBinding {
    ContextDomainId domain;
    AllocationId allocation;
    std::int64_t offset         = -1;
    std::size_t bytes           = 0;
    std::uint64_t binding_epoch = 0;
};

struct RecordUsage {
    std::uint64_t stride_bytes = 0;
    std::uint64_t live_records = 0, pending_retirement_records = 0;
    std::uint64_t cached_records = 0;
};

// Record-relative fields supplied by the layout owner. Copies preserve represented content.
struct RecordRangeCopy {
    std::uint32_t source_rank      = 0;
    std::uint32_t destination_rank = 0;
    std::size_t source_offset      = 0;
    std::size_t destination_offset = 0;
    std::size_t bytes              = 0;
};

// Owns one rank-local replica. The domain outlives all record owners and reservations.
class RecordLease {
public:
    RecordLease() noexcept = default;
    ~RecordLease();
    RecordLease(RecordLease&&) noexcept;
    RecordLease& operator=(RecordLease&&) noexcept;
    RecordLease(const RecordLease&)            = delete;
    RecordLease& operator=(const RecordLease&) = delete;

    [[nodiscard]] bool valid() const noexcept { return pool_ != nullptr; }

    [[nodiscard]] bool belongs_to(const ContextPool& pool) const noexcept { return pool_ == &pool; }

    [[nodiscard]] ContextDomainId domain() const noexcept { return domain_; }

    [[nodiscard]] AllocationId allocation() const noexcept { return allocation_; }

    [[nodiscard]] ContextRecordBinding binding() const;
    void set_use(ContextUse use) noexcept;
    void defer_retirement();
    void cancel_retirement();
    void reset() noexcept;

private:
    RecordLease(ContextPool&, ContextDomainId, AllocationId) noexcept;
    ContextPool* pool_ = nullptr;
    ContextDomainId domain_;
    AllocationId allocation_;
    bool retiring_      = false;
    ContextUse use_     = ContextUse::Request;
    RecordUsage* usage_ = nullptr;
    void account(bool add) noexcept;
    void set_retiring(bool retiring) noexcept;
    void forget() noexcept;
    friend class RecordReservation;
    friend class RecordDomains;
};

struct RecordDemand {
    std::uint32_t layout = 0;
    ContextMemory memory = ContextMemory::Device;
    std::uint32_t count  = 0;
    // Empty selects all parts of the layout. A nonempty list selects only missing local replicas.
    std::vector<std::uint32_t> ranks;
};

struct RecordRelease {
    ContextDomainId domain;
    AllocationId allocation;
    friend bool operator==(RecordRelease, RecordRelease) = default;
};

struct DomainAssessment {
    ContextDomainId domain;
    PoolAssessment placement;
};

struct RecordAssessment {
    std::vector<DomainAssessment> domains;
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] bool feasible() const noexcept;
};

// A share of an atomic bundle. Each part owns its exact unmaterialized count; splitting the
// control object does not release positions to the public pool.
class RecordReservation {
public:
    RecordReservation() noexcept = default;
    ~RecordReservation();
    RecordReservation(RecordReservation&&) noexcept;
    RecordReservation& operator=(RecordReservation&&) noexcept;
    RecordReservation(const RecordReservation&)            = delete;
    RecordReservation& operator=(const RecordReservation&) = delete;

    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] bool contributes_to(const RecordAssessment&,
                                      bool metadata_pressure) const noexcept;
    [[nodiscard]] std::uint32_t remaining(std::uint32_t rank) const noexcept;

    [[nodiscard]] std::span<const std::uint32_t> ranks() const noexcept { return ranks_; }

    [[nodiscard]] RecordLease materialize(std::uint32_t rank);
    [[nodiscard]] std::vector<RecordLease> materialize();
    void restore(RecordLease&&);
    void defer_return(RecordLease&);
    void prepare_metadata_returns(MetadataOwner&, std::size_t count);
    void attach_metadata(MetadataReservation);
    [[nodiscard]] std::uint32_t take_metadata(MetadataOwner&);
    [[nodiscard]] MetadataReservation take_metadata_reservation(MetadataOwner&);
    void restore_metadata(MetadataOwner&, std::uint32_t);
    [[nodiscard]] RecordReservation split(std::uint32_t count);
    void reserve_merge_capacity(std::size_t additional_claims,
                                std::size_t additional_metadata_tokens);
    void reserve_merge(const RecordReservation&);
    void merge(RecordReservation&&);
    void shrink(std::uint32_t count);
    void clear() noexcept;

private:
    struct Claim {
        ContextPool* pool = nullptr;
        ContextDomainId domain;
        RecordClassId record_class;
        std::shared_ptr<PoolReservation> reservation;
        std::uint32_t count = 0;
    };

    std::vector<MetadataReservation> metadata_;
    std::vector<Claim> claims_;
    std::vector<std::uint32_t> ranks_;
    friend class RecordDomains;
};

struct RecordReserveResult {
    RecordAssessment assessment;
    std::optional<std::vector<RecordReservation>> reservations;
};

// Physical ledger for every rank and memory domain in one Session. Schema/layout IDs are local
// to this directory; the allocator only receives record sizes, alignments and counts.
class RecordDomains {
public:
    RecordDomains(std::span<const ContextRankSpec>, std::span<const ContextRecordLayout>);
    ~RecordDomains();
    RecordDomains(const RecordDomains&)            = delete;
    RecordDomains& operator=(const RecordDomains&) = delete;

    [[nodiscard]] std::span<const ContextRankSpec> ranks() const noexcept { return ranks_; }

    [[nodiscard]] const ContextRecordLayout& layout(std::uint32_t id) const;
    [[nodiscard]] ContextPool& pool(ContextDomainId);
    [[nodiscard]] const ContextPool& pool(ContextDomainId) const;
    // Logical directories classify an owned replica; equal physical classes may serve
    // different layouts. Classification never changes allocation or reservation policy.
    void classify(RecordLease&, std::uint32_t layout);
    [[nodiscard]] const RecordUsage& usage(ContextDomainId, std::uint32_t layout) const;
    [[nodiscard]] RecordAssessment assess(std::span<const RecordDemand>,
                                          std::span<const RecordRelease> = {}) const;
    [[nodiscard]] RecordReserveResult reserve(std::span<const RecordDemand>);
    [[nodiscard]] std::optional<RecordReservation> reserve(const RecordDemand&);
    void prepare_returns(RecordReservation&, const RecordDemand&);
    [[nodiscard]] std::size_t metadata_bytes() const;

private:
    struct Domain;
    std::vector<ContextRankSpec> ranks_;
    std::vector<ContextRecordLayout> layouts_;
    std::vector<std::unique_ptr<Domain>> domains_;
    [[nodiscard]] Domain& domain(ContextDomainId);
    [[nodiscard]] const Domain& domain(ContextDomainId) const;
};

} // namespace ninfer::models::context
