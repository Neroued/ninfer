#include "models/common/context/directory/records.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::context {

RecordLease::RecordLease(ContextPool& pool, ContextDomainId domain,
                         AllocationId allocation) noexcept
    : pool_(&pool), domain_(domain), allocation_(allocation) {}

RecordLease::~RecordLease() { reset(); }

RecordLease::RecordLease(RecordLease&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)), domain_(other.domain_),
      allocation_(other.allocation_), retiring_(other.retiring_), use_(other.use_),
      usage_(std::exchange(other.usage_, nullptr)) {}

RecordLease& RecordLease::operator=(RecordLease&& other) noexcept {
    if (this != &other) {
        reset();
        pool_       = std::exchange(other.pool_, nullptr);
        domain_     = other.domain_;
        allocation_ = other.allocation_;
        retiring_   = other.retiring_;
        use_        = other.use_;
        usage_      = std::exchange(other.usage_, nullptr);
    }
    return *this;
}

ContextRecordBinding RecordLease::binding() const {
    if (!pool_) throw std::logic_error("context record owner is empty");
    const auto value = pool_->resolve(allocation_);
    if (value.offset > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("context record offset exceeds execution address range");
    return {domain_, allocation_, static_cast<std::int64_t>(value.offset),
            static_cast<std::size_t>(value.bytes), value.binding_epoch};
}

void RecordLease::defer_retirement() {
    if (!pool_ || retiring_) return;
    pool_->defer_release(allocation_);
    set_retiring(true);
}

void RecordLease::cancel_retirement() {
    if (!pool_ || !retiring_) return;
    pool_->cancel_deferred_release(allocation_);
    set_retiring(false);
}

void RecordLease::account(bool add) noexcept {
    if (!usage_) return;
    auto update = [add](std::uint64_t& count) {
        if (add)
            ++count;
        else
            --count;
    };
    update(retiring_ ? usage_->pending_retirement_records : usage_->live_records);
    if (!retiring_ && use_ == ContextUse::Cache) update(usage_->cached_records);
}

void RecordLease::set_use(ContextUse use) noexcept {
    if (use_ == use) return;
    account(false);
    use_ = use;
    account(true);
}

void RecordLease::set_retiring(bool retiring) noexcept {
    account(false);
    retiring_ = retiring;
    account(true);
}

void RecordLease::reset() noexcept {
    if (!pool_) return;
    if (retiring_)
        pool_->retire(allocation_);
    else
        pool_->release(allocation_);
    forget();
}

void RecordLease::forget() noexcept {
    account(false);
    usage_ = nullptr;
    pool_  = nullptr;
}

bool RecordAssessment::ready() const noexcept {
    return std::all_of(domains.begin(), domains.end(), [](const auto& value) {
        return value.placement.status == PoolAvailability::Ready;
    });
}

bool RecordAssessment::feasible() const noexcept {
    return std::all_of(domains.begin(), domains.end(),
                       [](const auto& value) { return value.placement.feasible(); });
}

RecordReservation::~RecordReservation() { clear(); }

RecordReservation::RecordReservation(RecordReservation&& other) noexcept
    : metadata_(std::move(other.metadata_)), claims_(std::move(other.claims_)),
      ranks_(std::move(other.ranks_)) {
    other.claims_.clear();
    other.ranks_.clear();
}

RecordReservation& RecordReservation::operator=(RecordReservation&& other) noexcept {
    if (this != &other) {
        clear();
        metadata_ = std::move(other.metadata_);
        claims_   = std::move(other.claims_);
        ranks_    = std::move(other.ranks_);
        other.claims_.clear();
        other.ranks_.clear();
    }
    return *this;
}

bool RecordReservation::ready() const noexcept {
    return std::all_of(claims_.begin(), claims_.end(), [](const auto& claim) {
        return claim.count == 0 || claim.reservation->ready();
    });
}

bool RecordReservation::empty() const noexcept {
    return std::all_of(claims_.begin(), claims_.end(),
                       [](const auto& claim) { return claim.count == 0; }) &&
           std::all_of(metadata_.begin(), metadata_.end(),
                       [](const auto& metadata) { return metadata.remaining() == 0; });
}

bool RecordReservation::contributes_to(const RecordAssessment& pressure,
                                       bool metadata_pressure) const noexcept {
    if (metadata_pressure && std::any_of(metadata_.begin(), metadata_.end(),
                                         [](const auto& item) { return item.remaining() != 0; }))
        return true;
    for (const auto& claim : claims_) {
        if (!claim.count || !claim.reservation->valid()) continue;
        for (const auto& domain : pressure.domains)
            if (domain.domain == claim.domain && domain.placement.reclaim_bytes) return true;
    }
    return false;
}

std::uint32_t RecordReservation::remaining(std::uint32_t rank) const noexcept {
    std::uint32_t count = 0;
    for (const auto& claim : claims_)
        if (claim.domain.rank == rank) count += claim.count;
    return count;
}

RecordLease RecordReservation::materialize(std::uint32_t rank) {
    for (auto& claim : claims_) {
        if (claim.domain.rank != rank || !claim.count) continue;
        auto allocation = claim.reservation->materialize(claim.record_class);
        --claim.count;
        return RecordLease(*claim.pool, claim.domain, allocation);
    }
    throw std::logic_error("context record reservation has no remaining rank-local record");
}

std::vector<RecordLease> RecordReservation::materialize() {
    std::vector<RecordLease> result;
    result.reserve(ranks_.size());
    for (const auto rank : ranks_) {
        if (!remaining(rank)) throw std::logic_error("context record bundle is incomplete");
    }
    if (!ready()) throw std::logic_error("context record reservation is not ready");
    for (const auto rank : ranks_) result.push_back(materialize(rank));
    return result;
}

void RecordReservation::restore(RecordLease&& record) {
    if (!record.valid()) throw std::invalid_argument("cannot restore an empty context record");
    const auto binding = record.pool_->resolve(record.allocation_);
    for (auto& claim : claims_) {
        if (claim.pool != record.pool_ || claim.record_class != binding.record_class) continue;
        claim.pool->return_to_reservation(record.allocation_, *claim.reservation);
        ++claim.count;
        record.forget();
        return;
    }
    throw std::invalid_argument("context record does not belong to this reservation layout");
}

void RecordReservation::defer_return(RecordLease& record) {
    for (auto& claim : claims_) {
        if (claim.pool != record.pool_ ||
            claim.record_class != record.pool_->resolve(record.allocation_).record_class)
            continue;
        claim.pool->defer_to_reservation(record.allocation_, *claim.reservation);
        ++claim.count;
        record.set_retiring(true);
        return;
    }
    throw std::logic_error("retiring record has no prepared return reservation");
}

void RecordReservation::prepare_metadata_returns(MetadataOwner& owner, std::size_t count) {
    for (auto& metadata : metadata_)
        if (metadata.owner() == &owner) {
            metadata.prepare_returns(count);
            return;
        }
    MetadataReservation prepared(owner, {});
    prepared.prepare_returns(count);
    metadata_.push_back(std::move(prepared));
}

void RecordReservation::attach_metadata(MetadataReservation metadata) {
    metadata_.push_back(std::move(metadata));
}

std::uint32_t RecordReservation::take_metadata(MetadataOwner& owner) {
    for (auto& metadata : metadata_)
        if (metadata.owner() == &owner && metadata.remaining()) return metadata.take(owner);
    throw std::logic_error("context reservation has no logical descriptor claim");
}

void RecordReservation::restore_metadata(MetadataOwner& owner, std::uint32_t id) {
    for (auto& metadata : metadata_)
        if (metadata.owner() == &owner) {
            metadata.restore(owner, id);
            return;
        }
    throw std::logic_error("context reservation has no matching metadata owner");
}

MetadataReservation RecordReservation::take_metadata_reservation(MetadataOwner& owner) {
    for (auto& metadata : metadata_)
        if (metadata.owner() == &owner && metadata.remaining()) return std::move(metadata);
    throw std::logic_error("context reservation has no control descriptor claim");
}

RecordReservation RecordReservation::split(std::uint32_t count) {
    for (const auto rank : ranks_)
        if (remaining(rank) < count)
            throw std::logic_error("context split exceeds reserved records");
    RecordReservation result;
    result.claims_.reserve(claims_.size());
    result.ranks_ = ranks_;
    result.metadata_.reserve(metadata_.size());
    auto logical = count;
    for (auto& metadata : metadata_) {
        const auto taken = std::min<std::size_t>(logical, metadata.remaining());
        if (taken) result.metadata_.push_back(metadata.split(taken));
        logical -= taken;
    }
    for (const auto rank : ranks_) {
        auto left = count;
        for (auto& claim : claims_) {
            if (claim.domain.rank != rank || !left) continue;
            const auto taken = std::min(claim.count, left);
            if (taken) {
                result.claims_.push_back(
                    {claim.pool, claim.domain, claim.record_class, claim.reservation, taken});
                claim.count -= taken;
                left -= taken;
            }
        }
    }
    return result;
}

void RecordReservation::reserve_merge_capacity(std::size_t claims, std::size_t metadata) {
    claims_.reserve(claims_.size() + claims);
    metadata_.reserve(metadata_.size() + metadata);
}

void RecordReservation::reserve_merge(const RecordReservation& other) {
    for (std::size_t i = claims_.size(); i > 0; --i) {
        const auto& candidate = claims_[i - 1];
        if (!candidate.count &&
            std::any_of(claims_.begin(), claims_.begin() + i - 1, [&](const auto& item) {
                return item.pool == candidate.pool && item.record_class == candidate.record_class;
            }))
            claims_.erase(claims_.begin() + i - 1);
    }
    for (std::size_t i = metadata_.size(); i > 0; --i) {
        const auto& candidate = metadata_[i - 1];
        if (!candidate.remaining() &&
            std::any_of(metadata_.begin(), metadata_.begin() + i - 1,
                        [&](const auto& item) { return item.owner() == candidate.owner(); }))
            metadata_.erase(metadata_.begin() + i - 1);
    }
    metadata_.reserve(metadata_.size() + other.metadata_.size());
    claims_.reserve(claims_.size() + other.claims_.size());
    auto ranks = ranks_.size();
    for (const auto rank : other.ranks_)
        ranks += std::find(ranks_.begin(), ranks_.end(), rank) == ranks_.end();
    ranks_.reserve(ranks);
}

void RecordReservation::merge(RecordReservation&& other) {
    reserve_merge(other);
    for (auto& metadata : other.metadata_) metadata_.push_back(std::move(metadata));
    other.metadata_.clear();
    for (const auto rank : other.ranks_)
        if (std::find(ranks_.begin(), ranks_.end(), rank) == ranks_.end()) ranks_.push_back(rank);
    for (auto& claim : other.claims_) claims_.push_back(std::move(claim));
    other.claims_.clear();
    other.ranks_.clear();
}

void RecordReservation::shrink(std::uint32_t count) {
    auto keep = count;
    for (auto& metadata : metadata_) {
        const auto retained = std::min<std::size_t>(keep, metadata.remaining());
        if (retained < metadata.remaining()) metadata.shrink(retained);
        keep -= retained;
    }
    for (const auto rank : ranks_) {
        auto excess = remaining(rank) > count ? remaining(rank) - count : 0;
        for (auto& claim : claims_) {
            if (claim.domain.rank != rank || !excess) continue;
            const auto released = std::min(claim.count, excess);
            const auto current  = claim.reservation->remaining(claim.record_class);
            claim.reservation->shrink(claim.record_class, current - released);
            claim.count -= released;
            excess -= released;
        }
    }
}

void RecordReservation::clear() noexcept {
    for (auto& claim : claims_) {
        if (!claim.count || !claim.reservation->valid()) continue;
        const auto current = claim.reservation->remaining(claim.record_class);
        claim.reservation->shrink(claim.record_class, current - claim.count);
        claim.count = 0;
    }
    claims_.clear();
    ranks_.clear();
    metadata_.clear();
}

struct RecordDomains::Domain {
    struct Class {
        std::uint32_t layout;
        RecordClassId record_class;
        RecordUsage usage;
    };

    ContextDomainId id;
    std::unique_ptr<ContextPool> pool;
    std::vector<Class> classes;

    RecordClassId find(std::uint32_t layout) const {
        const auto it = std::find_if(classes.begin(), classes.end(),
                                     [&](const auto& value) { return value.layout == layout; });
        if (it == classes.end())
            throw std::invalid_argument("record layout has no part in this rank");
        return it->record_class;
    }
};

RecordDomains::RecordDomains(std::span<const ContextRankSpec> ranks,
                             std::span<const ContextRecordLayout> layouts)
    : ranks_(ranks.begin(), ranks.end()), layouts_(layouts.begin(), layouts.end()) {
    for (std::size_t i = 0; i < layouts_.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j)
            if (layouts_[i].id == layouts_[j].id)
                throw std::invalid_argument("duplicate context record layout");
        for (std::size_t j = 0; j < layouts_[i].parts.size(); ++j) {
            const auto rank = layouts_[i].parts[j].rank;
            if (std::none_of(ranks_.begin(), ranks_.end(),
                             [&](const auto& item) { return item.rank == rank; }))
                throw std::invalid_argument("record layout names an unknown context rank");
            for (std::size_t k = 0; k < j; ++k)
                if (layouts_[i].parts[k].rank == rank)
                    throw std::invalid_argument("duplicate rank in context record layout");
        }
    }
    for (std::size_t i = 0; i < ranks_.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j)
            if (ranks_[i].rank == ranks_[j].rank)
                throw std::invalid_argument("duplicate context rank");
        for (const auto memory : {ContextMemory::Device, ContextMemory::Host}) {
            auto value = std::make_unique<Domain>();
            value->id  = {ranks_[i].rank, memory};
            std::vector<RecordSpec> specs;
            for (const auto& layout : layouts_) {
                const auto part =
                    std::find_if(layout.parts.begin(), layout.parts.end(),
                                 [&](const auto& item) { return item.rank == value->id.rank; });
                if (part == layout.parts.end()) continue;
                specs.push_back(part->record);
                value->classes.push_back({layout.id, {}});
            }
            const auto bytes =
                memory == ContextMemory::Device ? ranks_[i].device_bytes : ranks_[i].host_bytes;
            value->pool = std::make_unique<ContextPool>(static_cast<std::uint32_t>(domains_.size()),
                                                        bytes, specs);
            for (std::size_t j = 0; j < specs.size(); ++j)
                value->classes[j].record_class = value->pool->record_class(j);
            const auto occupancy = value->pool->usage();
            for (auto& item : value->classes)
                item.usage.stride_bytes = occupancy.classes.at(item.record_class.index).stride;
            domains_.push_back(std::move(value));
        }
    }
}

RecordDomains::~RecordDomains() = default;

const RecordUsage& RecordDomains::usage(ContextDomainId id, std::uint32_t layout) const {
    const auto& local = domain(id);
    const auto found  = std::find_if(local.classes.begin(), local.classes.end(),
                                     [=](const auto& item) { return item.layout == layout; });
    if (found == local.classes.end()) throw std::invalid_argument("unknown context record layout");
    return found->usage;
}

void RecordDomains::classify(RecordLease& record, std::uint32_t layout) {
    if (!record.valid() || !record.belongs_to(pool(record.domain())))
        throw std::invalid_argument("context record belongs to a different domain");
    const auto& local = domain(record.domain());
    if (local.find(layout) != record.pool_->resolve(record.allocation_).record_class)
        throw std::invalid_argument("context record classification has a different geometry");
    auto* target = &const_cast<RecordUsage&>(usage(record.domain(), layout));
    if (record.usage_ == target) return;
    record.account(false);
    record.usage_ = target;
    record.account(true);
}

const ContextRecordLayout& RecordDomains::layout(std::uint32_t id) const {
    const auto it = std::find_if(layouts_.begin(), layouts_.end(),
                                 [&](const auto& value) { return value.id == id; });
    if (it == layouts_.end()) throw std::invalid_argument("unknown context record layout");
    return *it;
}

RecordDomains::Domain& RecordDomains::domain(ContextDomainId id) {
    return const_cast<Domain&>(std::as_const(*this).domain(id));
}

const RecordDomains::Domain& RecordDomains::domain(ContextDomainId id) const {
    const auto it = std::find_if(domains_.begin(), domains_.end(),
                                 [&](const auto& value) { return value->id == id; });
    if (it == domains_.end()) throw std::invalid_argument("unknown context memory domain");
    return **it;
}

ContextPool& RecordDomains::pool(ContextDomainId id) { return *domain(id).pool; }

const ContextPool& RecordDomains::pool(ContextDomainId id) const { return *domain(id).pool; }

namespace {
bool selected_rank(const RecordDemand& demand, std::uint32_t rank) {
    return demand.ranks.empty() ||
           std::find(demand.ranks.begin(), demand.ranks.end(), rank) != demand.ranks.end();
}
} // namespace

RecordAssessment RecordDomains::assess(std::span<const RecordDemand> demands,
                                       std::span<const RecordRelease> releases) const {
    RecordAssessment result;
    for (const auto& demand : demands) {
        const auto& spec = layout(demand.layout);
        for (const auto rank : demand.ranks)
            if (std::none_of(spec.parts.begin(), spec.parts.end(),
                             [&](const auto& part) { return part.rank == rank; }))
                throw std::invalid_argument(
                    "context demand selects a rank outside its record placement");
    }
    for (const auto& value : domains_) {
        std::vector<RecordRequest> requests;
        for (const auto& demand : demands) {
            if (!demand.count || demand.memory != value->id.memory ||
                !selected_rank(demand, value->id.rank))
                continue;
            const auto& spec = layout(demand.layout);
            if (std::none_of(spec.parts.begin(), spec.parts.end(),
                             [&](const auto& part) { return part.rank == value->id.rank; }))
                continue;
            requests.push_back({value->find(demand.layout), demand.count});
        }
        if (requests.empty()) continue;
        std::vector<AllocationId> freed;
        for (const auto& release : releases)
            if (release.domain == value->id) freed.push_back(release.allocation);
        result.domains.push_back({value->id, value->pool->assess(requests, freed)});
    }
    return result;
}

RecordReserveResult RecordDomains::reserve(std::span<const RecordDemand> demands) {
    RecordReserveResult result{assess(demands), std::nullopt};
    if (!result.assessment.feasible()) return result;
    std::vector<RecordReservation> reservations(demands.size());
    for (const auto& value : domains_) {
        std::vector<RecordRequest> requests;
        std::vector<std::size_t> indices;
        for (std::size_t i = 0; i < demands.size(); ++i) {
            const auto& demand = demands[i];
            if (!demand.count || demand.memory != value->id.memory ||
                !selected_rank(demand, value->id.rank))
                continue;
            const auto& spec = layout(demand.layout);
            if (std::none_of(spec.parts.begin(), spec.parts.end(),
                             [&](const auto& part) { return part.rank == value->id.rank; }))
                continue;
            requests.push_back({value->find(demand.layout), demand.count});
            indices.push_back(i);
        }
        if (requests.empty()) continue;
        auto reserved = value->pool->reserve(requests);
        if (!reserved.reservation) {
            reservations.clear();
            result.assessment = assess(demands);
            return result;
        }
        auto owner = std::make_shared<PoolReservation>(std::move(*reserved.reservation));
        for (std::size_t i = 0; i < indices.size(); ++i) {
            auto& reservation = reservations[indices[i]];
            reservation.claims_.push_back(
                {value->pool.get(), value->id, requests[i].record_class, owner, requests[i].count});
            reservation.ranks_.push_back(value->id.rank);
        }
    }
    result.reservations = std::move(reservations);
    return result;
}

std::optional<RecordReservation> RecordDomains::reserve(const RecordDemand& demand) {
    auto result = reserve(std::span(&demand, 1));
    if (!result.reservations) return std::nullopt;
    return std::move(result.reservations->front());
}

void RecordDomains::prepare_returns(RecordReservation& reservation, const RecordDemand& demand) {
    for (const auto& part : layout(demand.layout).parts) {
        if (!selected_rank(demand, part.rank)) continue;
        auto& local             = domain({part.rank, demand.memory});
        const auto record_class = local.find(demand.layout);
        auto found              = std::find_if(
            reservation.claims_.begin(), reservation.claims_.end(), [&](const auto& claim) {
                return claim.pool == local.pool.get() && claim.record_class == record_class;
            });
        if (found == reservation.claims_.end()) {
            const RecordRequest empty{record_class, 0};
            auto result = local.pool->reserve(std::span(&empty, 1));
            if (!result.reservation)
                throw std::logic_error("empty return reservation was rejected");
            auto owner = std::make_shared<PoolReservation>(std::move(*result.reservation));
            reservation.claims_.push_back(
                {local.pool.get(), local.id, record_class, std::move(owner), 0});
            found = reservation.claims_.end() - 1;
            if (std::find(reservation.ranks_.begin(), reservation.ranks_.end(), part.rank) ==
                reservation.ranks_.end())
                reservation.ranks_.push_back(part.rank);
        }
        found->reservation->prepare_returns(record_class, demand.count);
    }
}

std::size_t RecordDomains::metadata_bytes() const {
    std::size_t bytes = sizeof(*this);
    for (const auto& value : domains_)
        bytes += sizeof(Domain) + value->classes.capacity() * sizeof(Domain::Class) +
                 value->pool->usage().metadata_bytes;
    return bytes;
}

} // namespace ninfer::models::context
