#include "models/common/context/directory.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>

namespace {
thread_local std::ptrdiff_t fail_at = -1;
}

// Exercise exception unwinding of the real multi-participant reservation, at each host allocation.
void* operator new(std::size_t bytes) {
    if (fail_at >= 0 && fail_at-- == 0) throw std::bad_alloc();
    if (void* ptr = std::malloc(std::max<std::size_t>(bytes, 1))) return ptr;
    throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept { std::free(ptr); }

void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }

namespace {
using namespace ninfer;
using namespace ninfer::models::context;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

ContextDirectorySpec spec(unsigned blocks = 512, unsigned logical = 4096) {
    return {.ranks = {{0, 256ULL * blocks, 0}, {1, 256ULL * blocks, 0}},
            .state = {logical, {{0, {{0, 256, 256}, {1, 256, 256}}}}},
            .pools = {{0, logical, logical, 128, 4, {{0, {256, 256}}, {1, {256, 256}}}}},
            .metadata_budget_bytes = 96U << 20};
}

void complete(ContextTransaction& tx) {
    tx.prepare_actions();
    tx.mark_submitted();
    tx.mark_completed();
    tx.publish();
}

HistoryOwner bind(ContextDirectory& d, int row, unsigned peak,
                  const HistoryPrefix* prefix = nullptr) {
    auto& h             = d.pool(0).histories();
    const auto front    = prefix ? prefix->frontier() : 0;
    const auto snapshot = h.binding_demand(prefix, front, peak);
    ContextDemand demand;
    const auto id = demand.add_history(0, snapshot);
    auto rights   = d.reserve(demand);
    require(rights && rights.reservation->ready(), "fixture requires immediate records");
    auto pending = h.prepare_binding(prefix, front, peak, row, rights.reservation->take(id), false);
    auto tx      = d.begin_transaction();
    tx.add_binding(0, std::move(pending));
    complete(tx);
    return tx.binding(0)->commit();
}

void write(ContextDirectory& d, HistoryOwner& owner, unsigned end) {
    auto& h = d.pool(0).histories();
    auto tx = d.begin_transaction();
    h.prepare_write(owner, h.committed_frontier(owner.handle()), end, tx);
    complete(tx);
    h.commit_frontier(owner.handle(), end);
}

void adopt(HistoryDirectory& h, HistoryId writer, RecordReservation claim) {
    h.prepare_growth(writer, claim);
    h.adopt_growth(writer, std::move(claim));
}

OptionalAdmissionWork anchor_when_full(unsigned blocks, bool cow) {
    ContextDirectory d(spec(blocks));
    auto& h       = d.pool(0).histories();
    auto original = bind(d, 0, 63);
    write(d, original, 63);
    auto prefix     = h.retain_prefix(original.handle(), 63);
    auto checkpoint = h.logical_page(original.handle(), 0);
    std::optional<HistoryOwner> clone;
    if (cow) {
        require(h.binding_demand(&prefix, 63, 64).tail_copies == 1,
                "shared anchor was not priced as COW");
        clone = bind(d, 1, 64, &prefix);
        require(h.logical_page(clone->handle(), 0) != checkpoint,
                "shared write reused checkpoint's page");
    }
    auto& owner   = clone ? *clone : original;
    const auto id = owner.handle();
    require(h.write_pages(id, 63, 64) == 0 && h.write_pages(id, 63, 65) == 1,
            "private page-boundary demand");
    ContextDemand anchor;
    auto anchor_id = anchor.add_growth(h, id, 0);
    auto permit    = d.reserve(anchor);
    require(bool(permit), "required anchor permit");
    adopt(h, id, permit.reservation->take(anchor_id));
    const auto busy = clone ? 2U : 1U;
    auto full = d.records().reserve(RecordDemand{1, ContextMemory::Device, blocks - busy, {1}});
    require(bool(full), "fill only rank 1");
    const auto reserved = [&](unsigned rank) {
        return d.records().pool({rank, ContextMemory::Device}).usage().reserved_bytes;
    };
    const auto r0 = reserved(0), r1 = reserved(1);
    ContextDemand optional;
    optional.add_growth(h, id, 1);
    OptionalAdmissionWork work;
    require(!d.try_reserve_optional(optional, &work), "exhausted rank granted optional page");
    require(work.metadata.examined == 0 && reserved(0) == r0 && reserved(1) == r1,
            "rejected growth modified resource rights");
    write(d, owner, 64);
    require(h.committed_frontier(id) == 64 && prefix.frontier() == 63 &&
                h.logical_page(prefix.history(), 0) == checkpoint,
            "denied optional growth invalidated required anchor or snapshot");
    return work;
}

enum class Donor { State, Page, History, Chunk };

OptionalAdmissionWork metadata_pressure(Donor donor, unsigned occupied_segments) {
    ContextDirectory d(spec(512, 16384));
    auto& pages     = d.pool(0).pages();
    auto& histories = d.pool(0).histories();
    // Every unrelated segment remains occupied during the trial.
    auto occupied = d.states().reserve_descriptors(occupied_segments * 256);
    if (donor == Donor::State) {
        auto idle = d.states().reserve_descriptors(256);
    } else if (donor == Donor::Page) {
        auto idle = pages.reserve_descriptors(256, 1);
    } else if (donor == Donor::History) {
        auto idle = histories.reserve_descriptors(256);
    } else {
        RecordReservation prepared;
        histories.reserve_control({{}, 64, 0}, prepared);
        prepared.clear();
        // Isolate the 256-chunk segment. The empty history descriptor segment goes first.
        require(histories.return_empty_segments(1).segments == 1, "chunk donor setup");
    }
    ContextDemand demand;
    if (donor == Donor::State)
        demand.add_history(0, 1);
    else
        demand.add_state({.count = 1});
    const auto bytes = d.assess(demand).metadata_required_bytes;
    auto& budget     = d.metadata_budget();
    require(bytes > 0 && budget.available() > bytes, "metadata pressure setup");
    const auto hold = budget.available() - bytes + 1;
    budget.reserve(hold);
    OptionalAdmissionWork work;
    auto result = d.try_reserve_optional(demand, &work);
    require(result && result->ready() && work.metadata.segments == 1,
            "failed to fund grant from one known idle segment");
    budget.cancel(hold);
    if (donor == Donor::Chunk) {
        RecordReservation restored;
        histories.reserve_control({{}, 64, 0}, restored);
    }
    return work;
}

void same_directory_donation() {
    ContextDirectory d(spec());
    auto older = d.states().reserve_descriptors(256);
    auto newer = d.states().reserve_descriptors(256);
    older.reset();
    newer.reset();
    ContextDemand demand;
    auto id = demand.add_state({.count = 1});
    demand.add_history(0, 1);
    auto& budget        = d.metadata_budget();
    const auto required = d.assess(demand).metadata_required_bytes;
    require(required && required < budget.available(), "same-directory fixture");
    const auto hold = budget.available() - required + 1;
    budget.reserve(hold);
    OptionalAdmissionWork work;
    auto grant = d.try_reserve_optional(demand, &work);
    require(grant && work.metadata.examined == 2 && work.metadata.segments == 1,
            "did not preserve the quoted empty segment and return its idle neighbour");
    auto state = d.states().reserve_destination(grant->take(id));
    require(state && d.states().release(*state), "grant lost its selected descriptor");
    budget.cancel(hold);
}

struct JointGrowth {
    ContextDirectory d{spec()};
    HistoryDirectory& h = d.pool(0).histories();
    HistoryOwner a = bind(d, 0, 63), b = bind(d, 1, 63);
    HistoryPrefix first, second;
    ContextDemand demand;
    ContextDemandId aid, bid;
    std::array<std::size_t, 2> reserved;
    unsigned state_available, page_available;
    PageId initial;
    std::optional<std::size_t> metadata_bytes;

    explicit JointGrowth(bool prewarm = true) {
        write(d, a, 63);
        write(d, b, 63);
        first   = h.retain_prefix(a.handle(), 63);
        second  = h.retain_prefix(b.handle(), 63);
        initial = h.logical_page(a.handle(), 0);
        // F+1 fits both private tails. Keep one previously granted recovery page for a.
        ContextDemand required;
        auto item  = required.add_growth(h, a.handle(), 1);
        auto grant = d.reserve(required);
        adopt(h, a.handle(), grant.reservation->take(item));
        demand.add_state({.count = 1});
        aid = demand.add_growth(h, a.handle(), 40);
        bid = demand.add_growth(h, b.handle(), 40);
        // Materialize reusable directory segments once, so the comparison below isolates
        // metadata owned by the actual grant from the directory's idle storage.
        if (prewarm) {
            {
                auto warm = d.try_reserve_optional(demand);
                require(bool(warm), "warm metadata storage");
            }
            metadata_bytes = d.metadata_budget().allocated();
        }
        state_available = d.states().available_descriptors();
        page_available  = d.pool(0).pages().available_descriptors();
        for (unsigned rank = 0; rank < 2; ++rank)
            reserved[rank] = d.records().pool({rank, ContextMemory::Device}).usage().reserved_bytes;
    }

    void unchanged() {
        for (unsigned rank = 0; rank < 2; ++rank)
            require(d.records().pool({rank, ContextMemory::Device}).usage().reserved_bytes ==
                        reserved[rank],
                    "partial transaction leaked rank-local records");
        require(d.states().available_descriptors() == state_available &&
                    d.pool(0).pages().available_descriptors() == page_available,
                "partial transaction leaked descriptors");
        require(!metadata_bytes || d.metadata_budget().allocated() == *metadata_bytes,
                "unpublished writer metadata remained charged");
        require(h.reserved_growth_pages(a.handle()) == 1 &&
                    h.reserved_growth_pages(b.handle()) == 0 &&
                    !h.growth_contributes(b.handle(), {}, true) && first.frontier() == 63 &&
                    second.frontier() == 63 && h.logical_page(first.history(), 0) == initial,
                "partial transaction changed either history or the required recovery grant");
    }
};

void atomic_grants() {
    // Stop at the first successful run: every preceding allocation site faulted exactly once.
    for (bool prewarm : {false, true}) {
        for (std::ptrdiff_t point = 0;; ++point) {
            JointGrowth fixture(prewarm);
            auto& d      = fixture.d;
            bool success = false;
            try {
                fail_at      = point;
                auto granted = d.try_reserve_optional(fixture.demand);
                fail_at      = -1;
                require(granted.has_value(), "control fixture didn't fit");
                auto taken = granted->take(fixture.bid);
                success    = true;
            } catch (const std::bad_alloc&) { fail_at = -1; }
            fixture.unchanged();
            if (success) break;
            {
                auto retry = d.try_reserve_optional(fixture.demand);
                require(bool(retry), "failed grant prevented a later retry");
            }
            fixture.unchanged();
        }
    }
    JointGrowth f;
    {
        auto issued  = f.d.try_reserve_optional(f.demand);
        auto partial = issued->take(f.bid);
        partial.shrink(1);
        partial.shrink(0);
        require(partial.empty(), "fully returned grant retained writer controls");
    }
    f.unchanged();
    auto grant = f.d.try_reserve_optional(f.demand);
    adopt(f.h, f.a.handle(), grant->take(f.aid));
    adopt(f.h, f.b.handle(), grant->take(f.bid));
    require(f.h.reserved_growth_pages(f.a.handle()) == 41 &&
                f.h.reserved_growth_pages(f.b.handle()) == 40,
            "joint controls not adopted");
    f.h.shrink_growth(f.a.handle(), 1);
    f.h.settle_reservation(f.b.handle(), 64);
    require(f.h.reserved_growth_pages(f.a.handle()) == 1 &&
                !f.h.growth_contributes(f.b.handle(), {}, true),
            "unused optional branch/chunks survived settlement");
}

} // namespace

int main() {
    try {
        const auto small = anchor_when_full(8, false), large = anchor_when_full(256, true);
        require(small.physical_classes && small.physical_classes == large.physical_classes,
                "physical probe grew with pool size");
        for (auto kind : {Donor::State, Donor::Page, Donor::History, Donor::Chunk}) {
            const auto a = metadata_pressure(kind, 1), b = metadata_pressure(kind, 32);
            require(a.physical_classes && a.physical_classes == b.physical_classes &&
                        a.metadata.examined == b.metadata.examined &&
                        a.metadata.segments == b.metadata.segments,
                    "metadata work grew with unrelated directory segments");
        }
        same_directory_donation();
        atomic_grants();
    } catch (const std::exception& e) {
        fail_at = -1;
        std::cerr << e.what() << '\n';
        return 1;
    }
}
