#include "core/context_pool.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <typename F>
void rejects(F&& function, std::string_view message) {
    try {
        function();
    } catch (const std::exception&) { return; }
    expect(false, message);
}

PoolReservation reserve(ContextPool& pool, std::initializer_list<RecordRequest> demand) {
    auto result = pool.reserve(demand);
    if (!result.reservation) throw std::runtime_error("expected reservation was rejected");
    return std::move(*result.reservation);
}

void accounting(const ContextPool& pool) {
    const auto usage = pool.usage();
    expect(usage.live_bytes + usage.reserved_bytes + usage.pending_retirement_bytes +
                   usage.free_class_bytes + usage.free_block_bytes + usage.padding_bytes +
                   usage.tail_bytes ==
               usage.capacity_bytes,
           "physical occupancy categories partition backing capacity");
}

void geometry_and_atomic_admission() {
    const std::array specs{RecordSpec{64, 8}, RecordSpec{24, 8}, RecordSpec{40, 8},
                           RecordSpec{24, 8}};
    const auto geometry = describe_context_pool(specs);
    expect(geometry.spec_classes[1] == geometry.spec_classes[3] && geometry.classes.size() == 3,
           "identical record specifications share one class");
    expect(geometry.block_bytes == 64 && geometry.record_limit(195) == 6,
           "geometry describes shared blocks and one common descriptor upper bound");
    const std::array demand{RecordRequest{geometry.spec_classes[0], 2},
                            RecordRequest{geometry.spec_classes[1], 1},
                            RecordRequest{geometry.spec_classes[2], 1}};
    expect(geometry.required_bytes(demand) == 256,
           "working-set planning includes each indivisible class block");
    ContextPool pool(7, 195, specs);
    const auto large = pool.record_class(0), small = pool.record_class(1),
               medium = pool.record_class(2);
    auto initial      = reserve(pool, {{small, 1}, {medium, 1}});
    const auto a = initial.materialize(small), b = initial.materialize(medium);
    const auto revision = pool.revision();
    const auto before   = pool.usage();
    const std::array extra{RecordRequest{large, 2}};
    expect(!pool.assess(extra).feasible() && !pool.reserve(extra).reservation,
           "aggregate free bytes cannot admit a mixed-class bundle without enough whole blocks");
    expect(pool.revision() == revision && pool.usage().live_bytes == before.live_bytes,
           "failed whole-bundle reservation leaves the ledger unchanged");
    const std::array releases{a, b, a};
    expect(pool.assess(extra, releases).feasible() && pool.valid(a) && pool.valid(b),
           "projected releases are deduplicated and do not release the source");
    pool.release(a);
    pool.release(b);
    auto next = reserve(pool, {{large, 2}});
    expect(next.ready() && next.remaining() == 2,
           "the same released conditions realize the assessment");
    next.cancel();
    expect(pool.usage().free_block_bytes == 192 && pool.usage().tail_bytes == 3,
           "empty blocks return to common capacity while the backing tail remains explicit");
    accounting(pool);
    expect(pool.usage().metadata_bytes <= geometry.metadata_bytes(pool.capacity_bytes()),
           "the geometry budget covers materialized allocator metadata");
    rejects(
        [] {
            const std::array bad{RecordSpec{1, 3}};
            (void)describe_context_pool(bad);
        },
        "invalid record alignment is rejected by the allocator");
}

void reservation_lifetime_and_resize() {
    const std::array specs{RecordSpec{64, 16}, RecordSpec{16, 16}};
    ContextPool pool(3, 128, specs);
    const auto large = pool.record_class(0), small = pool.record_class(1);
    auto owned          = reserve(pool, {{small, 4}});
    const auto live     = owned.materialize(small);
    const auto original = pool.resolve(live);
    const std::array too_large{RecordRequest{large, 2}};
    expect(!pool.resize(owned, too_large).feasible() && owned.remaining(small) == 3,
           "failed resize preserves the complete original permit");
    const std::array adjusted{RecordRequest{small, 1}, RecordRequest{large, 1}};
    expect(pool.resize(owned, adjusted).feasible() && owned.remaining() == 2,
           "resize atomically exchanges reserved classes without freeing the live object");
    expect(pool.resolve(live).offset == original.offset,
           "a reservation adjustment does not relocate unrelated live data when no maintenance is "
           "needed");
    auto moved = std::move(owned);
    expect(!owned.valid() && moved.ready(), "move transfers reservation ownership");
    moved.shrink(large, 0);
    moved.cancel();
    expect(pool.usage().reserved_bytes == 0 && pool.usage().live_bytes == 16,
           "reservation cancellation releases only its unmaterialized records");
    pool.release(live);
    auto recycled          = reserve(pool, {{small, 1}});
    const auto replacement = recycled.materialize(small);
    expect(!pool.valid(live) && replacement != live,
           "recycled descriptors do not revive stale allocation IDs");
    rejects([&] { (void)pool.resolve(live); },
            "stale identity cannot resolve to a replacement payload");
    pool.return_to_reservation(replacement, recycled);
    expect(
        !pool.valid(replacement) && recycled.remaining(small) == 1,
        "return-to-reservation invalidates the previous allocation without releasing its position");
    const auto again = recycled.materialize(small);
    pool.release(again);
    accounting(pool);
}

struct FragmentedPool {
    std::array<RecordSpec, 2> specs{{{64, 16}, {16, 16}}};
    ContextPool pool{1, 192, specs};
    RecordClassId large = pool.record_class(0), small = pool.record_class(1);
    std::vector<AllocationId> live;
    std::array<unsigned char, 192> bytes{};

    FragmentedPool() {
        auto records = reserve(pool, {{small, 12}});
        for (int i = 0; i < 12; ++i) {
            auto allocation    = records.materialize(small);
            const auto binding = pool.resolve(allocation);
            if ((binding.offset / 16) % 2 == 0)
                pool.release(allocation);
            else {
                live.push_back(allocation);
                std::memset(bytes.data() + binding.offset, static_cast<int>(allocation.index + 1),
                            binding.bytes);
            }
        }
    }

    void copy() {
        const auto moves = pool.maintenance_moves();
        for (const auto& move : moves) {
            for (const auto& other : moves)
                expect(move.destination_offset + move.bytes <= other.source_offset ||
                           other.source_offset + other.bytes <= move.destination_offset,
                       "maintenance sources and destinations are globally disjoint");
            std::memcpy(bytes.data() + move.destination_offset, bytes.data() + move.source_offset,
                        move.bytes);
        }
    }

    void verify() {
        for (const auto allocation : live) {
            const auto binding = pool.resolve(allocation);
            expect(std::all_of(bytes.begin() + binding.offset,
                               bytes.begin() + binding.offset + binding.bytes,
                               [&](auto byte) { return byte == allocation.index + 1; }),
                   "relocation preserves payload associated with stable allocation identity");
        }
        accounting(pool);
    }
};

void compaction_and_cancel() {
    FragmentedPool f;
    const std::array demand{RecordRequest{f.large, 1}};
    const auto assessment = f.pool.assess(demand);
    expect(assessment.status == PoolAvailability::NeedsMaintenance && assessment.move_bytes == 32,
           "reclaiming one sparse block chooses only the minimum live payload");
    {
        auto permit = reserve(f.pool, {{f.large, 1}});
        expect(!permit.ready() && f.pool.has_maintenance(),
               "future positions await the reserved maintenance plan");
        expect(f.pool.usage().future_reserved_bytes == 64 && f.pool.usage().relocation_bytes == 32,
               "future use is distinct from temporary physical relocation occupancy");
        accounting(f.pool);
        permit.cancel();
    }
    expect(!f.pool.has_maintenance(),
           "unsubmitted reservation cancellation restores the old placement");
    f.verify();
    auto permit = reserve(f.pool, {{f.large, 1}});
    std::vector<RecordBinding> before;
    for (auto allocation : f.live) before.push_back(f.pool.resolve(allocation));
    f.pool.begin_maintenance();
    rejects([&] { f.pool.cancel_maintenance(); }, "submitted maintenance cannot be rolled back");
    rejects([&] { (void)f.pool.resolve(f.live.front()); },
            "physical bindings are unavailable inside maintenance");
    f.copy();
    f.pool.complete_maintenance();
    expect(permit.ready(),
           "copy and binding completion makes all promised positions materializable");
    std::size_t moved = 0;
    for (const auto& old : before) {
        const auto now = f.pool.resolve(old.allocation);
        if (old.offset != now.offset) {
            ++moved;
            expect(now.binding_epoch == old.binding_epoch + 1,
                   "relocation advances binding epoch without changing ID");
        } else
            expect(now.binding_epoch == old.binding_epoch, "unmoved bindings retain their epoch");
    }
    expect(moved == 2, "compaction does not relocate unrelated blocks");
    const auto state = permit.materialize(f.large);
    f.verify();
    f.pool.release(state);
}

void cancellation_after_submit_and_retirement() {
    FragmentedPool f;
    auto permit         = reserve(f.pool, {{f.large, 1}});
    const auto retiring = f.pool.maintenance_moves().front().allocation;
    f.pool.begin_maintenance();
    f.pool.defer_release(retiring);
    permit.cancel();
    f.copy();
    f.pool.complete_maintenance();
    expect(f.pool.valid(retiring) && f.pool.usage().pending_retirement_bytes == 16,
           "a cancelled live owner stays charged through submitted relocation");
    f.pool.retire(retiring);
    std::erase(f.live, retiring);
    expect(f.pool.usage().reserved_bytes == 0 && f.pool.usage().free_block_bytes == 64,
           "request cancellation completes submitted movement then releases its future positions");
    f.verify();

    const std::array specs{RecordSpec{64, 16}};
    ContextPool pool(2, 64, specs);
    const auto record_class = pool.record_class(0);
    auto initial            = reserve(pool, {{record_class, 1}});
    const auto source       = initial.materialize(record_class);
    const auto offset       = pool.resolve(source).offset;
    auto handover           = reserve(pool, {});
    handover.prepare_returns(record_class, 1);
    pool.defer_to_reservation(source, handover);
    expect(!handover.ready() && pool.usage().pending_retirement_bytes == 64,
           "retirement handover retains exclusive occupancy while the consumer is active");
    const std::array demand{RecordRequest{record_class, 1}};
    expect(!pool.reserve(demand).reservation, "another request cannot steal a reserved handover");
    pool.retire(source);
    expect(!pool.valid(source) && handover.ready() && pool.usage().reserved_bytes == 64,
           "retirement changes pending occupancy directly into the original permit");
    const auto destination = handover.materialize(record_class);
    expect(pool.resolve(destination).offset == offset,
           "retirement reuse never releases and reacquires the record");
    pool.defer_release(destination);
    expect(pool.assess(demand).status == PoolAvailability::WaitingForRetirement,
           "pending retirement is reported as a dependency instead of immediate capacity");
    pool.retire(destination);
    expect(pool.assess(demand).feasible(), "retired unclaimed records return to shared capacity");
    accounting(pool);
}

void unmaterialized_relocation() {
    const std::array specs{RecordSpec{64, 16}, RecordSpec{16, 16}};
    ContextPool pool(9, 192, specs);
    const auto large = pool.record_class(0), small = pool.record_class(1);
    std::array<PoolReservation, 12> records;
    for (auto& record : records) record = reserve(pool, {{small, 1}});
    for (std::size_t i = 1; i < records.size(); i += 2) records[i].cancel();
    auto target = reserve(pool, {{large, 1}});
    expect(pool.has_maintenance() && pool.maintenance_moves().empty(),
           "unmaterialized reservations relocate without payload copies");
    target.shrink(large, 0);
    expect(!pool.has_maintenance() && !target.valid(),
           "shrinking every future claim cancels an unsubmitted maintenance plan");
    target = reserve(pool, {{large, 1}});
    pool.begin_maintenance();
    records.front().cancel();
    pool.complete_maintenance();
    std::vector<RecordBinding> bindings{pool.resolve(target.materialize(large))};
    for (std::size_t i = 2; i < records.size(); i += 2)
        bindings.push_back(pool.resolve(records[i].materialize(small)));
    for (std::size_t i = 0; i < bindings.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            expect(bindings[i].offset + bindings[i].bytes <= bindings[j].offset ||
                       bindings[j].offset + bindings[j].bytes <= bindings[i].offset,
                   "retargeted reservations preserve disjoint promised positions");
    accounting(pool);
    for (const auto& binding : bindings) pool.release(binding.allocation);
    expect(pool.usage().free_block_bytes == pool.capacity_bytes(),
           "reservation cancellation during metadata relocation releases all records");
}

void retire_while_maintenance_waits() {
    FragmentedPool f;
    const auto retiring = f.live.front();
    auto permit         = reserve(f.pool, {{f.large, 1}});
    f.pool.defer_release(retiring);
    f.pool.retire(retiring);
    std::erase(f.live, retiring);
    f.pool.begin_maintenance();
    f.copy();
    f.pool.complete_maintenance();
    f.verify();
    expect(permit.ready(),
           "existing consumers can retire while maintenance waits for its borrow gate");
}

void failed_maintenance_drain() {
    for (const bool discard_owner : {false, true}) {
        FragmentedPool f;
        std::vector<RecordBinding> before;
        for (auto allocation : f.live) before.push_back(f.pool.resolve(allocation));
        auto permit = reserve(f.pool, {{f.large, 1}});
        f.pool.begin_maintenance();
        const auto partial = f.pool.maintenance_moves().front();
        std::memcpy(f.bytes.data() + partial.destination_offset,
                    f.bytes.data() + partial.source_offset, partial.bytes);
        if (discard_owner) permit.cancel();
        f.pool.abandon_maintenance_after_failure();
        expect(!permit.valid() && !f.pool.has_maintenance(),
               "failed drained movement discards its unmaterialized permit");
        for (const auto& old : before) {
            const auto current = f.pool.resolve(old.allocation);
            expect(current.offset == old.offset && current.binding_epoch == old.binding_epoch,
                   "failed movement published a partial destination binding");
        }
        f.verify();
    }
}
} // namespace

int main() {
    try {
        geometry_and_atomic_admission();
        reservation_lifetime_and_resize();
        compaction_and_cancel();
        cancellation_after_submit_and_retirement();
        unmaterialized_relocation();
        retire_while_maintenance_waits();
        failed_maintenance_drain();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
