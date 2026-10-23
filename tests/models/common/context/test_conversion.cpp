#include "models/common/context/directory.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
namespace context = ninfer::models::context;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& operation, const char* message) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

struct Fixture {
    static constexpr std::size_t capacity = 64 * 1024;
    context::ContextDirectory directory{
        {.ranks = {{0, capacity, 0}, {1, capacity, 0}},
         .state = {32, {{0, {{0, 64, 16}, {1, 32, 16}}}, {1, {{0, 32, 16}, {1, 64, 16}}}}},
         .pools = {{.pool_id       = 0,
                    .logical_pages = 64,
                    .histories     = 16,
                    .page_capacity = 16,
                    .table_rows    = 3,
                    .parts         = {{0, {256, 16}}, {1, {256, 16}}}}},
         .metadata_budget_bytes = 16 * 1024 * 1024}};
    std::array<std::vector<std::byte>, 2> backing{
        std::vector<std::byte>(capacity, std::byte{0x7f}),
        std::vector<std::byte>(capacity, std::byte{0x7f})};

    context::HistoryDirectory& histories() { return directory.pool(0).histories(); }

    context::PageDirectory& pages() { return directory.pool(0).pages(); }

    std::byte* bytes(std::uint32_t rank, context::ContextRange range) {
        require(range.memory == context::ContextMemory::Device && range.offset >= 0 &&
                    std::size_t(range.offset) + range.bytes <= backing.at(rank).size(),
                "action escaped its backing");
        return backing[rank].data() + range.offset;
    }

    std::byte* bytes(context::ContextRecordBinding record) {
        return bytes(record.domain.rank, {record.domain.memory, record.offset, record.bytes});
    }

    void execute(context::ContextTransaction& transaction) {
        // Independent byte interpreter. Physical CUDA/NCCL execution has its own transfer tests.
        for (unsigned rank = 0; rank < 2; ++rank) {
            for (const auto& action : transaction.actions(rank).operations) {
                if (const auto* peer = std::get_if<context::PeerCopy>(&action)) {
                    if (!peer->send) continue;
                    const context::PeerCopy* receiver = nullptr;
                    for (const auto& other : transaction.actions(peer->peer).operations)
                        if (const auto* match = std::get_if<context::PeerCopy>(&other);
                            match && !match->send && match->peer == rank &&
                            match->resource == peer->resource) {
                            require(!receiver, "peer transfer has two matching receivers");
                            receiver = match;
                        }
                    require(receiver && receiver->ranges.size() == peer->ranges.size(),
                            "peer ranges do not match");
                    for (std::size_t i = 0; i < peer->ranges.size(); ++i) {
                        require(peer->ranges[i].bytes == receiver->ranges[i].bytes,
                                "peer byte counts differ");
                        std::memcpy(bytes(peer->peer, receiver->ranges[i]),
                                    bytes(rank, peer->ranges[i]), peer->ranges[i].bytes);
                    }
                } else if (const auto* copy = std::get_if<context::RecordCopy>(&action)) {
                    require(copy->source.bytes == copy->destination.bytes,
                            "local copy byte counts differ");
                    std::memcpy(bytes(rank, copy->destination), bytes(rank, copy->source),
                                copy->source.bytes);
                } else if (const auto* clear = std::get_if<context::RecordClear>(&action)) {
                    std::memset(bytes(rank, clear->destination), 0, clear->destination.bytes);
                }
            }
        }
    }

    void complete(context::ContextTransaction& transaction) {
        transaction.prepare_actions();
        transaction.mark_submitted();
        execute(transaction);
        transaction.mark_completed();
        transaction.publish();
    }

    context::RecordReservation reserve(const context::HistoryBindingDemand& demand) {
        context::ContextDemand bundle;
        const auto key = bundle.add_history(0, demand);
        auto result    = directory.reserve(bundle);
        require(result && result.reservation->ready(), "history fixture could not reserve");
        return result.reservation->take(key);
    }

    context::RecordReservation reserve_state(unsigned variant) {
        context::ContextDemand bundle;
        const auto key = bundle.add_state({.count = 1, .variant = variant});
        auto result    = directory.reserve(bundle);
        require(result && result.reservation->ready(), "State fixture could not reserve");
        return result.reservation->take(key);
    }

    context::HistoryOwner history(std::uint32_t frontier, unsigned rank) {
        const std::array placement{rank};
        const auto demand = histories().binding_demand(nullptr, 0, frontier, {}, placement);
        auto transaction  = directory.begin_transaction();
        transaction.add_binding(0, histories().prepare_binding(nullptr, 0, frontier, rank,
                                                               reserve(demand), false, placement));
        complete(transaction);
        auto owner = transaction.binding(0)->commit();
        auto write = directory.begin_transaction();
        histories().prepare_write(owner, 0, frontier, write);
        complete(write);
        histories().commit_frontier(owner.handle(), frontier);
        return owner;
    }
};

void state_conversion() {
    Fixture fixture;
    auto& states         = fixture.directory.states();
    const auto source    = states.reserve_destination(fixture.reserve_state(0), 0).value();
    const auto from_zero = states.device_binding(source, 0);
    const auto from_one  = states.device_binding(source, 1);
    std::memset(fixture.bytes(from_zero), 0x11, 32);
    std::memset(fixture.bytes(from_zero) + 32, 0x33, 32);
    std::memset(fixture.bytes(from_one), 0x22, 32);
    states.complete_initialization(source);
    states.freeze(source);
    const auto epoch = states.content_epoch(source);
    const std::array fields{context::RecordRangeCopy{0, 0, 0, 0, 32},
                            context::RecordRangeCopy{1, 1, 0, 0, 32},
                            context::RecordRangeCopy{0, 1, 32, 32, 32}};
    auto transaction = fixture.directory.begin_transaction();
    transaction.convert_state(source, 1, fixture.reserve_state(1), fields);
    const auto destination = transaction.state_destination().image();
    rejects([&] { (void)states.begin_write(source, destination); },
            "ordinary fork accepted a layout conversion");
    require(states.role(destination) == context::StateRole::ReservedDestination &&
                states.content_epoch(destination) == 0 && states.source_pins(source) == 1,
            "conversion published before retirement or lost its source lease");
    transaction.prepare_actions();
    transaction.mark_submitted();
    rejects([&] { transaction.publish(); }, "in-flight conversion was published");
    fixture.execute(transaction);
    transaction.mark_completed();
    transaction.publish();
    require(states.variant(destination) == 1 && states.content_epoch(destination) == epoch &&
                states.source_pins(source) == 0,
            "layout conversion changed the mathematical content epoch");
    const auto to_zero = states.device_binding(destination, 0);
    const auto to_one  = states.device_binding(destination, 1);
    require(std::memcmp(fixture.bytes(to_zero), fixture.bytes(from_zero), 32) == 0 &&
                std::memcmp(fixture.bytes(to_one), fixture.bytes(from_one), 32) == 0 &&
                std::memcmp(fixture.bytes(to_one) + 32, fixture.bytes(from_zero) + 32, 32) == 0,
            "rank-local fields or owner tail did not reach their target layout");
    (void)transaction.state_destination().release();
    states.complete_write(destination, destination);
    require(states.content_epoch(destination) != epoch, "native write preserved a stale epoch");
    require(states.release(destination) && states.release(source), "State conversion cleanup");
}

void history_conversion() {
    Fixture fixture;
    auto source      = fixture.history(96, 0);
    auto prefix      = fixture.histories().retain_prefix(source.handle(), 80);
    const auto first = fixture.histories().logical_page(source.handle(), 0);
    const auto tail  = fixture.histories().logical_page(source.handle(), 1);
    std::memset(fixture.bytes(fixture.pages().physical(first, 0)), 0x44, 256);
    std::memset(fixture.bytes(fixture.pages().physical(tail, 0)), 0x55, 64);
    std::memset(fixture.bytes(fixture.pages().physical(tail, 0)) + 64, 0x66, 192);
    const std::array placement{1U};
    rejects([&] { (void)fixture.histories().binding_demand(&prefix, 80, 128, {}, placement); },
            "ordinary binding accepted a different history placement");
    const auto demand = fixture.histories().conversion_demand(prefix, 80, 128, placement);
    require(demand.device_pages() == 2 && demand.logical_pages == 2 &&
                demand.records.front().ranks == std::vector{1U},
            "conversion demand did not reserve all destination history pages");
    auto transaction = fixture.directory.begin_transaction();
    transaction.add_binding(0, fixture.histories().prepare_conversion(
                                   prefix, 80, 128, 1, fixture.reserve(demand), placement,
                                   {{{0, 1, 0, 0, 256}}, {{0, 1, 0, 0, 64}}, 16}));
    const auto destination = transaction.binding(0)->history();
    require(fixture.histories().committed_frontier(destination) == 0 &&
                fixture.histories().mapped_pages(destination) == 0 &&
                fixture.pages().source_pins(tail) == 1,
            "history conversion exposed incomplete coverage or failed to lease a writer tail");
    rejects([&] { (void)transaction.binding(0)->commit(); },
            "history conversion bound before retirement");
    fixture.complete(transaction);
    auto converted = transaction.binding(0)->commit();
    require(fixture.histories().committed_frontier(converted.handle()) == 80 &&
                fixture.histories().placement(converted.handle()).front() == 1,
            "history conversion changed frontier or placement");
    const auto target_first = fixture.histories().logical_page(converted.handle(), 0);
    const auto target_tail  = fixture.histories().logical_page(converted.handle(), 1);
    require(target_first != first && target_tail != tail &&
                fixture.pages().content_epoch(target_first) == fixture.pages().content_epoch(first),
            "conversion reused source physical identity or changed content epoch");
    const auto* target = fixture.bytes(fixture.pages().physical(target_tail, 1));
    require(std::all_of(target, target + 64, [](auto byte) { return byte == std::byte{0x55}; }) &&
                std::all_of(target + 64, target + 256,
                            [](auto byte) { return byte == std::byte{0x7f}; }),
            "partial-page copy read the source writer suffix");
    require(fixture.pages().source_pins(tail) == 0 &&
                fixture.histories().committed_frontier(source.handle()) == 96,
            "conversion changed or pinned the original writer after publication");
    require(fixture.histories().released_pages(source.handle()).empty() &&
                fixture.histories().released_histories(source.handle()).empty(),
            "active-owner release quote counted snapshot-retained pages");
    prefix.reset();
    require(fixture.histories().released_pages(source.handle()).size() == 2 &&
                fixture.histories().released_histories(source.handle()).size() == 1,
            "active-owner release quote missed unshared pages");
    source.reset();
    converted.reset();
    require(fixture.pages().occupied() == 0, "converted history pages leaked");
}

void cancellation() {
    Fixture fixture;
    auto source            = fixture.history(16, 0);
    auto prefix            = fixture.histories().retain_prefix(source.handle(), 16);
    const auto source_page = fixture.histories().logical_page(source.handle(), 0);
    const std::array placement{1U};
    for (const bool submitted : {false, true}) {
        const auto before = fixture.directory.records()
                                .pool({1, context::ContextMemory::Device})
                                .usage()
                                .live_bytes;
        auto transaction  = fixture.directory.begin_transaction();
        const auto demand = fixture.histories().conversion_demand(prefix, 16, 64, placement);
        transaction.add_binding(
            0, fixture.histories().prepare_conversion(prefix, 16, 64, 1, fixture.reserve(demand),
                                                      placement, {{}, {{0, 1, 0, 0, 64}}, 16}));
        const auto target = transaction.binding(0)->history();
        if (submitted) {
            transaction.prepare_actions();
            transaction.mark_submitted();
            transaction.cancel();
            rejects([&] { transaction.discard(); }, "cancel released an in-flight destination");
            fixture.execute(transaction);
            transaction.mark_completed();
            rejects([&] { transaction.publish(); }, "cancelled transfer was published");
        }
        transaction.discard();
        require(!fixture.histories().valid(target) &&
                    fixture.pages().source_pins(source_page) == 0 &&
                    fixture.pages().occupied() == 1 &&
                    fixture.directory.records()
                            .pool({1, context::ContextMemory::Device})
                            .usage()
                            .live_bytes == before,
                "discard failed to release unpublished history destinations");
    }
}
} // namespace

int main() {
    try {
        state_conversion();
        history_conversion();
        cancellation();
        std::cout << "context conversion tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
