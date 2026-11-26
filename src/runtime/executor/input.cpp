#include "runtime/executor/input.h"

#include <set>
#include <stdexcept>

namespace ninfer::runtime {

auto InputTransfers::find(Key key) -> std::list<Transfer>::iterator {
    auto& transfers = peers_.at(key.first).transfers;
    const auto it = std::find_if(transfers.begin(), transfers.end(),
                                 [&](const auto& transfer) { return transfer.id == key.second; });
    if (it == transfers.end()) throw std::logic_error("unknown execution input transfer");
    return it;
}

void InputTransfers::validate_frames(std::span<const InputResource> resources,
                                     const distributed::TransportLimits& limits) {
    namespace wire = distributed::wire;
    constexpr auto traffic = distributed::Traffic::Input;
    constexpr auto field = sizeof(std::uint64_t);
    for (const auto& resource : resources) {
        limits.validate_outbound_frame(resource.rank, wire::Kind::InputBegin,
                                       2 * field + resource.metadata.size(), traffic);
        limits.validate_outbound_frame(resource.rank, wire::Kind::InputEnd, field, traffic);
        for (const auto part : resource.parts)
            if (!part.empty())
                limits.validate_outbound_frame(resource.rank, wire::Kind::InputChunk,
                                               4 * field + std::min(part.size(), kChunkBytes),
                                               traffic);
    }
}

std::vector<InputTransfers::Key> InputTransfers::begin(std::vector<InputResource> resources) {
    std::vector<Key> keys;
    keys.reserve(resources.size());
    std::set<distributed::RankId> batch_peers;
    for (const auto& resource : resources) {
        const Key key{resource.rank, resource.id};
        const auto peer = peers_.find(key.first);
        if (!batch_peers.insert(resource.rank).second)
            throw std::logic_error("one execution cannot require multiple inputs on one peer");
        if (!resource.id ||
            (peer != peers_.end() &&
             std::any_of(peer->second.transfers.begin(), peer->second.transfers.end(),
                         [&](const auto& transfer) { return transfer.id == key.second; })))
            throw std::logic_error("duplicate or zero execution input transfer");
        keys.push_back(key);
    }
    std::size_t inserted = 0;
    try {
        for (auto& resource : resources) {
            peers_[resource.rank].transfers.push_back(Transfer{std::move(resource)});
            ++inserted;
        }
    } catch (...) {
        release(std::span(keys).first(inserted));
        throw;
    }
    return keys;
}

void InputTransfers::acknowledge_chunks(distributed::RankId rank, std::uint64_t count) {
    auto& inflight = peers_.at(rank).in_flight_chunks;
    if (!count || count > inflight)
        throw std::logic_error("input chunk credit exceeds in-flight count");
    inflight -= static_cast<std::size_t>(count);
}

void InputTransfers::complete(distributed::RankId rank, std::uint64_t id) {
    auto& transfer = *find({rank, id});
    if (transfer.phase != Phase::AwaitReady)
        throw std::logic_error("execution input readiness precedes its payload");
    transfer.phase    = transfer.released ? Phase::Release : Phase::Ready;
    transfer.resource = {};
}

bool InputTransfers::ready(std::span<const Key> keys) const {
    return std::all_of(keys.begin(), keys.end(), [&](const auto& key) {
        const auto& transfers = peers_.at(key.first).transfers;
        return !transfers.empty() && transfers.front().id == key.second &&
               transfers.front().phase == Phase::Ready;
    });
}

bool InputTransfers::has_pending_releases() const noexcept {
    return std::any_of(peers_.begin(), peers_.end(), [](const auto& entry) {
        const auto& transfers = entry.second.transfers;
        return !transfers.empty() && transfers.front().released;
    });
}

void InputTransfers::release(std::span<const Key> keys) {
    for (const auto& key : keys) {
        auto transfer = find(key);
        if (transfer->phase == Phase::Begin) {
            peers_.at(key.first).transfers.erase(transfer);
        } else {
            transfer->released = true;
            if (transfer->phase == Phase::Ready) transfer->phase = Phase::Release;
        }
    }
}

} // namespace ninfer::runtime
