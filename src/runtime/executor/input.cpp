#include "runtime/executor/input.h"

#include <algorithm>

namespace ninfer::runtime {
std::vector<InputTransfers::Key> InputTransfers::begin(std::vector<InputResource> resources) {
    std::vector<Key> keys;
    for (auto& resource : resources) {
        Key key{resource.rank, resource.id};
        if (!resource.id || !transfers_.try_emplace(key, Transfer{std::move(resource)}).second)
            throw std::logic_error("duplicate execution input transfer");
        keys.push_back(key);
    }
    return keys;
}

void InputTransfers::complete(distributed::RankId rank, std::uint64_t id) {
    auto& transfer = transfers_.at({rank, id});
    if (transfer.phase != Phase::AwaitReady)
        throw std::logic_error("execution input readiness precedes its payload");
    transfer.phase    = Phase::Ready;
    transfer.resource = {};
}

bool InputTransfers::ready(std::span<const Key> keys) const {
    return std::all_of(keys.begin(), keys.end(),
                       [&](const auto& key) { return transfers_.at(key).phase == Phase::Ready; });
}

void InputTransfers::release(std::uint64_t id) {
    std::erase_if(transfers_, [&](const auto& item) { return item.first.second == id; });
}

} // namespace ninfer::runtime
