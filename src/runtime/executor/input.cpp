#include "runtime/executor/input.h"

#include <algorithm>

namespace ninfer::runtime {
namespace wire = distributed::wire;

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

void InputTransfers::progress(distributed::Transport& transport, std::uint64_t epoch) {
    constexpr std::size_t chunk_bytes = 256ULL << 10;
    for (auto& [key, transfer] : transfers_) {
        auto& resource = transfer.resource;
        while (transfer.phase != Phase::AwaitReady && transfer.phase != Phase::Ready) {
            if (transfer.phase == Phase::Chunks) {
                while (transfer.part < resource.parts.size() &&
                       transfer.offset == resource.parts[transfer.part].size()) {
                    ++transfer.part;
                    transfer.offset = 0;
                }
                if (transfer.part == resource.parts.size()) transfer.phase = Phase::End;
            }
            std::size_t bytes = 0;
            if (transfer.phase == Phase::Chunks)
                bytes =
                    std::min(chunk_bytes, resource.parts[transfer.part].size() - transfer.offset);
            if (!transfer.pending) {
                wire::Encoder out;
                out.u64(resource.id);
                auto kind = wire::Kind::InputEnd;
                if (transfer.phase == Phase::Begin) {
                    kind = wire::Kind::InputBegin;
                    out.bytes(resource.metadata);
                } else if (transfer.phase == Phase::Chunks) {
                    kind = wire::Kind::InputChunk;
                    out.u64(transfer.part);
                    out.u64(transfer.offset);
                    out.bytes(resource.parts[transfer.part].subspan(transfer.offset, bytes));
                }
                transfer.pending = std::make_shared<const wire::Message>(
                    wire::Message{kind, epoch, std::move(out).take()});
            }
            const distributed::OutgoingMessage message{key.first, transfer.pending};
            if (!transport.try_send({&message, 1}, distributed::Traffic::Input)) return;
            transfer.pending.reset();
            switch (transfer.phase) {
            case Phase::Begin:
                transfer.phase = Phase::Chunks;
                break;
            case Phase::Chunks:
                transfer.offset += bytes;
                break;
            case Phase::End:
                transfer.phase = Phase::AwaitReady;
                break;
            default:
                break;
            }
        }
    }
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
