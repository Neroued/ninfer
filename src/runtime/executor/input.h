#pragma once

#include "distributed/wire.h"
#include "distributed/transport.h"

#include <algorithm>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::runtime {

// Immutable model input, encoded by its native codec. Parts borrow the shared owner;
// transport copies only bounded chunks and never interprets their model contents.
struct InputResource {
    distributed::RankId rank;
    std::uint64_t id;
    distributed::wire::Bytes metadata;
    std::vector<std::span<const std::byte>> parts;
    std::shared_ptr<const void> owner;
};

struct InputRelease {
    std::uint64_t id;
};

class InputTransfers {
public:
    using Key = std::pair<distributed::RankId, std::uint64_t>;
    [[nodiscard]] std::vector<Key> begin(std::vector<InputResource>);
    template <class Sender>
    void progress(Sender&, std::uint64_t epoch);
    void complete(distributed::RankId, std::uint64_t id);
    [[nodiscard]] bool ready(std::span<const Key>) const;
    void release(std::uint64_t id);

private:
    enum class Phase { Begin, Chunks, End, AwaitReady, Ready };

    struct Transfer {
        InputResource resource;
        Phase phase      = Phase::Begin;
        std::size_t part = 0, offset = 0;
        std::shared_ptr<const distributed::wire::Message> pending;
    };

    std::map<Key, Transfer> transfers_;
};

template <class Sender>
void InputTransfers::progress(Sender& transport, std::uint64_t epoch) {
    namespace wire                    = distributed::wire;
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

} // namespace ninfer::runtime
