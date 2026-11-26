#pragma once

#include "distributed/wire.h"
#include "distributed/transport.h"

#include <algorithm>
#include <list>
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

class InputTransfers {
public:
    static constexpr std::size_t kMaxInFlightChunksPerPeer = 4;
    static constexpr std::size_t kChunkBytes               = 256ULL << 10;
    using Key = std::pair<distributed::RankId, std::uint64_t>;
    static void validate_frames(std::span<const InputResource>,
                                const distributed::TransportLimits&);
    [[nodiscard]] std::vector<Key> begin(std::vector<InputResource>);
    template <class Sender>
    void progress(Sender&, std::uint64_t epoch);
    void acknowledge_chunks(distributed::RankId, std::uint64_t count);
    void complete(distributed::RankId, std::uint64_t id);
    [[nodiscard]] bool ready(std::span<const Key>) const;
    [[nodiscard]] bool has_pending_releases() const noexcept;
    // Unstarted transfers can be dropped. A started transfer completes its receive protocol
    // before release, even if its Submit is cancelled.
    void release(std::span<const Key>);

private:
    enum class Phase { Begin, Chunks, End, AwaitReady, Ready, Release };

    struct Transfer {
        InputResource resource;
        std::uint64_t id = resource.id;
        Phase phase      = Phase::Begin;
        std::size_t part = 0, offset = 0;
        bool released = false;
        std::shared_ptr<const distributed::wire::Message> pending;
    };

    struct Peer {
        std::list<Transfer> transfers;
        std::size_t in_flight_chunks = 0;
    };

    [[nodiscard]] auto find(Key key) -> std::list<Transfer>::iterator;

    std::map<distributed::RankId, Peer> peers_;
};

template <class Sender>
void InputTransfers::progress(Sender& transport, std::uint64_t epoch) {
    namespace wire = distributed::wire;
    for (auto& [peer, state] : peers_) {
        if (state.transfers.empty()) continue;
        auto& transfer = state.transfers.front();
        if (transfer.phase == Phase::Release) {
            if (!transfer.pending) {
                wire::Encoder out;
                out.u64(transfer.id);
                transfer.pending = std::make_shared<const wire::Message>(
                    wire::Message{wire::Kind::InputRelease, epoch, std::move(out).take()});
            }
            const distributed::OutgoingMessage message{peer, transfer.pending};
            if (transport.try_send({&message, 1}, distributed::Traffic::Continuation))
                state.transfers.pop_front();
            continue;
        }
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
            if (transfer.phase == Phase::Chunks &&
                state.in_flight_chunks >= kMaxInFlightChunksPerPeer) {
                break;
            }
            std::size_t bytes = 0;
            if (transfer.phase == Phase::Chunks)
                bytes =
                    std::min(kChunkBytes, resource.parts[transfer.part].size() - transfer.offset);
            if (!transfer.pending) {
                wire::Encoder out;
                out.u64(transfer.id);
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
            const distributed::OutgoingMessage message{peer, transfer.pending};
            if (!transport.try_send({&message, 1}, distributed::Traffic::Input)) break;
            transfer.pending.reset();
            switch (transfer.phase) {
            case Phase::Begin:
                transfer.phase = Phase::Chunks;
                break;
            case Phase::Chunks:
                transfer.offset += bytes;
                ++state.in_flight_chunks;
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
