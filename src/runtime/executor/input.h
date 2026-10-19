#pragma once

#include "distributed/wire.h"
#include "distributed/transport.h"

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
    void progress(distributed::Transport&, std::uint64_t epoch);
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

} // namespace ninfer::runtime
