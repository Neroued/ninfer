#pragma once

#include "models/common/context/recovery.h"
#include "models/common/qwen/prefix_identity.h"

namespace ninfer::models::qwen {

// Both Qwen Programs expose their immutable identity through the record, keeping native
// continuation layouts and backend frontiers outside the shared recovery algorithm.
template <class Store>
struct RecoveryStoreView {
    const Store& store;

    const std::uint64_t& revision() const { return store.revision(); }

    std::size_t capacity() const { return store.size(); }

    auto handle(std::uint32_t index) const { return store.handle(index); }

    bool valid(context::CheckpointHandle handle) const { return store.valid(handle); }

    std::uint32_t frontier(context::CheckpointHandle handle) const {
        return store.at(handle).frontier();
    }

    context::RecoveryKey key(context::CheckpointHandle handle, std::uint32_t frontier) const {
        const auto identity = store.at(handle).recovery_identity();
        return {identity.digests.at(frontier), identity.domain};
    }

    bool prefix(context::CheckpointHandle earlier, context::CheckpointHandle later) const {
        return recovery_prefix_matches(store.at(earlier).recovery_identity(),
                                       store.at(later).recovery_identity());
    }
};

} // namespace ninfer::models::qwen
