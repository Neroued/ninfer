#pragma once
#include "models/common/context/prefix.h"
#include "models/common/context/recovery.h"

namespace ninfer::models::context {
template <class Store>
struct RecoveryStoreView {
    const Store& store;
    const PrefixStore& prefixes;

    const std::uint64_t& revision() const { return store.revision(); }

    std::size_t capacity() const { return store.metadata_extent(); }

    auto handle(std::uint32_t i) const { return store.handle(i); }

    bool valid(CheckpointHandle h) const { return store.valid(h); }

    std::uint32_t frontier(CheckpointHandle h) const { return store.at(h).frontier(); }

    auto relations() const {
        std::vector<PrefixId> ids(capacity());
        for (std::uint32_t i = 0; i < ids.size(); ++i)
            if (valid(handle(i))) ids[i] = store.at(handle(i)).snapshot.input_prefix().id();
        return prefixes.relations(ids);
    }
};
} // namespace ninfer::models::context
