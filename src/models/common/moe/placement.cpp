#include "models/common/moe/placement.h"

#include <set>

namespace ninfer::models::moe {

std::uint64_t ExpertDomain::bank_signature(std::size_t b) const {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix     = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) {
            hash ^= (word >> (i * 8)) & 255U;
            hash *= 1099511628211ULL;
        }
    };
    mix(b);
    mix(experts);
    mix(intermediate);
    mix(members.size());
    const auto& parts = banks.at(b);
    for (std::size_t peer = 0; peer < members.size(); ++peer) {
        mix(members[peer]);
        mix(peer_owners.at(peer));
        const auto& p = parts.at(peer);
        mix(p.intermediate.begin);
        mix(p.intermediate.size);
        mix(p.ids.size());
        for (auto id : p.ids) mix(id);
    }
    return hash;
}

void ExpertDomain::validate() const {
    if (!experts || !intermediate || members.empty() || banks.empty() ||
        std::set<distributed::RankId>(members.begin(), members.end()).size() != members.size())
        throw std::invalid_argument("invalid expert dispatch domain");
    if (peer_owners.size() != members.size())
        throw std::invalid_argument("missing expert peer ownership");
    std::uint32_t next_owner = 0;
    for (auto owner : peer_owners) {
        if (owner > next_owner) throw std::invalid_argument("invalid expert peer ownership");
        if (owner == next_owner) ++next_owner;
    }
    for (const auto& bank : banks) {
        if (bank.size() != members.size())
            throw std::invalid_argument("invalid expert dispatch peers");
        std::vector<std::vector<IndexRange>> slices(experts);
        for (const auto& p : bank) {
            const auto r = p.intermediate;
            if (p.ids.empty() || !r.size || r.begin >= intermediate ||
                r.size > intermediate - r.begin)
                throw std::invalid_argument("invalid expert dispatch slice");
            for (const auto e : p.ids) {
                if (e >= experts) throw std::invalid_argument("invalid dispatch expert ID");
                slices[e].push_back(r);
            }
        }
        for (auto& s : slices) {
            std::ranges::sort(s, {}, &IndexRange::begin);
            std::uint32_t next = 0;
            for (const auto r : s) {
                if (r.begin != next)
                    throw std::invalid_argument("gap or overlap in expert dispatch slices");
                next += r.size;
            }
            if (next != intermediate)
                throw std::invalid_argument("incomplete expert dispatch slices");
        }
    }
}

void ExpertDomain::validate_local(distributed::RankId rank,
                                  std::span<const ExpertPartition> local) const {
    const auto it = std::ranges::find(members, rank);
    if (it == members.end() || local.size() != banks.size())
        throw std::invalid_argument("expert dispatch differs from local rank");
    const auto peer = static_cast<std::size_t>(it - members.begin());
    for (std::size_t b = 0; b < banks.size(); ++b) {
        const auto& p = banks[b].at(peer);
        if (p.ids != local[b].ids || p.intermediate != local[b].intermediate)
            throw std::invalid_argument("expert dispatch differs from local weight slots");
    }
}

} // namespace ninfer::models::moe
