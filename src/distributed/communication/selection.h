#pragma once

#include "distributed/communication/implementation.h"

namespace ninfer::distributed::detail {

// Preference ranges express measured choices; backend support only expresses capability.
// Candidates have already been restricted by policy and resolved against every peer.
[[nodiscard]] std::vector<CollectiveRoute>
select_collectives(std::span<const CollectiveRoute> preferences,
                   std::span<Backend* const> candidates);

[[nodiscard]] GroupRouting select_routing(const GroupRequirements&, const GroupDescription&,
                                          CommunicationPolicy, std::span<Backend* const> candidates,
                                          bool peer_work);

} // namespace ninfer::distributed::detail
