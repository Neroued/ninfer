#pragma once

#include "distributed/bootstrap.h"

namespace ninfer::distributed {

void encode_communication_preparation(wire::Encoder&, const CommunicationPreparation&);
[[nodiscard]] CommunicationPreparation decode_communication_preparation(wire::Decoder&);

// Matching coordinator/worker preparation. The caller starts it once all workers can declare
// their requirements. Connection and device warmup are separated by all-rank completion.
void prepare_communication(ExecutionGroup&, std::chrono::milliseconds timeout);
// False means Shutdown was received. The caller completes its ordinary shutdown path.
[[nodiscard]] bool prepare_communication(WorkerConnection&, std::span<const GroupRequirements>,
                                         CommunicationPolicy = CommunicationPolicy::Auto);

} // namespace ninfer::distributed
