#pragma once

#include "distributed/wire.h"
#include "models/common/context/contract.h"

namespace ninfer::models::context {

void encode_actions(distributed::wire::Encoder&, const ContextActions&);
[[nodiscard]] ContextActions decode_actions(distributed::wire::Decoder&);

} // namespace ninfer::models::context
