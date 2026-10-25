#pragma once

#include "distributed/wire.h"
#include "models/common/context/contract.h"
#include "models/common/context/directory.h"

namespace ninfer::models::context {

void encode_actions(distributed::wire::Encoder&, const ContextActions&);
[[nodiscard]] ContextActions decode_actions(distributed::wire::Decoder&);

void encode_directory(distributed::wire::Encoder&, const ContextDirectorySpec&);
[[nodiscard]] ContextDirectorySpec decode_directory(distributed::wire::Decoder&);

} // namespace ninfer::models::context
