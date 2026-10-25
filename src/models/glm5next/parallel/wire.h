#pragma once

#include "models/glm5next/parallel/preparation.h"
#include "distributed/wire.h"

namespace ninfer::models::glm5next {

void encode_rank_load(distributed::wire::Encoder&, const RankLoadRequest&);
[[nodiscard]] RankLoadRequest decode_rank_load(distributed::wire::Decoder&);

void encode(distributed::wire::Encoder&, const RankResourceDescription&);
[[nodiscard]] RankResourceDescription decode_rank_resources(distributed::wire::Decoder&);

} // namespace ninfer::models::glm5next
