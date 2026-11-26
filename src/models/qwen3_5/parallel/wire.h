#pragma once

#include "distributed/wire.h"
#include "models/qwen3_5/parallel/preparation.h"

namespace ninfer::models::qwen3_5 {

void encode_rank_load(distributed::wire::Encoder&, const RankLoadRequest&);
[[nodiscard]] RankLoadRequest decode_rank_load(distributed::wire::Decoder&);

void encode(distributed::wire::Encoder&, const RankExecutionDescription&);
[[nodiscard]] RankExecutionDescription decode_rank_resources(distributed::wire::Decoder&);

} // namespace ninfer::models::qwen3_5
