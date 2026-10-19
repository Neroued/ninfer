#pragma once

#include "models/qwen4_exp/parallel/plan.h"
#include "distributed/wire.h"

namespace ninfer::models::qwen4_exp {

struct RankResourceLayout;

struct RankLoadRequest {
    artifact::ArtifactId artifact;
    RankPlan rank;
    EngineOptions options;
    std::size_t host_reservation_bytes = 0;
};

void encode_rank_load(distributed::wire::Encoder&, const RankLoadRequest&);
[[nodiscard]] RankLoadRequest decode_rank_load(distributed::wire::Decoder&);

void encode_rank_layout(distributed::wire::Encoder&, const RankResourceLayout&);
[[nodiscard]] RankResourceLayout decode_rank_layout(distributed::wire::Decoder&);

} // namespace ninfer::models::qwen4_exp
