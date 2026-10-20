#pragma once

#include "models/qwen4_exp/parallel/plan.h"
#include "models/qwen4_exp/session/planning.h"
#include "distributed/wire.h"
#include "runtime/executor/contract.h"

namespace ninfer::models::qwen4_exp {

struct RankExecutionDescription {
    SessionPlan session;
    runtime::ExecutionCapacity capacity;
    RankMemoryBudget budget;
    RankSnapshot observation;
};

struct RankLoadRequest {
    artifact::ArtifactId artifact;
    RankPlan rank;
    EngineOptions options;
    std::size_t host_reservation_bytes = 0;
    bool control_owner                 = false;
};

void encode_rank_load(distributed::wire::Encoder&, const RankLoadRequest&);
[[nodiscard]] RankLoadRequest decode_rank_load(distributed::wire::Decoder&);

void encode(distributed::wire::Encoder&, const RankExecutionDescription&);
[[nodiscard]] RankExecutionDescription decode_rank_execution(distributed::wire::Decoder&);

} // namespace ninfer::models::qwen4_exp
