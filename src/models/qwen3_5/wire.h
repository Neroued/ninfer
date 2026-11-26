#pragma once

#include "models/common/qwen/vision/wire.h"
#include "distributed/wire.h"
#include "models/common/batch.h"
#include "models/common/speculative/round_plan.h"
#include "models/qwen3_5/work.h"
#include "runtime/executor/protocol.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5 {

using qwen::VisionInputBindings;

struct WorkContract {
    std::uint32_t max_context         = 0;
    std::uint32_t max_concurrency     = 1;
    std::uint32_t prefill_chunk       = 0;
    std::uint32_t draft_window        = 0;
    SpeculativeBackend backend        = SpeculativeBackend::None;
    std::uint64_t vocabulary          = 0;
    std::uint32_t public_tokens       = 0;
    bool vision_enabled               = false;
    std::uint32_t vision_max_features = 0;
    std::uint32_t control_token_limit = 0;
    std::uint32_t kv_pool_count       = 1;
    std::shared_ptr<const BatchExecutionPlan> batches;
    std::shared_ptr<const speculative::DecodeGraphRoutes> decode_routes;

    void validate_decode_route(const speculative::RoundRoute&, std::uint32_t owners,
                               std::size_t rows) const;
    void validate(const Work&, runtime::FrameKind) const;
    void validate(const Work&, const runtime::DecisionRequest&,
                  const runtime::DecisionInput&) const;
    void validate(const Work&, const Result&, const Commit&) const;
    [[nodiscard]] runtime::WorkPayloadCapacity payload_capacity() const;
};

void encode_work(distributed::wire::Encoder&, const Work&, const VisionInputBindings&);
[[nodiscard]] Work decode_work(distributed::wire::Decoder&, const VisionInputBindings&,
                               const WorkContract&);
void encode_commit(distributed::wire::Encoder&, const Commit&);
[[nodiscard]] Commit decode_commit(distributed::wire::Decoder&);
void encode_result(distributed::wire::Encoder&, const Result&);
[[nodiscard]] Result decode_result(distributed::wire::Decoder&);

} // namespace ninfer::models::qwen3_5
