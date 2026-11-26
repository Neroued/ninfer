#pragma once

#include "models/common/qwen/vision/wire.h"
#include "distributed/wire.h"
#include "models/qwen4_exp/work.h"

namespace ninfer::models::qwen4_exp {

// A single frozen window mapping survives release of the raw production payload.
using qwen::VisionInputBindings;

void encode_work(distributed::wire::Encoder&, const Work&, const VisionInputBindings&,
                 std::uint32_t rank);
[[nodiscard]] Work decode_work(distributed::wire::Decoder&, const VisionInputBindings&);
void encode_commit(distributed::wire::Encoder&, const Commit&, std::uint32_t rank);
[[nodiscard]] Commit decode_commit(distributed::wire::Decoder&);
void encode_result(distributed::wire::Encoder&, const Result&);
[[nodiscard]] Result decode_result(distributed::wire::Decoder&);

} // namespace ninfer::models::qwen4_exp
