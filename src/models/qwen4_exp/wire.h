#pragma once

#include "distributed/wire.h"
#include "models/qwen4_exp/work.h"

namespace ninfer::models::qwen4_exp {

// A single frozen window mapping survives release of the raw production payload.
class VisionInputBindings {
public:
    void install(qwen::VisionWork);
    void release_payload(std::uint64_t);
    [[nodiscard]] qwen::VisionWork get(std::uint64_t id, bool produce) const;
private:
    std::optional<qwen::VisionWork> current_;
};

void encode_work(distributed::wire::Encoder&, const Work&, const VisionInputBindings&,
                 std::uint32_t rank);
[[nodiscard]] Work decode_work(distributed::wire::Decoder&, const VisionInputBindings&);
void encode_commit(distributed::wire::Encoder&, const Commit&, std::uint32_t rank);
[[nodiscard]] Commit decode_commit(distributed::wire::Decoder&);
void encode_result(distributed::wire::Encoder&, const Result&);
[[nodiscard]] Result decode_result(distributed::wire::Decoder&);

} // namespace ninfer::models::qwen4_exp
