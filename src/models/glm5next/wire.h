#pragma once
#include "distributed/wire.h"
#include "models/glm5next/work.h"

namespace ninfer::models::glm5next {
// Retain the frozen mapping after the producer payload has retired.
class VisionInputBindings {
public:
    void install(vision::Work);
    void release_payload(std::uint64_t);
    [[nodiscard]] vision::Work get(std::uint64_t id, bool produce) const;
private:
    std::optional<vision::Work> current_;
};

void encode_work(distributed::wire::Encoder&, const Work&, std::uint32_t rank);
Work decode_work(distributed::wire::Decoder&, const VisionInputBindings& = {});
void encode_commit(distributed::wire::Encoder&, const Commit&, std::uint32_t rank);
Commit decode_commit(distributed::wire::Decoder&);
void encode_result(distributed::wire::Encoder&, const Result&);
Result decode_result(distributed::wire::Decoder&);
} // namespace ninfer::models::glm5next
