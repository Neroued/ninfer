#pragma once
#include "models/glm5next/work.h"
#include "distributed/wire.h"

namespace ninfer::models::glm5next {
void encode_work(distributed::wire::Encoder&, const Work&, std::uint32_t rank);
Work decode_work(distributed::wire::Decoder&);
void encode_commit(distributed::wire::Encoder&, const Commit&, std::uint32_t rank);
Commit decode_commit(distributed::wire::Decoder&);
void encode_result(distributed::wire::Encoder&, const Result&);
Result decode_result(distributed::wire::Decoder&);
} // namespace ninfer::models::glm5next
