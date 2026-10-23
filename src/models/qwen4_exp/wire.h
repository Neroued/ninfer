#pragma once

#include "distributed/wire.h"
#include "models/qwen4_exp/work.h"

#include <map>

namespace ninfer::models::qwen4_exp {

// Installed Host inputs have session-scoped identities. A decoded binding acquires its own
// shared owner, so removing the lookup entry does not release an input still used by Program.
class VisionInputBindings {
public:
    void install(std::uint64_t, std::shared_ptr<const qwen::VisionExecutionInput>);
    void release(std::uint64_t);
    [[nodiscard]] std::uint64_t id(const std::shared_ptr<const qwen::VisionExecutionInput>&) const;
    [[nodiscard]] std::shared_ptr<const qwen::VisionExecutionInput> get(std::uint64_t) const;
private:
    std::map<std::uint64_t, std::shared_ptr<const qwen::VisionExecutionInput>> inputs_;
};

void encode_work(distributed::wire::Encoder&, const Work&, const VisionInputBindings&,
                 std::uint32_t rank);
[[nodiscard]] Work decode_work(distributed::wire::Decoder&, const VisionInputBindings&);
void encode_commit(distributed::wire::Encoder&, const Commit&, std::uint32_t rank);
[[nodiscard]] Commit decode_commit(distributed::wire::Decoder&);
void encode_result(distributed::wire::Encoder&, const Result&);
[[nodiscard]] Result decode_result(distributed::wire::Decoder&);

} // namespace ninfer::models::qwen4_exp
