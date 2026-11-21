#pragma once

#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/runtime_types.h"

namespace ninfer::test::flash_next {
namespace q = models::qwen4_exp;

inline void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void exercise(q::Session&, runtime::Executor<q::ExecutionTypes>&, models::qwen::Frontend&,
              const EngineOptions&, ExecutionPath, unsigned hidden);

class HistoryReference {
public:
    void input(q::detail::RankProgramImpl&, const q::Commit&);
    void compare(q::detail::RankProgramImpl&);
private:
    struct Prefix {
        std::vector<TokenId> ids;
        std::vector<std::array<std::int32_t, 3>> coordinates;
        std::vector<std::uint16_t> hidden;
    };

    void compare_repair(q::detail::RankProgramImpl&);
    std::map<std::uint64_t, Prefix> prefixes_;
    std::vector<unsigned> adopted_;
};
} // namespace ninfer::test::flash_next
