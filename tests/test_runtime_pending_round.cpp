#include "runtime/contract/pending_round.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

using ninfer::runtime::FinishDisposition;
using ninfer::runtime::PendingRound;
using ninfer::runtime::TokenId;

int check(bool condition, std::string_view message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

struct FakeOwner {
    enum class State {
        Pending,
        Active,
        Resident,
        Invalid,
    };

    using Round = PendingRound<FakeOwner>;

    [[nodiscard]] Round open(std::span<const TokenId> tokens) {
        pending_tokens = tokens.size();
        state          = State::Pending;
        return Round(
            *this, epoch, tokens,
            {is_live_callback, commit_all_callback, commit_prefix_callback, discard_callback});
    }

    static bool is_live_callback(const FakeOwner& owner, std::uint64_t round_epoch) noexcept {
        return round_epoch == owner.epoch && owner.state == State::Pending;
    }

    void replace_with_new_round(std::size_t token_count) noexcept {
        ++epoch;
        pending_tokens = token_count;
        committed      = 0;
        state          = State::Pending;
    }

    static void commit_all_callback(FakeOwner& owner, std::uint64_t round_epoch) {
        if (round_epoch != owner.epoch || owner.state != State::Pending) {
            throw std::logic_error("stale commit callback");
        }
        owner.committed = owner.pending_tokens;
        owner.state     = State::Active;
        ++owner.epoch;
    }

    static FinishDisposition commit_prefix_callback(FakeOwner& owner, std::uint64_t round_epoch,
                                                    std::size_t count) {
        if (round_epoch != owner.epoch || owner.state != State::Pending) {
            throw std::logic_error("stale prefix callback");
        }
        owner.committed = count;
        owner.state     = State::Resident;
        ++owner.epoch;
        return FinishDisposition::Resident;
    }

    static void discard_callback(FakeOwner& owner, std::uint64_t round_epoch) noexcept {
        if (round_epoch != owner.epoch || owner.state != State::Pending) {
            ++owner.stale_discards;
            return;
        }
        ++owner.matching_discards;
        owner.committed = 0;
        owner.state     = State::Invalid;
        ++owner.epoch;
    }

    State state                   = State::Pending;
    std::uint64_t epoch           = 1;
    std::size_t pending_tokens    = 0;
    std::size_t committed         = 0;
    std::size_t matching_discards = 0;
    std::size_t stale_discards    = 0;
};

int test_move_and_commit_all() {
    const std::array<TokenId, 3> tokens{11, 12, 13};
    FakeOwner owner;
    auto source      = owner.open(tokens);
    const auto epoch = owner.epoch;
    auto destination = std::move(source);

    int failures = 0;
    failures += check(source.tokens().empty(), "moved-from round is not inert");
    failures +=
        check(destination.tokens().size() == tokens.size(), "moved-to round lost its tokens");
    std::move(destination).commit_all();
    failures += check(owner.state == FakeOwner::State::Active, "commit_all did not activate owner");
    failures += check(owner.committed == tokens.size(), "commit_all did not commit the full round");
    failures += check(owner.epoch == epoch + 1, "commit_all did not advance owner epoch");
    failures += check(owner.matching_discards == 0,
                      "resolved or moved-from round discarded during destruction");
    return failures;
}

int test_partial_finish() {
    const std::array<TokenId, 4> tokens{21, 22, 23, 24};
    FakeOwner owner;
    auto round = owner.open(tokens);

    const FinishDisposition disposition = std::move(round).commit_prefix_and_finish(2);
    int failures                        = 0;
    failures += check(disposition == FinishDisposition::Resident,
                      "partial finish returned the wrong disposition");
    failures += check(owner.state == FakeOwner::State::Resident,
                      "partial finish did not make owner resident");
    failures += check(owner.committed == 2, "partial finish committed the wrong prefix");
    failures +=
        check(owner.matching_discards == 0, "resolved partial round discarded during destruction");
    return failures;
}

int test_unresolved_destructor_discards() {
    const std::array<TokenId, 1> tokens{31};
    FakeOwner owner;
    const auto epoch = owner.epoch;
    {
        auto unresolved = owner.open(tokens);
        (void)unresolved;
    }

    int failures = 0;
    failures += check(owner.state == FakeOwner::State::Invalid,
                      "unresolved round destructor did not invalidate owner");
    failures += check(owner.matching_discards == 1,
                      "unresolved round destructor did not discard exactly once");
    failures +=
        check(owner.epoch == epoch + 1, "unresolved round destructor did not advance owner epoch");
    return failures;
}

int test_stale_destructor_cannot_touch_new_round() {
    const std::array<TokenId, 2> old_tokens{41, 42};
    const std::array<TokenId, 3> new_tokens{51, 52, 53};
    FakeOwner owner;

    std::optional<FakeOwner::Round> stale;
    stale.emplace(owner.open(old_tokens));
    owner.replace_with_new_round(new_tokens.size());
    const auto new_epoch = owner.epoch;
    auto current         = owner.open(new_tokens);

    bool stale_tokens_rejected = false;
    try {
        (void)stale->tokens();
    } catch (const std::logic_error&) { stale_tokens_rejected = true; }

    stale.reset();
    int failures = 0;
    failures += check(stale_tokens_rejected, "stale round exposed a newer token buffer");
    failures += check(owner.state == FakeOwner::State::Pending,
                      "stale destructor changed the new round state");
    failures += check(owner.epoch == new_epoch, "stale destructor advanced the new round epoch");
    failures += check(owner.matching_discards == 0, "stale destructor discarded the new round");
    failures += check(owner.stale_discards == 1, "fake owner did not observe the stale epoch");

    std::move(current).commit_all();
    failures += check(owner.state == FakeOwner::State::Active,
                      "new round was not resolvable after stale destruction");
    failures +=
        check(owner.committed == new_tokens.size(), "new round committed the wrong token count");
    return failures;
}

} // namespace

int main() {
    const int failures = test_move_and_commit_all() + test_partial_finish() +
                         test_unresolved_destructor_discards() +
                         test_stale_destructor_cannot_touch_new_round();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
