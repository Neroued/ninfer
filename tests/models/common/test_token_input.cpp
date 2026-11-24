#include "models/common/token_input.h"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;
using namespace ninfer::models;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid token history was accepted");
}

void counts() {
    TokenCountBuilder builder;
    constexpr std::uint32_t vocabulary                = 16;
    const std::vector<std::vector<TokenId>> histories = {
        {},
        {3, 3, 8, 3},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0},
        {15, 2, 15},
        {3}};
    for (const auto& history : histories) {
        const auto snapshot = builder.build(history, vocabulary);
        snapshot.validate(vocabulary, history.size());
        std::vector<std::int32_t> restored(vocabulary, -1), expected(vocabulary);
        for (auto id : history) ++expected[id];
        snapshot.restore(restored);
        require(restored == expected, "restored committed counts differ from token histogram");
    }
    const std::array<TokenId, 3> invalid{1, 1, 16};
    rejects([&] { (void)builder.build(invalid, vocabulary); });
    const std::array<TokenId, 1> next{1};
    std::vector<std::int32_t> restored(vocabulary);
    builder.build(next, vocabulary).restore(restored);
    require(restored[1] == 1, "failed count preparation leaked into the next residency");
    rejects([&] { TokenCounts(TokenCounts::Sparse{{2, 1}, {2, 1}}).validate(vocabulary, 2); });
    rejects([&] { TokenCounts(TokenCounts::Sparse{{2, 3}}).validate(vocabulary, 2); });
    rejects([&] { TokenCounts(TokenCounts::Dense(vocabulary, -1)).validate(vocabulary, 16); });
}

void control() {
    for (std::size_t n : {1, 2, 17}) {
        std::vector<TokenId> input(n, 7), expected(n, 7), actual(n);
        input[0]        = 99;
        expected.back() = 12;
        const ControlTokens tokens{input, 12};
        const auto split = n / 2;
        tokens.copy(std::span(actual).first(split));
        tokens.copy(std::span(actual).subspan(split), split);
        require(actual == expected, "control output retained the anchor or lost its final token");
    }
}
} // namespace

int main() {
    try {
        counts();
        control();
        std::cout << "Token inputs passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
