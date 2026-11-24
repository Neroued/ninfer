#include "ops/linear/add/runner.h"
#include "ops/linear_common.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear_add;

struct Profile {
    std::int32_t n;
    std::int32_t k;
    std::int32_t boundary;
    std::uint32_t seed;
};

void add_profile(Suite& suite, const Profile& profile) {
    constexpr std::array a16_graph_tokens{4, 128};
    constexpr std::array a8_graph_tokens{128, 193, 257, 512, 513, 1025};
    std::vector<Invocation> invocations;
    const auto add = [&](std::int32_t tokens, ops::LinearPolicy policy) {
        const auto graphs = policy == ops::LinearPolicy::A16Only
                                ? std::span<const int>(a16_graph_tokens)
                                : std::span<const int>(a8_graph_tokens);
        const bool replay = std::find(graphs.begin(), graphs.end(), tokens) != graphs.end();
        invocations.push_back(Invocation{
            .tokens = tokens,
            .policy = policy,
            .graph  = replay ? GraphReplay::OriginalNegative : GraphReplay::None,
        });
    };
    for (const int tokens : {1, 2, 26}) { add(tokens, ops::LinearPolicy::A16Only); }
    for (const int tokens :
         {profile.boundary - 1, profile.boundary, 48, 65, 1024, 8, 16, 32, 64, 96, 128, 129}) {
        add(tokens, ops::LinearPolicy::AllowA8);
    }
    for (const int tokens : {17, 18, 191, 192, 193, 255, 256, 257, 383, 384, 385, 511, 512, 513,
                             767, 768, 769, 1023, 1025}) {
        add(tokens, ops::LinearPolicy::AllowA8);
    }
    for (int tokens = 2; tokens <= 24; ++tokens) { add(tokens, ops::LinearPolicy::A16Only); }
    for (const int tokens : {31, 32, 33, 63, 64, 65, 127, 128, 129, 1024}) {
        add(tokens, ops::LinearPolicy::A16Only);
    }
    add_cases(suite, QType::FP8_E4M3FN_ROW_BF16,
              InvocationCase{
                  .n           = profile.n,
                  .k           = profile.k,
                  .seed        = profile.seed,
                  .invocations = invocations,
                  .fixture     = FixturePattern::Hashed,
                  .rows        = Samples::Uniform32,
                  .columns     = Samples::Uniform32,
              });

    auto test =
        test::linear_suite::make_case("fp8", profile.n, profile.k, 0, ops::LinearPolicy::A16Only,
                                      test::linear_suite::Execution::Eager, "workspace-intervals");
    test.policies = 7;
    suite.add(std::move(test), [profile] {
        int failures = 0;
        for (auto policy :
             {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
            for (auto [first, last] :
                 {std::pair{1, 48}, std::pair{1, 1024}, std::pair{1000, 1048}}) {
                const auto capacity = ops::linear_add_workspace_capacity_bytes(
                    QType::FP8_E4M3FN_ROW_BF16, profile.n, profile.k, policy, first, last);
                for (int t = first; t <= last; ++t) {
                    if (ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16,
                                                                 profile.n, profile.k, policy, t,
                                                                 t) > capacity)
                        ++failures;
                }
            }
        }
        return failures;
    });
}

} // namespace

void ninfer::test::linear_add::add_fp8_cases(Suite& suite) {
    add_profile(suite, {5120, 6144, 17, 861U});
    add_profile(suite, {5120, 17408, 20, 863U});
}
