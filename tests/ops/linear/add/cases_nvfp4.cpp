#include "ops/linear/add/runner.h"
#include "ops/linear_common.h"

#include <array>
#include <cstdint>

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
    const std::array invocations{
        Invocation{.tokens = 1, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 4, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 5, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 8, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 16, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 17, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 24, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 25, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 32, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 33, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 48, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 49, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 64, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 65, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 128,
                   .policy = ops::LinearPolicy::A16Only,
                   .graph  = GraphReplay::OriginalTwice},
        Invocation{.tokens = 129, .policy = ops::LinearPolicy::A16Only},
        Invocation{.tokens = 1024, .policy = ops::LinearPolicy::A16Only},
        Invocation{
            .tokens = profile.boundary,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 17,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 8,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 16,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 32,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 64,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 96,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{.tokens = 128,
                   .policy = ops::LinearPolicy::AllowA4,
                   .graph  = GraphReplay::OriginalTwice},
        Invocation{
            .tokens = 129,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 1023,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 1024,
            .policy = ops::LinearPolicy::AllowA4,
        },
        Invocation{
            .tokens = 1025,
            .policy = ops::LinearPolicy::AllowA4,
        },
    };
    add_cases(suite, QType::NVFP4,
              InvocationCase{
                  .n           = profile.n,
                  .k           = profile.k,
                  .seed        = profile.seed,
                  .invocations = invocations,
                  .fixture     = FixturePattern::Hashed,
                  .rows        = Samples::BoundarySeven,
                  .columns     = Samples::BoundarySeven,
              });
}

} // namespace

void ninfer::test::linear_add::add_nvfp4_cases(Suite& suite) {
    add_profile(suite, {5120, 6144, 17, 811U});
    add_profile(suite, {5120, 17408, 8, 821U});
}
