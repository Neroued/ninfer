#include "ops/linear_add/linear_add_test_common.h"
#include "ops/linear_common.h"

#include <array>
#include <cstdint>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear_add;

struct Profile {
    std::int32_t n;
    std::int32_t k;
    std::int32_t first_a4;
    std::uint32_t seed;
};

int run_profile(const Profile& profile) {
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
        Invocation{.tokens     = profile.first_a4,
                   .policy     = ops::LinearPolicy::AllowA4,
                   .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 17, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{.tokens = 8, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 16, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 32, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        // 32 and 33 straddle the K=17408 narrow-tile seam: the last width M32N32 takes and the
        // first that returns to M32N64.
        Invocation{
            .tokens = 33, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 64, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 96, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{.tokens     = 128,
                   .policy     = ops::LinearPolicy::AllowA4,
                   .arithmetic = Arithmetic::A4,
                   .graph      = GraphReplay::OriginalTwice},
        Invocation{
            .tokens = 129, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 1023, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 1024, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
        Invocation{
            .tokens = 1025, .policy = ops::LinearPolicy::AllowA4, .arithmetic = Arithmetic::A4},
    };
    return run_cases("NVFP4 linear_add", WeightFormat::NVFP4,
                     InvocationCase{
                         .n               = profile.n,
                         .k               = profile.k,
                         .seed            = profile.seed,
                         .invocations     = invocations,
                         .fixture         = FixturePattern::Hashed,
                         .rows            = Samples::BoundarySeven,
                         .columns         = Samples::BoundarySeven,
                         .scan_all_finite = false,
                     });
}

} // namespace

int main() {
    return ninfer::test::linear_common::run_test_main("NVFP4 linear_add", [] {
        return run_profile(Profile{.n = 5120, .k = 6144, .first_a4 = 17, .seed = 811U}) +
               run_profile(Profile{.n = 5120, .k = 17408, .first_a4 = 8, .seed = 821U});
    });
}
