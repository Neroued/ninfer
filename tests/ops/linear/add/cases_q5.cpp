#include "ops/linear/add/runner.h"
#include "ops/linear_common.h"

#include <array>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::QType;

void q5_a16_conformance(ninfer::test::linear_suite::Suite& suite) {
    // Exercise the production transitions, including the same transitions in
    // a composite tail and the 192-token limit on that tail.
    constexpr std::array<std::int32_t, 13> starts{5,  9,   17,  25,  33,  49, 65,
                                                  97, 129, 161, 193, 257, 513};
    constexpr std::array<std::int32_t, 13> interiors{1,  2,   3,   7,   13,  20,  40,
                                                     80, 112, 144, 224, 768, 1024};
    constexpr std::array<std::int32_t, 10> graphs{13, 40, 80, 112, 128, 192, 513, 545, 609, 705};
    constexpr std::array<std::int32_t, 0> no_starts{};
    constexpr std::array<std::int32_t, 7> full{8, 13, 32, 40, 80, 112, 192};
    for (const auto k : {6144, 17408}) {
        ninfer::test::linear_add::add_shape(suite, QType::Q5_G64_FP16,
                                            ShapeCase{.n                = 5120,
                                                      .k                = k,
                                                      .seed             = 401U,
                                                      .route_starts     = starts,
                                                      .route_interiors  = interiors,
                                                      .graph_tokens     = graphs,
                                                      .composite_offset = 512});
        ninfer::test::linear_add::add_shape(suite, QType::Q5_G64_FP16,
                                            ShapeCase{.n               = 5120,
                                                      .k               = k,
                                                      .seed            = 401U,
                                                      .route_starts    = no_starts,
                                                      .route_interiors = full,
                                                      .full_output     = true});
    }
}

} // namespace

void ninfer::test::linear_add::add_q5_cases(Suite& suite) { q5_a16_conformance(suite); }
