#include "ops/linear/split/runner.h"

namespace ninfer::test::linear_split {

void add_segmented_cases(Suite& suite) {
    constexpr ReductionCriterion a16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
    const quantized_weight::PatternedWeightOptions small{
        quantized_weight::RowSplitScalePattern::Small,
        quantized_weight::RowSplitCodePattern::Hashed};

    struct Fixture {
        Profile profile;
        std::vector<int> tokens;
    };

    const std::vector<Fixture> fixtures{
        {{2560,
          {{QType::Q8_G32_FP16, 12288, 317U, small}},
          {6144, 6144},
          {{0, 6144, 256}, {6144, 6144, 256}},
          a16},
         {1,   2,   4,   8,   15,  16,  17,  31,  32,   33,   34,   35,  39,  40,
          41,  63,  64,  65,  79,  80,  81,  95,  96,   97,   103,  104, 105, 109,
          110, 111, 112, 113, 127, 128, 129, 512, 1024, 2048, 4096, 4097}},
        {{2560,
          {{QType::BF16, 1664, 331U}},
          {512, 512, 512, 128},
          {{0, 512, 256}, {512, 512, 256}, {1024, 512, 128}, {1536, 128, 128}},
          a16},
         {1,  2,  4,  8,   15,  16,  17,  47,  48,  49,   63,   64,   65,
          95, 96, 97, 127, 128, 129, 511, 512, 513, 1024, 2048, 4096, 4097}},
        {{2560,
          {{QType::BF16, 13952, 331U}},
          {6144, 512, 6144, 512, 512, 128},
          {{0, 6144, 256},
           {6144, 512, 256},
           {6656, 6144, 256},
           {12800, 512, 256},
           {13312, 512, 128},
           {13824, 128, 128}},
          a16},
         {1,  2,  4,  7,  8,  9,   15,  16,  17,  31,  32,  33,   47,   48,   49,  63,
          64, 65, 95, 96, 97, 127, 128, 129, 511, 512, 513, 1024, 2048, 4096, 4097}},
    };
    for (const auto& fixture : fixtures) {
        std::vector<Invocation> calls{{.t = 0, .validate_operands = true}};
        for (int t : fixture.tokens) {
            calls.push_back(
                {.t            = t,
                 .graph        = t == 1 || t == 8 || t == 32 || t == 64 || t == 128 || t == 129,
                 .input_seed   = 317U,
                 .changed_seed = 348U,
                 .columns      = Columns::HeadBoundaries,
                 .full_rows    = t <= 8,
                 .convenience  = true});
        }
        calls.push_back({.t             = 17,
                         .graph         = true,
                         .input_seed    = 317U,
                         .changed_seed  = 348U,
                         .columns       = Columns::HeadBoundaries,
                         .convenience   = true,
                         .shared_output = true});
        add_profile(suite, fixture.profile, std::move(calls));
    }
}
} // namespace ninfer::test::linear_split
