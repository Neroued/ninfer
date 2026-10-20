#include "ops/linear/split/runner.h"
#include "core/platform.h"

#include <set>

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

    std::vector<Fixture> fixtures{
#if NINFER_TARGET_SPARK
        {{2560,
          {{QType::BF16, 640, 331U}},
          {256, 256, 128},
          {{0, 256, 256}, {256, 256, 256}, {512, 128, 128}},
          a16},
         {1, 4, 8, 9, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 4096}},
        {{2560,
          {{QType::BF16, 1152, 331U}},
          {256, 256, 512, 128},
          {{0, 256, 256}, {256, 256, 256}, {512, 512, 128}, {1024, 128, 128}},
          a16},
         {1,   4,   7,   8,   9,   32,   63,   64,   65,   127,  128,  129, 255,
          256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4096}},
        {{2560,
          {{QType::BF16, 7296, 331U}},
          {256, 256, 128, 3072, 3072, 512},
          {{0, 256, 256},
           {256, 256, 256},
           {512, 128, 128},
           {640, 3072, 256},
           {3712, 3072, 256},
           {6784, 512, 128}},
          a16},
         {1,   4,   7,   8,   9,   32,  63,   64,   65,   95,   96,  97,
          127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2048, 4096}},
#endif
        {{2560,
          {{QType::BF16, 1152, 331U}},
          {512, 512, 128},
          {{0, 512, 256}, {512, 512, 256}, {1024, 128, 128}},
          a16},
         {1,   4,   8,   16,  17,  48,  49,   64,   65,   96,   97,
          128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 4096}},
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
          {512, 512, 128, 6144, 6144, 512},
          {{0, 512, 256},
           {512, 512, 256},
           {1024, 128, 128},
           {1152, 6144, 256},
           {7296, 6144, 256},
           {13440, 512, 128}},
          a16},
         {1,  2,  4,  7,  8,  9,   15,  16,  17,  31,  32,  33,   47,   48,   49,  63,
          64, 65, 95, 96, 97, 127, 128, 129, 511, 512, 513, 1024, 2048, 4096, 4097}},
    };
#if NINFER_TARGET_SPARK
    fixtures.push_back({{4096,
                         {{QType::BF16, 288, 373U}},
                         {32, 128, 128},
                         {{0, 32, 32}, {32, 128, 32}, {160, 128, 32}},
                         a16},
                        {1, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4097}});
    fixtures.push_back(
        {{4096,
          {{QType::Q8_G32_FP16, 12288, 367U, small}},
          {4096, 4096, 4096},
          {{0, 4096, 128}, {4096, 4096, 128}, {8192, 4096, 128}},
          a16},
         {1, 4, 6, 8, 16, 32, 40, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 4097}});
    fixtures.push_back({{2560,
                         {{QType::Q8_G32_FP16, 8192, 347U, small}},
                         {5120, 3072},
                         {{0, 1024, 128}, {1024, 1024, 128}, {2048, 3072, 128}, {5120, 3072, 128}},
                         a16},
                        {1,   2,   4,   6,   8,   9,   15,   16,   17,   23,   24,   25,  31,  32,
                         33,  39,  40,  41,  63,  64,  65,   95,   96,   97,   127,  128, 129, 191,
                         192, 193, 256, 511, 512, 513, 1023, 1024, 1025, 2048, 4096, 4097}});
    fixtures.push_back({{2560,
                         {{QType::Q8_G32_FP16, 6144, 353U, small}},
                         {3072, 3072},
                         {{0, 3072, 256}, {3072, 3072, 256}},
                         a16},
                        {1,   2,   4,   6,   8,   9,   15,   16,   17,   23,   24,   25,  31,  32,
                         33,  39,  40,  41,  63,  64,  65,   95,   96,   97,   127,  128, 129, 191,
                         192, 193, 256, 511, 512, 513, 1023, 1024, 1025, 2048, 4096, 4097}});
#endif
    for (const auto& fixture : fixtures) {
        std::set<int> tokens(fixture.tokens.begin(), fixture.tokens.end());
        std::set<int> graph_tokens{1, 8, 32, 64, 128, 129};
        if (fixture.profile.k == 4096) {
            for (int m = 1; m <= 128; ++m) tokens.insert(m);
            for (int b : {4, 8, 16, 32, 40, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096}) {
                tokens.insert({b - 1, b, b + 1});
                graph_tokens.insert({b, b + 1});
            }
        }
        if (fixture.profile.weights.size() == 1 && fixture.profile.weights[0].type == QType::BF16 &&
            fixture.profile.weights[0].rows == 1664) {
            for (int end : {32,  64,   96,   128,  160,  256,  336,  448,  560,  640,
                            896, 1024, 1152, 1280, 1408, 1440, 1664, 1856, 2048, 4096}) {
                tokens.insert({end - 1, end, end + 1});
                graph_tokens.insert({end, end + 1});
            }
            // Full and partial token tiles for each new warp-store shape.
            tokens.insert({288, 384});
            graph_tokens.insert({288, 384});
        }
        if (fixture.profile.weights.size() == 1 && fixture.profile.weights[0].type == QType::BF16 &&
            fixture.profile.weights[0].rows == 13952) {
            for (int end : {256, 512, 3072, 3839, 4096}) {
                tokens.insert({end - 1, end, end + 1});
                graph_tokens.insert({end, end + 1});
            }
            graph_tokens.insert(1024);
        }
        if (fixture.profile.weights.size() == 1 &&
            fixture.profile.weights[0].type == QType::Q8_G32_FP16) {
            tokens.insert({191, 192, 193, 1023, 1025, 1151, 1152, 1153});
            graph_tokens.insert({192, 1024, 1152});
        }
        std::vector<Invocation> calls{{.t = 0, .validate_operands = true}};
        for (int t : tokens) {
            calls.push_back({.t            = t,
                             .graph        = graph_tokens.contains(t),
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
        if (fixture.profile.k == 4096) {
            calls.push_back({.t            = 17,
                             .graph        = true,
                             .input_seed   = 317U,
                             .changed_seed = 348U,
                             .columns      = Columns::HeadBoundaries,
                             .convenience  = false});
            calls.push_back({.t = 0, .workspace_intervals = true});
        }
        add_profile(suite, fixture.profile, std::move(calls));
    }
}
} // namespace ninfer::test::linear_split
