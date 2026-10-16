#include "ops/linear/split/runner.h"

namespace ninfer::test::linear_split {

void add_two_output_cases(Suite& suite) {
    using Policy = ops::LinearPolicy;
    constexpr ReductionCriterion paired_a16{3.0e-3, 4.0e-3, 3.5e-3};
    constexpr ReductionCriterion a16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
    {
        Profile profile{5120,
                        {{QType::Q4_G64_FP16, 4096, 409U}, {QType::Q5_G64_FP16, 12288, 419U}},
                        {10240, 6144},
                        {{0, 4096}, {4096, 6144}, {10240, 6144}},
                        paired_a16};
        std::vector<Invocation> calls;
        for (int t : {1,  2,  3,  4,  5,  6,  7,  8,  9,   10,  11,  12,
                      13, 15, 16, 17, 32, 33, 64, 65, 127, 128, 129, 193})
            calls.push_back({.t = t, .input_seed = 401U});
        for (int t : {7, 8, 9, 10, 12, 13, 33, 65, 129})
            calls.push_back({.t                    = t,
                             .graph                = true,
                             .input_seed           = 701U,
                             .changed_seed         = 811U,
                             .eager_before_capture = true});
        add_profile(suite, profile, std::move(calls));
    }
    for (int k : {2048, 2560}) {
        const int tail = k == 2560 ? 6144 : 4096;
        Profile profile{k,
                        {{QType::Q8_G32_FP16, 4096 + 2 * tail, 503U}},
                        {4096 + tail, tail},
                        {{0, 2048}, {2048, 2048}, {4096, tail}, {4096 + tail, tail}},
                        a16,
                        1.0 / 256.0,
                        Binding::ProjectionInputs,
                        {2048, 2048, tail, tail}};
        const std::vector<int> points =
            k == 2048
                ? std::vector<int>{1, 2, 97}
                : std::vector<int>{1,  2,  7,  8,   9,   15,  16,  17,  23,  24,   25,   31,
                                   32, 33, 39, 40,  41,  47,  48,  49,  62,  63,   64,   65,
                                   95, 96, 97, 127, 128, 129, 511, 512, 513, 1024, 2048, 4096};
        std::vector<Invocation> calls;
        for (int t : points) calls.push_back({.t = t, .input_seed = 501U});
        if (k == 2560)
            for (int t : {1, 8, 16, 24, 32, 40, 48, 63, 64, 96, 128, 512, 513})
                calls.push_back({.t = t, .graph = true, .input_seed = 501U, .changed_seed = 1501U});
        calls.push_back({.t = 0, .workspace_intervals = true});
        add_profile(suite, profile, std::move(calls));
    }
    {
        Profile profile{5120,
                        {{QType::NVFP4,
                          16384,
                          607U,
                          {.weight_scale_divisor = 0.125F, .input_scale_divisor = 3.5F}}},
                        {10240, 6144},
                        {{0, 2048}, {2048, 2048}, {4096, 6144}, {10240, 6144}},
                        paired_a16,
                        4.0e-3};
        std::vector<Invocation> calls{
            {.t = 1, .input_seed = 601U, .columns = Columns::Seven},
            {.t = 4, .policy = Policy::AllowA8, .input_seed = 601U, .columns = Columns::Seven}};
        for (int t : {1, 2, 17, 1023, 1024, 1025})
            calls.push_back(
                {.t = t, .policy = Policy::AllowA4, .input_seed = 601U, .columns = Columns::Seven});
        add_profile(suite, profile, std::move(calls));
    }
    {
        Profile profile{5120,
                        {{QType::FP8_E4M3FN_ROW_BF16, 16384, 613U}},
                        {10240, 6144},
                        {{0, 2048}, {2048, 2048}, {4096, 6144}, {10240, 6144}},
                        a16};
        std::vector<Invocation> calls;
        for (int t : {5, 8, 16, 24, 32, 33, 64, 65, 96, 97, 128, 129, 1, 2})
            calls.push_back({.t = t, .input_seed = 617U, .columns = Columns::Seven});
        for (int t : {1,   2,   4,   8,   16,  17,  18,  32,  33,  48,  64,  65,  127,  128, 129,
                      191, 192, 193, 255, 256, 257, 383, 384, 385, 511, 512, 513, 1024, 1025})
            calls.push_back({.t           = t,
                             .policy      = t == 17 ? Policy::AllowA4 : Policy::AllowA8,
                             .input_seed  = 617U,
                             .columns     = Columns::Seven,
                             .sample_rows = t >= 17 ? 31 : 7});
        for (int t : {4, 17, 128, 385, 512, 1025})
            calls.push_back({.t           = t,
                             .policy      = Policy::AllowA8,
                             .graph       = true,
                             .input_seed  = 617U,
                             .columns     = Columns::Seven,
                             .sample_rows = t >= 17 ? 31 : 7});
        calls.push_back({.t = 0, .policy = Policy::AllowA8, .workspace_intervals = true});
        add_profile(suite, profile, std::move(calls));
    }
}
} // namespace ninfer::test::linear_split
