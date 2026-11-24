#include "ops/linear/split/runner.h"

namespace ninfer::test::linear_split {

void add_multi_output_cases(Suite& suite) {
    using Policy = ops::LinearPolicy;
    constexpr ReductionCriterion a16{2.9e-3, 4.0e-3, 4.5e-3};
    constexpr ReductionCriterion fp8_a16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
    const std::vector<int> segments{6144, 1024, 6144, 1024};
    const std::vector<Region> regions{{0, 6144}, {6144, 1024}, {7168, 6144}, {13312, 1024}};
    const std::vector<int> bulk{129, 191, 192, 193, 256, 257, 287,  288,
                                289, 383, 384, 385, 512, 513, 1024, 1025};
    {
        Profile profile{5120,
                        {{QType::Q4_G64_FP16, 7168, 103U}, {QType::Q5_G64_FP16, 7168, 107U}},
                        segments,
                        regions,
                        a16,
                        1.0 / 256.0,
                        Binding::ProjectionInputs};
        std::vector<Invocation> calls;
        for (int t = 1; t <= 128; ++t) calls.push_back({.t = t, .input_seed = 101U});
        for (int t : bulk) calls.push_back({.t = t, .input_seed = 101U});
        for (int t :
             {1, 7, 8, 9, 12, 13, 16, 32, 63, 64, 65, 96, 104, 105, 127, 128, 129, 192, 193, 17})
            calls.push_back({.t = t, .graph = true, .input_seed = 101U, .sample_rows = 31});
        add_profile(suite, profile, std::move(calls));
    }
    {
        Profile profile{5120, {{QType::BF16, 14336, 313U}}, segments, regions, a16};
        std::vector<Invocation> calls;
        for (int t :
             {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  15,  16,  17,  31,  32,  33,
              34, 63, 64, 65, 66, 95, 96, 97, 98, 127, 128, 129, 130, 191, 192, 193, 194, 1024})
            calls.push_back({.t           = t,
                             .input_seed  = 317U,
                             .columns     = Columns::Projection,
                             .full_rows   = t == 1,
                             .convenience = true});
        for (int t : {4, 5, 16, 17, 32, 33, 64, 65, 96, 97, 128, 129, 192, 193})
            calls.push_back({.t           = t,
                             .graph       = true,
                             .input_seed  = 317U,
                             .columns     = Columns::Projection,
                             .convenience = true});
        calls.push_back({.t = 0, .workspace_intervals = true});
        add_profile(suite, profile, std::move(calls));
    }
    {
        Profile profile{5120,
                        {{QType::NVFP4,
                          14336,
                          331U,
                          {.weight_scale_divisor = 0.125F, .input_scale_divisor = 3.5F}}},
                        segments,
                        regions,
                        a16,
                        1.0 / 256.0,
                        Binding::Nvfp4Inputs};
        std::vector<Invocation> calls;
        for (int t : {1, 2, 4, 8, 16, 20, 32, 33})
            calls.push_back({.t = t, .input_seed = 337U, .omit_divisors = t == 1});
        calls.push_back({.t = 4, .policy = Policy::AllowA8, .input_seed = 337U});
        calls.push_back(
            {.t = 4, .policy = Policy::AllowA8, .input_seed = 337U, .omit_divisors = true});
        for (int t : {4, 17, 1023, 1024, 1025})
            calls.push_back(
                {.t = t, .policy = Policy::AllowA4, .input_seed = 337U, .sample_rows = 31});
        add_profile(suite, profile, std::move(calls));
    }
    {
        Profile profile{5120,
                        {{QType::FP8_E4M3FN_ROW_BF16, 14336, 349U}},
                        segments,
                        regions,
                        fp8_a16,
                        1.0 / 256.0,
                        Binding::ProjectionInputs};
        std::vector<Invocation> calls;
        for (auto policy : {Policy::A16Only, Policy::AllowA8}) {
            const int sample_rows = policy == Policy::A16Only ? 7 : 31;
            for (int t = 1; t <= 128; ++t)
                calls.push_back(
                    {.t = t, .policy = policy, .input_seed = 101U, .sample_rows = sample_rows});
            for (int t : bulk)
                calls.push_back(
                    {.t = t, .policy = policy, .input_seed = 101U, .sample_rows = sample_rows});
            for (int t : {1,   4,   8,   16,  17,  32,  33,  64,  65,  96,  97,
                          128, 129, 192, 193, 288, 289, 385, 512, 513, 1025})
                calls.push_back({.t           = t,
                                 .policy      = policy,
                                 .graph       = true,
                                 .input_seed  = 101U,
                                 .sample_rows = 31});
            calls.push_back({.t = 0, .policy = policy, .workspace_intervals = true});
        }
        for (int t : {1, 17, 129})
            calls.push_back({.t           = t,
                             .policy      = Policy::AllowA4,
                             .graph       = t == 17,
                             .input_seed  = 101U,
                             .sample_rows = 31});
        add_profile(suite, profile, std::move(calls));
    }
    {
        Profile profile{2048,
                        {{QType::Q8_G32_FP16, 9216, 211U}},
                        {4096, 512, 4096, 512},
                        {{0, 4096}, {4096, 512}, {4608, 4096}, {8704, 512}},
                        a16};
        std::vector<Invocation> calls;
        for (int t : {1, 2, 17, 48, 64, 65, 129})
            calls.push_back({.t = t, .input_seed = 201U, .convenience = true});
        add_profile(suite, profile, std::move(calls));
    }
    for (int k : {2048, 5120}) {
        Profile profile{k,
                        {{QType::Q8_G32_FP16, 6144, k == 2048 ? 307U : 313U}},
                        {4096, 1024, 1024},
                        {{0, 4096}, {4096, 1024}, {5120, 1024}},
                        a16,
                        1.0 / 256.0,
                        Binding::LinearInputs};
        std::vector<Invocation> calls;
        if (k == 2048) {
            for (int t : {1, 2, 97, 193, 289, 321, 385, 449})
                calls.push_back({.t = t, .input_seed = 301U, .convenience = true});
        } else {
            for (int t = 1; t <= 128; ++t)
                calls.push_back({.t = t, .input_seed = 301U, .convenience = true});
            for (int t : {129, 192, 193, 1024})
                calls.push_back({.t = t, .input_seed = 301U, .convenience = true});
            for (int t : {1, 8, 16, 48, 49, 53, 54, 63, 64, 65, 96, 97, 112, 127, 128, 129})
                calls.push_back({.t           = t,
                                 .graph       = true,
                                 .input_seed  = 301U,
                                 .sample_rows = 31,
                                 .convenience = true});
        }
        add_profile(suite, profile, std::move(calls));
    }
}
} // namespace ninfer::test::linear_split
