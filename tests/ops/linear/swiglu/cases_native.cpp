#include "core/platform.h"
#include "ops/linear/swiglu/runner.h"
#include "ninfer/ops/linear_swiglu.h"

#include <array>
#include <utility>

namespace ninfer::test::linear_swiglu {

void add_native_cases(Suite& suite) {
#if NINFER_TARGET_SPARK
    constexpr std::array tokens{1, 2, 7, 17, 129, 513};
    constexpr std::array graph{33};
    constexpr std::array dense_tokens{
        1,   2,    3,    4,    7,    8,    9,    12,   15,   16,   17,   24,   31,  32,
        33,  47,   48,   49,   63,   64,   65,   96,   127,  128,  129,  191,  192, 193,
        255, 256,  257,  319,  320,  321,  351,  352,  353,  383,  384,  385,  511, 512,
        513, 1023, 1024, 1025, 2047, 2048, 2049, 3071, 3072, 3073, 4095, 4096, 4097};
    constexpr std::array dense_graph{1, 2, 4, 8, 16, 33, 49, 129, 257, 320, 321, 385, 2049, 4097};
    constexpr std::array shared_tokens{
        1,   2,   3,    4,    7,    8,    9,    16,   17,   24,   25,  31,  32,  33,
        47,  48,  49,   63,   64,   65,   95,   96,   97,   127,  128, 129, 191, 192,
        193, 255, 256,  257,  383,  384,  385,  511,  512,  513,  639, 640, 641, 767,
        768, 769, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097};
    constexpr std::array shared_graph{1, 4, 8, 17, 33, 49, 65, 129, 193, 385, 513, 641, 4097};
    for (int rows : {2048, 12288}) {
        add_profile(
            suite, {QType::MXFP8, rows, 4096, rows / 2, 3301U, ops::LinearPolicy::AllowA8, 10.F},
            rows == 12288 ? std::span<const int>(dense_tokens)
                          : std::span<const int>(shared_tokens),
            rows == 12288 ? std::span<const int>(dense_graph) : std::span<const int>(shared_graph));
        add_profile(suite,
                    {QType::MXFP8, rows, 4096, rows / 2, 3307U, ops::LinearPolicy::A16Only, 3.F},
                    std::array{17, 257});
    }
    add_profile(suite, {QType::MXFP8, 12288, 4096, 6144, 3301U, ops::LinearPolicy::AllowA8},
                std::array{8, 257, 384, 4097}, std::array{320});
    add_profile(suite, {QType::MXFP8, 2048, 4096, 1024, 3301U, ops::LinearPolicy::AllowA8},
                std::array{4, 33, 129, 193, 385, 641, 4097}, std::array{65, 513});
    for (int rows : {2048, 12288}) {
        suite.add(linear_suite::make_case("mxfp8", rows, 4096, 0, ops::LinearPolicy::AllowA8,
                                          linear_suite::Execution::Eager, "capacity-intervals"),
                  [rows] {
                      int failures = 0;
                      for (auto [first, last] :
                           {std::pair{1, 48}, std::pair{33, 129}, std::pair{129, 321},
                            std::pair{257, 385}, std::pair{321, 2049}, std::pair{1, 4097}}) {
                          const auto peak = ops::linear_swiglu_workspace_capacity_bytes(
                              QType::MXFP8, rows, 4096, ops::LinearPolicy::AllowA8, first, last);
                          for (int t = first; t <= last; ++t)
                              if (ops::linear_swiglu_workspace_capacity_bytes(
                                      QType::MXFP8, rows, 4096, ops::LinearPolicy::AllowA8, t, t) >
                                  peak)
                                  ++failures;
                      }
                      return failures;
                  });
    }
    add_profile(suite,
                {QType::FP8_E4M3FN_BLOCK128X128_FP32, 2048, 4096, 1024, 3313U,
                 ops::LinearPolicy::A16Only, 10.F},
                shared_tokens, shared_graph);
    add_profile(suite,
                {QType::FP8_E4M3FN_BLOCK128X128_FP32, 2048, 4096, 1024, 3313U,
                 ops::LinearPolicy::A16Only, 10.F},
                std::array{575, 576, 577, 959, 960, 961}, std::array{577, 961});
    add_profile(
        suite,
        {QType::FP8_E4M3FN_BLOCK128X128_FP32, 2048, 4096, 1024, 3313U, ops::LinearPolicy::A16Only},
        std::array{1, 33, 65, 129, 193, 257, 385, 641, 4097}, std::array{65, 513});
    add_profile(suite,
                {QType::NVFP4_MUL, 8192, 1024, 4096, 3319U, ops::LinearPolicy::A16Only, 10.F},
                std::array{1, 2, 7, 8, 9, 16, 17, 64, 65, 128, 129, 513, 4096}, std::array{33, 129});
    add_profile(suite,
                {QType::NVFP4_MUL, 20480, 4096, 10240, 3323U, ops::LinearPolicy::A16Only, 10.F},
                std::array{1, 2, 3, 16, 17, 32, 33, 64, 65, 96, 128, 129, 256, 4096},
                std::array{2, 65, 129});
    add_profile(suite,
                {QType::NVFP4_MUL, 20480, 4096, 10240, 3329U, ops::LinearPolicy::A16Only, 0.F},
                std::array{1, 33, 65, 129});
#endif
}

} // namespace ninfer::test::linear_swiglu
