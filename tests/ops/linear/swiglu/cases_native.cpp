#include "core/platform.h"
#include "ops/linear/swiglu/runner.h"

#include <array>

namespace ninfer::test::linear_swiglu {

void add_native_cases(Suite& suite) {
#if NINFER_TARGET_SPARK
    constexpr std::array tokens{1, 2, 7, 17, 129, 513};
    constexpr std::array graph{33};
    for (int rows : {2048, 12288}) {
        add_profile(suite,
                    {QType::MXFP8, rows, 4096, rows / 2, 3301U, ops::LinearPolicy::AllowA8, 10.F},
                    tokens, graph);
        add_profile(suite,
                    {QType::MXFP8, rows, 4096, rows / 2, 3307U, ops::LinearPolicy::A16Only, 3.F},
                    std::array{17, 257});
    }
    add_profile(suite,
                {QType::FP8_E4M3FN_BLOCK128X128_FP32, 2048, 4096, 1024, 3313U,
                 ops::LinearPolicy::A16Only, 10.F},
                tokens, graph);
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
