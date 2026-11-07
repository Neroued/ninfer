#include "ops/linear/projection/runner.h"
#include "core/platform.h"
#include <array>

namespace ninfer::test::linear {
namespace {
template <QType Format>
quantized_weight::PackedWeight native_weight(int n, int k, std::uint32_t seed) {
    if constexpr (Format == QType::NVFP4_MUL)
        return quantized_weight::make_patterned_weight(
            Format, n, k, seed, {.weight_scale = 0.031713F, .input_scale = 0.01931F});
    else
        return quantized_weight::make_patterned_weight(Format, n, k, seed);
}

#if NINFER_TARGET_SPARK
template <QType Format>
quantized_weight::PackedWeight native_four_byte_scales(int n, int k, std::uint32_t seed) {
    auto weight = native_weight<Format>(n, k, seed);
    // The public format permits a scale plane aligned to four bytes, independently of codes.
    weight.payload.insert(weight.payload.begin() + weight.geometry.scale_offset, 4, 0xa5);
    weight.geometry.scale_offset += 4;
    weight.geometry.bytes += 4;
    weight.scale_plane_offset += 4;
    weight.weight =
        weight.device_weight({weight.payload.data(), weight.payload.size(), Device::cpu()});
    return weight;
}
#endif
} // namespace

void add_native_cases(Suite& suite) {
#if NINFER_TARGET_SPARK
    const std::array<Invocation, 6> mx_calls{
        {{1, CallForm::A16Convenience},
         {5, CallForm::Policy, ops::LinearPolicy::AllowA8},
         {17, CallForm::Policy, ops::LinearPolicy::A16Only},
         {129, CallForm::Policy, ops::LinearPolicy::AllowA8},
         {513, CallForm::Policy, ops::LinearPolicy::AllowA8},
         {33, CallForm::Policy, ops::LinearPolicy::AllowA8, true}}};
    const std::array<std::array<int, 2>, 14> mx_shapes{{{12288, 4096},
                                                        {4096, 4096},
                                                        {4096, 6144},
                                                        {77440, 4096},
                                                        {2048, 4096},
                                                        {4096, 1024},
                                                        {2208, 4096},
                                                        {2176, 4096},
                                                        {640, 4096},
                                                        {20480, 1536},
                                                        {16384, 1536},
                                                        {4096, 16384},
                                                        {3072, 1024},
                                                        {1024, 1024}}};
    for (auto [n, k] : mx_shapes) {
        std::vector<Invocation> calls(mx_calls.begin(), mx_calls.end());
        if (n == 12288 && k == 4096) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            // Complete and partial tiles and the transition to quantized activation workspace.
            calls.insert(calls.end(), {{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {2, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {3, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {7, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {8, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {9, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {16, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {31, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {48, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {64, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {128, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {192, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {193, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {320, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {321, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {4096, CallForm::Policy, ops::LinearPolicy::AllowA8, true}});
            const std::array<Invocation, 4> unaligned_calls{
                {{129, CallForm::Policy, ops::LinearPolicy::AllowA8},
                 {257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                 {512, CallForm::Policy, ops::LinearPolicy::AllowA8},
                 {513, CallForm::Policy, ops::LinearPolicy::AllowA8, true}}};
            add_shape(suite, QType::MXFP8, ActivationRecipe::Hashed,
                      native_four_byte_scales<QType::MXFP8>,
                      {n, k, 1461U, Comparison::Sampled, true, unaligned_calls});
        }
        if (n == 4096 && k == 4096) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            calls.insert(calls.end(), {{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {2, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {4, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {64, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {128, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {256, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {257, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {384, CallForm::Policy, ops::LinearPolicy::AllowA8},
                                       {385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
                                       {4096, CallForm::Policy, ops::LinearPolicy::AllowA8, true}});
        }
        if (n == 4096 && k == 6144) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1, 4, 8, 16, 17, 32, 48, 49, 128, 192, 193, 256, 257, 384, 512, 2048})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 9, 385, 2049, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 77440 && k == 4096) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1, 4, 8, 16, 17, 32, 64, 128, 192, 193, 256, 257, 384, 512})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 9, 65, 385, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 2048 && k == 4096) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t :
                 {1, 4, 8, 16, 32, 48, 49, 64, 128, 192, 193, 256, 257, 384, 512, 768, 1024, 1536})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 9, 65, 385, 769, 1025, 1537, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 4096 && k == 1024) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1, 4, 8, 16, 17, 32, 64, 128, 256, 257, 384, 512, 2048})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 5, 9, 385, 1025, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 2208 && k == 4096) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1, 4, 8, 16, 24, 32, 48, 64, 128, 256, 257, 512, 640})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 9, 65, 641, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 2176 && k == 4096) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1, 8, 16, 24, 32, 40, 64, 128, 256, 257, 320, 512, 640, 704})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {9, 41, 65, 321, 641, 705, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 640 && k == 4096) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 576, 1024})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 65, 577, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 20480 && k == 1536) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1, 4, 8, 16, 32, 48, 64, 80, 88, 127, 128, 192, 193, 384, 512, 640, 768,
                          1024, 2048})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 9, 65, 385, 641, 769, 1025, 2049, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 16384 && k == 1536) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1,   4,   8,   16,  32,  64,  96,   128,  192,  193,
                          256, 257, 384, 512, 640, 768, 1024, 2048, 2049, 3712})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 9, 65, 385, 641, 769, 1025, 3713, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 4096 && k == 16384) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1,   2,   4,   8,   16,  24,   32,   48,   64,   96,   128, 192,
                          256, 384, 512, 640, 768, 1024, 1152, 1280, 1536, 2048, 3072})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {9, 49, 65, 385, 641, 769, 1025, 1153, 1281, 1537, 2049, 3073, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 3072 && k == 1024) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t :
                 {1, 4, 8, 32, 64, 96, 128, 192, 256, 257, 384, 512, 1024, 1152, 2048, 3072})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 65, 129, 385, 1025, 1153, 2049, 3073, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        if (n == 1024 && k == 1024) {
            add_workspace_cases(suite, QType::MXFP8, n, k);
            for (int t : {1, 4, 8, 16, 32, 64, 96, 97, 128, 256, 257, 384, 512, 768, 1024, 2048})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
            for (int t : {2, 65, 385, 769, 1025, 2049, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
        }
        add_shape(suite, QType::MXFP8, ActivationRecipe::Hashed, native_weight<QType::MXFP8>,
                  {n, k, 1451U, Comparison::Sampled, true, calls});
    }
    const std::array<Invocation, 5> a16_calls{
        {{1, CallForm::A16Convenience},
         {5},
         {129},
         {513},
         {33, CallForm::Policy, ops::LinearPolicy::A16Only, true}}};
    for (auto [n, k] : std::array<std::array<int, 2>, 5>{
             {{2048, 4096}, {512, 4096}, {16384, 1536}, {4096, 16384}, {4096, 1024}}}) {
        std::vector<Invocation> calls(a16_calls.begin(), a16_calls.end());
        if (n == 2048 && k == 4096) {
            for (int t : {3, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 896, 960,
                          1024, 2048, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
            for (int t : {2, 65, 129, 193, 257, 385, 641, 769, 897, 961, 1025})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
        }
        if (n == 512 && k == 4096) {
            for (int t : {4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
            for (int t : {49, 65, 97, 129, 193, 385, 513, 1025})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
        }
        if (n == 16384 && k == 1536) {
            for (int t : {4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1152,
                          2048, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
            for (int t : {17, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1153, 2049})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
            const std::array<Invocation, 3> unaligned_calls{{
                {129, CallForm::Policy, ops::LinearPolicy::A16Only},
                {193, CallForm::Policy, ops::LinearPolicy::A16Only, true},
                {257, CallForm::Policy, ops::LinearPolicy::A16Only, true},
            }};
            add_shape(suite, QType::FP8_E4M3FN_BLOCK128X128_FP32, ActivationRecipe::Hashed,
                      native_four_byte_scales<QType::FP8_E4M3FN_BLOCK128X128_FP32>,
                      {n, k, 1471U, Comparison::Sampled, true, unaligned_calls});
        }
        if (n == 4096 && k == 16384) {
            add_workspace_cases(suite, QType::FP8_E4M3FN_BLOCK128X128_FP32, n, k);
            for (int t :
                 {4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
            for (int t : {17, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 2049})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
            for (int t : {2, 104, 640, 897, 1281, 1408})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
            for (int t : {641, 1409})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
            for (int t : {640, 1408}) calls.push_back({t, CallForm::A16Convenience});
            const std::array<Invocation, 3> unaligned_calls{{
                {257, CallForm::Policy, ops::LinearPolicy::A16Only, true},
                {640, CallForm::Policy, ops::LinearPolicy::A16Only, true},
                {4096, CallForm::Policy, ops::LinearPolicy::A16Only},
            }};
            add_shape(suite, QType::FP8_E4M3FN_BLOCK128X128_FP32, ActivationRecipe::Hashed,
                      native_four_byte_scales<QType::FP8_E4M3FN_BLOCK128X128_FP32>,
                      {n, k, 1473U, Comparison::Sampled, true, unaligned_calls});
            const std::array<Invocation, 2> borrowed_calls{{
                {640, CallForm::Policy, ops::LinearPolicy::A16Only},
                {1408, CallForm::Policy, ops::LinearPolicy::A16Only, true},
            }};
            ShapeCase borrowed{n, k, 1475U, Comparison::Sampled, true, borrowed_calls};
            borrowed.workspace_prefix = 17;
            add_shape(suite, QType::FP8_E4M3FN_BLOCK128X128_FP32, ActivationRecipe::Hashed,
                      native_weight<QType::FP8_E4M3FN_BLOCK128X128_FP32>, borrowed);
        }
        if (n == 4096 && k == 1024) {
            for (int t :
                 {2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
            for (int t : {3, 9, 17, 65, 97, 129, 193, 257, 385, 513, 769, 1025})
                calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
            const std::array<Invocation, 2> unaligned_calls{{
                {129, CallForm::Policy, ops::LinearPolicy::A16Only, true},
                {257, CallForm::Policy, ops::LinearPolicy::A16Only, true},
            }};
            add_shape(suite, QType::FP8_E4M3FN_BLOCK128X128_FP32, ActivationRecipe::Hashed,
                      native_four_byte_scales<QType::FP8_E4M3FN_BLOCK128X128_FP32>,
                      {n, k, 1477U, Comparison::Sampled, true, unaligned_calls});
        }
        add_shape(suite, QType::FP8_E4M3FN_BLOCK128X128_FP32, ActivationRecipe::Hashed,
                  native_weight<QType::FP8_E4M3FN_BLOCK128X128_FP32>,
                  {n, k, 1457U, Comparison::Sampled, true, calls});
    }
    const std::array<Invocation, 12> short_k_calls{
        {{1, CallForm::A16Convenience},
         {2},
         {4},
         {5},
         {16},
         {17},
         {32},
         {33, CallForm::Policy, ops::LinearPolicy::A16Only, true},
         {64},
         {65},
         {129},
         {4096}}};
    add_shape(suite, QType::NVFP4_MUL, ActivationRecipe::Hashed, native_weight<QType::NVFP4_MUL>,
              {8192, 1024, 1459U, Comparison::Sampled, true, short_k_calls});
    const std::array<Invocation, 16> narrow_n_calls{
        {{1, CallForm::A16Convenience},
         {2},
         {65},
         {96},
         {97, CallForm::Policy, ops::LinearPolicy::A16Only, true},
         {192},
         {193},
         {384},
         {385},
         {512},
         {768},
         {769},
         {1024},
         {1152},
         {1153},
         {4096}}};
    add_shape(suite, QType::NVFP4_MUL, ActivationRecipe::Hashed, native_weight<QType::NVFP4_MUL>,
              {1024, 4096, 1459U, Comparison::Sampled, true, narrow_n_calls});
    const std::array<Invocation, 20> square_calls{
        {{1, CallForm::A16Convenience},
         {2},
         {8},
         {9},
         {16},
         {17},
         {32},
         {33},
         {64},
         {65},
         {72},
         {73},
         {96},
         {97},
         {128},
         {129, CallForm::Policy, ops::LinearPolicy::A16Only, true},
         {193},
         {257},
         {513},
         {4096}}};
    add_shape(suite, QType::NVFP4_MUL, ActivationRecipe::Hashed, native_weight<QType::NVFP4_MUL>,
              {4096, 4096, 1459U, Comparison::Sampled, true, square_calls});
    const std::array<Invocation, 16> wide_n_calls{
        {{1, CallForm::A16Convenience},
         {2},
         {3},
         {4},
         {5},
         {16},
         {17},
         {32},
         {33},
         {64},
         {65, CallForm::Policy, ops::LinearPolicy::A16Only, true},
         {96},
         {128},
         {129},
         {256},
         {4096}}};
    add_shape(suite, QType::NVFP4_MUL, ActivationRecipe::Hashed, native_weight<QType::NVFP4_MUL>,
              {20480, 4096, 1459U, Comparison::Sampled, true, wide_n_calls});
    const std::array<Invocation, 21> long_k_calls{
        {{1, CallForm::A16Convenience},
         {2},
         {4},
         {8},
         {9},
         {16},
         {17},
         {32},
         {33, CallForm::Policy, ops::LinearPolicy::A16Only, true},
         {64},
         {65},
         {96},
         {97},
         {128},
         {129},
         {192},
         {193},
         {256},
         {257},
         {512},
         {4096}}};
    add_shape(suite, QType::NVFP4_MUL, ActivationRecipe::Hashed, native_weight<QType::NVFP4_MUL>,
              {4096, 10240, 1459U, Comparison::Sampled, true, long_k_calls});
#endif
}
} // namespace ninfer::test::linear
