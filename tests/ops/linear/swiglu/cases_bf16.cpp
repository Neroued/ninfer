#include "core/platform.h"
#include <array>
#include "ops/linear/swiglu/runner.h"
#include "ops/linear/weights.h"
#include "ops/float_codec.h"
#include "ninfer/ops/linear_swiglu.h"
#include "cuda_fixture.h"
#include "core/buffer.h"

#include <iostream>
#include <set>

namespace ninfer::test::linear_swiglu {
namespace {
#if NINFER_TARGET_RTX5090
constexpr int kGateUpRows = 18432;
constexpr int kInputRows  = 2304;
constexpr int kOutputRows = kGateUpRows / 2;

int verify_ordinary_only() {
    for (auto policy :
         {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
        if (ops::linear_swiglu_workspace_capacity_bytes(QType::BF16, kGateUpRows, kInputRows,
                                                        policy, 1, 4097) != 0) {
            std::cerr << "BF16 SwiGLU expected zero workspace\n";
            return 1;
        }
    }

    const auto& env   = cuda_environment();
    const auto device = env.context.device();
    const auto packed = linear_weights::make_bf16_weight(kGateUpRows, kInputRows, 2401U);
    DeviceBuffer weights(device, packed.payload.size());
    weights.copy_from_host(packed.payload.data(), packed.payload.size());
    const auto weight = packed.device_weight(weights.span());
    DeviceBuffer input(device, kInputRows * sizeof(std::uint16_t));
    DeviceBuffer output(device, kOutputRows * sizeof(std::uint16_t));
    DeviceBuffer workspace(device, 256);
    const Tensor x(input.span(), DType::BF16, {kInputRows, 1});
    Tensor y(output.span(), DType::BF16, {kOutputRows, 1});
    MemoryArena arena(workspace.span());
    try {
        ops::linear_swiglu(x, weight, y, 10.f, ops::LinearPolicy::A16Only, arena,
                           env.default_execution);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "BF16 SwiGLU accepted clamped [18432, 2304]\n";
    return 1;
}
#endif

quantized_weight::PackedWeight clamp_weight(int recipe) {
    auto packed = linear_weights::make_bf16_weight(2048, 4096, 2101U);
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    for (int row = 0; row < 2048; ++row) {
        const float value =
            row < 1024 ? (recipe == 0 ? -10.5f : 20.f) : (recipe == 2 ? -20.f : 20.f);
        const auto bits = f32_to_bf16(value);
        std::memcpy(packed.payload.data() + std::size_t(row) * 4096 * 2, &bits, 2);
    }
    return packed;
}
} // namespace

void add_bf16_cases(Suite& suite) {
#if NINFER_TARGET_RTX5090
    constexpr auto a16 = ops::LinearPolicy::A16Only;
    std::set<int> points{1, 4, 8, 2048, 4096, 4097};
    for (int end : {1, 16, 32, 64, 96, 128, 144, 192, 256, 384, 640, 768, 960, 1024, 1056, 1152}) {
        for (int t : {end - 1, end, end + 1})
            if (t > 0) points.insert(t);
    }
    const std::vector<int> tokens(points.begin(), points.end());
    add_profile(suite, {QType::BF16, kGateUpRows, kInputRows, kOutputRows, 2401U, a16}, tokens,
                tokens);
    for (auto policy : {ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4}) {
        add_profile(suite, {QType::BF16, kGateUpRows, kInputRows, kOutputRows, 2401U, policy},
                    std::array{1, 8, 17, 129, 257}, std::array{16, 129});
    }
    for (int sign : {-1, 1}) {
        add_negative_gate(
            suite,
            {QType::BF16, kGateUpRows, kInputRows, kOutputRows, std::uint32_t(2409 + sign), a16},
            std::array{1, 4, 17, 33, 65, 97, 129, 145, 257, 385, 1024}, std::array{0}, [sign] {
                auto packed = linear_weights::make_bf16_weight(kGateUpRows, kInputRows, 2409U);
                std::fill(packed.payload.begin(), packed.payload.end(), 0);
                for (int row = 0; row < kGateUpRows; ++row) {
                    const auto bits = f32_to_bf16(row < kOutputRows ? sign * 20.f : 20.f);
                    std::memcpy(packed.payload.data() + std::size_t(row) * kInputRows * 2, &bits,
                                sizeof(bits));
                }
                return packed;
            });
    }
    suite.add(linear_suite::make_case("bf16", kGateUpRows, kInputRows, 1, a16,
                                      linear_suite::Execution::Eager, "ordinary-only"),
              verify_ordinary_only);
#endif
#if NINFER_TARGET_SPARK
    add_profile(suite, {QType::BF16, 12288, 4096, 6144, 2351U, ops::LinearPolicy::A16Only},
                std::array{1, 2, 7, 8, 16, 17, 31, 32, 33, 63, 64, 65, 96, 97, 127, 128, 129},
                std::array{7, 33, 65, 97, 128, 129});
    for (int sign : {-1, 1}) {
        add_negative_gate(suite,
                          {QType::BF16, 12288, 4096, 6144, std::uint32_t(2359 + sign),
                           ops::LinearPolicy::A16Only},
                          std::array{1, 7, 33, 65, 128}, std::array{0}, [sign] {
                              auto packed = linear_weights::make_bf16_weight(12288, 4096, 2359U);
                              std::fill(packed.payload.begin(), packed.payload.end(), 0);
                              for (int row = 0; row < 12288; ++row) {
                                  const auto bits = f32_to_bf16(row < 6144 ? sign * 20.f : 20.f);
                                  std::memcpy(packed.payload.data() + std::size_t(row) * 4096 * 2,
                                              &bits, 2);
                              }
                              return packed;
                          });
    }
    add_profile(suite, {QType::BF16, 20480, 4096, 10240, 2201U, ops::LinearPolicy::A16Only, 10.f},
                std::array{1,   4,   7,   8,    9,    15,   16,   17,   31,   32,
                           33,  63,  64,  65,   127,  128,  129,  255,  256,  257,
                           511, 512, 513, 1024, 2048, 2559, 2560, 2561, 4096, 4097},
                std::array{8, 129, 513});
    add_profile(suite, {QType::BF16, 20480, 4096, 10240, 2203U, ops::LinearPolicy::A16Only},
                std::array{1, 8, 17, 33, 65, 129, 513}, std::array{33});
    constexpr std::array tokens{1,   2,   6,   8,   16,  17,   32,   64,  128,
                                129, 256, 257, 512, 513, 2048, 4096, 4097};
    add_profile(suite, {QType::BF16, 2048, 4096, 1024, 2101U, ops::LinearPolicy::A16Only, 10.f},
                tokens,
                std::array{1, 4, 8, 16, 17, 32, 64, 128, 129, 256, 512, 513, 1024, 2048, 4096});
    add_profile(suite, {QType::BF16, 2048, 4096, 1024, 2103U, ops::LinearPolicy::A16Only},
                std::array{1, 4, 8, 16, 17, 32, 64, 128, 129, 256, 512, 513, 1024, 2048, 4096},
                std::array{4, 129});
    for (auto policy : {ops::LinearPolicy::AllowA8, ops::LinearPolicy::AllowA4})
        add_profile(
            suite, {QType::BF16, 2048, 4096, 1024, std::uint32_t(2111 + int(policy)), policy, 10.f},
            std::array{1, 16, 17, 128, 129, 512, 513, 4096}, std::array{17, 513});
    for (int recipe = 0; recipe < 3; ++recipe)
        add_negative_gate(
            suite,
            {QType::BF16, 2048, 4096, 1024, std::uint32_t(2121 + recipe),
             ops::LinearPolicy::A16Only, 10.f},
            std::array{1, 4, 8, 16, 17, 32, 64, 128, 129, 256, 512, 513, 1024, 2048, 4096},
            std::array{0}, [recipe] { return clamp_weight(recipe); });
    add_negative_gate(suite,
                      {QType::BF16, 2048, 4096, 1024, 2131U, ops::LinearPolicy::A16Only, 3.f},
                      std::array{1, 17, 513}, std::array{0}, [] { return clamp_weight(1); });
#endif
}
} // namespace ninfer::test::linear_swiglu
