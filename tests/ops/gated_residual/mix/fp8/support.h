#pragma once
#include "ops/gated_residual/support.h"
#include "ops/quantized_weight.h"
#include "ops/direct_bf16_weight.h"

namespace ninfer::test::hc::mix::fp8 {
inline constexpr int H = 5120, R = 640, K = 4 * H, Catalog = 7;
enum class Pattern { Dense, Mapping, Saturation, Zero };

struct Weights {
    Weights(Pattern);
    Pattern pattern;
    quantized_weight::PackedWeight down, up;
    direct_bf16_weight::HostWeight inject;
    std::vector<float> wd, wu;
    GuardedDeviceBuffer dd, du, dj;
    ops::HcFp8MixWeights bound;
};

std::vector<float> input(Pattern, int phase = 0);

struct Ref {
    std::vector<double> output, injection;
};

Ref reference(const Weights&, std::span<const float>);

struct Fixture {
    Fixture(Weights&, int, bool, ops::LinearPolicy);
    Fixture(Weights&, int, bool, ops::LinearPolicy, ops::LinearPolicy);
    void upload(std::span<const float>, cudaStream_t);
    void invoke(CudaExecutionView);
    int check(std::span<const float>, const Ref&, const std::string&) const;
    Weights& weights;
    int m;
    bool final;
    ops::LinearPolicy policy, up_policy;
    GuardedDeviceBuffer dz, dx, dj, scratch;
    MemoryArena arena;
    Tensor z, x, j;
    std::vector<std::uint16_t> words;
};
} // namespace ninfer::test::hc::mix::fp8
