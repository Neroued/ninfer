#pragma once

#include "core/buffer.h"
#include "ninfer/ops/gdn_control.h"
#include "ops/op_tester.h"
#include "ops/suite.h"
#include "ops/op_check.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <limits>

namespace ninfer::test::gdn_control {

inline constexpr float kEps = 1.0e-6F;

enum class Storage { Parent, Pair };
enum class Pattern {
    Random,
    ZeroInput,
    ZeroA,
    ZeroB,
    Sparse,
    Cancellation,
    GateEdges,
    TinyNormInput,
    LargeNormInput,
    ZeroNormGain
};
enum class Execution { Eager, Graph };

struct Profile {
    int heads, input_rows;
    Storage storage = Storage::Parent;
};

inline constexpr std::array kProfiles{Profile{48, 5120}, Profile{48, 5120, Storage::Pair},
                                      Profile{32, 2048}, Profile{48, 2560}};

struct Case {
    int columns;
    Pattern pattern     = Pattern::Random;
    Execution execution = Execution::Eager;
    bool prefix         = false;
};

std::string label(Profile profile, Case test_case, bool normalize_input);
std::vector<std::uint16_t> bits(const std::vector<float>& values);

class Weights {
public:
    explicit Weights(Profile profile, Pattern pattern);
    void upload();
    int check(std::string_view name) const;
    Profile profile;
    std::vector<float> a, b;
    DeviceBuffer first;
    DeviceBuffer second;
    ops::ProjectionWeights prepared;
private:
    std::vector<std::uint16_t> first_bits_, second_bits_;
};

struct Fixture {
    Fixture(Profile profile, Case test_case, bool normalize_input);
    void upload();
    void advance();
    void repaint(cudaStream_t stream);
    void invoke(MemoryArena& arena, CudaExecutionView execution);
    int check(std::string_view name) const;
    Profile profile;
    Case test_case;
    bool normalize_input;
    Weights weights;
    std::size_t capacity, prefix;
    std::vector<float> x, norm_gain, a_log, dt_bias;
    DeviceBuffer device_x;
    DeviceBuffer device_norm_weight;
    DeviceBuffer device_a_log;
    DeviceBuffer device_dt_bias;
    GuardedDeviceBuffer g, beta, normalized, scratch;
    Tensor input, norm_weight, a_log_tensor, dt_bias_tensor, normalized_tensor;
    ops::GdnControlOutputs outputs;
    MemoryArena arena;
};

std::size_t workspace_capacity(const ops::ProjectionWeights& weights, bool normalize_input,
                               int first, int last);


inline constexpr double kFp32UnitRoundoff = std::numeric_limits<float>::epsilon() / 2.0;

// Against the complete FP64 formula, expf contributes up to two ULP from the correctly
// rounded result and log1pf one ULP. Allow 16u for decay's addition, two exponentials,
// logarithm and multiplication; sigmoid's exponential, addition and reciprocal need 8u.
// Absolute floors cover small outputs, where relative error is poorly conditioned.
inline constexpr PointwiseCriterion kDecay{1.0e-7, 16.0 * kFp32UnitRoundoff};
inline constexpr PointwiseCriterion kSigmoid{1.0e-8, 8.0 * kFp32UnitRoundoff};

// The longest registered BF16 dot has K=5120, with about 320 FP32 MMA accumulation
// steps. gamma(320) supplies a 19.1 ppm accumulation-depth estimate; add decay's 16u
// epilogue budget. This finite gross cap is a numerical quality limit, not a worst-case
// bound for arbitrary dot-product conditioning. Keep the stricter 1.4 ppm relative L2
// criterion to reject distributed error. The whole projection family uses this budget.
inline constexpr double kProjectionAccumulationDepth = 5120.0 / 16.0;
inline constexpr double kProjectionAccumulationError =
    kProjectionAccumulationDepth * kFp32UnitRoundoff /
    (1.0 - kProjectionAccumulationDepth * kFp32UnitRoundoff);
inline constexpr ReductionCriterion kProjection{1.4e-6, 5.0e-7,
                                                kProjectionAccumulationError + kDecay.relative};

inline constexpr ReductionCriterion kNormalized{1.75e-3, 1.0e-4, 4.0e-3};
inline constexpr ReductionCriterion kNormControl{8.0e-4, 1.5e-4, 2.0e-3};

std::vector<Case> projection_cases(Profile profile);
std::vector<Case> norm_cases(Profile profile);

struct Reference {
    std::vector<double> g, beta, normalized;
};

Reference projection_reference(const Fixture& fixture);
int verify_projection(std::string_view name, const Fixture& fixture, const Reference& reference);
double softplus(double value);
double sigmoid(double value);
void add_projection_cases(OpSuite& suite);
int run(Profile profile, Case test_case, bool normalize_input, CudaExecutionView execution);
int validate(Profile profile, bool normalize_input, CudaExecutionView execution);
int validate_capacity(Profile profile, bool normalize_input);
} // namespace ninfer::test::gdn_control
