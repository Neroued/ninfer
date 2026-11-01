#pragma once

#include "core/device.h"
#include "ninfer/ops/gated_residual.h"
#include "ops/op_tester.h"
#include "ops/suite.h"
#include "ops/op_check.h"

namespace ninfer::test::hc {
inline constexpr int kWidth = 2560, kStreams = 4;
inline constexpr float kEpsilon = 1.0e-6F;
enum class Form { ExpandNorm, GroupedNorm, Combine, CombineNorm, AddNorm };
enum class Pattern { Random, Affine, Zero, Constant, Cancellation, Midpoint, Small };
const char* name(Form form);

struct Fixture {
    Fixture(Form form, int columns, bool in_place);
    void set(Pattern pattern, int phase = 0);
    void upload(cudaStream_t stream);
    void invoke(CudaExecutionView execution);
    int check(const std::string& label) const;

    Form form;
    int columns;
    bool in_place;
    float epsilon = kEpsilon;
    std::vector<float> h, block, weight, injection;
    GuardedDeviceBuffer dh, db, dw, dj, dr, dz;
    Tensor th, tb, tw, tj, tr, tz;
};

// BF16 RNE has unit roundoff 2^-8. Coherent storage error can approach that
// bound even on constant, unit-scale inputs; random-input RMS limits are not
// universal bounds. Leave the small remaining margin for FP32 FMA/reduction/rsqrt.
// Nonfinite results fail. These budgets cover all forms/routes and all cases.
inline constexpr PointwiseCriterion kResidual{1.0e-7, 4.0e-3};
inline constexpr ReductionCriterion kNormalized{4.0e-3, 1.0e-6, 4.0e-3};
// Mix includes BF16 u/output storage and FP32 dot reductions.
// sigmoid is 1/4-Lipschitz; injection's derivative with respect to its dot is at most 1/8.
// The output L2 budget covers staging and sigmoid sensitivity, with a finite gross cap. Injection
// uses its own pointwise budget. The same A16 budgets apply to both forms and every production
// route.
inline constexpr ReductionCriterion kMixOutput{8.0e-3, 1.0e-5, 1.6e-2};
inline constexpr PointwiseCriterion kMixInjection{2.0e-6, 2.0e-5};

struct Reference {
    std::vector<double> residual, normalized;
    std::vector<std::uint16_t> residual_words;
};

Reference reference(const Fixture& fixture);
std::vector<double> grouped_norm(std::span<const double> input, std::span<const float> weight,
                                 int columns, float epsilon);
int verify(const Fixture& fixture, const std::string& label, bool exact_residual);
void add_norm_cases(OpSuite& suite);
void add_mix_cases(OpSuite& suite);
int validate(Form form, CudaExecutionView execution);
} // namespace ninfer::test::hc
