#pragma once
#include "ninfer/ops/mhc.h"
#include "ops/op_tester.h"
#include "ops/op_check.h"
#include "ops/suite.h"
#include "ops/direct_bf16_weight.h"

namespace ninfer::test::mhc {
inline constexpr int D = 4096, S = 4, N = 24, K = D * S;
enum class Form { Pre, Post, ExpandPre, PostPre, PostMeanNorm };
enum class Pattern { General, Zero, Saturated, Unbalanced, Midpoint, Small };
const char* name(Form f);
using CapacityQuery = std::size_t (*)(const ops::MhcGeometry&, int, int, int);
CapacityQuery capacity_query(Form f);
bool has_pre(Form f);

// One BF16 output boundary permits unit roundoff 2^-8. FP32 coefficient outputs
// have a separate reduction budget, including dot cancellation and Sinkhorn divisions.
// Every criterion rejects non-finites and is shared by all cases and production routes.
inline constexpr ReductionCriterion kBf16{4.e-3, 2.e-6, 4.1e-3};
inline constexpr ReductionCriterion kCoefficients{2.e-5, 2.e-6, 3.e-5, 2.e-7};
// Fused outputs include error propagated through the residual BF16 boundary. A second
// independent pre reference uses the published residual with the strict FP32 budget.
inline constexpr ReductionCriterion kFusedCoefficients{1.e-3, 2.e-6, 2.e-3, 2.e-7};
inline constexpr ReductionCriterion kFinalHidden{4.e-3, 2.e-6, 1.2e-2};

struct Fixture {
    Fixture(Form form, int columns, bool in_place, std::size_t tensor_guard_bytes = 256);
    void set(Pattern pattern, int phase = 0);
    void upload(cudaStream_t stream);
    void invoke(CudaExecutionView e, MemoryArena* workspace = nullptr);
    float final_epsilon = 1.e-5F;
    int preserved(const std::string& label) const;
    Form form;
    int m;
    bool alias;
    ops::MhcPreParameters parameters{1.e-5F, 1.e-5F, 1.e-6F, 1.e-6F, 2.F, 20};
    std::vector<float> x, y, f, base, scale, gamma, q, c;
    GuardedDeviceBuffer dx, dy, df, db, ds, dg, dq, dc, dr, du, dnq, dnc;
    Tensor tx, ty, tr;
    ops::MhcPreWeights weights;
    ops::MhcCoefficients coefficients;
    ops::MhcPreOutputs outputs;
    std::size_t capacity;
    GuardedDeviceBuffer scratch;
    MemoryArena arena;
};

struct PreReference {
    std::vector<double> u, q, c;
};

PreReference pre_reference(const Fixture& f, std::span<const double> x);
std::vector<double> post_reference(const Fixture& f, int m);
double round_bf16(double x);
int verify(const Fixture& f, const std::string& label);
int validate(Form form, CudaExecutionView e);
void add_cases(OpSuite& suite);
} // namespace ninfer::test::mhc
