#pragma once

#include "core/tensor.h" // QType, QuantLayout, Weight

#include <cstdint>
#include <string>

namespace ninfer::ops::detail {

enum class LinearFormat {
    DenseBF16,
    DenseFP32,
    GenericUnsupported,
};

enum class ShapeFamily {
    DenseCtrl48x5120,
    MtpFc5120x10240,
    MtpKV1024x5120,
    MtpAttnIn14336x5120,
    AttnInQKV7168x5120,
    GdnInQK4096x5120,
    Proj6144x5120,
    Out5120x6144,
    MlpGateUp34816x5120,
    MlpDown5120x17408,
    Generic,
};

enum class LinearRegime { T1, SmallT, LargeT };

enum class LinearBackendKind { Gemv, Gemm, Reference };

enum class LinearPolicyId {
    GenericDenseGemv,
    GenericDenseGemm,
};

struct LinearPlanKey {
    LinearFormat format;
    ShapeFamily shape;
    LinearRegime regime;
};

struct LinearPlan {
    LinearBackendKind backend;
    LinearPolicyId policy;
    const char* plan_id;    // stable coarse identity (policy granularity in Phase 1)
    bool uses_tensor_cores; // derived metadata, reports only
};

// Host classification for the remaining BF16/FP32 compatibility planner.
LinearFormat classify_format(const Weight& w);
ShapeFamily classify_shape(std::int32_t n, std::int32_t k);
LinearRegime classify_regime(LinearFormat fmt, ShapeFamily shape, std::int32_t t);

// Quantized pure Linear formats are owned by format-local plans and cannot be represented here.
LinearPlan resolve_plan(LinearPlanKey key);

// Names / identity for logs and (future) benchmarks. No behavioral role.
const char* format_name(LinearFormat f);
const char* shape_name(ShapeFamily s);
const char* regime_name(LinearRegime r);
const char* policy_name(LinearPolicyId p);
std::string plan_id_string(LinearPlanKey key, const LinearPlan& plan);

} // namespace ninfer::ops::detail
