#pragma once

#include "core/tensor.h"   // QType, QuantLayout, Weight

#include <cstdint>
#include <string>

namespace ninfer::kernels::detail {

enum class LinearFormat {
    Q4G64_RowSplit,
    Q5G64_RowSplit,
    Q6G64_RowSplit,
    W8G32_RowSplit,
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
    LmHeadVocabx5120,
    Generic,
};

enum class LinearRegime { T1, SmallT, LargeT };

enum class LinearBackendKind { Gemv, Gemm, Reference };

enum class LinearPolicyId {
    RowsplitLowbitGemmSmallt,
    RowsplitLowbitGemmMma,
    RowsplitW8G32GemmMma,
    GenericDenseGemv,
    GenericDenseGemm,
    MlpGateUp34816Q4RowsplitGemv,
    AttnInQKV7168Q4RowsplitGemv,
    AttnInQKV7168Q5RowsplitGemv,
    GdnInQK4096Q4RowsplitGemv,
    MlpDownQ5RowsplitGemv,
    LmHeadQ6RowsplitGemv,
    LmHeadQ4RowsplitGemv,
    Proj6144Q5RowsplitGemv,
    Out6144Q5RowsplitGemv,
};

struct LinearPlanKey {
    LinearFormat format;
    ShapeFamily  shape;
    LinearRegime regime;
};

struct LinearPlan {
    LinearBackendKind backend;
    LinearPolicyId    policy;
    const char*       plan_id;           // stable coarse identity (policy granularity in Phase 1)
    bool              uses_tensor_cores;  // derived metadata, reports only
};

// Host classification. classify_format returns GenericUnsupported for any (qtype, layout) the
// wrapper does not accept; the wrapper validation rejects those before dispatch.
LinearFormat classify_format(const Weight& w);
ShapeFamily  classify_shape(std::int32_t n, std::int32_t k);
LinearRegime classify_regime(LinearFormat fmt, ShapeFamily shape, std::int32_t t);

// Phase 1 registry: every key resolves to a Generic (reference) plan.
LinearPlan resolve_plan(LinearPlanKey key);

// Names / identity for logs and (future) benchmarks. No behavioral role.
const char* format_name(LinearFormat f);
const char* shape_name(ShapeFamily s);
const char* regime_name(LinearRegime r);
const char* policy_name(LinearPolicyId p);
std::string plan_id_string(LinearPlanKey key, const LinearPlan& plan);

} // namespace ninfer::kernels::detail
