#pragma once

#include "core/arena.h"
#include "core/cpu/executor.h"
#include "core/device.h"
#include "core/weight_bank.h"
#include "ninfer/ops/projection_weights.h"
#include "ninfer/ops/weight_input.h"

#include <optional>

namespace ninfer::ops {

// Every SwiGLU uses gate_up [2I,H] with gate rows first, then up rows; down is [H,I].
struct SwiGLUProfile {
    std::int32_t hidden = 0, intermediate = 0;
    QType gate_up_type = QType::BF16, down_type = QType::BF16;
    QuantLayout gate_up_layout = QuantLayout::Contiguous, down_layout = QuantLayout::Contiguous;
    LinearPolicy gate_up_policy = LinearPolicy::A16Only, down_policy = LinearPolicy::A16Only;
    // Absent: SiLU(gate)*up. Present: SiLU(min(gate,L))*clamp(up,-L,L), L finite and positive.
    std::optional<float> clamp_limit;
    friend bool operator==(const SwiGLUProfile&, const SwiGLUProfile&) = default;
};

struct RouteProfile {
    std::int32_t hidden = 0, experts = 0, top_k = 0;
};

struct RoutedSwiGLUProfile {
    SwiGLUProfile ffn;
    std::int32_t experts = 0, top_k = 0;
};

// NVFP4 A4 calibration divisors for x and the FP32 SiLU(gate)*up product, respectively.
// Matrix divisors remain independently addressed by logical expert in the bank.
// Block-FP8 weights and their dynamic activation groups have no tensor divisors.
struct ExpertActivationDivisors {
    std::optional<float> gate_up, down;
};

// Combines activation permissions and calibration across Uses evaluated by one expert call.
// NVFP4 A4 requires the same positive divisor for every included input.
class ExpertInputUsage {
public:
    void include(const WeightInput&);

    [[nodiscard]] LinearPolicy policy() const noexcept { return policy_; }

    [[nodiscard]] std::optional<float> activation_divisor(QType format) const;

private:
    LinearPolicy policy_ = LinearPolicy::AllowA4;
    std::optional<float> divisor_;
    bool missing_divisor_ = false, unequal_divisors_ = false;
};

struct CudaSwiGLUExperts {
    ExpertBankDeviceView bank;
    ExpertActivationDivisors activation_divisors;
};

struct CpuSwiGLUExperts {
    WeightBankView gate_up, down;
};

struct SwiGLUWeights {
    SingleProjectionWeight gate_up, down;
};

// Complete unscaled FFN y=down*(activation(gate*x,up*x)), coefficient exactly one.
// x and destination are contiguous 16-byte-aligned BF16 [H,T], T>0. Gate/up is [2I,H],
// down [H,I]; inputs and native weight planes are immutable and disjoint from destination
// and live caller scratch. Profile and prepared weights must agree. No observable product
// rounding boundary, residual, routes or coefficient tensor. Writes every destination cell.
// Capacity covers the inclusive extent interval from a 256-byte-aligned arena cursor.
// No hidden allocation/transfer/readback; enqueues on execution and supports Graph replay.
// Supports registered BF16, Q8-G32/FP16 and NVFP4 profiles with an explicit finite positive clamp
// limit.
[[nodiscard]] std::size_t swiglu_workspace_capacity_bytes(const SwiGLUProfile& profile,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);
void swiglu(const Tensor& x, const SwiGLUProfile& profile, const SwiGLUWeights& weights,
            Tensor& destination, MemoryArena& workspace, CudaExecutionView execution);

struct ScaledSwiGLUTerm {
    SwiGLUWeights weights;
    Tensor coefficients; // Contiguous FP32 [T].
};

struct ExpertRoutes {
    Tensor ids;          // Contiguous I32 [K,T]; distinct logical IDs within each token.
    Tensor coefficients; // Contiguous FP32 [K,T]; consumed without renormalization.
};

enum class ExpertSubsetKind : std::uint8_t { All, None, MaskOnes, MaskZeros };

struct ExpertSubset {
    ExpertSubsetKind kind = ExpertSubsetKind::All;
    Tensor mask; // I32 [E], 0/1, on the execution device; unused by All and None.
};

enum class MoeEpilogue : std::uint8_t { Write, AddResidual };

struct MoeProfile {
    RoutedSwiGLUProfile routed;
    SwiGLUProfile shared;
};

struct MoeWeights {
    Weight control; // BF16 [E+1,H]: routed logits, then one independent shared logit.
    CudaSwiGLUExperts routed;
    SwiGLUWeights shared;
};

// Pure L2 read-ahead; the referenced CUDA storage outlives the submitted device work.
struct MoeExecutionHints {
    const void* read_ahead       = nullptr;
    std::size_t read_ahead_bytes = 0;
};

[[nodiscard]] SwiGLUProfile swiglu_profile(const SwiGLUWeights& weights);
[[nodiscard]] SwiGLUProfile swiglu_profile(LinearProfile gate_up, LinearProfile down);
[[nodiscard]] RoutedSwiGLUProfile
routed_swiglu_profile(const WeightBankLayout& gate_up, const WeightBankLayout& down,
                      std::int32_t top_k, LinearPolicy gate_up_policy, LinearPolicy down_policy);
[[nodiscard]] RoutedSwiGLUProfile
routed_swiglu_profile(const WeightBankView& gate_up, const WeightBankView& down, std::int32_t top_k,
                      LinearPolicy gate_up_policy, LinearPolicy down_policy);

// Execution admits registered geometries and format pairs; unsupported profiles are rejected.
// For each BF16 x[:,t], l=control*x; select K greatest l[0:E], descending, ties by lower ID.
// Publish their IDs and softmax coefficients, and sigmoid(l[E]) separately. Shapes are
// x [H,T], routes [K,T], shared_coefficients [T]; all contiguous, T>0. x/routes and encoded
// matrix planes are 16-byte aligned; shared coefficients are 4-byte aligned. Every output is
// written. Outputs and live scratch are mutually disjoint and overlap no input. Enqueues on
// execution without host synchronization; CUDA Graph capture/replay is supported.
[[nodiscard]] std::size_t topk_softmax_route_workspace_capacity_bytes(const RouteProfile& profile,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);
void topk_softmax_route(const Tensor& x, const RouteProfile& profile, const Weight& control,
                        ExpertRoutes& routes, Tensor& shared_coefficients, MemoryArena& workspace,
                        CudaExecutionView execution);

// Corrected sigmoid routing over all global experts; no shared logit.
struct SigmoidRouteProfile {
    RouteProfile route;
    float scaling_factor;
    float normalization_epsilon;
};

// Native BF16 control [E,H] produces logits=control*x without an intermediate BF16 cast.
// Choice is sigmoid(logit)+correction_bias[e]; TopK is descending, ties by lower global ID.
// Coefficients are scaling_factor*sigmoid(selected_logit)/(sum_selected_sigmoid+epsilon):
// correction bias affects selection only. Both normalization parameters are positive finite.
// Supports registered H/E/K geometries. Bias is contiguous FP32[E], routes I32/FP32[K,T].
// x/output/weights follow the alignment, immutability, disjoint caller-scratch and Graph
// contract above; there is no BF16 logit boundary, hidden allocation or host readback.
[[nodiscard]] std::size_t topk_sigmoid_route_workspace_capacity_bytes(const SigmoidRouteProfile&,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);
void topk_sigmoid_route(const Tensor& x, const SigmoidRouteProfile&, const Weight& control,
                        const Tensor& correction_bias, ExpertRoutes& routes, MemoryArena&,
                        CudaExecutionView);

struct SigmoidMoeProfile {
    SigmoidRouteProfile routing;
    RoutedSwiGLUProfile routed;
    SwiGLUProfile shared;
};

struct SigmoidMoeWeights {
    Weight control;
    Tensor correction_bias;
    CudaSwiGLUExperts routed;
    SwiGLUWeights shared;
};

// Complete local contribution: routed_swiglu(topk_sigmoid_route(x),subset,x)+shared(x).
// Shared is unscaled, exactly once, including zero routed hits. Only final output is BF16.
// Supports registered block-scaled routed and native BF16 shared profiles,
// both with explicit finite positive clamp limits.
// Destination is contiguous BF16[H,T], Write; no residual, rank, communication or shared gate.
// Router, native bank/subset and shared semantics follow the profiles above; all matrices and
// control inputs are immutable and disjoint from output/live caller scratch. Graph replay may
// change x/bias/routes-inducing data, mask and index contents after prior consumers finish.
// None skips router/routed inputs and computes unscaled shared; those unused views may be empty.
// No hidden allocation/transfer/readback; workspace covers every positive T in the interval.
[[nodiscard]] std::size_t topk_sigmoid_swiglu_moe_workspace_capacity_bytes(const SigmoidMoeProfile&,
                                                                           ExpertSubsetKind,
                                                                           std::int32_t min_tokens,
                                                                           std::int32_t max_tokens);
void topk_sigmoid_swiglu_moe(const Tensor&, const SigmoidMoeProfile&, const SigmoidMoeWeights&,
                             ExpertSubset, Tensor& destination, MemoryArena&, CudaExecutionView);

// F_e(x)=down_e*(activation(gate_e*x,up_e*x)), with the explicit profile clamp when present.
// Sum coefficient[j,t]*F_ids[j,t](x[:,t]) over subset. Registered clamped block-scaled
// geometries support Write to BF16 or FP32, without a scaled shared term.
// The shared overload adds coefficients[t]*F_shared(x[:,t]) exactly once, including for None.
// x is contiguous BF16 [H,T], destination is contiguous FP32 or BF16 [H,T], T>0. Both,
// routes and encoded planes are 16-byte aligned; mask/shared coefficients are 4-byte aligned.
// Write reads no old destination; AddResidual requires BF16 and adds its old value before
// final rounding. None writes zero without accessing routed weights/routes (or computes shared).
// Route IDs are in [0,E); unselected weights and index entries are never read. Callers publish
// and retain every selected weight through execution. All views belong to the execution GPU.
// Destination and live scratch are mutually disjoint and overlap no input. Intermediate precision
// is private; there is no semantic staging cast. No hidden allocation, transfer or host readback
// is performed.
// Fixed-profile Graph replay may change route/mask/index contents after prior consumers finish.
[[nodiscard]] std::size_t
routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile, ExpertSubsetKind subset,
                                       DType output_type, MoeEpilogue epilogue,
                                       std::int32_t min_tokens, std::int32_t max_tokens);
[[nodiscard]] std::size_t routed_swiglu_workspace_capacity_bytes(
    const RoutedSwiGLUProfile& profile, const SwiGLUProfile& shared, ExpertSubsetKind subset,
    DType output_type, MoeEpilogue epilogue, std::int32_t min_tokens, std::int32_t max_tokens);
void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue epilogue, Tensor& destination,
                   MemoryArena& workspace, CudaExecutionView execution);
void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, const ScaledSwiGLUTerm& shared, MoeEpilogue epilogue,
                   Tensor& destination, MemoryArena& workspace, CudaExecutionView execution);

// CPU contribution of the same routed formula. x is BF16 [H,T], destination is FP32 [H,T];
// all tensors are contiguous CPU views with element alignment. Encoded bank planes have
// 16-byte alignment. Route coefficients are finite; masks contain only 0/1. None writes zero
// without accessing routes or banks. Other subsets read only their selected expert payloads.
// Destination is overwritten and, together with live scratch, overlaps no input or each other.
// Intermediate precision is private; no activation quantization or semantic staging cast is
// introduced. The caller retains banks, tensors and scratch until synchronous return. No hidden
// allocation, transfer or thread creation occurs. Shared experts and residual addition are
// separate computations. Requires an AVX2/FMA/F16C worker team for nonempty contributions.
// NVFP4 consumes the native swizzled scales and each expert's matrix divisor directly.
// Capacity covers the inclusive token interval from a 256-byte-aligned local cursor; it is
// independent of worker count and excludes destination storage. None requires no scratch.
[[nodiscard]] std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                                 ExpertSubsetKind subset,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);
void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CpuSwiGLUExperts& experts, const ExpertRoutes& routes, ExpertSubset subset,
                   Tensor& destination, MemoryArena& workspace, CpuExecutionView execution);

// Complete top-k-softmax routed SwiGLU plus sigmoid-gated shared SwiGLU, with the same routing,
// expert and residual formulas above. TopK is selected globally, then restricted to subset without
// renormalizing its coefficients. The local shared term is added exactly once. Destination is
// BF16 [H,T]. Only selected routed weights/index entries must be readable; None reads neither.
// Only destination and caller scratch are mutated; they are mutually disjoint and overlap no
// input. Hints affect cache warming only. CUDA Graph capture/replay is supported.
// Capacity queries cover every positive T in their inclusive interval, from an aligned local
// workspace cursor, with no dependence on route values. Output storage is excluded from scratch.
[[nodiscard]] std::size_t
topk_softmax_swiglu_moe_workspace_capacity_bytes(const MoeProfile& profile, ExpertSubsetKind subset,
                                                 MoeEpilogue epilogue, std::int32_t min_tokens,
                                                 std::int32_t max_tokens);
void topk_softmax_swiglu_moe(const Tensor& x, const MoeProfile& profile, const MoeWeights& weights,
                             ExpertSubset subset, MoeEpilogue epilogue, Tensor& destination,
                             MemoryArena& workspace, CudaExecutionView execution,
                             const MoeExecutionHints& hints = {});

} // namespace ninfer::ops
