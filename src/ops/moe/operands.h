#pragma once

#include "ninfer/ops/moe.h"
#include "ops/common/host/operands.h"

#include <array>

namespace ninfer::ops::detail {

void validate_route_profile(const RouteProfile& profile);
void validate_expert_profile(const RoutedSwiGLUProfile& profile);
void validate_shared_profile(const SwiGLUProfile& profile, int hidden);
void validate_moe_profile(const MoeProfile& profile);
void validate_interval(int first, int last);
void validate_subset_kind(ExpertSubsetKind kind);
void validate_output(DType type, MoeEpilogue epilogue);

class MoeOperands {
public:
    MoeOperands(const char* operation, CudaExecutionView execution);
    int input(const Tensor& x, int hidden);
    void output(const Tensor& y, int hidden, int tokens, MoeEpilogue epilogue);
    void control(const Weight& control, const RouteProfile& profile);
    void routes(const ExpertRoutes& routes, int top_k, int tokens, bool writing);
    void coefficient(const Tensor& value, int tokens, bool writing);
    void subset(ExpertSubset subset, int experts);
    void experts(const CudaSwiGLUExperts& weights, const RoutedSwiGLUProfile& profile);
    void shared(const SwiGLUWeights& weights, const SwiGLUProfile& profile);
    void workspace(const MemoryArena& workspace, std::size_t bytes);
    void finish() const;

private:
    void read(const void* pointer, std::size_t bytes);
    void write(const Tensor& tensor);
    void matrix(const Weight& weight, int rows, int columns, QType type);
    const char* op_;
    Device device_;
    std::array<common::operand::Range, 32> reads_{};
    std::array<common::operand::Range, 8> writes_{};
    std::size_t read_count_ = 0, write_count_ = 0;
};

} // namespace ninfer::ops::detail
