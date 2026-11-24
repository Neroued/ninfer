#pragma once

#include "ops/moe/kernels/cpu/invocation.h"

namespace ninfer::ops::detail::moe_routed_cpu {

struct ShapePolicy {
    std::size_t (*capacity)(int tokens);
    void (*launch)(const moe_cpu::Invocation&, MemoryArena&, CpuExecutionView);
};

enum class Isa { Avx2, Avx512, Avx512Bf16 };
Isa select_isa(CpuFeatures features);
const ShapePolicy& shape_dispatch(const RoutedSwiGLUProfile& profile);
const ShapePolicy& q4_q5_h2048_i512_e256_k8();
const ShapePolicy& q4_q6_h2048_i512_e256_k8();
const ShapePolicy& nvfp4_h2560_i640_e512_k10();
void validate_capacity(ExpertSubsetKind subset, int first, int last);

} // namespace ninfer::ops::detail::moe_routed_cpu
