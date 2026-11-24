#include "ops/moe/routed_swiglu/cpu/dispatch.h"
#include "ops/moe/kernels/cpu/nvfp4/work.h"

namespace ninfer::ops::detail::moe_routed_cpu {
namespace {
namespace kernel          = moe_cpu::nvfp4;
constexpr int token_block = 256;

kernel::Plan select_plan(CpuFeatures features) {
    switch (select_isa(features)) {
    case Isa::Avx512Bf16:
        return {.run = kernel::run_avx512_bf16, .token_block = token_block, .max_columns = 8};
    case Isa::Avx512:
        return {.run = kernel::run_avx512, .token_block = token_block, .max_columns = 8};
    case Isa::Avx2:
        return {.run = kernel::run_avx2, .token_block = token_block, .max_columns = 2};
    }
    __builtin_unreachable();
}

std::size_t capacity(int tokens) { return kernel::capacity(tokens, token_block); }

void launch(const moe_cpu::Invocation& args, MemoryArena& arena, CpuExecutionView execution) {
    kernel::launch(args, select_plan(execution.features()), arena, execution);
}
} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() {
    static const ShapePolicy policy{.capacity = capacity, .launch = launch};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed_cpu
