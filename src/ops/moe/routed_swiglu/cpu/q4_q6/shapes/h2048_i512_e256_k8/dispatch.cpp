#include "ops/moe/routed_swiglu/cpu/dispatch.h"
#include "ops/moe/kernels/cpu/groupwise_int/work.h"

namespace ninfer::ops::detail::moe_routed_cpu {
namespace {
namespace kernel          = moe_cpu::groupwise_int;
constexpr int token_block = 8;

kernel::Plan select_plan(CpuFeatures features) {
    kernel::Plan plan{.token_block = token_block, .down_bits = 6};
    switch (select_isa(features)) {
    case Isa::Avx512Bf16:
        plan.kernel      = &kernel::avx512_bf16;
        plan.max_columns = 4;
        break;
    case Isa::Avx512:
        plan.kernel      = &kernel::avx512;
        plan.max_columns = 4;
        break;
    case Isa::Avx2:
        plan.kernel      = &kernel::avx2;
        plan.max_columns = 2;
        break;
    }
    return plan;
}

std::size_t capacity(int tokens) { return kernel::capacity(tokens, token_block); }

void launch(const moe_cpu::Invocation& args, MemoryArena& arena, CpuExecutionView execution) {
    kernel::launch(args, select_plan(execution.features()), arena, execution);
}
} // namespace

const ShapePolicy& q4_q6_h2048_i512_e256_k8() {
    static const ShapePolicy policy{.capacity = capacity, .launch = launch};
    return policy;
}
} // namespace ninfer::ops::detail::moe_routed_cpu
