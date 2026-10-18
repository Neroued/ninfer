#include "core/nvtx.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/operands.h"
#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/workspace.h"
#include "ops/moe/kernels/cuda/groupwise_int/workspace.h"
#include "ops/moe/kernels/cuda/block_scaled/workspace.h"

namespace ninfer::ops {
namespace {
namespace kernel       = detail::moe_cuda;
namespace integer      = kernel::groupwise_int;
namespace block_scaled = kernel::block_scaled;
namespace impl         = detail::moe_routed;

struct CallStorage {
    integer::ExpertStorage integer;
    block_scaled::Storage block_scaled;
};

template <class Arena>
CallStorage allocate(Arena& arena, const impl::Plan& plan, int tokens, ExpertSubsetKind subset,
                     int shared_intermediate, bool split_activation) {
    const bool shared = shared_intermediate != 0;
    if (plan.implementation == kernel::ExpertImplementation::BlockScaled) {
        CallStorage out;
        out.block_scaled =
            block_scaled::allocate(arena, plan.block_scaled, tokens,
                                   subset != ExpertSubsetKind::None, shared_intermediate);
        return out;
    }
    if (subset == ExpertSubsetKind::None) {
        if (!shared) { return {}; }
        return {.integer = integer::allocate_shared(arena, tokens)};
    }
    return {.integer = plan.integer.grouped
                           ? integer::allocate_grouped(arena, tokens, plan.integer.group, shared,
                                                       split_activation)
                           : integer::allocate_token(arena, tokens, shared)};
}

std::size_t capacity(const RoutedSwiGLUProfile& profile, const SwiGLUProfile* shared,
                     ExpertSubsetKind subset, DType type, MoeEpilogue mode, int first, int last) {
    const auto& shape = impl::shape_dispatch(profile);
    detail::validate_interval(first, last);
    detail::validate_subset_kind(subset);
    detail::validate_output(type, mode);
    if (shared) { detail::validate_shared_profile(*shared, profile.ffn.hidden); }
    const int shared_intermediate = shared ? shared->intermediate : 0;
    return detail::interval_capacity(
        first, last, shape.allocation_boundaries, impl::kSliceTokens, [&](int tokens) {
            WorkspaceLayoutBuilder layout;
            (void)allocate(layout, shape.select(tokens, shared_intermediate), tokens, subset,
                           shared_intermediate, profile.ffn.gate_up_type == QType::Q8_G32_FP16);
            return layout.peak_bytes(1);
        });
}

void execute(const Tensor& x, const RoutedSwiGLUProfile& profile, const CudaSwiGLUExperts& experts,
             const ExpertRoutes& routes, ExpertSubset subset, const ScaledSwiGLUTerm* shared,
             MoeEpilogue mode, Tensor& destination, MemoryArena& workspace,
             CudaExecutionView execution) {
    const auto& shape = impl::shape_dispatch(profile);
    detail::MoeOperands operands("routed_swiglu", execution);
    const int tokens = operands.input(x, profile.ffn.hidden);
    operands.output(destination, profile.ffn.hidden, tokens, mode);
    operands.subset(subset, profile.experts);
    if (subset.kind != ExpertSubsetKind::None) {
        operands.routes(routes, profile.top_k, tokens, false);
        operands.experts(experts, profile);
    }
    std::optional<SwiGLUProfile> shared_profile;
    if (shared) {
        shared_profile = swiglu_profile(shared->weights);
        operands.shared(shared->weights, *shared_profile);
        operands.coefficient(shared->coefficients, tokens, false);
    }
    operands.workspace(workspace, capacity(profile, shared_profile ? &*shared_profile : nullptr,
                                           subset.kind, destination.dtype, mode, tokens, tokens));
    operands.finish();
    nvtx::ScopedRange range(nvtx::Name::RoutedSwiGLU, nvtx::Category::Moe, tokens);
    if (subset.kind == ExpertSubsetKind::None && !shared) {
        if (mode == MoeEpilogue::Write) { kernel::launch_zero(destination, execution); }
        return;
    }
    const int shared_intermediate = shared_profile ? shared_profile->intermediate : 0;
    for (std::int64_t start = 0; start < tokens; start += impl::kSliceTokens) {
        const int offset = static_cast<int>(start);
        const int count  = std::min(tokens - offset, impl::kSliceTokens);
        const auto input = x.slice(1, offset, count);
        auto output      = destination.slice(1, offset, count);
        const ExpertRoutes selection =
            subset.kind == ExpertSubsetKind::None
                ? ExpertRoutes{}
                : ExpertRoutes{routes.ids.slice(1, offset, count),
                               routes.coefficients.slice(1, offset, count)};
        std::optional<ScaledSwiGLUTerm> term;
        if (shared) {
            term = ScaledSwiGLUTerm{shared->weights, shared->coefficients.slice(0, offset, count)};
        }
        auto scope         = workspace.scope();
        const auto plan    = shape.select(count, shared_intermediate);
        const auto buffers = allocate(workspace, plan, count, subset.kind, shared_intermediate,
                                      profile.ffn.gate_up_type == QType::Q8_G32_FP16);
        if (plan.implementation == kernel::ExpertImplementation::BlockScaled) {
            block_scaled::launch(input, experts, selection, subset, term ? &*term : nullptr, mode,
                                 output, plan.block_scaled, buffers.block_scaled, execution);
            continue;
        }
        const auto& storage = buffers.integer;
        if (subset.kind == ExpertSubsetKind::None) {
            integer::launch_shared_swiglu(input, *term, mode, output, storage, plan.integer.shared,
                                          execution);
            continue;
        }
        if (plan.integer.grouped) {
            integer::launch_group_count(selection, subset, storage.groups, execution);
            integer::launch_group_scan(selection, plan.integer.group, storage.groups, execution);
            if (plan.integer.group.adaptive) {
                integer::launch_token_swiglu(input, profile, experts, selection, subset,
                                             term ? &*term : nullptr, mode, output,
                                             plan.integer.token, storage, {}, execution, true);
            }
            integer::launch_group_index(selection, storage.groups, execution);
            integer::launch_grouped_swiglu(input, profile, experts, selection,
                                           term ? &*term : nullptr, mode, output,
                                           plan.integer.group, storage, execution);
        } else {
            integer::launch_token_swiglu(input, profile, experts, selection, subset,
                                         term ? &*term : nullptr, mode, output, plan.integer.token,
                                         storage, {}, execution);
        }
    }
}
} // namespace

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   ExpertSubsetKind subset, DType type,
                                                   MoeEpilogue mode, int first, int last) {
    return capacity(profile, nullptr, subset, type, mode, first, last);
}

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   const SwiGLUProfile& shared,
                                                   ExpertSubsetKind subset, DType type,
                                                   MoeEpilogue mode, int first, int last) {
    return capacity(profile, &shared, subset, type, mode, first, last);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue mode, Tensor& destination,
                   MemoryArena& workspace, CudaExecutionView execution) {
    execute(x, profile, experts, routes, subset, nullptr, mode, destination, workspace, execution);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, const ScaledSwiGLUTerm& shared, MoeEpilogue mode,
                   Tensor& destination, MemoryArena& workspace, CudaExecutionView execution) {
    execute(x, profile, experts, routes, subset, &shared, mode, destination, workspace, execution);
}
} // namespace ninfer::ops
