#include "core/nvtx.h"
#include "core/platform.h"
#include "ops/moe/kernels/cuda/block_scaled/clamped/workspace.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/operands.h"
#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/routed_swiglu/cuda/contribution.h"
#include "ops/moe/workspace.h"
#include "ops/moe/kernels/cuda/groupwise_int/workspace.h"
#include "ops/moe/kernels/cuda/block_scaled/workspace.h"

namespace ninfer::ops {
namespace {
namespace kernel       = detail::moe_cuda;
namespace integer      = kernel::groupwise_int;
namespace block_scaled = kernel::block_scaled;
namespace impl         = detail::moe_routed;
namespace clamped      = block_scaled::clamped;

struct CallStorage {
    integer::ExpertStorage integer;
    block_scaled::Storage block_scaled;
    clamped::Storage clamped;
#if NINFER_TARGET_RTX5090
    impl::softcap::Storage softcap;
    impl::mxfp8_softcap::Storage mxfp8;
#endif
};

template <class Arena>
CallStorage allocate(Arena& arena, const impl::Plan& plan, int tokens, ExpertSubsetKind subset,
                     int shared_intermediate, bool split_activation, bool partial_output) {
    const bool shared = shared_intermediate != 0;
    if (plan.implementation == impl::Implementation::ClampedBlockScaled)
        return {.clamped = clamped::allocate(arena, plan.clamped, tokens,
                                             subset != ExpertSubsetKind::None, partial_output)};
    if (partial_output)
        throw std::invalid_argument("routed contribution requires a clamped native profile");
#if NINFER_TARGET_RTX5090
    if (plan.implementation == impl::Implementation::SoftcapNvfp4)
        return {.softcap = impl::softcap::allocate(arena, plan.softcap, tokens,
                                                   subset != ExpertSubsetKind::None)};
    if (plan.implementation == impl::Implementation::SoftcapMxfp8)
        return {.mxfp8 = impl::mxfp8_softcap::allocate(arena, plan.mxfp8, tokens,
                                                       subset != ExpertSubsetKind::None)};
#endif
    if (plan.implementation == impl::Implementation::BlockScaled) {
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
                     ExpertSubsetKind subset, DType type, MoeEpilogue mode, int first, int last,
                     bool partial_output) {
    const auto& shape = impl::shape_dispatch(profile);
    detail::validate_interval(first, last);
    detail::validate_subset_kind(subset);
    detail::validate_output(type, mode);
    if (shared && !shape.scaled_shared)
        throw std::invalid_argument("routed_swiglu: this geometry has no scaled shared term");
    if (mode != MoeEpilogue::Write && !shape.accumulate)
        throw std::invalid_argument("routed_swiglu: this geometry requires Write");
    if (shared) { detail::validate_shared_profile(*shared, profile.ffn.hidden); }
    const int shared_intermediate = shared ? shared->intermediate : 0;
    const auto bytes              = [&](int tokens) {
        WorkspaceLayoutBuilder layout;
        const auto plan = shape.select(profile.ffn, tokens, shared_intermediate);
        (void)allocate(layout, plan, tokens, subset, shared_intermediate,
                                    profile.ffn.gate_up_type == QType::Q8_G32_FP16, partial_output);
        return layout.peak_bytes(1);
    };
    const int slice  = shape.slice_tokens(subset);
    const auto range = [&](int begin, int end) {
        return detail::interval_capacity(begin, end, shape.allocation_boundaries, slice, bytes);
    };
#if NINFER_TARGET_RTX5090
    if (profile.ffn.softcap && last > slice) {
        // All complete slices share scratch with the final slice. Remainders
        // wrap across consecutive calls and can use a different family.
        const int first_slice = (first - 1) / slice, last_slice = (last - 1) / slice;
        const int low = (first - 1) % slice + 1, high = (last - 1) % slice + 1;
        if (first_slice == last_slice) return std::max(bytes(slice), range(low, high));
        if (last_slice - first_slice > 1) return range(1, slice);
        return std::max(range(low, slice), range(1, high));
    }
#endif
    return range(first, last);
}

void execute(const Tensor& x, const RoutedSwiGLUProfile& profile, const CudaSwiGLUExperts& experts,
             const ExpertRoutes& routes, ExpertSubset subset, const ScaledSwiGLUTerm* shared,
             MoeEpilogue mode, Tensor& destination, MemoryArena& workspace,
             CudaExecutionView execution, int* valid_tokens) {
    const auto& shape = impl::shape_dispatch(profile);
    if (shared && !shape.scaled_shared)
        throw std::invalid_argument("routed_swiglu: this geometry has no scaled shared term");
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
    auto workspace_scope = workspace.scope();
    const auto bytes = capacity(profile, shared_profile ? &*shared_profile : nullptr, subset.kind,
                                destination.dtype, mode, tokens, tokens, valid_tokens != nullptr);
    MemoryArena scratch(bytes ? workspace.alloc_bytes(bytes)
                              : MemorySpan{nullptr, 0, workspace.device()});
    operands.workspace(scratch.storage());
    operands.finish();
    nvtx::ScopedRange range(nvtx::Name::RoutedSwiGLU, nvtx::Category::Moe, tokens);
    if (subset.kind == ExpertSubsetKind::None && !shared) {
        if (mode == MoeEpilogue::Write) { kernel::launch_zero(destination, execution); }
        return;
    }
    const int shared_intermediate = shared_profile ? shared_profile->intermediate : 0;
    const int slice               = shape.slice_tokens(subset.kind);
    for (std::int64_t start = 0; start < tokens; start += slice) {
        const int offset = static_cast<int>(start);
        const int count  = std::min(tokens - offset, slice);
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
        auto scope      = scratch.scope();
        const auto plan = shape.select(profile.ffn, count, shared_intermediate);
        auto buffers =
            allocate(scratch, plan, count, subset.kind, shared_intermediate,
                     profile.ffn.gate_up_type == QType::Q8_G32_FP16, valid_tokens != nullptr);
        if (valid_tokens) buffers.clamped.active_tokens = valid_tokens + offset;
#if NINFER_TARGET_RTX5090
        if (plan.implementation == impl::Implementation::SoftcapNvfp4) {
            impl::softcap::launch(input, profile.ffn, experts, selection, subset, mode, output,
                                  plan.softcap, buffers.softcap, execution);
            continue;
        }
        if (plan.implementation == impl::Implementation::SoftcapMxfp8) {
            impl::mxfp8_softcap::launch(input, profile.ffn, experts, selection, subset, mode,
                                        output, plan.mxfp8, buffers.mxfp8, execution);
            continue;
        }
#endif
#if NINFER_TARGET_SPARK
        if (plan.implementation == impl::Implementation::ClampedBlockScaled) {
            clamped::launch(input, experts, selection, subset, mode, output, plan.clamped,
                            buffers.clamped, execution);
            continue;
        }
#endif
        if (plan.implementation == impl::Implementation::BlockScaled) {
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
    return capacity(profile, nullptr, subset, type, mode, first, last, false);
}

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                   const SwiGLUProfile& shared,
                                                   ExpertSubsetKind subset, DType type,
                                                   MoeEpilogue mode, int first, int last) {
    return capacity(profile, &shared, subset, type, mode, first, last, false);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue mode, Tensor& destination,
                   MemoryArena& workspace, CudaExecutionView execution) {
    execute(x, profile, experts, routes, subset, nullptr, mode, destination, workspace, execution,
            nullptr);
}

void routed_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                   const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, const ScaledSwiGLUTerm& shared, MoeEpilogue mode,
                   Tensor& destination, MemoryArena& workspace, CudaExecutionView execution) {
    execute(x, profile, experts, routes, subset, &shared, mode, destination, workspace, execution,
            nullptr);
}

namespace detail::moe_routed {
std::size_t contribution_workspace_capacity_bytes(const RoutedSwiGLUProfile& profile,
                                                  ExpertSubsetKind subset, int first, int last) {
    return capacity(profile, nullptr, subset, DType::FP32, MoeEpilogue::Write, first, last, true);
}

RoutedSwiGLUContribution execute_contribution(const Tensor& x, const RoutedSwiGLUProfile& profile,
                                              const CudaSwiGLUExperts& experts,
                                              const ExpertRoutes& routes, ExpertSubset subset,
                                              Tensor& values, int* valid_tokens,
                                              MemoryArena& workspace, CudaExecutionView execution) {
    if (!valid_tokens || subset.kind == ExpertSubsetKind::None || values.dtype != DType::FP32)
        throw std::invalid_argument("invalid private routed contribution output");
    const auto& shape       = shape_dispatch(profile);
    const int tokens        = x.ne[1];
    const bool dense_vector = tokens > 0 && tokens <= shape.slice_tokens(subset.kind) &&
                              shape.select(profile.ffn, tokens, 0).clamped.direct_assignments;
    execute(x, profile, experts, routes, subset, nullptr, MoeEpilogue::Write, values, workspace,
            execution, dense_vector ? nullptr : valid_tokens);
    return {static_cast<const float*>(values.data), dense_vector ? nullptr : valid_tokens};
}
} // namespace detail::moe_routed
} // namespace ninfer::ops
