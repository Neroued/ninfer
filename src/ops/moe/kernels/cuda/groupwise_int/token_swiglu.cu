#include "ops/moe/kernels/cuda/groupwise_int/expert.cuh"
#include "ops/moe/kernels/cuda/groupwise_int/simt.cuh"

#include <algorithm>
#include <type_traits>

namespace ninfer::ops::detail::moe_cuda::groupwise_int {
namespace {
enum class TokenTraversal { Single, Token, Adaptive };

template <class Shape, class Codec, class Access, bool Routed, bool Shared, bool Masked, int Paths,
          TokenTraversal Traversal>
__global__ void token_gate_up(const __nv_bfloat16* __restrict__ input, const int* __restrict__ ids,
                              Access weights, PackedExpertMatrix shared,
                              const int* __restrict__ mask, int member,
                              float* __restrict__ activation, int tokens,
                              const GroupedInfo* __restrict__ info, bool programmatic_input) {
    constexpr int H = Shape::hidden, I = Shape::intermediate, K = Shape::top_k;
    constexpr int Contributions = int(Routed) * K + int(Shared);
    constexpr int PathBlocks    = (Contributions + Paths - 1) / Paths;
    constexpr bool Adaptive     = Traversal == TokenTraversal::Adaptive;
    constexpr bool Single       = Traversal == TokenTraversal::Single;
    __shared__ __align__(16) __nv_bfloat16 x[H];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if constexpr (Adaptive) {
        if (!info->direct) { return; }
    }
    if constexpr (Single) {
        if (!tid) { pdl::trigger_dependents(); }
    }
    const int total = tokens * PathBlocks * I;
    const int begin = Adaptive ? int(blockIdx.x) : int(blockIdx.y) * I + int(blockIdx.x);
    const int step  = Adaptive ? int(gridDim.x) : total;
    for (int work = begin; work < total; work += step) {
        const int token_path = work / I, row = work % I;
        const int token    = Single ? 0 : token_path / PathBlocks;
        const int path     = Single ? warp : (token_path % PathBlocks) * Paths + warp;
        const auto* source = input + std::int64_t(token) * H;
        for (int v = tid; v < H / 8; v += Paths * 32) {
            common::store_vec(x + v * 8, common::load_vec<uint4>(source + v * 8));
        }
        __syncthreads();
        if constexpr (Routed && !Adaptive) {
            if (programmatic_input && path < K) { pdl::wait_for_dependencies(); }
        }
        float gate = 0, up = 0;
        if (path < Contributions) {
            if constexpr (std::is_same_v<Codec, Q8Codec>) {
                ExpertPlanes matrix{shared.codes, nullptr, shared.scales, 0};
                bool selected = true;
                if constexpr (Routed) {
                    if (path < K) {
                        const int expert = ids[token * K + path];
                        if constexpr (Masked) { selected = mask[expert] == member; }
                        if (selected) { matrix = weights.template resolve<2 * I>(expert); }
                    }
                }
                if (selected) {
                    dot_two_rows<Q8Codec, H>(matrix.codes, matrix.high, matrix.scales,
                                             matrix.row + row, matrix.row + I + row, x, 0, H, gate,
                                             up);
                }
            } else {
                if (path < K) {
                    const int expert = ids[token * K + path];
                    bool selected    = true;
                    if constexpr (Masked) { selected = mask[expert] == member; }
                    if (selected) {
                        const auto matrix = weights.template resolve<2 * I>(expert);
                        dot_two_rows<Codec, H>(matrix.codes, matrix.high, matrix.scales,
                                               matrix.row + row, matrix.row + I + row, x, 0, H,
                                               gate, up);
                    }
                } else if constexpr (Shared) {
                    dot_two_rows<Q8Codec, H>(shared.codes, nullptr, shared.scales, row, I + row, x,
                                             0, H, gate, up);
                }
            }
            if (!lane) {
                activation[(std::int64_t(token) * Contributions + path) * I + row] =
                    common::silu(gate) * up;
            }
        }
        if constexpr (Adaptive) {
            __syncthreads();
        } else {
            break;
        }
    }
}

template <class Shape, class Codec, class Access, class Output, bool Routed, bool Shared,
          bool Masked, int Rows, TokenTraversal Traversal>
__global__ void token_down(const int* __restrict__ ids, const float* __restrict__ coefficients,
                           Access weights, PackedExpertMatrix shared, const int* __restrict__ mask,
                           int member, const float* __restrict__ beta,
                           const float* __restrict__ activation, Output output, int tokens,
                           const GroupedInfo* __restrict__ info, MoeExecutionHints hints) {
    constexpr int H = Shape::hidden, I = Shape::intermediate, K = Shape::top_k;
    constexpr int Contributions = int(Routed) * K + int(Shared);
    constexpr bool Adaptive     = Traversal == TokenTraversal::Adaptive;
    constexpr bool Single       = Traversal == TokenTraversal::Single;
    __shared__ float values[Contributions][Rows];
    if constexpr (Adaptive) {
        if (!info->direct) { return; }
    }
    if constexpr (Single) { pdl::wait_for_dependencies(); }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int total = tokens * (H / Rows);
    const int begin = Adaptive ? int(blockIdx.x) : int(blockIdx.y) * (H / Rows) + int(blockIdx.x);
    const int step  = Adaptive ? int(gridDim.x) : total;
    for (int work = begin; work < total; work += step) {
        const int token = Single ? 0 : work / (H / Rows), row = (work % (H / Rows)) * Rows;
        const float* act  = activation + (std::int64_t(token) * Contributions + warp) * I;
        float dot[Rows]   = {};
        float coefficient = 0;
        if constexpr (std::is_same_v<Codec, Q8Codec>) {
            ExpertPlanes matrix{shared.codes, nullptr, shared.scales, 0};
            bool selected = true;
            if constexpr (Routed) {
                if (warp < K) {
                    const int expert = ids[token * K + warp];
                    if constexpr (Masked) { selected = mask[expert] == member; }
                    if (selected) {
                        matrix      = weights.template resolve<H>(expert);
                        coefficient = coefficients[token * K + warp];
                    }
                } else if constexpr (Shared) {
                    coefficient = beta[token];
                }
            } else {
                coefficient = beta[token];
            }
            if (selected) {
                dot_fp32_rows<Q8Codec, Rows, I>(matrix.codes, matrix.high, matrix.scales,
                                                matrix.row + row, act, 0, I / Q8Codec::kGroupK,
                                                dot);
            }
        } else {
            if (warp < K) {
                const int expert = ids[token * K + warp];
                bool selected    = true;
                if constexpr (Masked) { selected = mask[expert] == member; }
                if (selected) {
                    const auto matrix = weights.template resolve<H>(expert);
                    dot_fp32_rows<Codec, Rows, I>(matrix.codes, matrix.high, matrix.scales,
                                                  matrix.row + row, act, 0, I / Codec::kGroupK,
                                                  dot);
                    coefficient = coefficients[token * K + warp];
                }
            } else if constexpr (Shared) {
                dot_fp32_rows<Q8Codec, Rows, I>(shared.codes, nullptr, shared.scales, row, act, 0,
                                                I / Q8Codec::kGroupK, dot);
                coefficient = beta[token];
            }
        }
        if (!lane) {
#pragma unroll
            for (int r = 0; r < Rows; ++r) { values[warp][r] = coefficient * dot[r]; }
        }
        if constexpr (Single) {
            common::prefetch_l2(hints.read_ahead, hints.read_ahead_bytes,
                                (std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128);
        }
        __syncthreads();
        if (!warp && lane < Rows) {
            const auto index = std::int64_t(token) * H + row + lane;
            float sum        = output.initial(index);
#pragma unroll
            for (int p = 0; p < Contributions; ++p) { sum += values[p][lane]; }
            output.store(index, sum);
        }
        if constexpr (Adaptive) {
            __syncthreads();
        } else {
            break;
        }
    }
}

template <class Access, bool Routed, bool Shared, bool Masked, TokenTraversal Traversal>
void token_launch(const Tensor& x, const RoutedSwiGLUProfile& profile, Access gate_up, Access down,
                  const ExpertRoutes& routes, ExpertSubset subset, const ScaledSwiGLUTerm* shared,
                  MoeEpilogue epilogue, Tensor& destination, const TokenPlan& plan,
                  const ExpertStorage& storage, const MoeExecutionHints& hints,
                  CudaExecutionView execution) {
    constexpr bool adaptive     = Traversal == TokenTraversal::Adaptive;
    constexpr bool single       = Traversal == TokenTraversal::Single;
    using S                     = SwiGLUH2048I512E256K8;
    constexpr int Contributions = int(Routed) * S::top_k + int(Shared);
    const auto* ids             = static_cast<const int*>(routes.ids.data);
    const auto* alpha           = static_cast<const float*>(routes.coefficients.data);
    const auto* input           = static_cast<const __nv_bfloat16*>(x.data);
    const auto* mask            = static_cast<const int*>(subset.mask.data);
    const int member            = subset.kind == ExpertSubsetKind::MaskOnes ? 1 : 0;
    const auto shared_gu   = Shared ? matrix(shared->weights.gate_up.weight) : PackedExpertMatrix{};
    const auto shared_down = Shared ? matrix(shared->weights.down.weight) : PackedExpertMatrix{};
    const auto* beta  = Shared ? static_cast<const float*>(shared->coefficients.data) : nullptr;
    const auto* info  = adaptive ? storage.groups.info : nullptr;
    const auto stream = execution.stream();
    const auto gate   = [&]<class Codec, int Paths>() {
        const dim3 grid =
            adaptive ? dim3(5 * S::intermediate)
                       : dim3(S::intermediate, x.ne[1] * ((Contributions + Paths - 1) / Paths));
        const auto kernel =
            token_gate_up<S, Codec, Access, Routed, Shared, Masked, Paths, Traversal>;
        if (plan.programmatic_input) {
            CUDA_CHECK(pdl::launch_dependent({grid, dim3(Paths * 32), 0, stream}, kernel, input,
                                               ids, gate_up, shared_gu, mask, member,
                                               storage.token_activation, x.ne[1], info, true));
        } else {
            kernel<<<grid, Paths * 32, 0, stream>>>(input, ids, gate_up, shared_gu, mask, member,
                                                      storage.token_activation, x.ne[1], info, false);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    const auto gate_codec = [&]<class Codec>() {
        if constexpr (!Routed) {
            gate.template operator()<Codec, 1>();
        } else if constexpr (single) {
            gate.template operator()<Codec, Contributions>();
        } else if constexpr (adaptive) {
            gate.template operator()<Codec, Shared ? 3 : 4>();
        } else {
            switch (plan.paths_per_block) {
            case 1:
                gate.template operator()<Codec, 1>();
                break;
            case 3:
                gate.template operator()<Codec, 3>();
                break;
            case 4:
                gate.template operator()<Codec, 4>();
                break;
            default:
                gate.template operator()<Codec, Contributions>();
                break;
            }
        }
    };
    if constexpr (!Routed) {
        gate_codec.template operator()<Q8Codec>();
    } else if constexpr (adaptive) {
        gate_codec.template operator()<Q4Codec>();
    } else if (profile.ffn.gate_up_type == QType::Q4_G64_FP16) {
        gate_codec.template operator()<Q4Codec>();
    } else {
        gate_codec.template operator()<Q8Codec>();
    }

    with_output(destination, epilogue, [&](auto output) {
        const auto finish = [&]<class Codec, int Rows>() {
            const dim3 grid =
                adaptive ? dim3(5 * (S::hidden / 4)) : dim3(S::hidden / Rows, x.ne[1]);
            const auto kernel = token_down<S, Codec, Access, decltype(output), Routed, Shared,
                                           Masked, Rows, Traversal>;
            if constexpr (single) {
                CUDA_CHECK(pdl::launch_dependent({grid, dim3(Contributions * 32), 0, stream},
                                                 kernel, ids, alpha, down, shared_down, mask,
                                                 member, beta, storage.token_activation, output,
                                                 x.ne[1], info, hints));
            } else {
                kernel<<<grid, Contributions * 32, 0, stream>>>(
                    ids, alpha, down, shared_down, mask, member, beta, storage.token_activation,
                    output, x.ne[1], info, hints);
                CUDA_CHECK(cudaGetLastError());
            }
        };
        const auto codec = [&]<class Codec>() {
            if constexpr (single) {
                finish.template operator()<Codec, 1>();
            } else if constexpr (adaptive) {
                finish.template operator()<Codec, 4>();
            } else if (plan.output_rows == 1) {
                finish.template operator()<Codec, 1>();
            } else if (plan.output_rows == 2) {
                finish.template operator()<Codec, 2>();
            } else {
                finish.template operator()<Codec, 4>();
            }
        };
        if constexpr (!Routed) {
            codec.template operator()<Q8Codec>();
        } else if constexpr (adaptive) {
            if (profile.ffn.down_type == QType::Q5_G64_FP16) {
                codec.template operator()<Q5Codec>();
            } else {
                codec.template operator()<Q6Codec>();
            }
        } else {
            switch (profile.ffn.down_type) {
            case QType::Q5_G64_FP16:
                codec.template operator()<Q5Codec>();
                break;
            case QType::Q6_G64_FP16:
                codec.template operator()<Q6Codec>();
                break;
            default:
                codec.template operator()<Q8Codec>();
                break;
            }
        }
    });
}

} // namespace

void launch_token_swiglu(const Tensor& x, const RoutedSwiGLUProfile& profile,
                         const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                         ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue epilogue,
                         Tensor& destination, const TokenPlan& plan, const ExpertStorage& storage,
                         const MoeExecutionHints& hints, CudaExecutionView execution,
                         bool adaptive) {
    const auto run = [&]<TokenTraversal Traversal>() {
        if (subset.kind == ExpertSubsetKind::None) {
            token_launch<PackedExpertMatrix, false, true, false, Traversal>(
                x, profile, {}, {}, {}, subset, shared, epilogue, destination, plan, storage, hints,
                execution);
            return;
        }
        with_expert_addresses(experts, profile, [&](auto gu, auto down) {
            const auto invoke = [&]<bool Shared, bool Masked>() {
                token_launch<decltype(gu), true, Shared, Masked, Traversal>(
                    x, profile, gu, down, routes, subset, shared, epilogue, destination, plan,
                    storage, hints, execution);
            };
            if (shared) {
                if (subset.kind == ExpertSubsetKind::All) {
                    invoke.template operator()<true, false>();
                } else {
                    invoke.template operator()<true, true>();
                }
            } else {
                if (subset.kind == ExpertSubsetKind::All) {
                    invoke.template operator()<false, false>();
                } else {
                    invoke.template operator()<false, true>();
                }
            }
        });
    };
    if (adaptive) {
        run.template operator()<TokenTraversal::Adaptive>();
    } else if (plan.single_token) {
        run.template operator()<TokenTraversal::Single>();
    } else {
        run.template operator()<TokenTraversal::Token>();
    }
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
