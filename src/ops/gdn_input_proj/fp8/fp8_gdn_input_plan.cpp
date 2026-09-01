#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_launch.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kVoltaCutlassMinT = 33;

enum class Fp8GdnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8GdnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 gdn_input_proj: unsupported policy");
    }
    return tokens >= 8 ? Fp8GdnInputRoute::A8 : Fp8GdnInputRoute::A16;
}

} // namespace

std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    if (resolve_route(policy, max_tokens) == Fp8GdnInputRoute::A8) {
        return fp8_a8_workspace_capacity_bytes(max_tokens, Fp8GdnInputGeometry::kInputRows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (max_tokens >= kVoltaCutlassMinT) {
        return fp8_gdn_input_cutlass_workspace_bytes(max_tokens);
    }
#endif
    return 0;
}

void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                WorkspaceArena* workspace, cudaStream_t stream) {
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
#ifdef NINFER_VOLTA_BUILD
    if (x.ne[1] >= kVoltaCutlassMinT) {
        if (workspace == nullptr) {
            throw std::invalid_argument("fp8 Volta GDN prefill requires caller workspace");
        }
        fp8_gdn_input_cutlass_sm70_launch(x, weight, qkv, z, *workspace, stream);
        return;
    }
    const bool qpn = fp8_volta_qpn_supported(weight.n, weight.k, kFp8VoltaQpnMaxTokens);
    const std::int32_t kChunk =
        qpn ? kFp8VoltaQpnMaxTokens : kFp8LinearSmallTMax<Fp8GdnInputGeometry>;
#else
    constexpr std::int32_t kChunk   = kFp8LinearSmallTMax<Fp8GdnInputGeometry>;
#endif
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* qkv_output =
            static_cast<std::uint8_t*>(qkv.data) +
            static_cast<std::int64_t>(token_begin) * kQkvRows * sizeof(std::uint16_t);
        auto* z_output = static_cast<std::uint8_t*>(z.data) +
                         static_cast<std::int64_t>(token_begin) * kZRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor qkv_chunk(qkv_output, DType::BF16, {kQkvRows, active});
        Tensor z_chunk(z_output, DType::BF16, {kZRows, active});
#ifdef NINFER_VOLTA_BUILD
        if (fp8_volta_qpn_supported(weight.n, weight.k, active)) {
            launch_fp8_gdn_input_volta_qpn(input_chunk, weight, qkv_chunk, z_chunk, stream);
            continue;
        }
#endif
        if (active == 1) {
            fp8_gdn_input_decode_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
        } else {
            fp8_gdn_input_small_t_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
        }
    }
}

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                   = workspace.scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    fp8_gdn_input_a8_launch(x, weight, qkv, z, scratch, stream);
}

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8GdnInputRoute::A16) {
        fp8_gdn_input_a16_dispatch(x, weight, qkv, z, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 gdn_input_proj requires caller workspace");
    }
    fp8_gdn_input_a8_dispatch(x, weight, qkv, z, *workspace, stream);
}

} // namespace ninfer::ops::detail
