#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Nvfp4GdnInputRoute::A16; }
    if (policy == LinearPolicy::AllowA4) { return Nvfp4GdnInputRoute::W4A4; }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                cudaStream_t stream) {
    constexpr std::int32_t kChunk   = kNvfp4LastSmallT;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
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
        if (active == 1) {
            nvfp4_gdn_input_decode_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
        } else {
            nvfp4_gdn_input_small_t_launch(input_chunk, weight, qkv_chunk, z_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
#ifdef NINFER_NVFP4_W4A4
    return resolve_route(policy, max_tokens) == Nvfp4GdnInputRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4GdnInputGeometry::kInputRows)
               : 0;
#else
    if (resolve_route(policy, max_tokens) == Nvfp4GdnInputRoute::W4A4) {
        throw std::invalid_argument(
            "nvfp4 W4A4 gdn_input_proj requires the Blackwell FP4 tensor-core build "
            "(CMAKE_CUDA_ARCHITECTURES=120a)");
    }
    return 0;
#endif
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4GdnInputRoute::A16) {
        launch_a16(x, weight, qkv, z, stream);
        return;
    }
#ifdef NINFER_NVFP4_W4A4
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 gdn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_w4a4_launch(x, weight, qkv, z, scratch, stream);
#else
    throw std::invalid_argument(
        "nvfp4 W4A4 gdn_input_proj requires the Blackwell FP4 tensor-core build "
        "(CMAKE_CUDA_ARCHITECTURES=120a)");
#endif
}

} // namespace ninfer::ops::detail
