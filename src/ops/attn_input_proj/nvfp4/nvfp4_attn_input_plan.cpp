#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4AttnInputRoute::A16;
    }
    if (!allows_a4(policy)) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    return tokens >= 4 ? Nvfp4AttnInputRoute::W4A4 : Nvfp4AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kChunk  = 32;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* query = static_cast<std::uint8_t*>(q.data) +
                      static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* output_gate = static_cast<std::uint8_t*>(gate.data) +
                            static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* key = static_cast<std::uint8_t*>(k.data) +
                    static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        auto* value = static_cast<std::uint8_t*>(v.data) +
                      static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor query_chunk(query, DType::BF16, {kQRows, active});
        Tensor gate_chunk(output_gate, DType::BF16, {kQRows, active});
        Tensor key_chunk(key, DType::BF16, {kKvRows, active});
        Tensor value_chunk(value, DType::BF16, {kKvRows, active});
        if (active == 1) {
            nvfp4_attn_input_decode_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                           value_chunk, stream);
        } else {
            nvfp4_attn_input_small_t_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                            value_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Nvfp4AttnInputRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4N14336K5120::kInputRows)
               : 0;
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
