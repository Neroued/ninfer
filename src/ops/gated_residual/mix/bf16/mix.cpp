#include "ninfer/ops/gated_residual.h"
#include "ops/gated_residual/mix/bf16/implementation.h"

namespace ninfer::ops {
namespace mix = detail::hc::mix::bf16;

namespace {
Weight describe_weight(const LinearProfile& profile) {
    Weight w;
    w.qtype  = profile.qtype;
    w.layout = profile.layout;
    w.ndim   = 2;
    w.n = w.shape[0] = w.padded_shape[0] = profile.n;
    w.k = w.shape[1] = w.padded_shape[1] = profile.k;
    if (profile.n > 0 && profile.k > 0)
        w.payload_bytes = std::uint64_t(profile.n) * profile.k * sizeof(std::uint16_t);
    return w;
}

} // namespace

void hc_mix(const Tensor& normalized, const HcBf16MixWeights& weights, MemoryArena& workspace,
            Tensor& block_input, Tensor& injection, CudaExecutionView execution) {
    mix::execute({mix::Kind::Block, weights.down_injection, weights.up}, normalized, workspace,
                 block_input, &injection, execution);
}

void hc_mix(const Tensor& normalized, const HcBf16FinalMixWeights& weights, MemoryArena& workspace,
            Tensor& sample_hidden, CudaExecutionView execution) {
    mix::execute({mix::Kind::Final, weights.down, weights.up}, normalized, workspace, sample_hidden,
                 nullptr, execution);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcBf16MixWeights& weights, std::int32_t first,
                                            std::int32_t last) {
    return mix::capacity({mix::Kind::Block, weights.down_injection, weights.up}, first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcBf16FinalMixWeights& weights, std::int32_t first,
                                            std::int32_t last) {
    return mix::capacity({mix::Kind::Final, weights.down, weights.up}, first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcBf16MixProfile& p, std::int32_t first,
                                            std::int32_t last) {
    return hc_mix_workspace_capacity_bytes(
        HcBf16MixWeights{describe_weight(p.down_injection), describe_weight(p.up)}, first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcBf16FinalMixProfile& p, std::int32_t first,
                                            std::int32_t last) {
    return hc_mix_workspace_capacity_bytes(
        HcBf16FinalMixWeights{describe_weight(p.down), describe_weight(p.up)}, first, last);
}

} // namespace ninfer::ops
