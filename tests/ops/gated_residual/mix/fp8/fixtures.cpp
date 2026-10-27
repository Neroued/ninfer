#include "support.h"

namespace ninfer::test::hc::mix::fp8 {
namespace {
quantized_weight::PackedWeight matrix(int n, int k, Pattern p, bool up) {
    auto w =
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, n, k, up ? 719 : 701);
    for (int r = 0; r < n; ++r) {
        const float scale = p == Pattern::Dense
                                ? (up ? 4.F : 1.F) * (1.F + .25F * (r % 4)) / std::sqrt(float(k))
                                : (p == Pattern::Saturation ? (up ? 64.F : 16.F) : 1.F);
        quantized_weight::detail::store_u16_le(w.payload, w.scale_plane_offset + 2 * r,
                                               f32_to_bf16(scale));
        for (int c = 0; c < k; ++c) {
            std::uint32_t h =
                std::uint32_t(r) * 0x9e3779b9U + std::uint32_t(c) * 0x85ebca6bU + (up ? 719 : 701);
            h ^= h >> 16;
            h *= 0x7feb352dU;
            h ^= h >> 15;
            auto code = std::uint8_t(0x28 + 4 * ((h >> 8) % 6));
            if (h & 1) code |= 0x80;
            if (p != Pattern::Dense) {
                const int selected = up ? r % R : (r % 4) * H + (r * 7 + 3) % H;
                code               = (p != Pattern::Zero && c == selected)
                                         ? std::uint8_t((r % 3 == 0) ? 0xb8 : 0x38)
                                         : 0;
            }
            w.payload[std::size_t(r) * k + c] = code;
        }
    }
    return w;
}

std::size_t capacity(Weights& w, int m, bool final, ops::LinearPolicy p, ops::LinearPolicy up) {
    auto b        = w.bound;
    b.down.policy = p;
    b.up.policy   = up;
    return final
               ? ops::hc_mix_workspace_capacity_bytes(ops::HcFp8FinalMixWeights{b.down, b.up}, m, m)
               : ops::hc_mix_workspace_capacity_bytes(b, m, m);
}
} // namespace

Weights::Weights(Pattern p)
    : pattern(p), down(matrix(R, K, p, false)), up(matrix(K, R, p, true)),
      inject(direct_bf16_weight::make_patterned(4, K, 733)), wd(std::size_t(R) * K),
      wu(std::size_t(K) * R), dd(down.payload.size()), du(up.payload.size()),
      dj(inject.bits.size() * 2) {
    for (int r = 0; r < R; ++r)
        for (int k = 0; k < K; ++k)
            wd[std::size_t(r) * K + k] = quantized_weight::logical_weight_fp64(down, r, k);
    for (int r = 0; r < K; ++r)
        for (int k = 0; k < R; ++k)
            wu[std::size_t(r) * R + k] = quantized_weight::logical_weight_fp64(up, r, k);
    for (auto& x : inject.bits) x = f32_to_bf16(bf16_to_f32(x) * .125F);
    if (p != Pattern::Dense) {
        std::fill(inject.bits.begin(), inject.bits.end(), 0);
        for (int s = 0; s < 4; ++s)
            inject.bits[s * K + s * H + 23] = f32_to_bf16(
                p == Pattern::Zero ? 0.F : float(s - 1) * (p == Pattern::Saturation ? 512.F : 2.F));
    }
    dd.copy_from_host(down.payload.data(), dd.bytes());
    du.copy_from_host(up.payload.data(), du.bytes());
    dj.copy_from_host(inject.bits.data(), dj.bytes());
    bound = {{down.device_weight(dd.span()), ops::LinearPolicy::AllowA8},
             {up.device_weight(du.span()), ops::LinearPolicy::AllowA8},
             inject.device_weight(dj.span())};
}

std::vector<float> input(Pattern p, int phase) {
    std::vector<float> x(std::size_t(K) * Catalog);
    fill_uniform(x, 881 + phase, -1.F, 1.F);
    if (p == Pattern::Saturation)
        for (auto& a : x) a = std::copysign(2.F, a);
    if (p == Pattern::Zero) std::fill(x.begin(), x.end(), 0.F);
    round_to_bf16(x);
    return x;
}

Fixture::Fixture(Weights& w, int n, bool f, ops::LinearPolicy p) : Fixture(w, n, f, p, p) {}

Fixture::Fixture(Weights& w, int n, bool f, ops::LinearPolicy p, ops::LinearPolicy up)
    : weights(w), m(n), final(f), policy(p), up_policy(up), dz(std::size_t(K) * m * 2),
      dx(std::size_t(H) * m * 2), dj(final ? 0 : std::size_t(4) * m * 4),
      scratch(capacity(w, m, final, p, up)), arena(scratch.span()),
      z(dz.span(), DType::BF16, {H, 4, m}), x(dx.span(), DType::BF16, {H, m}),
      j(final ? Tensor{} : Tensor(dj.span(), DType::FP32, {4, m})), words(std::size_t(K) * m) {}

void Fixture::upload(std::span<const float> catalog, cudaStream_t stream) {
    for (int m0 = 0; m0 < m; ++m0)
        for (int k = 0; k < K; ++k)
            words[std::size_t(m0) * K + k] = f32_to_bf16(catalog[(m0 % Catalog) * K + k]);
    cuda_check(
        cudaMemcpyAsync(dz.data(), words.data(), words.size() * 2, cudaMemcpyHostToDevice, stream),
        "HC input upload");
}

void Fixture::invoke(CudaExecutionView e) {
    auto b        = weights.bound;
    b.down.policy = policy;
    b.up.policy   = up_policy;
    if (final)
        ops::hc_mix(z, ops::HcFp8FinalMixWeights{b.down, b.up}, arena, x, e);
    else
        ops::hc_mix(z, b, arena, x, j, e);
}
} // namespace ninfer::test::hc::mix::fp8
