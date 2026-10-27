#include "support.h"
#include "cuda_fixture.h"

namespace ninfer::test::hc::mix::bf16 {
const char* name(Kind kind) { return kind == Kind::Block ? "block-mix" : "final-mix"; }

namespace {
direct_bf16_weight::HostWeight matrix(Kind kind, Pattern pattern, bool up) {
    const int rows    = up ? kTotal : kRank + (kind == Kind::Block ? kStreams : 0);
    const int columns = up ? kRank : kTotal;
    std::vector<float> values(std::size_t(rows) * columns);
    if (pattern == Pattern::Dense || (pattern == Pattern::InjectionOnly && up)) {
        const float scale = std::sqrt(3.F / columns) * (up ? 4.F : 1.F);
        fill_uniform(values, up ? 4117U : 4101U, -scale, scale);
    } else if (pattern == Pattern::RowMapping || pattern == Pattern::Saturation) {
        if (up) {
            for (int s = 0; s < kStreams; ++s)
                for (int d = 0; d < kWidth; ++d) {
                    const int r = (13 * d + 29 * s) % kRank;
                    values[std::size_t(s * kWidth + d) * kRank + r] =
                        pattern == Pattern::Saturation ? (d % 3 == 0 ? -64.F : 64.F)
                                                       : float((d + 3 * s) % 9 - 4) * 0.5F;
                }
        } else {
            for (int r = 0; r < kRank; ++r)
                values[std::size_t(r) * kTotal + (r % kStreams) * kWidth + (7 * r + 3) % kWidth] =
                    r % 2 ? -4.F : 4.F;
        }
    }
    if (!up && kind == Kind::Block && pattern != Pattern::Dense && pattern != Pattern::Zero) {
        constexpr float normal[]{-8.F, 0.F, 8.F, 16.F};
        constexpr float saturated[]{-512.F, 0.F, 32.F, 512.F};
        for (int s = 0; s < kStreams; ++s)
            values[std::size_t(kRank + s) * kTotal + s * kWidth + 23] =
                pattern == Pattern::Saturation ? saturated[s] : normal[s];
    }
    direct_bf16_weight::HostWeight result{rows, columns, {}};
    result.bits.resize(values.size());
    std::transform(values.begin(), values.end(), result.bits.begin(), f32_to_bf16);
    return result;
}

std::vector<float> decode(const std::vector<std::uint16_t>& words) {
    std::vector<float> values(words.size());
    std::transform(words.begin(), words.end(), values.begin(), bf16_to_f32);
    return values;
}

std::size_t capacity_for(const Weights& w, int first, int last) {
    if (w.kind == Kind::Block)
        return ops::hc_mix_workspace_capacity_bytes(ops::HcBf16MixWeights{w.down, w.up}, first, last);
    return ops::hc_mix_workspace_capacity_bytes(ops::HcBf16FinalMixWeights{w.down, w.up}, first, last);
}
} // namespace

Weights::Weights(Kind k, Pattern p)
    : kind(k), pattern(p), host_down(matrix(k, p, false)), host_up(matrix(k, p, true)),
      down_values(decode(host_down.bits)), up_values(decode(host_up.bits)),
      device_down(host_down.bits.size() * 2), device_up(host_up.bits.size() * 2),
      down(host_down.device_weight(device_down.span())),
      up(host_up.device_weight(device_up.span())) {
    device_down.copy_from_host(host_down.bits.data(), device_down.bytes());
    device_up.copy_from_host(host_up.bits.data(), device_up.bytes());
}

int Weights::check() const {
    int failed = device_down.verify_guards("Mix down") + device_up.verify_guards("Mix up");
    failed += verify_exact("Mix down preserved",
                           from_device<std::uint16_t>(device_down.data(), host_down.bits.size()),
                           host_down.bits);
    failed += verify_exact("Mix up preserved",
                           from_device<std::uint16_t>(device_up.data(), host_up.bits.size()),
                           host_up.bits);
    return failed;
}

Fixture::Fixture(Weights& w, int m)
    : weights(w), columns(m), capacity(capacity_for(w, m, m)), input_words(std::size_t(kTotal) * m),
      input(input_words.size() * 2), output(std::size_t(kWidth) * m * 2),
      injection(w.kind == Kind::Block ? std::size_t(kStreams) * m * 4 : 0), scratch(capacity),
      workspace(scratch.span()), z(input.span(), DType::BF16, {kWidth, kStreams, m}),
      x(output.span(), DType::BF16, {kWidth, m}),
      j(w.kind == Kind::Block ? Tensor(injection.span(), DType::FP32, {kStreams, m}) : Tensor{}) {}

void Fixture::upload(std::span<const float> catalog, cudaStream_t stream) {
    cuda_synchronize(stream);
    const int count = int(catalog.size() / kTotal);
    for (int m = 0; m < columns; ++m)
        for (int d = 0; d < kTotal; ++d)
            input_words[std::size_t(m) * kTotal + d] =
                f32_to_bf16(catalog[(m % count) * kTotal + d]);
    input.copy_from_host(input_words.data(), input.bytes());
    cuda_check(cudaStreamSynchronize(nullptr), "Mix upload completion");
    cuda_check(cudaMemsetAsync(output.data(), 0xff, output.bytes(), stream), "poison Mix output");
    if (injection.bytes())
        cuda_check(cudaMemsetAsync(injection.data(), 0xff, injection.bytes(), stream),
                   "poison injection");
}

void Fixture::invoke(CudaExecutionView e) {
    if (weights.kind == Kind::Block)
        ops::hc_mix(z, ops::HcBf16MixWeights{weights.down, weights.up}, workspace, x, j, e);
    else
        ops::hc_mix(z, ops::HcBf16FinalMixWeights{weights.down, weights.up}, workspace, x, e);
}

int Fixture::check(const std::string& label) const {
    int failed = input.verify_guards(label) + output.verify_guards(label) +
                 injection.verify_guards(label) + scratch.verify_guards(label);
    failed +=
        verify_exact((label + " input preserved").c_str(),
                     from_device<std::uint16_t>(input.data(), input_words.size()), input_words);
    if (workspace.used() != 0 || workspace.peak_used() != capacity) {
        std::cerr << label << ": workspace scope/capacity mismatch\n";
        ++failed;
    }
    return failed;
}

std::size_t Fixture::query(int first, int last) const { return capacity_for(weights, first, last); }

std::vector<float> input_catalog(Pattern pattern, int phase) {
    std::vector<float> values(kTotal * kCatalogColumns);
    fill_uniform(values, 4201U + phase, -1.F, 1.F);
    for (int m = 0; m < kCatalogColumns; ++m)
        for (int s = 0; s < kStreams; ++s) {
            const auto base = (m * kStreams + s) * kWidth;
            for (int d = 0; d < kWidth; ++d) values[base + d] *= float(s + 1) * 0.5F;
            if (pattern != Pattern::Dense && pattern != Pattern::Zero) values[base + 23] = 1.F;
        }
    round_to_bf16(values);
    return values;
}
} // namespace ninfer::test::hc::mix
