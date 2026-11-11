#include "support.h"
#include "cuda_fixture.h"

namespace ninfer::test::hc {
const char* name(Form form) {
    switch (form) {
    case Form::ExpandNorm:
        return "expand-norm";
    case Form::GroupedNorm:
        return "grouped-rmsnorm";
    case Form::Combine:
        return "combine";
    case Form::CombineNorm:
        return "combine-norm";
    case Form::AddNorm:
        return "add-norm";
    }
    throw std::logic_error("invalid HC form");
}

Fixture::Fixture(Form f, int m, bool alias, int w)
    : form(f), columns(m), in_place(alias), width(w), h(std::size_t(width) * kStreams * m),
      block(std::size_t(width) * m), weight(width * kStreams), injection(kStreams * m),
      dh(h.size() * 2), db(block.size() * 2), dw(weight.size() * 2), dj(injection.size() * 4),
      dr(h.size() * 2), dz(h.size() * 2), th(dh.span(), DType::BF16, {width, kStreams, m}),
      tb(db.span(), DType::BF16, {width, m}), tw(dw.span(), DType::BF16, {width, kStreams}),
      tj(dj.span(), DType::FP32, {kStreams, m}),
      tr(alias ? dh.span() : dr.span(), DType::BF16, {width, kStreams, m}),
      tz(dz.span(), DType::BF16, {width, kStreams, m}) {}

void Fixture::set(Pattern pattern, int phase) {
    epsilon = kEpsilon;
    fill_uniform(h, 3101U + phase, -1.0F, 1.0F);
    fill_uniform(block, 3201U + phase, -0.75F, 0.75F);
    fill_uniform(weight, 3301U + phase, -0.25F, 0.25F);
    fill_uniform(injection, 3401U + phase, 0.0F, 2.0F);
    for (int m = 0; m < columns; ++m)
        for (int s = 0; s < kStreams; ++s) {
            const float scale = std::ldexp(1.0F, s - 1);
            const auto base   = (std::size_t(m) * kStreams + s) * width;
            for (int d = 0; d < width; ++d) h[base + d] *= scale;
        }
    if (pattern == Pattern::Affine) {
        for (int s = 0; s < kStreams; ++s)
            for (int d = 0; d < width; ++d)
                weight[s * width + d] = d % 3 == 0 ? -1.0F : (d % 3 == 1 ? -1.5F : 0.0F);
        for (std::size_t i = 0; i < injection.size(); ++i) injection[i] = float((i + phase) % 3);
    } else if (pattern == Pattern::Zero) {
        std::fill(h.begin(), h.end(), 0.0F);
        std::fill(block.begin(), block.end(), 0.0F);
        for (std::size_t i = 0; i < block.size(); i += 2) block[i] = -0.0F;
    } else if (pattern == Pattern::Constant) {
        std::fill(h.begin(), h.end(), 1.0F);
        std::fill(block.begin(), block.end(), 1.0F);
        std::fill(weight.begin(), weight.end(), 1.0F / 256.0F);
        std::fill(injection.begin(), injection.end(), 0.0F);
    } else if (pattern == Pattern::Cancellation) {
        for (int m = 0; m < columns; ++m)
            for (int s = 0; s < kStreams; ++s)
                for (int d = 0; d < width; ++d) {
                    const float value                              = float((d % 17) - 8) / 8.0F;
                    block[std::size_t(m) * width + d]              = value;
                    h[(std::size_t(m) * kStreams + s) * width + d] = -value;
                    injection[m * kStreams + s] = s == 0 ? 1.0F : 1.0F + float(s) / 16.0F;
                }
    } else if (pattern == Pattern::Midpoint) {
        std::fill(h.begin(), h.end(), 1.0F);
        for (std::size_t i = 0; i < block.size(); ++i)
            block[i] = (i % 4 == 0 ? 1.0F : (i % 4 == 1 ? -1.0F : 0.5F)) /
                       (form == Form::AddNorm ? 256.0F : 1.0F);
        for (std::size_t i = 0; i < injection.size(); ++i)
            injection[i] = (i + phase) % 3 == 2
                               ? 1.0F / 256.0F
                               : std::nextafter(1.0F / 256.0F, (i + phase) % 3 ? 0.0F : 1.0F);
    } else if (pattern == Pattern::Gain) {
        std::fill(h.begin(), h.end(), 1.0F);
        std::fill(block.begin(), block.end(), form == Form::ExpandNorm ? 1.0F : 0.0F);
        std::fill(weight.begin(), weight.end(), 3.0F / 256.0F);
        std::fill(injection.begin(), injection.end(), 0.0F);
        epsilon = 1.0e-3F;
    } else if (pattern == Pattern::Small) {
        for (auto& value : h) value *= 1.0e-4F;
        for (auto& value : block) value *= 1.0e-4F;
    }
    round_to_bf16(h);
    round_to_bf16(block);
    round_to_bf16(weight);
}

namespace {
void upload_bf16(GuardedDeviceBuffer& buffer, const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(), f32_to_bf16);
    buffer.copy_from_host(bits.data(), bits.size() * 2);
}

int unchanged(const std::string& label, const GuardedDeviceBuffer& buffer,
              const std::vector<float>& values, bool bf16) {
    if (bf16) {
        std::vector<std::uint16_t> expected(values.size());
        std::transform(values.begin(), values.end(), expected.begin(), f32_to_bf16);
        return verify_exact(label.c_str(), from_device<std::uint16_t>(buffer.data(), values.size()),
                            expected);
    }
    std::vector<std::uint32_t> expected(values.size());
    std::memcpy(expected.data(), values.data(), values.size() * sizeof(float));
    return verify_exact(label.c_str(), from_device<std::uint32_t>(buffer.data(), values.size()),
                        expected);
}
} // namespace

void Fixture::upload(cudaStream_t stream) {
    cuda_synchronize(stream);
    upload_bf16(dh, h);
    upload_bf16(db, block);
    upload_bf16(dw, weight);
    dj.copy_from_host(injection.data(), injection.size() * 4);
    cuda_check(cudaStreamSynchronize(nullptr), "fixture upload completion");
    cuda_check(cudaMemsetAsync(dr.data(), 0xff, dr.bytes(), stream), "poison residual");
    cuda_check(cudaMemsetAsync(dz.data(), 0xff, dz.bytes(), stream), "poison normalized");
}

void Fixture::invoke(CudaExecutionView e) {
    switch (form) {
    case Form::ExpandNorm:
        ops::hc_expand_norm(tb, tw, epsilon, tr, tz, e);
        break;
    case Form::GroupedNorm:
        ops::hc_grouped_rmsnorm(th, tw, epsilon, tz, e);
        break;
    case Form::Combine:
        ops::hc_combine(th, tb, tj, tr, e);
        break;
    case Form::CombineNorm:
        ops::hc_combine_norm(th, tb, tj, tw, epsilon, tr, tz, e);
        break;
    case Form::AddNorm:
        ops::hc_combine_norm(th, tb, tw, epsilon, tr, tz, e);
        break;
    }
}

int Fixture::check(const std::string& label) const {
    int failed = dh.verify_guards(label) + db.verify_guards(label) + dw.verify_guards(label) +
                 dj.verify_guards(label) + dr.verify_guards(label) + dz.verify_guards(label);
    if (!in_place) failed += unchanged(label + " residual input", dh, h, true);
    failed += unchanged(label + " embedding/block", db, block, true);
    failed += unchanged(label + " norm weight", dw, weight, true);
    failed += unchanged(label + " injection", dj, injection, false);
    return failed;
}
} // namespace ninfer::test::hc
