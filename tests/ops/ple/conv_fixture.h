#pragma once

#include "cuda_fixture.h"
#include "core/buffer.h"
#include "fixture.h"
#include "ninfer/ops/ple.h"

namespace ninfer::test::ple {

// BF16 RNE contributes at most ~2^-8 relative error. The absolute allowance
// covers FP32 arithmetic near cancellation at the declared fixture magnitudes.
inline constexpr PointwiseCriterion conv_criterion{2e-6, 4e-3};
inline constexpr ReductionCriterion norm_criterion{4e-3, 2e-6, 4e-3};

struct ConvFixture {
    int tokens, batch, capacity, slots;
    Words residual, gated, kernel, norm, history, records, updated, normalized;
    std::vector<int> valid, source, offset;
    DeviceBuffer d_valid;
    DeviceBuffer d_source;
    DeviceBuffer d_offset;
    Tensor y;
    bool inplace;

    ConvFixture(int t, int b, int layout = 0, bool in_place = false)
        : tokens(t), batch(b), capacity(t + 5), slots(b + 3), residual({H, C, t, b}),
          gated({H, C, t, b}), kernel({D, 4}, layout == 1), norm({H, C}, layout == 1),
          history({D, 9, slots}, layout == 1, history_strides(slots, layout)),
          records({D, capacity, b}, layout == 1, column_strides(capacity, b, layout)),
          updated({H, C, t, b}), normalized({H, C, t, b}), valid(b), source(b), offset(b),
          d_valid(ninfer::test::cuda_environment().context.device(), b * 4),
          d_source(ninfer::test::cuda_environment().context.device(), b * 8),
          d_offset(ninfer::test::cuda_environment().context.device(), b * 4),
          y(in_place ? residual.tensor : updated.tensor), inplace(in_place) {}

    ops::PleBatchView batch_view() { return {Tensor(d_valid.span(), DType::I32, {batch})}; }

    ops::PleConvStateView state_view() {
        return {history_field(history.tensor, history.device->span(),
                              Tensor(d_source.span(), DType::I64, {batch}))};
    }

    ops::PleConvRecordWindowView record_view() {
        return {records.tensor, Tensor(d_offset.span(), DType::I32, {batch})};
    }

    void prepare(int phase = 0, bool masked = true, float scale = 1.f) {
        residual.fill(101, scale);
        gated.fill(103, scale);
        kernel.fill(107, .25f);
        norm.fill(109, .5f);
        history.fill(113, scale);
        records.fill(127, scale);
        for (int s = 0; s < C; ++s) {
            norm.host[norm.index(0, s)] = f32_to_bf16(-1.f);
            norm.host[norm.index(1, s)] = f32_to_bf16(.00390625f * (2 * s + 1));
        }
        for (int b = 0; b < batch; ++b) {
            valid[b]  = !masked                ? tokens
                        : (b + phase) % 3 == 1 ? 0
                                               : std::max(1, tokens - (b + phase) % 5);
            source[b] = valid[b] ? (b + phase + 1) % slots : -91;
            offset[b] = (b * 2 + phase) % 5;
            for (int t = 0; t < capacity; ++t)
                if (t < offset[b] || t >= offset[b] + valid[b])
                    for (int c = 0; c < D; ++c) records.host[records.index(c, t, b)] = 0x7fc1;
            for (int t = valid[b]; t < tokens; ++t)
                for (int s = 0; s < C; ++s)
                    for (int d = 0; d < H; ++d) {
                        residual.host[residual.index(d, s, t, b)] = 0x7fc2;
                        gated.host[gated.index(d, s, t, b)]       = 0x7fc3;
                    }
        }
        // A short prefix history also exercises initialized leading zeros.
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < D; ++c) history.host[history.index(c, j, 1)] = 0;
        for (Words* w :
             {&residual, &gated, &kernel, &norm, &history, &records, &updated, &normalized})
            w->upload();
        upload_i32(d_valid, valid);
        upload_offsets(d_source, history.tensor, source);
        upload_i32(d_offset, offset);
        // Fixture uploads use the default stream; the Op uses a nonblocking compute stream.
        cuda_synchronize();
    }

    void launch(bool with_norm, CudaExecutionView execution) {
        if (with_norm)
            ops::ple_conv_add_norm(residual.tensor, gated.tensor, kernel.tensor, batch_view(),
                                   state_view(), record_view(), norm.tensor, epsilon, y,
                                   normalized.tensor, execution);
        else
            ops::ple_conv_add(residual.tensor, gated.tensor, kernel.tensor, batch_view(),
                              state_view(), record_view(), y, execution);
    }

    int verify(bool with_norm, const std::string& label) const {
        const auto words = from_device<std::uint16_t>(y.data, std::size_t(D) * tokens * batch);
        std::vector<double> got(words.size()), reference(words.size());
        int failures = 0;
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < tokens; ++t)
                for (int c = 0; c < D; ++c) {
                    const auto i = (std::size_t(b) * tokens + t) * D + c;
                    got[i]       = bf16_to_f32(words[i]);
                    if (t >= valid[b]) {
                        if (words[i] != 0) ++failures;
                        continue;
                    }
                    double sum = 0;
                    for (int j = 0; j < 4; ++j) {
                        const int pos  = t - 9 + j * 3;
                        const double n = pos < 0 ? value(history, c, 9 + pos, source[b])
                                                 : value(records, c, offset[b] + pos, b);
                        sum += value(kernel, c, j) * n;
                    }
                    reference[i] = value(residual, c % H, c / H, t, b) +
                                   value(gated, c % H, c / H, t, b) + sum / (1. + std::exp(-sum));
                }
        failures += verify_pointwise(label + " Y", got, reference, conv_criterion);
        if (with_norm) {
            for (int b = 0; b < batch; ++b)
                for (int t = 0; t < tokens; ++t)
                    for (int s = 0; s < C; ++s) {
                        const auto base = ((std::size_t(b) * tokens + t) * C + s) * H;
                        double sum      = 0;
                        for (int d = 0; d < H; ++d) sum += got[base + d] * got[base + d];
                        const double inv = 1. / std::sqrt(sum / H + epsilon);
                        for (int d = 0; d < H; ++d)
                            reference[base + d] =
                                t < valid[b] ? got[base + d] * inv * (1. + value(norm, d, s)) : 0.;
                    }
            const auto z = from_device<std::uint16_t>(normalized.tensor.data, words.size());
            for (std::size_t i = 0; i < z.size(); ++i) {
                got[i]      = bf16_to_f32(z[i]);
                const int b = i / (std::size_t(tokens) * D), t = i / D % tokens;
                if (t >= valid[b] && z[i] != 0) ++failures;
            }
            failures += verify_reduction(label + " norm(actual Y)", got, reference, norm_criterion);
        }
        for (const Words* w : {&gated, &kernel, &norm, &history, &records})
            failures += w->preserved(label.c_str());
        if (!inplace) failures += residual.preserved(label.c_str());
        for (const Words* w : {&residual, &updated, &normalized})
            failures += w->device->verify_guards(label);
        failures += verify_exact("valid", from_device<int>(d_valid, batch), valid);
        failures += verify_exact("source", from_device<std::int64_t>(d_source, batch),
                                 history_offsets(history.tensor, source));
        failures += verify_exact("offset", from_device<int>(d_offset, batch), offset);
        if (failures) std::cerr << label << ": failures=" << failures << '\n';
        return failures;
    }
};

} // namespace ninfer::test::ple
