#pragma once

#include "reference.h"
#include "ninfer/ops/video_delta_attention.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include <array>
#include <cstring>
#include <memory>
#include <random>

namespace ninfer::test::vda {
inline constexpr ReductionCriterion OutputCriterion{6e-3, 1e-5, 1.5e-2};
inline constexpr ReductionCriterion StateCriterion{4e-3, 1e-5, 1.2e-2};

struct TestCase {
    int frames, tokens_per_frame, heads;
    int batch = 1, chunk_frames = 5, radius = 1, frame_offset = 1;
    bool correlated = false, pitched = false;
};

inline vda_ref::Input make_input(const TestCase& c, unsigned seed) {
    vda_ref::Input a{c.frames,       c.tokens_per_frame, c.heads,       c.batch,
                     c.chunk_frames, c.radius,           c.frame_offset};
    const auto n = std::size_t(c.frames) * c.tokens_per_frame * c.heads * c.batch * 128;
    a.q.resize(n);
    a.k.resize(n);
    a.v.resize(n);
    a.gate.resize(n);
    a.alpha.resize(std::size_t(c.frames) * c.heads * c.batch * 128);
    a.beta.resize(n / 128);
    a.initial.resize(std::size_t(c.heads) * c.batch * 128 * 128);
    a.weight.resize(128);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> un(-1, 1);
    for (std::size_t row = 0; row < n / 128; ++row) {
        float qs = 0, ks = 0;
        for (int d = 0; d < 128; ++d) {
            a.q[row * 128 + d] = un(rng);
            a.k[row * 128 + d] =
                c.correlated ? std::sin(float(d) * .17f) + un(rng) * 1e-3f : un(rng);
            qs += a.q[row * 128 + d] * a.q[row * 128 + d];
            ks += a.k[row * 128 + d] * a.k[row * 128 + d];
            a.v[row * 128 + d]    = un(rng) * .4f;
            a.gate[row * 128 + d] = un(rng) * 3;
        }
        for (int d = 0; d < 128; ++d) {
            a.q[row * 128 + d] /= std::sqrt(qs);
            a.k[row * 128 + d] /= std::sqrt(ks);
        }
    }
    for (auto& x : a.alpha) x = .94f + .059f * (un(rng) * .5f + .5f);
    for (auto& x : a.beta) x = .1f + .8f * (un(rng) * .5f + .5f);
    for (auto& x : a.initial) x = un(rng) * .1f;
    for (auto& x : a.weight) x = 1 + un(rng) * .1f;
    round_to_bf16(a.q);
    round_to_bf16(a.k);
    round_to_bf16(a.v);
    round_to_bf16(a.gate);
    round_to_bf16(a.weight);
    return a;
}

struct VectorData {
    std::array<std::int64_t, 4> strides;
    std::unique_ptr<GuardedDeviceBuffer> data;
    Tensor tensor;
    std::vector<std::uint16_t> words;
    int offset;

    VectorData(const std::vector<float>& x, int h, int t, int b, bool pitched) {
        strides[0]       = 2;
        strides[1]       = 256 + (pitched ? 2 : 0);
        strides[2]       = h * strides[1] + (pitched ? 6 : 0);
        strides[3]       = std::max(1, t) * strides[2] + (pitched ? 10 : 0);
        offset           = pitched ? 2 : 0;
        const auto bytes = std::max<std::size_t>(256, offset + std::size_t(b) * strides[3]);
        data             = std::make_unique<GuardedDeviceBuffer>(bytes);
        words.assign(bytes / 2, 0x7fc0);
        for (int ib = 0; ib < b; ++ib)
            for (int it = 0; it < t; ++it)
                for (int ih = 0; ih < h; ++ih)
                    for (int d = 0; d < 128; ++d)
                        words[(offset + ib * strides[3] + it * strides[2] + ih * strides[1]) / 2 +
                              d] = f32_to_bf16(x[((std::size_t(ib) * t + it) * h + ih) * 128 + d]);
        data->copy_from_host(words.data(), bytes);
        tensor =
            Tensor(static_cast<std::byte*>(data->data()) + offset,
                   cuda_environment().context.device(), DType::BF16, {128, h, std::max(t, 1), b});
        tensor.ne[2] = t;
        std::copy(strides.begin(), strides.end(), tensor.nb);
    }

    int intact(const char* name) {
        return data->verify_guards(name) +
               verify_exact(name, from_device<std::uint16_t>(data->data(), words.size()), words);
    }

    void update(const std::vector<float>& values) {
        const int heads = tensor.ne[1], tokens = tensor.ne[2], batch = tensor.ne[3];
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < tokens; ++t)
                for (int h = 0; h < heads; ++h)
                    for (int d = 0; d < 128; ++d)
                        words[(offset + b * strides[3] + t * strides[2] + h * strides[1]) / 2 + d] =
                            f32_to_bf16(
                                values[((std::size_t(b) * tokens + t) * heads + h) * 128 + d]);
        data->copy_from_host(words.data(), words.size() * sizeof(std::uint16_t));
    }
};

struct Fixture {
    vda_ref::Input input;
    VectorData q, k, v, gate;
    DeviceBuffer alpha, beta, initial, weight;
    GuardedDeviceBuffer output, scratch;
    MemoryArena workspace;
    Tensor retention_tensor, beta_tensor, initial_tensor, weight_tensor, output_tensor;

    Fixture(const TestCase& c, unsigned seed = 17)
        : input(make_input(c, seed)),
          q(input.q, c.heads, c.frames * c.tokens_per_frame, c.batch, c.pitched),
          k(input.k, c.heads, c.frames * c.tokens_per_frame, c.batch, c.pitched),
          v(input.v, c.heads, c.frames * c.tokens_per_frame, c.batch, c.pitched),
          gate(input.gate, c.heads, c.frames * c.tokens_per_frame, c.batch, c.pitched),
          alpha(to_device_f32(input.alpha)), beta(to_device_f32(input.beta)),
          initial(to_device_f32(input.initial)), weight(to_device_bf16(input.weight)),
          output(std::max<std::size_t>(256, input.q.size() * 2)),
          scratch(std::max<std::size_t>(256, ops::video_delta_attention_workspace_capacity_bytes(
                                                 c.heads, c.tokens_per_frame, 0, c.frames, c.batch,
                                                 {c.chunk_frames, c.radius, c.frame_offset}))),
          workspace(scratch.span()),
          initial_tensor(initial.span(), DType::FP32, {128, 128, c.heads, c.batch}),
          weight_tensor(weight.span(), DType::BF16, {128}) {
        if (!c.frames) {
            alpha = DeviceBuffer(cuda_environment().context.device(), 256);
            beta  = DeviceBuffer(cuda_environment().context.device(), 256);
        }
        retention_tensor       = Tensor(alpha.data(), alpha.device(), DType::FP32,
                                        {128, c.heads, std::max(c.frames, 1), c.batch});
        retention_tensor.ne[2] = c.frames;
        beta_tensor            = Tensor(beta.data(), beta.device(), DType::FP32,
                                        {c.heads, std::max(c.frames * c.tokens_per_frame, 1), c.batch});
        beta_tensor.ne[1]      = c.frames * c.tokens_per_frame;
        output_tensor = Tensor(output.data(), cuda_environment().context.device(), DType::BF16,
                               {128, c.heads, std::max(c.frames * c.tokens_per_frame, 1), c.batch});
        output_tensor.ne[2] = c.frames * c.tokens_per_frame;
        output.fill(0xff);
        scratch.fill(0xff);
    }

    void call(bool gated, CudaExecutionView ex) {
        ops::VideoDeltaAttentionWindow win{input.chunk, input.radius, input.offset};
        if (gated)
            ops::video_delta_attention_sigmoid_gated(q.tensor, k.tensor, v.tensor, retention_tensor,
                                                     beta_tensor, initial_tensor, weight_tensor,
                                                     gate.tensor, 1e-6f, input.per_frame, win,
                                                     workspace, output_tensor, ex);
        else
            ops::video_delta_attention(q.tensor, k.tensor, v.tensor, retention_tensor, beta_tensor,
                                       initial_tensor, input.per_frame, win, workspace,
                                       output_tensor, ex);
    }

    int verify(bool gated) {
        cuda_synchronize();
        auto ref = vda_ref::evaluate(input, gated);
        auto got = from_device_bf16(output_tensor.data, input.q.size());
        int fail = 0;
        for (int b = 0; b < input.batch; ++b)
            for (int h = 0; h < input.heads; ++h)
                for (int f = 0; f < input.frames; ++f) {
                    std::vector<double> x, r;
                    for (int u = 0; u < input.per_frame; ++u)
                        for (int d = 0; d < 128; ++d) {
                            auto i = input.vi(b, f * input.per_frame + u, h, d);
                            x.push_back(got[i]);
                            r.push_back(ref[i]);
                        }
                    const auto name = "VDA B" + std::to_string(b) + "/H" + std::to_string(h) +
                                      "/F" + std::to_string(f);
                    fail += verify_reduction(name.c_str(), x, r, OutputCriterion);
                }
        fail += q.intact("q");
        fail += k.intact("k");
        fail += v.intact("v");
        fail += gate.intact("gate");
        fail += verify_exact("initial immutable",
                             from_device<float>(initial_tensor.data, input.initial.size()),
                             input.initial);
        fail += verify_exact("alpha immutable",
                             from_device<float>(retention_tensor.data, input.alpha.size()),
                             input.alpha);
        fail += verify_exact("beta immutable",
                             from_device<float>(beta_tensor.data, input.beta.size()), input.beta);
        fail += verify_exact("norm weight immutable",
                             from_device_bf16(weight_tensor.data, input.weight.size()),
                             std::vector<double>(input.weight.begin(), input.weight.end()));
        fail += workspace.used() != 0;
        if (!input.frames) {
            fail += verify_exact("empty output untouched",
                                 from_device<std::uint8_t>(output.data(), output.span().bytes),
                                 std::vector<std::uint8_t>(output.span().bytes, 0xff));
        }
        return fail + output.verify_guards("output") + scratch.verify_guards("workspace");
    }
};

} // namespace ninfer::test::vda
