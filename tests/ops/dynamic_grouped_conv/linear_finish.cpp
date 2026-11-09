#include "ops/suite.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "core/platform.h"
#include "core/decode_graph.h"
#include "ninfer/ops/dynamic_grouped_conv.h"

#include <array>
#include <algorithm>
#include <cmath>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int H = 4096, G = H / 16, MaxM = 128;
constexpr ReductionCriterion local_criterion{1. / 256, 1. / 256, 2. / 256};
constexpr ReductionCriterion sum_criterion{1. / 256, 1., 2. / 256};

std::vector<std::uint16_t> input(int k, int seed) {
    std::vector<std::uint16_t> x(std::size_t(k) * MaxM);
    for (int c = 0; c < MaxM; ++c)
        for (int h = 0; h < k; ++h) {
            std::uint32_t z =
                std::uint32_t(h) * 0x9e3779b9U + std::uint32_t(c + seed) * 0x85ebca6bU;
            z ^= z >> 16;
            z *= 0x7feb352dU;
            z ^= z >> 15;
            x[std::size_t(c) * k + h] = f32_to_bf16((int(z % 257) - 128) / 256.f);
        }
    return x;
}

std::vector<double> project(const direct_bf16_weight::HostWeight& w,
                            const std::vector<std::uint16_t>& x) {
    std::vector<double> z(std::size_t(H) * MaxM);
    parallel_rows(H, std::uint64_t(H) * w.k * MaxM, [&](int a, int b) {
        for (int h = a; h < b; ++h)
            for (int c = 0; c < MaxM; ++c) {
                double sum = 0;
                for (int k = 0; k < w.k; ++k)
                    sum += double(bf16_to_f32(w.bits[std::size_t(h) * w.k + k])) *
                           double(bf16_to_f32(x[std::size_t(c) * w.k + k]));
                z[std::size_t(c) * H + h] = sum;
            }
    });
    return z;
}

std::vector<double> finish(const std::vector<double>& z, const std::vector<std::uint16_t>& base,
                           const std::vector<std::uint16_t>& delta, int width, int batch,
                           int sign) {
    std::vector<double> y(std::size_t(H) * width * batch);
    for (int c = 0; c < width * batch; ++c)
        for (int h = 0; h < H; ++h) {
            auto di      = c * 2 * G + h / 16;
            auto j       = std::size_t(c) * H + h;
            double value = (double(bf16_to_f32(base[2 * H + h])) + bf16_to_f32(delta[di])) * z[j];
            if (c % width)
                value +=
                    (double(bf16_to_f32(base[3 * H + h])) + bf16_to_f32(delta[di + G])) * z[j - H];
            y[j] = sign * value;
        }
    return y;
}

int run(int k, std::span<const std::pair<int, int>> cases) {
    const auto x0 = input(k, 101), x1 = input(k, 307);
    auto neg = x0;
    for (auto& v : neg) v ^= 0x8000;
    std::vector<std::uint16_t> base(4 * H);
    for (int h = 0; h < H; ++h) {
        base[h]         = f32_to_bf16(3.f);
        base[H + h]     = f32_to_bf16(-4.f);
        base[2 * H + h] = f32_to_bf16(.375f + (h % 13) / 4096.f);
        base[3 * H + h] = f32_to_bf16(-.15625f + (h % 17) / 8192.f);
    }
    auto delta = input(G * 2, 701);
    for (auto& v : delta) v = f32_to_bf16(bf16_to_f32(v) / 16.f);
    direct_bf16_weight::DeviceWeight w0(direct_bf16_weight::make_patterned(H, k, 809)),
        w1(direct_bf16_weight::make_patterned(H, k, 811));
    const auto z0 = project(w0.host, x0), z1 = project(w1.host, x1);
    auto dx0 = to_device(x0), dx1 = to_device(x1), db = to_device(base), dd = to_device(delta);
    Tensor b(db.span(), DType::BF16, {H, 2, 2});
    const auto pw0 = w0.view(), pw1 = w1.view();
    auto cuda    = cuda_environment().context.execution_view();
    int failures = 0;
    for (auto [width, batch] : cases) {
        const auto elements = std::size_t(H) * width * batch;
        GuardedDeviceBuffer out0(elements * 2), out1(elements * 2);
        Tensor a0(dx0.span(), DType::BF16, {k, width, batch}),
            a1(dx1.span(), DType::BF16, {k, width, batch});
        Tensor d(dd.span(), DType::BF16, {G, 2, width, batch});
        Tensor p0(out0.span(), DType::BF16, {H, width, batch}),
            p1(out1.span(), DType::BF16, {H, width, batch});
        const auto launch = [&] {
            ops::linear_dynamic_grouped_conv(a0, pw0, b, d, p0, cuda);
            ops::linear_dynamic_grouped_conv(a1, pw1, b, d, p1, cuda);
        };
        const auto verify = [&](int sign) {
            cuda_synchronize();
            auto e0 = finish(z0, base, delta, width, batch, sign),
                 e1 = finish(z1, base, delta, width, batch, 1);
            auto v0 = from_device_bf16(out0.data(), elements),
                 v1 = from_device_bf16(out1.data(), elements);
            failures += verify_reduction("local rank0", v0, e0, local_criterion);
            failures += verify_reduction("local rank1", v1, e1, local_criterion);
            std::vector<double> actual(elements), expected(elements);
            for (std::size_t i = 0; i < elements; ++i) {
                const float residual = bf16_to_f32(f32_to_bf16((int(i % 127) - 63) / 512.f));
                const auto sum       = bf16_to_f32(f32_to_bf16(float(v0[i] + v1[i])));
                actual[i]            = bf16_to_f32(f32_to_bf16(sum + residual));
                expected[i]          = e0[i] + e1[i] + residual;
                // The two published partials, collective sum and residual add each have BF16
                // boundaries. Bound gross cancellation error by their represented magnitudes.
                const auto bound =
                    (std::abs(e0[i]) + std::abs(e1[i]) + std::abs(residual)) / 64 + 1. / 256;
                if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > bound)
                    ++failures;
            }
            failures +=
                verify_reduction("two shards plus one residual", actual, expected, sum_criterion);
            failures += out0.verify_guards("partial0");
            failures += out1.verify_guards("partial1");
        };
        out0.fill(0xff);
        out1.fill(0xff);
        cuda_synchronize();
        launch();
        verify(1);
        if ((width == 3 && batch == 3) || (width == 7 && batch == 7) ||
            (width == 16 && batch == 8)) {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(cuda.stream(), launch);
            executable.instantiate(definition);
            dx0.copy_from_host(neg.data(), dx0.size());
            executable.launch(cuda.stream());
            executable.launch(cuda.stream());
            verify(-1);
            failures += verify_exact("changed activation preserved",
                                     from_device<std::uint16_t>(dx0, neg.size()), neg);
            dx0.copy_from_host(x0.data(), dx0.size());
        }
    }
    failures += verify_exact("input0", from_device<std::uint16_t>(dx0, x0.size()), x0);
    failures += verify_exact("input1", from_device<std::uint16_t>(dx1, x1.size()), x1);
    failures += verify_exact("base", from_device<std::uint16_t>(db, base.size()), base);
    failures += verify_exact("delta", from_device<std::uint16_t>(dd, delta.size()), delta);
    failures += w0.verify_preserved("weight0");
    failures += w1.verify_preserved("weight1");
    return failures;
}
} // namespace

void add_linear_conv_finish_cases(OpSuite& suite) {
#if NINFER_TARGET_SPARK
    suite.add({"conv_finish/operands", {{"format", "bf16"}}}, [] {
        direct_bf16_weight::DeviceWeight weight(direct_bf16_weight::make_patterned(H, 2048, 821));
        DeviceBuffer input_storage(cuda_environment().context.device(), H * 2 * 2);
        DeviceBuffer output_storage(cuda_environment().context.device(), H * 2 * 2);
        DeviceBuffer base_storage(cuda_environment().context.device(), H * 4 * 2);
        DeviceBuffer delta_storage(cuda_environment().context.device(), 320 * 2 * 2 * 2);
        Tensor x(input_storage.span(), DType::BF16, {2048, 2});
        Tensor out(output_storage.span(), DType::BF16, {H, 2});
        Tensor alias(input_storage.span(), DType::BF16, {H, 2});
        Tensor base(base_storage.span(), DType::BF16, {H, 2, 2});
        Tensor delta(delta_storage.span(), DType::BF16, {G, 2, 2});
        Tensor wrong_delta(delta_storage.span(), DType::BF16, {320, 2, 2});
        int failures              = 0;
        const auto expect_invalid = [&](auto&& fn) {
            try {
                fn();
                ++failures;
            } catch (const std::invalid_argument&) {}
        };
        const auto cuda = cuda_environment().context.execution_view();
        const auto w    = weight.view();
        expect_invalid([&] { ops::linear_dynamic_grouped_conv(x, w, base, delta, alias, cuda); });
        expect_invalid(
            [&] { ops::linear_dynamic_grouped_conv(x, w, base, wrong_delta, out, cuda); });
        auto wrong_weight  = w;
        wrong_weight.qtype = QType::MXFP8;
        expect_invalid(
            [&] { ops::linear_dynamic_grouped_conv(x, wrong_weight, base, delta, out, cuda); });
        return failures;
    });
    for (int k : {2048, 6144}) {
        const std::vector<std::pair<int, int>> shapes{
            {2, 1}, {2, 8},  {3, 3},  {4, 1},  {4, 8},  {7, 7},  {8, 1},  {8, 2},  {8, 4}, {8, 8},
            {9, 8}, {11, 6}, {12, 8}, {13, 5}, {14, 7}, {15, 8}, {16, 1}, {16, 4}, {16, 8}};
        std::vector<OpCase> cases;
        for (auto [w, b] : shapes)
            cases.push_back({"conv_finish/k" + std::to_string(k) + "/w" + std::to_string(w) + "/b" +
                                 std::to_string(b),
                             {{"format", "bf16"}}});
        suite.add(std::move(cases), [k, shapes](auto selected) {
            std::vector<std::pair<int, int>> cases;
            for (auto i : selected) cases.push_back(shapes[i]);
            return run(k, cases);
        });
    }
#endif
}
