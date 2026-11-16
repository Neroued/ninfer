#include "support.h"
#include "core/decode_graph.h"

namespace ninfer::test::mhc {
namespace {
int run_tap(int columns, int batch, bool alias, bool irregular, bool midpoint,
            bool five_slices = false, DType gamma_type = DType::BF16) {
    const int m = columns * batch;
    auto e      = cuda_environment().context.execution_view();
    Fixture f(Form::PostPre, m, alias, 256, DType::BF16, DType::FP32, gamma_type);
    const std::int64_t column_stride = 5 * D + (irregular ? 1 : 0),
                       batch_stride  = column_stride * columns + (irregular ? 17 : 0);
    const auto words                 = std::size_t(batch_stride) * batch + 32;
    GuardedDeviceBuffer tap_buffer(words * 2);
    std::vector<std::uint16_t> untouched(words, 0x7fc1);
    std::vector<DecodeGraphDefinition> definitions(five_slices ? 5 : 1);
    std::vector<DecodeGraphExecutable> graphs(five_slices ? 5 : 1);
    auto tap_view = [&](int slice) {
        Tensor t(tap_buffer.span().subspan(slice * D * 2), DType::BF16, {D, columns, batch});
        t.nb[1] = column_stride * 2;
        t.nb[2] = batch_stride * 2;
        t.nb[3] = batch_stride * batch * 2;
        return t;
    };
    int failed = 0;
    for (int phase = 0; phase < 2; ++phase) {
        f.set(Pattern::General, phase);
        if (midpoint) {
            std::fill(f.x.begin(), f.x.end(), 1.F);
            std::fill(f.y.begin(), f.y.end(), 1.F);
            std::fill(f.c.begin(), f.c.end(), 0.F);
            for (int j = 0; j < m; ++j)
                for (int s = 0; s < 4; ++s) {
                    f.c[j * 16 + s * 4 + s] = 1;
                    f.q[j * 4 + s]          = (s < 3 ? 5.F : 3.F) / 1024;
                    if (s == 3) std::fill_n(f.x.begin() + std::size_t(j) * K + 3 * D, D, .9921875F);
                }
        }
        f.mean_tap = {};
        f.upload(e.stream());
        f.invoke(e);
        cuda_synchronize(e.stream());
        auto baseline_r = from_device<std::uint16_t>(f.tr.data, f.x.size());
        failed += verify(f, "without tap independent math");
        f.arena.reset_peak();
        tap_buffer.copy_from_host(untouched.data(), words * 2);
        for (int slice = 0; slice < (five_slices ? 5 : 1); ++slice) {
            const int selected = five_slices ? slice : 2;
            f.mean_tap         = tap_view(selected);
            f.upload(e.stream());
            if (phase == 0) {
                cuda_synchronize();
                definitions[slice].capture(e.stream(), [&] { f.invoke(e); });
                graphs[slice].instantiate(definitions[slice]);
            }
            graphs[slice].launch(e.stream());
            cuda_synchronize(e.stream());
            failed += verify_exact("tap leaves BF16 post unchanged",
                                   from_device<std::uint16_t>(f.tr.data, f.x.size()), baseline_r);
            auto actual = from_device<std::uint16_t>(tap_buffer.data(), words);
            std::vector<double> means(std::size_t(D) * m), published(means.size());
            for (int b = 0; b < batch; ++b)
                for (int c = 0; c < columns; ++c)
                    for (int d = 0; d < D; ++d) {
                        const auto logical = std::size_t(b * columns + c) * D + d;
                        const auto at = b * batch_stride + c * column_stride + selected * D + d;
                        for (int s = 0; s < 4; ++s)
                            means[logical] +=
                                double(bf16_to_f32(
                                    baseline_r[std::size_t(b * columns + c) * K + s * D + d])) /
                                4;
                        published[logical] = bf16_to_f32(actual[at]);
                        untouched[at]      = actual[at];
                        if (midpoint && published[logical] != 1.0078125) ++failed;
                    }
            failed += verify_reduction("post-BF16 mean tap", published, means, kBf16);
            failed += verify_exact("tap untouched feature slices and gaps", actual, untouched) +
                      tap_buffer.verify_guards("mean tap");
        }
        failed += verify(f, "tap post/pre independent math") + f.preserved("tap inputs");
        if (f.arena.used() != 256 || f.arena.peak_used() > 256 + f.capacity) ++failed;
        std::fill(untouched.begin(), untouched.end(), 0x7fc1);
    }
    return failed;
}

int validate_tap() {
    auto e = cuda_environment().context.execution_view();
    Fixture f(Form::PostPre, 6, false);
    GuardedDeviceBuffer buffer(5 * D * 6 * 2);
    Tensor valid(buffer.span(), DType::BF16, {D, 3, 2});
    valid.nb[1]   = 5 * D * 2;
    valid.nb[2]   = valid.nb[1] * 3;
    valid.nb[3]   = valid.nb[2] * 2;
    int failed    = 0;
    auto rejected = [&](Tensor view) {
        f.mean_tap = view;
        try {
            f.invoke(e);
            ++failed;
        } catch (const std::invalid_argument&) {}
        if (f.arena.used() != 256) ++failed;
    };
    auto v  = valid;
    v.ne[1] = 2;
    rejected(v);
    v       = valid;
    v.nb[0] = 4;
    rejected(v);
    v       = valid;
    v.nb[1] = 16;
    rejected(v);
    v = valid;
    v.nb[2] += 1;
    rejected(v);
    v       = valid;
    v.dtype = DType::FP32;
    rejected(v);
    v      = valid;
    v.data = f.tr.data;
    rejected(v);
    v      = valid;
    v.data = f.tx.data;
    rejected(v);
    v      = valid;
    v.data = f.outputs.block_input.data;
    rejected(v);
    v      = valid;
    v.data = f.scratch.data();
    rejected(v);
    return failed;
}
} // namespace

void add_tap_cases(OpSuite& suite) {
    for (auto [c, b] : {std::pair{1, 1},
                        {7, 1},
                        {16, 8},
                        {11, 3},
                        {129, 1},
                        {447, 1},
                        {448, 1},
                        {512, 1},
                        {513, 1},
                        {2048, 1},
                        {4096, 1}})
        suite.add("post-pre-tap/c" + std::to_string(c) + "/b" + std::to_string(b),
                  [=] { return run_tap(c, b, false, false, false); }, {{"suite", "mhc_tap"}});
    suite.add("post-pre-tap/odd-strides-inplace", [] { return run_tap(3, 2, true, true, false); },
              {{"suite", "mhc_tap"}});
    suite.add("post-pre-tap/post-bf16-seam", [] { return run_tap(7, 1, false, false, true); },
              {{"suite", "mhc_tap"}});
    suite.add("post-pre-tap/five-slices", [] { return run_tap(7, 2, false, true, false, true); },
              {{"suite", "mhc_tap"}});
    suite.add("post-pre-tap/fp32-gamma",
              [] { return run_tap(513, 1, true, false, false, false, DType::FP32); },
              {{"suite", "mhc_tap"}});
    suite.add("post-pre-tap/contracts", validate_tap, {{"suite", "mhc_tap"}});
}
} // namespace ninfer::test::mhc
