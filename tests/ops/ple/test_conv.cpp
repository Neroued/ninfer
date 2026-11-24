#include "ops/suite.h"
#include "conv_fixture.h"
#include "core/decode_graph.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::ple;

namespace {

int run(DeviceContext& device, int t, int b, int layout, bool inplace, bool norm,
        bool graph = false, float scale = 1.f) {
    ConvFixture f(t, b, layout, inplace);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        f.prepare(phase, true, scale);
        if (graph && !phase) {
            definition.capture(device.stream(), [&] { f.launch(norm, device.execution_view()); });
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(device.stream());
        else
            f.launch(norm, device.execution_view());
        device.synchronize();
        const auto label = std::string(norm ? "ple_conv_add_norm" : "ple_conv_add") +
                           " T=" + std::to_string(t) + " B=" + std::to_string(b) +
                           " layout=" + std::to_string(layout) +
                           " inplace=" + std::to_string(inplace) +
                           " phase=" + std::to_string(phase) + " scale=" + std::to_string(scale);
        failures += f.verify(norm, label);
    }
    return failures;
}

int validation(DeviceContext& device) {
    ConvFixture f(3, 2);
    f.prepare();
    int failures   = 0;
    Tensor partial = f.y;
    partial.data   = static_cast<char*>(f.residual.tensor.data) + 16;
    failures += rejects("partial output alias", [&] {
        ops::ple_conv_add(f.residual.tensor, f.gated.tensor, f.kernel.tensor, f.batch_view(),
                          f.state_view(), f.record_view(), partial, device.execution_view());
    });
    auto record         = f.record_view();
    record.values.nb[1] = 2;
    failures += rejects("overlapping record columns", [&] {
        ops::ple_conv_add(f.residual.tensor, f.gated.tensor, f.kernel.tensor, f.batch_view(),
                          f.state_view(), record, f.y, device.execution_view());
    });
    failures += rejects("invalid epsilon", [&] {
        ops::ple_conv_add_norm(f.residual.tensor, f.gated.tensor, f.kernel.tensor, f.batch_view(),
                               f.state_view(), f.record_view(), f.norm.tensor, 0, f.y,
                               f.normalized.tensor, device.execution_view());
    });
    failures += rejects("overlapping outputs", [&] {
        ops::ple_conv_add_norm(f.residual.tensor, f.gated.tensor, f.kernel.tensor, f.batch_view(),
                               f.state_view(), f.record_view(), f.norm.tensor, epsilon, f.y, f.y,
                               device.execution_view());
    });
    return failures;
}

int split_window(DeviceContext& device, bool norm) {
    ConvFixture f(17, 1);
    f.prepare(0, false);
    Words intermediate({D, 9, 1});
    intermediate.upload();
    auto first_count   = to_device(std::vector<int>{5});
    auto second_count  = to_device(std::vector<int>{12});
    auto second_offset = to_device(std::vector<int>{5});
    auto zero          = to_device(std::vector<std::int64_t>{0});
    const Tensor first(first_count.span(), DType::I32, {1}),
        second(second_count.span(), DType::I32, {1});
    const ops::PleConvStateView middle{history_field(
        intermediate.tensor, intermediate.device->span(), Tensor(zero.span(), DType::I64, {1}))};
    cuda_synchronize();
    const auto segment = [&](int start, int count, const Tensor& valid,
                             const ops::PleConvStateView& before, const Tensor& offsets) {
        const auto h = f.residual.tensor.slice(2, start, count);
        const auto u = f.gated.tensor.slice(2, start, count);
        auto y = f.y.slice(2, start, count), z = f.normalized.tensor.slice(2, start, count);
        if (norm)
            ops::ple_conv_add_norm(h, u, f.kernel.tensor, {valid}, before,
                                   {f.records.tensor, offsets}, f.norm.tensor, epsilon, y, z,
                                   device.execution_view());
        else
            ops::ple_conv_add(h, u, f.kernel.tensor, {valid}, before, {f.records.tensor, offsets},
                              y, device.execution_view());
    };
    segment(0, 5, first, f.state_view(), f.record_view().offsets);
    ops::ple_state_extract(f.state_view(), {f.records.tensor, f.batch_view().valid_columns}, first,
                           middle, {}, device.execution_view());
    segment(5, 12, second, middle, Tensor(second_offset.span(), DType::I32, {1}));
    device.synchronize();
    int failures = f.verify(norm, "PLE split windows at t=5");
    // The outer transaction can still choose a prefix preceding the second call.
    for (int a : {0, 2, 5, 9, 17}) {
        upload_i32(first_count, {a});
        cuda_synchronize();
        ops::ple_state_extract(f.state_view(), {f.records.tensor, f.batch_view().valid_columns},
                               first, middle, {}, device.execution_view());
        device.synchronize();
        auto expected = intermediate.host;
        for (int c = 0; c < D; ++c) {
            std::array<std::uint16_t, 26> sequence;
            for (int j = 0; j < 9; ++j)
                sequence[j] = f.history.host[f.history.index(c, j, f.source[0])];
            for (int j = 0; j < 17; ++j) sequence[9 + j] = f.records.host[f.records.index(c, j, 0)];
            for (int j = 0; j < 9; ++j) expected[intermediate.index(c, j, 0)] = sequence[a + j];
        }
        failures += verify_exact("PLE outer accepted prefix", intermediate.download(), expected);
    }
    return failures;
}

int output_in_history_gap(DeviceContext& device) {
    ConvFixture f(1, 1);
    f.prepare(0, false);
    Words storage({D, 9, 2});
    for (int j = 0; j < 9; ++j)
        for (int c = 0; c < D; ++c)
            storage.host[storage.index(c, j)] = f.history.host[f.history.index(c, j, f.source[0])];
    storage.upload();
    auto slot = to_device(std::vector<std::int64_t>{0});
    f.y = Tensor(storage.device->span().subspan(std::size_t(D) * 9 * 2), DType::BF16, {H, C, 1, 1});
    cuda_synchronize();
    ops::ple_conv_add_norm(f.residual.tensor, f.gated.tensor, f.kernel.tensor, f.batch_view(),
                           {history_field(storage.tensor, storage.device->span(),
                                          Tensor(slot.span(), DType::I64, {1}))},
                           f.record_view(), f.norm.tensor, epsilon, f.y, f.normalized.tensor,
                           device.execution_view());
    device.synchronize();
    int failures  = f.verify(true, "PLE output in unused history gap");
    auto expected = storage.host;
    const auto y  = from_device<std::uint16_t>(f.y.data, D);
    std::copy(y.begin(), y.end(), expected.begin() + 9 * D);
    failures += verify_exact("PLE other record regions preserved", storage.download(), expected);
    return failures + storage.device->verify_guards("PLE shared allocation");
}

int short_record_view(DeviceContext& device, bool norm) {
    ConvFixture f(512, 2);
    f.prepare();
    f.valid = {3, 0};
    upload_i32(f.d_valid, f.valid);
    // Physical T can exceed R: only the live prefix has to fit the record window.
    f.records.tensor.ne[1] = 3;
    cuda_synchronize();
    f.launch(norm, device.execution_view());
    device.synchronize();
    return f.verify(norm, "PLE padded T with R=3");
}

int ragged_validation() {
    const RaggedFixture ragged({2, 3});
    const int n = ragged.tokens();
    ConvFixture f(3, ragged.segments());
    f.prepare(0, false);
    Words residual({H, C, n}), gated({H, C, n}), records({D, n}), updated({H, C, n}),
        normalized({H, C, n});
    for (const auto* w : {&residual, &gated, &records, &updated, &normalized}) w->upload();
    cuda_synchronize();
    const auto ex     = cuda_environment().context.execution_view();
    const auto launch = [&](const Tensor& kernel, const ops::PleConvStateView& before, float eps,
                            Tensor y, Tensor z) {
        ops::ple_conv_add_norm(residual.tensor, gated.tensor, kernel, ragged.view, before,
                               records.tensor, f.norm.tensor, eps, y, z, ex);
    };
    const auto before = f.state_view();
    int failures      = rejects("ragged invalid epsilon", [&] {
        launch(f.kernel.tensor, before, 0.f, updated.tensor, normalized.tensor);
    });
    auto bad_history  = before;
    bad_history.history.field.tensor.region.offset += 1;
    failures += rejects("ragged history base alignment", [&] {
        launch(f.kernel.tensor, bad_history, epsilon, updated.tensor, normalized.tensor);
    });
    auto bad_kernel  = f.kernel.tensor;
    bad_kernel.ne[1] = 3;
    failures += rejects("ragged kernel shape", [&] {
        launch(bad_kernel, before, epsilon, updated.tensor, normalized.tensor);
    });
    auto overlapping_norm = normalized.tensor;
    overlapping_norm.data = records.tensor.data;
    failures += rejects("ragged output/record overlap", [&] {
        launch(f.kernel.tensor, before, epsilon, updated.tensor, overlapping_norm);
    });
    cuda_environment().context.synchronize();
    return failures + updated.preserved("rejected ragged convolution writes no output");
}

enum class RaggedConvLayout { Dense, ScalarPadded, WidePadded, PairedPadded };

int ragged_conv(std::vector<int> lengths, bool norm, RaggedConvLayout layout, bool inplace,
                bool graph) {
    RaggedFixture ragged(std::move(lengths));
    const int n = ragged.tokens(), rows = ragged.segments();
    const int width           = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    const bool scalar         = layout == RaggedConvLayout::ScalarPadded;
    const bool wide           = layout == RaggedConvLayout::WidePadded;
    const bool paired_padding = layout == RaggedConvLayout::PairedPadded;
    auto strides              = column_strides(n, 1, scalar ? 1 : wide ? 2 : 0);
    if (paired_padding) {
        constexpr std::int64_t stride = D + 2;
        strides                       = {1, stride, stride * n, stride * n};
    }
    ConvFixture f(width, rows, scalar ? 1 : 0);
    Words residual({H, C, n}), gated({H, C, n}), current({D, n}, scalar, strides),
        updated({H, C, n}), normalized({H, C, n});
    // The packed column stride is wide-aligned, while the unused batch stride is not.
    if (wide) current.tensor.nb[2] = n * current.tensor.nb[1] + 2;
    auto y      = inplace ? residual.tensor : updated.tensor;
    auto before = f.state_view();
    if (wide || paired_padding) before.history.field.tensor.region.alignment = 8;
    if (scalar) before.history.field.tensor.region.alignment = 1;
    auto& device      = cuda_environment().context;
    const auto ex     = device.execution_view();
    const auto launch = [&] {
        if (norm)
            ops::ple_conv_add_norm(residual.tensor, gated.tensor, f.kernel.tensor, ragged.view,
                                   before, current.tensor, f.norm.tensor, epsilon, y,
                                   normalized.tensor, ex);
        else
            ops::ple_conv_add(residual.tensor, gated.tensor, f.kernel.tensor, ragged.view, before,
                              current.tensor, y, ex);
    };
    const auto unpack = [&](const Tensor& tensor) {
        return ragged.unpack(from_device<std::uint16_t>(tensor.data, D * n), width, D);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        if (phase) rotate_replay_partition(ragged);
        f.prepare(phase, false);
        f.valid         = ragged.lengths;
        f.source.back() = f.source.front();
        upload_i32(f.d_valid, f.valid);
        upload_offsets(f.d_source, f.history.tensor, f.source);
        residual.host = ragged.pack(f.residual.host, width, D);
        gated.host    = ragged.pack(f.gated.host, width, D);
        for (int s = 0; s < rows; ++s)
            for (int t = 0; t < ragged.lengths[s]; ++t)
                std::copy_n(f.records.host.begin() + f.records.index(0, t + f.offset[s], s), D,
                            current.host.begin() + current.index(0, ragged.prefix[s] + t));
        for (const auto* data : {&residual, &gated, &current, &updated, &normalized})
            data->upload();
        cuda_synchronize();
        if (graph && phase == 0) {
            definition.capture(device.stream(), launch);
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(device.stream());
        else
            launch();
        device.synchronize();
        const auto out = unpack(y);
        const auto z   = norm ? unpack(normalized.tensor) : std::vector<std::uint16_t>{};
        failures += f.verify(norm, "ragged PLE convolution", &out, norm ? &z : nullptr);
        for (const auto* data : {&residual, &updated, &normalized})
            failures += data->device->verify_guards("ragged PLE convolution");
        failures += gated.preserved("ragged gated values") + current.preserved("ragged records");
        if (!inplace) failures += residual.preserved("ragged residual");
    }
    return failures;
}
} // namespace

void add_ple_conv_cases(ninfer::test::OpSuite& suite) {
    using Layout = RaggedConvLayout;
    suite.add("ple_conv/ragged/single",
              [] { return ragged_conv({17}, true, Layout::WidePadded, true, false); });
    suite.add("ple_conv/ragged/small",
              [] { return ragged_conv({1, 2, 5}, false, Layout::Dense, false, false); });
    suite.add("ple_conv/ragged/large_norm",
              [] { return ragged_conv({511, 513}, true, Layout::WidePadded, true, true); },
              {{"execution", "graph"}});
    suite.add("ple_conv/ragged/large_norm_paired_stride",
              [] { return ragged_conv({511, 513}, true, Layout::PairedPadded, true, false); });
    for (bool norm : {false, true}) {
        const auto name = std::string("ple_conv/ragged/") + (norm ? "norm" : "plain");
        suite.add(name + "/paired",
                  [=] {
                      return ragged_conv({1, 2, 5, 8, 9, 10, 17, 129}, norm, Layout::WidePadded,
                                         false, true);
                  },
                  {{"execution", "graph"}});
        suite.add(name + "/scalar", [=] {
            return ragged_conv({1, 2, 5, 8, 9, 10, 17, 129}, norm, Layout::ScalarPadded, true,
                               false);
        });
    }
    suite.add("ple_conv/contract", [] { return validation(cuda_environment().context); });
    suite.add("ple_conv/ragged/contract", ragged_validation);
    suite.add("ple_conv/history_gap",
              [] { return output_in_history_gap(cuda_environment().context); });
    const auto add = [&](int t, int b, int layout, bool inplace, bool norm, bool graph = false,
                         float scale = 1.f) {
        std::vector<std::pair<std::string, std::string>> tags{
            {"execution", graph ? "graph" : "eager"}};
        if (b > 8) tags.emplace_back("suite", "concurrency");
        suite.add(
            "ple_conv/t" + std::to_string(t) + "/b" + std::to_string(b) + "/layout" +
                std::to_string(layout) + (inplace ? "/alias" : "/disjoint") +
                (norm ? "/norm" : "/plain") + "/scale" + std::to_string(scale) +
                (graph ? "/graph" : "/eager"),
            [=] {
                return run(cuda_environment().context, t, b, layout, inplace, norm, graph, scale);
            },
            std::move(tags));
    };
    for (bool norm : {false, true}) {
        suite.add(std::string("ple_conv/short_record/") + (norm ? "norm" : "plain"),
                  [=] { return short_record_view(cuda_environment().context, norm); });
        suite.add(std::string("ple_conv/split_window/") + (norm ? "norm" : "plain"),
                  [=] { return split_window(cuda_environment().context, norm); });
        for (int t : {1, 2, 3, 4, 6, 8, 9, 10, 16, 17, 33})
            add(t, t % 2 ? 1 : 8, 0, t % 3 == 0, norm, t == 1 || t == 9 || t == 16);
        for (int layout : {1, 2}) add(13, 3, layout, true, norm, true);
        for (float scale : {0.f, 1e-5f}) add(4, 2, 0, false, norm, true, scale);
        add(3, 17, 0, true, norm, true);
        add(1024, 1, 0, false, norm);
        add(4097, 1, 0, true, norm);
        if (norm) {
            for (int delta : {-1, 0, 1})
                suite.add("ple_conv/sm_boundary/" + std::to_string(delta),
                          [=] {
                              auto& device = cuda_environment().context;
                              return run(device, device.multiprocessor_count() / 2 + delta, 1, 0,
                                         true, true, delta == 0);
                          },
                          {{"execution", delta == 0 ? "graph" : "eager"}});
            for (int t : {1023, 1025, 2303, 2304, 2305}) add(t, 1, 0, true, true, t == 1025);
        } else {
            for (int t : {127, 128, 129}) add(t, 1, 0, false, false);
        }
    }
}
