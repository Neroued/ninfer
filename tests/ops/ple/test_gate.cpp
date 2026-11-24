#include "ops/suite.h"
#include "gate_fixture.h"
#include "core/decode_graph.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::ple;

namespace {

int run(DeviceContext& device, int t, int b, GateLayout layout = GateLayout::Packed,
        GateSignal signal = GateSignal::Random, bool graph = false) {
    GateFixture f(t, b, layout);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    int failures = 0;
    for (int phase = 0; phase < (graph ? 2 : 1); ++phase) {
        f.prepare(signal, phase);
        if (graph && !phase) {
            definition.capture(device.stream(), [&] { f.launch(device.execution_view()); });
            executable.instantiate(definition);
        }
        if (graph)
            executable.launch(device.stream());
        else
            f.launch(device.execution_view());
        device.synchronize();
        failures += f.verify("ple_gate T=" + std::to_string(t) + " B=" + std::to_string(b) +
                             " layout=" + name(layout) + " signal=" + name(signal) +
                             " phase=" + std::to_string(phase));
    }
    return failures;
}

int validation(DeviceContext& device) {
    GateFixture f(3, 2);
    f.prepare();
    int failures      = 0;
    auto projected    = ops::PleProjectionView{f.key, f.value};
    auto parameters   = f.parameters();
    const auto invoke = [&](Tensor& output) {
        ops::ple_gate(projected, f.residual.tensor, parameters, f.batch_view(), f.record_view(),
                      output, device.execution_view());
    };
    failures += rejects("gate in-place residual", [&] { invoke(f.residual.tensor); });
    projected.key.nb[2] = H * 2;
    failures += rejects("overlapping key columns", [&] { invoke(f.gated.tensor); });
    projected.key          = f.key;
    projected.value.device = Device::cpu();
    failures += rejects("Host projection", [&] { invoke(f.gated.tensor); });
    projected.value        = f.value;
    parameters.rms_epsilon = 0;
    failures += rejects("gate epsilon", [&] { invoke(f.gated.tensor); });
    parameters                = f.parameters();
    parameters.norm_key.dtype = DType::FP32;
    failures += rejects("gate weight dtype", [&] { invoke(f.gated.tensor); });
    return failures;
}

int short_records(DeviceContext& device) {
    GateFixture f(17, 2, GateLayout::OddStride);
    f.prepare(GateSignal::Tiny);
    f.valid                = {3, 0};
    f.offsets              = {0, 3};
    f.records.tensor.ne[1] = 3;
    f.eps                  = 2e-3f;
    upload_i32(f.d_valid, f.valid);
    upload_i32(f.d_offsets, f.offsets);
    cuda_synchronize();
    f.launch(device.execution_view());
    device.synchronize();
    return f.verify("ple_gate R<T and independent epsilon");
}

int long_extent(DeviceContext& device) {
    // Exercise the real grid limit with only two nonzero reference columns. K, Q and V
    // share read-only storage; no large Host fixture or full-size FP64 oracle is needed.
    constexpr int capacity             = 16385;
    constexpr std::size_t column_bytes = D * sizeof(std::uint16_t);
    DeviceBuffer input(device.device(), capacity * column_bytes);
    DeviceBuffer gated(device.device(), capacity * column_bytes);
    DeviceBuffer records(device.device(), capacity * column_bytes);
    GateFixture reference(2, 1);
    reference.prepare(GateSignal::Random, 0, false);
    for (int t = 0; t < 2; ++t)
        for (int d = 0; d < H; ++d) {
            reference.projected.host[reference.index(reference.value, d, t, 0)] =
                reference.residual.host[reference.residual.index(d, 0, t)];
            for (int s = 0; s < C; ++s)
                reference.projected.host[reference.index(reference.key, d, s, t)] =
                    reference.residual.host[reference.residual.index(d, s, t)];
        }
    reference.projected.upload();
    auto valid   = to_device(std::vector<int>{capacity});
    int failures = 0;
    for (int t : {16383, 16384, 16385}) {
        input.fill();
        gated.fill(0x7f);
        records.fill(0x7f);
        for (int sample = 0; sample < 2; ++sample)
            cuda_check(
                cudaMemcpy(static_cast<char*>(input.data()) + (sample ? t - 1 : 0) * column_bytes,
                           reference.residual.host.data() + sample * D, column_bytes,
                           cudaMemcpyHostToDevice),
                "long PLE input");
        upload_i32(valid, {t});
        cuda_synchronize();
        Tensor x(input.span(), DType::BF16, {H, C, t, 1});
        Tensor v(input.span(), DType::BF16, {H, t, 1});
        v.nb[1] = column_bytes;
        v.nb[2] = std::int64_t(t) * column_bytes;
        Tensor u(gated.span(), DType::BF16, {H, C, t, 1});
        ops::ple_gate(
            {x, v}, x, reference.parameters(), {Tensor(valid.span(), DType::I32, {1})},
            {Tensor(records.span(), DType::BF16, {D, t, 1}), reference.record_view().offsets}, u,
            device.execution_view());
        device.synchronize();
        for (int sample = 0; sample < 2; ++sample) {
            const auto offset = (sample ? t - 1 : 0) * column_bytes;
            cuda_check(
                cudaMemcpy(static_cast<char*>(reference.gated.tensor.data) + sample * column_bytes,
                           static_cast<const char*>(gated.data()) + offset, column_bytes,
                           cudaMemcpyDeviceToDevice),
                "long PLE U sample");
            cuda_check(cudaMemcpy(static_cast<char*>(reference.records.tensor.data) +
                                      sample * column_bytes,
                                  static_cast<const char*>(records.data()) + offset, column_bytes,
                                  cudaMemcpyDeviceToDevice),
                       "long PLE record sample");
        }
        failures += reference.verify("ple_gate long extent T=" + std::to_string(t));
    }
    return failures;
}

int chain(DeviceContext& device) {
    constexpr int t = 13;
    GateFixture f(t, 1);
    f.prepare(GateSignal::Random, 0, false);
    Words history({D, 9, 1}), after({D, 9, 1}), kernel({D, 4}), norm({H, C});
    Words y({H, C, t, 1}), z({H, C, t, 1});
    history.fill(401);
    kernel.fill(409, .25f);
    norm.fill(419, .4f);
    for (const Words* w : {&history, &after, &kernel, &norm, &y, &z}) w->upload();
    auto zero     = to_device(std::vector<std::int64_t>{0});
    auto accepted = to_device(std::vector<int>{t});
    Tensor offsets(zero.span(), DType::I64, {1}), count(accepted.span(), DType::I32, {1});
    const ops::PleConvStateView before{
        history_field(history.tensor, history.device->span(), offsets)};
    const ops::PleConvStateView destination{
        history_field(after.tensor, after.device->span(), offsets)};
    cuda_synchronize();
    f.launch(device.execution_view());
    ops::ple_conv_add_norm(f.residual.tensor, f.gated.tensor, kernel.tensor, f.batch_view(), before,
                           f.record_view(), norm.tensor, epsilon, y.tensor, z.tensor,
                           device.execution_view());
    device.synchronize();
    int failures = f.verify("gate feeding convolution");
    const auto u = f.gated.download(), n = f.records.download(), actual_y = y.download(),
               actual_z = z.download();
    std::vector<double> got_y(actual_y.size()), ref_y(actual_y.size()), got_z(actual_y.size()),
        ref_z(actual_y.size());
    for (int token = 0; token < t; ++token)
        for (int s = 0; s < C; ++s) {
            const auto base = std::size_t(token * C + s) * H;
            double yy       = 0;
            for (int d = 0; d < H; ++d) {
                const int c = d + H * s;
                double conv = 0;
                for (int tap = 0; tap < 4; ++tap) {
                    const int pos  = token - 9 + 3 * tap;
                    const double v = pos < 0 ? value(history, c, 9 + pos)
                                             : bf16_to_f32(n[f.records.index(c, pos)]);
                    conv += value(kernel, c, tap) * v;
                }
                got_y[base + d] = bf16_to_f32(actual_y[base + d]);
                ref_y[base + d] = value(f.residual, d, s, token) + bf16_to_f32(u[base + d]) +
                                  conv / (1. + std::exp(-conv));
                yy += got_y[base + d] * got_y[base + d];
            }
            const double inv = 1. / std::sqrt(yy / H + epsilon);
            for (int d = 0; d < H; ++d) {
                got_z[base + d] = bf16_to_f32(actual_z[base + d]);
                ref_z[base + d] = got_y[base + d] * inv * (1. + value(norm, d, s));
            }
        }
    failures += verify_reduction("Gate/Conv Y", got_y, ref_y, gate_criterion);
    failures += verify_reduction("Gate/Conv norm(actual Y)", got_z, ref_z, gate_criterion);
    for (int a : {0, 2, 9, t}) {
        upload_i32(accepted, {a});
        cuda_synchronize();
        ops::ple_state_extract(before, {f.records.tensor, f.batch_view().valid_columns}, count,
                               destination, device.execution_view());
        device.synchronize();
        auto expected = after.host;
        for (int j = 0; j < 9; ++j)
            for (int c = 0; c < D; ++c) {
                const int pos = a - 9 + j;
                expected[after.index(c, j)] =
                    pos < 0 ? history.host[history.index(c, 9 + pos)] : n[f.records.index(c, pos)];
            }
        failures += verify_exact("Gate records accepted prefix", after.download(), expected);
    }
    return failures + history.preserved("gate chain history") +
           kernel.preserved("gate chain kernel") + norm.preserved("gate chain norm") +
           f.verify("gate chain inputs/records preserved");
}
} // namespace

void add_ple_gate_cases(ninfer::test::OpSuite& suite) {
    for (auto [label, run] :
         {std::pair<const char*, int (*)(DeviceContext&)>{"contract", validation},
          {"short_records", short_records},
          {"long_extent", long_extent},
          {"conv_handoff", chain}})
        suite.add(std::string("ple_gate/") + label,
                  [=] { return run(cuda_environment().context); });
    const auto add = [&](int t, int b, GateLayout layout = GateLayout::Packed,
                         GateSignal signal = GateSignal::Random, bool graph = false) {
        suite.add("ple_gate/t" + std::to_string(t) + "/b" + std::to_string(b) + "/layout" +
                      std::to_string(int(layout)) + "/signal" + std::to_string(int(signal)) +
                      (graph ? "/graph" : "/eager"),
                  [=] { return run(cuda_environment().context, t, b, layout, signal, graph); },
                  {{"execution", graph ? "graph" : "eager"}});
    };
    for (int b = 1; b <= 8; ++b)
        add(b % 2 ? 1 : 5, b, GateLayout::Packed, GateSignal::Random, b == 8);
    add(1, 1, GateLayout::Packed, GateSignal::Random, true);
    for (auto layout : {GateLayout::Separate, GateLayout::OddStride, GateLayout::Interleaved,
                        GateLayout::PairAligned, GateLayout::PaddedColumns})
        add(9, 3, layout, GateSignal::Random, true);
    for (auto signal : {GateSignal::ZeroKey, GateSignal::Correlated, GateSignal::Opposed,
                        GateSignal::Balanced, GateSignal::Tiny})
        add(3, 2, GateLayout::Packed, signal);
    for (int t : {2, 3, 8, 16, 33, 127, 128, 129, 256, 1024, 4097}) add(t, 1);
}
