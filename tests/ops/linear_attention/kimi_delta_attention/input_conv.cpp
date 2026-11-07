#include "ops/suite.h"
#include "ops/input_projection_test_common.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/kda_input_proj.h"
#include "ninfer/ops/kda_replay.h"

#include <array>
#include <tuple>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int C = 12288, H = 4096;
// Q8 A16 reconstruction and the BF16 public result: two unit roundoffs, with a gross cap.
constexpr ReductionCriterion a16_criterion{2.0 / 256, 1.0 / 256, 2.0 / 256};
constexpr ReductionCriterion a8_criterion{0.04, 1.0 / 256, 0.06};

int fold_records(const Tensor& conv, const std::vector<std::uint16_t>& words,
                 const std::vector<float>& history, const std::vector<int>& selected,
                 const std::vector<int>& valid, CudaExecutionView cuda) {
    const int width = conv.ne[1], batch = conv.ne[2];
    StateImageBuilder builder;
    const auto fields = plan_linear_attention_state(builder, {1, C, 3, 32, 128, 128, DType::BF16});
    const auto layout = builder.finish();
    DeviceBuffer arena(cuda.device(), layout.image_bytes * 2 * batch);
    CUDA_CHECK(cudaMemsetAsync(arena.data(), 0, arena.size(), cuda.stream()));
    StateImageStorage storage(arena.span(), layout);
    LinearAttentionStateView states(storage, fields);
    LayoutBuilder records_builder;
    const auto record_layout =
        plan_kda_replay_records(records_builder, {1, batch, width, C, 32, 32, 128, 128});
    DeviceBuffer frame(cuda.device(), records_builder.finish(256));
    CUDA_CHECK(cudaMemsetAsync(frame.data(), 0, frame.size(), cuda.stream()));
    KdaReplayRecords records(frame.span(), record_layout);
    records.conv = conv;
    ops::KdaReplayFoldPlan plan(records, states.all_layers_view());
    std::vector<ops::KdaReplayFoldRow> rows(batch);
    std::vector<std::uint16_t> initial(std::size_t(batch) * C * 3);
    for (int b = 0; b < batch; ++b) {
        rows[b]           = {static_cast<std::int64_t>(2 * b * layout.image_bytes),
                             static_cast<std::int64_t>((2 * b + 1) * layout.image_bytes), 0};
        const auto target = states.conv_tensor(0, rows[b].source_offset);
        auto* row         = initial.data() + std::size_t(b) * C * 3;
        for (std::size_t i = 0; i < C * 3; ++i)
            row[i] = f32_to_bf16(history[std::size_t(selected[b]) * C * 3 + i]);
        CUDA_CHECK(cudaMemcpyAsync(target.data, row, target.bytes(), cudaMemcpyHostToDevice,
                                   cuda.stream()));
    }
    int failures = 0;
    for (int accepted = 0; accepted <= width; ++accepted) {
        for (int b = 0; b < batch; ++b) {
            rows[b].commit_columns = std::min(accepted, valid[b]);
            const auto target      = states.conv_tensor(0, rows[b].destination_offset);
            CUDA_CHECK(cudaMemsetAsync(target.data, 0x35, target.bytes(), cuda.stream()));
        }
        plan.execute(rows, cuda);
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
        for (int b = 0; b < batch; ++b) {
            const auto target = states.conv_tensor(0, rows[b].destination_offset);
            std::vector<std::uint16_t> expected(C * 3, 0x3535);
            const int count = rows[b].commit_columns;
            if (count)
                for (int j = 0; j < 3; ++j)
                    for (int c = 0; c < C; ++c) {
                        const int t = count - 3 + j;
                        expected[j * C + c] =
                            t < 0 ? f32_to_bf16(
                                        history[(std::size_t(selected[b]) * 3 + t + 3) * C + c])
                                  : words[(std::size_t(b) * width + t) * C + c];
                    }
            failures += verify_exact("fold actual input records",
                                     from_device<std::uint16_t>(target.data, C * 3), expected);
        }
    }
    return failures;
}

int run_case(int width, int batch, bool in_place, bool graph, int capacity_last = 0,
             QType format             = QType::Q8_G32_FP16,
             ops::LinearPolicy policy = ops::LinearPolicy::A16Only,
             bool precision_boundary  = false) {
    auto packed = quantized_weight::make_patterned_weight(format, C, H, 1201);
    if (precision_boundary) {
        std::fill(packed.payload.begin(), packed.payload.end(), 0);
        std::fill(packed.payload.begin() + packed.geometry.scale_offset,
                  packed.payload.begin() + packed.geometry.scale_offset +
                      packed.geometry.scale_bytes,
                  127);
        for (int c = 0; c < C; ++c) {
            packed.payload[std::size_t(c) * H]     = 0x38;
            packed.payload[std::size_t(c) * H + 1] = 0x38;
        }
    }
    input_projection::DevicePackedWeight weight(std::move(packed));
    const ops::SingleProjectionWeight projection{weight.view(), policy};
    const auto criterion = policy == ops::LinearPolicy::AllowA8 ? a8_criterion : a16_criterion;
    auto& ctx            = cuda_environment().context;
    CudaStream stream(ctx);
    const auto ex = ctx.execution_view(stream.stream());
    auto xh       = input_projection::make_bf16_activation(H, width * batch, 1203);
    auto historyh = input_projection::make_bf16_activation(C, 3 * batch, 1205);
    std::vector<float> wh(C * 4);
    fill_uniform(wh, 1207, -0.7F, 0.7F);
    if (precision_boundary) {
        std::fill(xh.begin(), xh.end(), 0);
        for (int column = 0; column < width * batch; ++column) {
            xh[std::size_t(column) * H]     = 1;
            xh[std::size_t(column) * H + 1] = 1.F / 256;
        }
        std::fill(historyh.begin(), historyh.end(), 1);
        std::fill(wh.begin(), wh.end(), 0);
        std::fill(wh.begin() + 2 * C, wh.begin() + 3 * C, -1);
        std::fill(wh.begin() + 3 * C, wh.end(), 1);
    }
    const bool mx = format == QType::MXFP8;
    if (mx) round_to_bf16(wh);
    auto xb = to_device_bf16(xh), wb = mx ? to_device_bf16(wh) : to_device_f32(wh),
         hb = to_device_bf16(historyh);
    Tensor x(xb.span(), DType::BF16, {H, width, batch}),
        w(wb.span(), mx ? DType::BF16 : DType::FP32, {C, 4});
    Tensor history(hb.span(), DType::BF16, {C, 3, batch});
    RecordStateFixture source(history, 2), destination(history, 2);
    std::vector<std::int64_t> so(batch), ds(batch);
    std::vector<int> valid(batch), selected(batch);
    for (int b = 0; b < batch; ++b) {
        selected[b] = b;
        so[b]       = source.byte_offset(b);
        ds[b]       = in_place ? so[b] : destination.byte_offset(b);
        valid[b]    = std::max(1, width - b % width);
    }
    auto sob = to_device(so), dsb = to_device(ds), vb = to_device(valid);
    Tensor st(sob.span(), DType::I64, {batch}), dt(dsb.span(), DType::I64, {batch}),
        vt(vb.span(), DType::I32, {batch});
    const int first = capacity_last ? 1 : width, last = capacity_last ? capacity_last : width;
    const auto bytes = std::max(ops::kda_input_proj_conv_record_workspace_capacity_bytes(
                                    linear_profile(projection), batch, first, last),
                                ops::kda_input_proj_conv_update_workspace_capacity_bytes(
                                    linear_profile(projection), batch, first, last));
    GuardedDeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    MemoryArena workspace(scratch.span());
    std::array<GuardedDeviceBuffer, 3> record_outputs{
        GuardedDeviceBuffer(std::size_t(H) * width * batch * 2),
        GuardedDeviceBuffer(std::size_t(H) * width * batch * 2),
        GuardedDeviceBuffer(std::size_t(H) * width * batch * 2)};
    std::array<GuardedDeviceBuffer, 3> update_outputs{
        GuardedDeviceBuffer(std::size_t(H) * width * batch * 2),
        GuardedDeviceBuffer(std::size_t(H) * width * batch * 2),
        GuardedDeviceBuffer(std::size_t(H) * width * batch * 2)};
    const auto outputs = [&](auto& buffers) {
        return ops::KdaConvOutputs{Tensor(buffers[0].span(), DType::BF16, {H, width, batch}),
                                   Tensor(buffers[1].span(), DType::BF16, {H, width, batch}),
                                   Tensor(buffers[2].span(), DType::BF16, {H, width, batch})};
    };
    GuardedDeviceBuffer raw(std::size_t(C) * width * batch * 2);
    raw.fill(0x35);
    Tensor records(raw.span(), DType::BF16, {C, width, batch});
    cuda_synchronize();
    const auto record = [&] {
        ops::kda_input_proj_conv_record(x, projection, w, {source.view(st)}, vt,
                                        outputs(record_outputs), records, workspace, ex);
    };
    record();
    if (graph) {
        CUDA_CHECK(cudaStreamSynchronize(ex.stream()));
        cudaGraph_t g;
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaStreamBeginCapture(ex.stream(), cudaStreamCaptureModeThreadLocal));
        record();
        CUDA_CHECK(cudaStreamEndCapture(ex.stream(), &g));
        CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
        const std::vector<int> one(batch, 1);
        vb.copy_from_host(one.data(), vb.size());
        CUDA_CHECK(cudaGraphLaunch(exec, ex.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ex.stream()));
        for (int b = 0; b < batch; ++b) {
            selected[b] = batch - 1 - b;
            so[b]       = source.byte_offset(selected[b]);
            if (in_place) ds[b] = so[b];
        }
        sob.copy_from_host(so.data(), sob.size());
        if (in_place) dsb.copy_from_host(ds.data(), dsb.size());
        vb.copy_from_host(valid.data(), vb.size());
        raw.fill(0x35);
        CUDA_CHECK(cudaGraphLaunch(exec, ex.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ex.stream()));
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(g));
    }
    cuda_synchronize();
    source.copy_back();
    int failures =
        verify_exact("record source unchanged", from_device_bf16(hb.data(), historyh.size()),
                     std::vector<double>(historyh.begin(), historyh.end()));
    ops::kda_input_proj_conv_update(
        x, projection, w, {source.view(st), in_place ? source.view(dt) : destination.view(dt)}, vt,
        outputs(update_outputs), workspace, ex);
    cuda_synchronize();
    const auto raw_bits = from_device<std::uint16_t>(raw.data(), raw.bytes() / 2);
    std::array<std::vector<double>, 3> got;
    for (int i = 0; i < 3; ++i) {
        got[i] = from_device_bf16(record_outputs[i].data(), record_outputs[i].bytes() / 2);
        failures += verify_exact(
            "input update/record",
            from_device<std::uint16_t>(record_outputs[i].data(), record_outputs[i].bytes() / 2),
            from_device<std::uint16_t>(update_outputs[i].data(), update_outputs[i].bytes() / 2));
        failures += record_outputs[i].verify_guards("record QKV") +
                    update_outputs[i].verify_guards("update QKV");
    }
    const auto& final_history = in_place ? source : destination;
    for (int b = 0; b < batch; ++b) {
        const auto t      = final_history.tensor(in_place ? selected[b] : b);
        const auto actual = from_device<std::uint16_t>(t.data, C * 3);
        std::vector<std::uint16_t> expected(C * 3);
        for (int j = 0; j < 3; ++j)
            for (int c = 0; c < C; ++c) {
                const int pos = valid[b] - 3 + j;
                expected[j * C + c] =
                    pos < 0 ? f32_to_bf16(historyh[(selected[b] * 3 + pos + 3) * C + c])
                            : raw_bits[(std::size_t(b) * width + pos) * C + c];
            }
        failures += verify_exact("update history from actual records", actual, expected);
        for (int t = valid[b]; t < width; ++t) {
            for (int c = 0; c < C; ++c)
                if (raw_bits[(std::size_t(b) * width + t) * C + c] != 0x3535)
                    throw std::runtime_error("input record overwrote invalid suffix");
            for (const auto& plane : got)
                for (int c = 0; c < H; ++c)
                    if (plane[(std::size_t(b) * width + t) * H + c] != 0)
                        throw std::runtime_error("input output invalid suffix is nonzero");
        }
    }
    std::vector<double> reference, actual, projected_reference, projected_actual;
    const auto sampled = input_projection::sampled_rows(C, 31);
    for (int b = 0; b < batch; ++b)
        for (const int c : sampled) {
            std::vector<double> p(valid[b]);
            for (int t = 0; t < valid[b]; ++t)
                p[t] = quantized_weight::dot_fp64(weight.host, c,
                                                  xh.data() + (std::size_t(b) * width + t) * H, H);
            for (int t = 0; t < valid[b]; ++t) {
                double sum = 0;
                for (int tap = 0; tap < 4; ++tap) {
                    const int pos = t + tap - 3;
                    const double value =
                        pos < 0 ? historyh[(selected[b] * 3 + pos + 3) * C + c] : p[pos];
                    sum += double(wh[tap * C + c]) * value;
                }
                reference.push_back(sum / (1 + std::exp(-sum)));
                actual.push_back(got[c / H][(std::size_t(b) * width + t) * H + c % H]);
                projected_reference.push_back(p[t]);
                projected_actual.push_back(
                    bf16_to_f32(raw_bits[(std::size_t(b) * width + t) * C + c]));
            }
        }
    failures += verify_reduction("fused projection/conv/SiLU", actual, reference, criterion);
    if (precision_boundary) {
        const double p        = 1.0 / 256;
        const double expected = bf16_to_f32(f32_to_bf16(p / (1 + std::exp(-p))));
        for (const auto& plane : got)
            if (plane[0] != expected)
                throw std::runtime_error("projection was rounded before convolution");
    }
    failures +=
        verify_reduction("projected history", projected_actual, projected_reference, criterion);
    if (mx && width == 8 && batch == 8)
        failures += fold_records(records, raw_bits, historyh, selected, valid, ex);
    failures += source.verify_guards("source") + destination.verify_guards("destination") +
                raw.verify_guards("raw records") + scratch.verify_guards("workspace");
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("kda/input/single", [] { return run_case(1, 1, true, false); });
    suite.add("kda/input/window", [] { return run_case(3, 8, false, true); });
    suite.add("kda/input/window_boundary", [] { return run_case(8, 4, true, false); });
    suite.add("kda/input/batch_boundary", [] { return run_case(7, 5, true, false); });
    suite.add("kda/input/batched_materialized", [] { return run_case(8, 8, false, true); });
    suite.add("kda/input/materialized", [] { return run_case(9, 2, true, false); });
    suite.add("kda/input/prefill", [] { return run_case(129, 1, false, true); });
    suite.add("kda/input/workspace_interval", [] { return run_case(32, 1, false, false, 4096); });
    for (int width :
         {31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257, 511, 512, 513, 2047, 2048, 2049})
        suite.add("kda/input/prefill_t" + std::to_string(width),
                  [width] { return run_case(width, 1, false, false); });
    for (const auto [width, batch, in_place, graph] : {std::tuple{1, 1, true, false},
                                                       {1, 4, true, true},
                                                       {2, 2, true, true},
                                                       {3, 1, true, true},
                                                       {4, 1, true, true},
                                                       {5, 1, true, true},
                                                       {2, 4, false, false},
                                                       {3, 8, true, true},
                                                       {4, 8, true, true},
                                                       {5, 4, true, true},
                                                       {5, 8, true, true},
                                                       {6, 8, true, true},
                                                       {7, 4, true, true},
                                                       {7, 8, true, true},
                                                       {8, 8, false, true},
                                                       {9, 2, true, false},
                                                       {17, 1, true, true},
                                                       {31, 1, false, false},
                                                       {33, 1, true, true},
                                                       {9, 4, true, true},
                                                       {129, 1, false, true},
                                                       {24, 1, true, true},
                                                       {64, 1, true, true},
                                                       {65, 1, true, true},
                                                       {128, 1, true, true},
                                                       {16, 8, true, true},
                                                       {192, 1, false, false},
                                                       {193, 1, true, true},
                                                       {257, 1, false, true},
                                                       {385, 1, true, true},
                                                       {513, 1, false, true},
                                                       {1024, 1, false, false},
                                                       {2048, 1, false, false},
                                                       {4096, 1, false, false}})
        suite.add("kda/input/mxfp8/t" + std::to_string(width) + "b" + std::to_string(batch), [=] {
            return run_case(width, batch, in_place, graph, 0, QType::MXFP8,
                            ops::LinearPolicy::AllowA8);
        });
    suite.add("kda/input/mxfp8/workspace_interval", [] {
        return run_case(32, 1, false, false, 4096, QType::MXFP8, ops::LinearPolicy::AllowA8);
    });
    suite.add("kda/input/mxfp8/batched_workspace_interval", [] {
        return run_case(9, 8, true, true, 9, QType::MXFP8, ops::LinearPolicy::AllowA8);
    });
    suite.add("kda/input/mxfp8/no_projection_rounding", [] {
        return run_case(129, 1, true, false, 0, QType::MXFP8, ops::LinearPolicy::A16Only, true);
    });
    return suite.main(argc, argv);
}
