#include "ops/suite.h"
#include "ops/input_projection_test_common.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/kda_input_proj.h"

#include <array>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int C = 12288, H = 4096;
// Q8 A16 reconstruction and the BF16 public result: two unit roundoffs, with a gross cap.
constexpr ReductionCriterion criterion{2.0 / 256, 1.0 / 256, 2.0 / 256};

int run_case(int width, int batch, bool in_place, bool graph) {
    input_projection::DevicePackedWeight weight(
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, C, H, 1201));
    const ops::SingleProjectionWeight projection{weight.view(), ops::LinearPolicy::A16Only};
    auto& ctx = cuda_environment().context;
    CudaStream stream(ctx);
    const auto ex       = ctx.execution_view(stream.stream());
    const auto xh       = input_projection::make_bf16_activation(H, width * batch, 1203);
    const auto historyh = input_projection::make_bf16_activation(C, 3 * batch, 1205);
    std::vector<float> wh(C * 4);
    fill_uniform(wh, 1207, -0.7F, 0.7F);
    auto xb = to_device_bf16(xh), wb = to_device_f32(wh), hb = to_device_bf16(historyh);
    Tensor x(xb.span(), DType::BF16, {H, width, batch}), w(wb.span(), DType::FP32, {C, 4});
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
    const auto bytes = ops::kda_input_proj_conv_record_workspace_capacity_bytes(
        linear_profile(projection), batch, width, width);
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
        }
        sob.copy_from_host(so.data(), sob.size());
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
        const auto t      = final_history.tensor(b);
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
    failures +=
        verify_reduction("projected history", projected_actual, projected_reference, criterion);
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
    for (int width : {63, 64, 65, 255, 256, 257})
        suite.add("kda/input/prefill_t" + std::to_string(width),
                  [width] { return run_case(width, 1, false, false); });
    return suite.main(argc, argv);
}
