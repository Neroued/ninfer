#include "ops/suite.h"
#include "ops/ragged_fixture.h"
#include "core/decode_graph.h"
#include "core/platform.h"
#include "core/transfer.h"
#include "ops/input_projection_test_common.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/kda_input_proj.h"
#include "ninfer/ops/kda_replay.h"
#include "ninfer/ops/kimi_delta_attention.h"

#include <array>
#include <tuple>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int C = 12288, H = 4096;
// Q8 A16 reconstruction and the BF16 public result: two unit roundoffs, with a gross cap.
constexpr ReductionCriterion a16_criterion{2.0 / 256, 1.0 / 256, 2.0 / 256};
constexpr ReductionCriterion a8_criterion{0.04, 1.0 / 256, 0.06};
enum class InactiveRows { None, Mixed, All };

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
    KdaReplayRecords records(frame.span(), record_layout);
    copy_bytes_async({records.conv.data, records.conv.bytes(), cuda.device()},
                     {conv.data, conv.bytes(), cuda.device()}, cuda);
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
    // Populate the opaque recurrence records from the same zero S0 used by fold.
    std::vector<std::int64_t> sources;
    for (const auto& row : rows) sources.push_back(row.source_offset);
    auto source_buffer = to_device(sources), lengths = to_device(valid);
    Tensor offsets(source_buffer.span(), DType::I64, {batch});
    Tensor counts(lengths.span(), DType::I32, {batch});
    auto zeros = to_device_bf16(std::vector<float>(std::size_t(H) * width * batch));
    const Tensor input(zeros.span(), DType::BF16, {128, 32, width, batch});
    const Tensor beta(zeros.span(), DType::BF16, {32, width, batch});
    auto gates = to_device_f32(std::vector<float>(H));
    const Tensor a(gates.span(), DType::FP32, {32}), bias(gates.span(), DType::FP32, {128, 32});
    DeviceBuffer output(cuda.device(), input.bytes());
    Tensor out(output.span(), DType::BF16, {128, 32, width, batch});
    ops::kimi_delta_attention_replay_record(
        input, input, input, input, beta, a, bias, 0.0F, 0.08838834764831845F,
        states.layer_view(0, offsets).recurrent, counts, records.layer(0, batch).recurrence, out,
        cuda);
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

void set_unit_projection(quantized_weight::PackedWeight& packed, QType format) {
    std::fill(packed.payload.begin(), packed.payload.end(), 0);
    if (format == QType::MXFP8) {
        std::fill_n(packed.payload.begin() + packed.geometry.scale_offset,
                    packed.geometry.scale_bytes, 127);
    } else {
        for (std::size_t i = 0; i < packed.scale_plane_bytes; i += 2)
            packed.payload[packed.scale_plane_offset + i + 1] = 0x3c; // FP16 1
    }
    const std::uint8_t one = format == QType::MXFP8 ? 0x38 : 1;
    for (int c = 0; c < C; ++c) {
        packed.payload[std::size_t(c) * H]     = one;
        packed.payload[std::size_t(c) * H + 1] = one;
    }
}

int run_case(int width, int batch, bool in_place, bool graph, int capacity_last = 0,
             QType format             = QType::Q8_G32_FP16,
             ops::LinearPolicy policy = ops::LinearPolicy::A16Only, bool precision_boundary = false,
             InactiveRows inactive = InactiveRows::None) {
    auto packed = quantized_weight::make_patterned_weight(format, C, H, 1201);
    if (precision_boundary) set_unit_projection(packed, format);
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
        if (inactive == InactiveRows::All || (inactive == InactiveRows::Mixed && b % 3 == 0)) {
            valid[b] = 0;
            so[b] = ds[b] = -1;
        }
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
        for (int b = 0; b < batch; ++b) so[b] = source.byte_offset(b);
        sob.copy_from_host(so.data(), sob.size());
        vb.copy_from_host(one.data(), vb.size());
        CUDA_CHECK(cudaGraphLaunch(exec, ex.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ex.stream()));
        for (int b = 0; b < batch; ++b) {
            selected[b] = batch - 1 - b;
            so[b]       = valid[b] ? source.byte_offset(selected[b]) : -1;
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
    std::array<std::vector<std::uint16_t>, 3> got, updated;
    for (int i = 0; i < 3; ++i) {
        got[i] = from_device<std::uint16_t>(record_outputs[i].data(), record_outputs[i].bytes() / 2);
        updated[i] =
            from_device<std::uint16_t>(update_outputs[i].data(), update_outputs[i].bytes() / 2);
        failures += record_outputs[i].verify_guards("record QKV") +
                    update_outputs[i].verify_guards("update QKV");
    }
    std::vector<std::vector<std::uint16_t>> updated_history(batch);
    const auto& final_history = in_place ? source : destination;
    for (int b = 0; b < batch; ++b) {
        const int original = in_place ? selected[b] : b;
        const auto t = final_history.tensor(original);
        updated_history[b] = from_device<std::uint16_t>(t.data, C * 3);
        for (int j = 0; j < 3; ++j) {
            const int pos = valid[b] - 3 + j;
            if (pos >= 0) continue;
            const int slot = valid[b] ? selected[b] : original;
            for (int c = 0; c < C; ++c)
                if (updated_history[b][j * C + c] !=
                        f32_to_bf16(historyh[(slot * 3 + pos + 3) * C + c]))
                    throw std::runtime_error("update changed untouched history words");
        }
        for (int t = valid[b]; t < width; ++t) {
            for (int c = 0; c < C; ++c)
                if (raw_bits[(std::size_t(b) * width + t) * C + c] != 0x3535)
                    throw std::runtime_error("input record overwrote invalid suffix");
            for (int i = 0; i < 3; ++i)
                for (int c = 0; c < H; ++c) {
                    const auto index = (std::size_t(b) * width + t) * H + c;
                    if (got[i][index] != 0 || updated[i][index] != 0)
                        throw std::runtime_error("input output invalid suffix is not BF16 +0");
                }
        }
    }
    std::vector<double> reference, actual, actual_update, projected_reference, projected_actual,
        update_history_reference, update_history_actual;
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
                const auto index = (std::size_t(b) * width + t) * H + c % H;
                actual.push_back(bf16_to_f32(got[c / H][index]));
                actual_update.push_back(bf16_to_f32(updated[c / H][index]));
                if (t >= valid[b] - 3) {
                    update_history_reference.push_back(p[t]);
                    update_history_actual.push_back(
                        bf16_to_f32(updated_history[b][(t + 3 - valid[b]) * C + c]));
                }
                projected_reference.push_back(p[t]);
                projected_actual.push_back(
                    bf16_to_f32(raw_bits[(std::size_t(b) * width + t) * C + c]));
            }
        }
    if (!reference.empty()) {
        failures += verify_reduction("record projection/conv/SiLU", actual, reference, criterion);
        failures += verify_reduction("update projection/conv/SiLU", actual_update, reference,
                                     criterion);
        failures += verify_reduction("updated projected history", update_history_actual,
                                     update_history_reference, criterion);
    }
    if (precision_boundary) {
        const double p        = 1.0 / 256;
        const double expected = bf16_to_f32(f32_to_bf16(p / (1 + std::exp(-p))));
        for (const auto& plane : got)
            if (bf16_to_f32(plane[0]) != expected)
                throw std::runtime_error("projection was rounded before convolution");
    }
    if (!projected_reference.empty())
        failures +=
            verify_reduction("projected history", projected_actual, projected_reference, criterion);
    if (mx && width == 8 && batch == 8)
        failures += fold_records(records, raw_bits, historyh, selected, valid, ex);
    failures += source.verify_guards("source") + destination.verify_guards("destination") +
                raw.verify_guards("raw records") + scratch.verify_guards("workspace");
    return failures;
}

enum class RaggedState { InPlace, Mixed, Shared };

int ragged_projection(std::vector<int> lengths, QType format, ops::LinearPolicy policy,
                      RaggedState alias, bool graph = false, bool precision = false) {
    RaggedFixture ragged(std::move(lengths));
    const int n = ragged.tokens(), rows = ragged.segments(), slots = 2 * rows + 1;
    std::vector<int> source(rows), destination(rows);
    for (int s = 0; s < rows; ++s) {
        source[s] = alias == RaggedState::Shared ? slots - 1 : rows - 1 - s;
        destination[s] =
            alias == RaggedState::InPlace || (alias == RaggedState::Mixed && s % 2 == 0) ? source[s]
                                                                                         : rows + s;
    }
    auto xh            = input_projection::make_bf16_activation(H, n, 1273);
    const auto initial = input_projection::make_bf16_activation(C, 3 * slots, 1291);
    const bool mx      = format == QType::MXFP8;
    std::vector<float> wh(C * 4);
    fill_uniform(wh, 1301, -.7F, .7F);
    auto packed = quantized_weight::make_patterned_weight(format, C, H, 1201);
    if (precision) {
        set_unit_projection(packed, format);
        std::fill(xh.begin(), xh.end(), 0);
        for (int column = 0; column < n; ++column) xh[std::size_t(column) * H] = 1;
        // Each impulse produces p = 1 + 1/256, a BF16 tie that rounds to 1. The
        // two-tap difference exposes the residue at both interior and tile-edge columns.
        for (int column : {5, 127, 129}) xh[std::size_t(column) * H + 1] = 1.F / 256;
        std::fill(wh.begin(), wh.begin() + 2 * C, 0);
        std::fill(wh.begin() + 2 * C, wh.begin() + 3 * C, -1);
        std::fill(wh.begin() + 3 * C, wh.end(), 1);
    }
    if (mx) round_to_bf16(wh);
    input_projection::DevicePackedWeight weight(std::move(packed));
    const ops::SingleProjectionWeight projection{weight.view(), policy};
    auto dx = to_device_bf16(xh), dw = mx ? to_device_bf16(wh) : to_device_f32(wh);
    auto history = to_device_bf16(initial);
    RecordStateFixture records(Tensor(history.span(), DType::BF16, {C, 3, slots}), 2);
    const auto so = records.record_offsets(source), ds = records.record_offsets(destination);
    auto source_offsets = to_device(so), dest_offsets = to_device(ds);
    const auto src = records.view(Tensor(source_offsets.span(), DType::I64, {rows}));
    const auto dst = records.view(Tensor(dest_offsets.span(), DType::I64, {rows}));
    Tensor x(dx.span(), DType::BF16, {H, n}), w(dw.span(), mx ? DType::BF16 : DType::FP32, {C, 4});
    const auto output_words           = std::size_t(H) * n;
    constexpr std::size_t state_words = 3 * C;
    GuardedDeviceBuffer q(output_words * 2), k(output_words * 2), v(output_words * 2);
    for (auto* buffer : {&q, &k, &v}) buffer->fill(0xff);
    const auto bytes = ops::kda_input_proj_conv_update_ragged_workspace_capacity_bytes(
        linear_profile(projection), n, n, rows, rows);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena ws(scratch.span());
    auto& context = cuda_environment().context;
    CudaStream stream(context);
    const auto execution = context.execution_view(stream.stream());
    const auto invoke    = [&] {
        ops::kda_input_proj_conv_update(x, projection, w, {src, dst}, ragged.view,
                                           {Tensor(q.span(), DType::BF16, {H, n}),
                                         Tensor(k.span(), DType::BF16, {H, n}),
                                         Tensor(v.span(), DType::BF16, {H, n})},
                                           ws, execution);
    };
    if (graph) {
        const auto reverse_slots = [&](auto selected) {
            for (auto& s : selected) s = slots - 1 - s;
            return records.record_offsets(selected);
        };
        source_offsets.copy_from_host(reverse_slots(source).data(), source_offsets.size());
        dest_offsets.copy_from_host(reverse_slots(destination).data(), dest_offsets.size());
        DecodeGraphDefinition definition;
        definition.capture(execution.stream(), invoke);
        DecodeGraphExecutable executable;
        executable.instantiate(definition);
        source_offsets.copy_from_host(so.data(), source_offsets.size());
        dest_offsets.copy_from_host(ds.data(), dest_offsets.size());
        executable.launch(execution.stream());
        cuda_synchronize();
    } else {
        invoke();
    }
    records.copy_back();

    const std::array got{from_device_bf16(q.data(), output_words),
                         from_device_bf16(k.data(), output_words),
                         from_device_bf16(v.data(), output_words)};
    const auto state = from_device_bf16(history.data(), initial.size());
    std::vector<double> reference, actual, projected, stored;
    const auto sampled = input_projection::sampled_rows(C, n > 512 ? 13 : 31);
    int failures       = 0;
    for (int s = 0; s < rows; ++s) {
        const int start = ragged.prefix[s], length = ragged.lengths[s];
        const auto historic = [&](int pos, int c) {
            return initial[(3LL * source[s] + pos + 3) * C + c];
        };
        for (int j = 0; j < 3 - length; ++j)
            for (int c = 0; c < C; ++c)
                failures +=
                    state[(3LL * destination[s] + j) * C + c] != historic(length - 3 + j, c);
        for (int c : sampled) {
            std::vector<double> dot(length);
            for (int t = 0; t < length; ++t)
                dot[t] = quantized_weight::dot_fp64(weight.host, c,
                                                    xh.data() + std::size_t(start + t) * H, H);
            for (int t = 0; t < length; ++t) {
                double sum = 0;
                for (int j = 0; j < 4; ++j) {
                    const int index = t + j - 3;
                    sum += wh[j * C + c] * (index < 0 ? historic(index, c) : dot[index]);
                }
                reference.push_back(sum / (1.0 + std::exp(-sum)));
                actual.push_back(got[c / H][std::size_t(start + t) * H + c % H]);
            }
            for (int j = std::max(0, 3 - length); j < 3; ++j) {
                projected.push_back(dot[length - 3 + j]);
                stored.push_back(state[(3LL * destination[s] + j) * C + c]);
            }
        }
    }
    for (int slot = 0; slot < slots; ++slot)
        if (std::find(destination.begin(), destination.end(), slot) == destination.end())
            for (std::size_t i = slot * state_words; i < (slot + 1) * state_words; ++i)
                failures += state[i] != initial[i];

    if (precision) {
        const auto silu_bf16 = [](double x) {
            return bf16_to_f32(f32_to_bf16(x / (1.0 + std::exp(-x))));
        };
        for (const auto& plane : got)
            for (int column : {5, 127, 128, 129, 130}) {
                const double residue = (column == 128 || column == 130) ? -1.0 / 256 : 1.0 / 256;
                if (plane[std::size_t(column) * H] != silu_bf16(residue))
                    throw std::runtime_error("ragged projection rounded before convolution");
            }
    }
    const auto criterion = policy == ops::LinearPolicy::AllowA8 ? a8_criterion : a16_criterion;
    failures += verify_reduction("ragged projected convolution", actual, reference, criterion);
    failures += verify_reduction("ragged projected history", stored, projected, criterion);
    for (const auto* buffer : {&q, &k, &v, &scratch})
        failures += buffer->verify_guards("ragged KDA input");
    return failures + records.verify_guards("ragged KDA histories") +
           (ws.peak_used() != bytes || ws.used() != 0);
}

int ragged_capacity_intervals() {
    using namespace ops;
    constexpr auto query = kda_input_proj_conv_update_ragged_workspace_capacity_bytes;
    for (const auto profile :
         {LinearProfile{QType::Q8_G32_FP16, C, H, LinearPolicy::A16Only, QuantLayout::RowSplit},
          LinearProfile{QType::MXFP8, C, H, LinearPolicy::A16Only, QuantLayout::RowGroupScale},
          LinearProfile{QType::MXFP8, C, H, LinearPolicy::AllowA8, QuantLayout::RowGroupScale}}) {
        for (const auto [first, last, min_rows, max_rows] : {std::array{1, 128, 1, 1},
                                                             {1, 64, 1, 1},
                                                             {1, 64, 2, 3},
                                                             {33, 64, 2, 3},
                                                             {1, 128, 1, 7},
                                                             {1, 64, 20, 80},
                                                             {33, 129, 3, 70}}) {
            std::size_t peak = 0;
            for (int n = first; n <= last; ++n)
                for (int s = min_rows; s <= std::min(n, max_rows); ++s)
                    peak = std::max(peak, query(profile, n, n, s, s));
            if (peak != query(profile, first, last, min_rows, max_rows))
                throw std::runtime_error("ragged KDA workspace is not the exact interval peak");
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    using enum ops::LinearPolicy;
    const auto ragged = [&](const char* name, std::vector<int> lengths, QType format,
                            ops::LinearPolicy policy, RaggedState state, bool graph = false,
                            bool precision = false) {
        suite.add(std::string("kda/input/ragged/") + name, [=] {
            return ragged_projection(lengths, format, policy, state, graph, precision);
        });
    };
    constexpr auto q8 = QType::Q8_G32_FP16, mx = QType::MXFP8;
    ragged("q8_staged", {5, 7, 11}, q8, A16Only, RaggedState::InPlace, true);
    suite.add("kda/input/ragged/q8_first_tiled_on_spark", [] {
        constexpr int tokens = 33, segments = 2;
        const ops::LinearProfile profile{q8, C, H, A16Only, QuantLayout::RowSplit};
        const auto bytes = ops::kda_input_proj_conv_update_ragged_workspace_capacity_bytes(
            profile, tokens, tokens, segments, segments);
        // One physical tile needs six FP32 edge columns and six BF16 tail columns on Spark.
        constexpr std::size_t expected = NINFER_TARGET_SPARK
                                             ? C * (6 * sizeof(float) + 6 * sizeof(std::uint16_t))
                                             : C * tokens * sizeof(float);
        if (bytes != expected)
            throw std::runtime_error("unexpected ragged Q8 workspace at the Spark crossover");
        return ragged_projection({1, 32}, q8, A16Only, RaggedState::Mixed);
    });
    ragged("q8_tiles", {120, 20}, q8, A16Only, RaggedState::Shared);
    ragged("q8_skew", {4095, 1}, q8, A16Only, RaggedState::Mixed);
    ragged("q8_no_projection_rounding", {126, 5}, q8, A16Only, RaggedState::Mixed, true, true);
    ragged("mxfp8_a16_single", {128}, mx, A16Only, RaggedState::InPlace);
    ragged("mxfp8_a16_tiles", {127, 1, 2, 130}, mx, A16Only, RaggedState::Mixed, true);
    ragged("mxfp8_a16_no_projection_rounding", {126, 5}, mx, A16Only, RaggedState::Mixed, false,
           true);
    ragged("mxfp8_a8_single_window", {1}, mx, AllowA8, RaggedState::InPlace);
    ragged("mxfp8_a8_skew", {1, 4095}, mx, AllowA8, RaggedState::InPlace);
    ragged("mxfp8_a8_many_short", std::vector<int>(64, 1), mx, AllowA8, RaggedState::Shared, true);
    ragged("mxfp8_a8_no_projection_rounding", {126, 5}, mx, AllowA8, RaggedState::Mixed, true,
           true);
    suite.add("kda/input/ragged/workspace_intervals", ragged_capacity_intervals);
    suite.add("kda/input/single", [] { return run_case(1, 1, true, false); });
    suite.add("kda/input/window", [] { return run_case(3, 8, false, true); });
    suite.add("kda/input/window_boundary", [] { return run_case(8, 4, true, false); });
    suite.add("kda/input/batch_boundary", [] { return run_case(7, 5, true, false); });
    suite.add("kda/input/batched_materialized", [] { return run_case(8, 8, false, true); });
    suite.add("kda/input/materialized", [] { return run_case(9, 2, true, false); });
    suite.add("kda/input/prefill", [] { return run_case(129, 1, false, true); });
    suite.add("kda/input/workspace_interval", [] { return run_case(32, 1, false, false, 4096); });
    suite.add("kda/input/expanded/update",
              [] {
                  return run_case(1, 17, true, true, 0, QType::Q8_G32_FP16,
                                  ops::LinearPolicy::A16Only, false, InactiveRows::Mixed);
              },
              {{"suite", "concurrency"}});
    suite.add("kda/input/expanded/window",
              [] {
                  return run_case(3, 9, false, true, 0, QType::Q8_G32_FP16,
                                  ops::LinearPolicy::A16Only, false, InactiveRows::Mixed);
              },
              {{"suite", "concurrency"}});
    suite.add("kda/input/expanded/materialized",
              [] {
                  return run_case(4, 64, false, false, 0, QType::Q8_G32_FP16,
                                  ops::LinearPolicy::A16Only, false, InactiveRows::Mixed);
              },
              {{"suite", "concurrency"}});
    suite.add("kda/input/expanded/all_inactive",
              [] {
                  return run_case(4, 9, true, true, 0, QType::Q8_G32_FP16,
                                  ops::LinearPolicy::A16Only, false, InactiveRows::All);
              },
              {{"suite", "concurrency"}});
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
    suite.add("kda/input/mxfp8/workspace_boundaries",
              [] {
                  for (const auto policy :
                       {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
                      const ops::LinearProfile profile{QType::MXFP8, C, H, policy,
                                                       QuantLayout::RowGroupScale};
                      for (int batch : {1, 8, 64})
                          for (int last : {4, 8, 16, 128})
                              for (const auto query :
                                   {ops::kda_input_proj_conv_update_workspace_capacity_bytes,
                                    ops::kda_input_proj_conv_record_workspace_capacity_bytes}) {
                                  const auto capacity = query(profile, batch, 1, last);
                                  for (int width = 1; width <= last; ++width)
                                      if (query(profile, batch, width, width) > capacity)
                                          throw std::runtime_error(
                                              "KDA workspace interval omits a width");
                              }
                  }
                  return 0;
              },
              {{"suite", "concurrency"}});
    suite.add("kda/input/mxfp8/workspace_interval", [] {
        return run_case(32, 1, false, false, 4096, QType::MXFP8, ops::LinearPolicy::AllowA8);
    });
    suite.add("kda/input/mxfp8/batched_workspace_interval", [] {
        return run_case(9, 8, true, true, 9, QType::MXFP8, ops::LinearPolicy::AllowA8);
    });
    suite.add("kda/input/mxfp8/no_projection_rounding", [] {
        return run_case(129, 1, true, false, 0, QType::MXFP8, ops::LinearPolicy::A16Only, true);
    });
    for (const auto [width, batch] : {std::pair{1, 64}, {3, 17}, {6, 9}, {4, 17}, {4, 64}})
        suite.add("kda/input/mxfp8/expanded/t" + std::to_string(width) + "b" +
                      std::to_string(batch),
                  [=] {
                      return run_case(width, batch, true, true, 0, QType::MXFP8,
                                      ops::LinearPolicy::AllowA8, false, InactiveRows::Mixed);
                  },
                  {{"suite", "concurrency"}});
    suite.add("kda/input/mxfp8/expanded/whole_window",
              [] {
                  return run_case(128, 2, true, true, 0, QType::MXFP8, ops::LinearPolicy::AllowA8,
                                  false, InactiveRows::Mixed);
              },
              {{"suite", "concurrency"}});
    suite.add("kda/input/mxfp8/expanded/a16",
              [] {
                  return run_case(4, 64, false, true, 0, QType::MXFP8, ops::LinearPolicy::A16Only,
                                  false, InactiveRows::Mixed);
              },
              {{"suite", "concurrency"}});
    return suite.main(argc, argv);
}
