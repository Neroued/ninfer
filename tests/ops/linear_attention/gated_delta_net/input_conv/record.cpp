#include "support.h"

#include "cuda_fixture.h"
#include "ops/state_fixture.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

int verify_projection(const gdn_input_conv::Weights& weights, const gdn_input_conv::RecordCase& c,
                      const std::vector<float>& x, const std::vector<std::uint16_t>& conv_weight,
                      const std::vector<std::uint16_t>& history, const std::vector<int>& slots,
                      const std::vector<int>& valid, const GuardedBf16Tensor& record,
                      const GuardedBf16Tensor& query, const GuardedBf16Tensor& key,
                      const GuardedBf16Tensor& value, const GuardedBf16Tensor& z) {
    const auto p         = weights.profile;
    const int value_rows = p.value_rows(), channels = 2 * p.qk_rows + value_rows;
    // Keep the original 7- and 32-point samples within a denser A4 grid
    // so normwise quantization error is measured across more channels.
    const int sample_count  = c.policy == ops::LinearPolicy::AllowA4 ? 187 : c.width <= 4 ? 32 : 7;
    const std::string label = std::string(p.name) + " T=" + std::to_string(c.width) +
                              " B=" + std::to_string(c.batch) + " " +
                              gdn_input_conv::policy_name(c.policy);
    const auto& criterion = gdn_input_conv::projection_conv_criterion(p, c.policy);
    const auto projected  = record.bits();
    const auto z_bits     = z.bits();
    std::vector<double> record_actual, record_expected, z_actual, z_expected;
    int failures     = 0;
    int channel_base = 0;
    for (const auto* part : {&query, &key, &value}) {
        const int rows    = part == &value ? value_rows : p.qk_rows;
        const auto actual = part->bits();
        std::vector<double> expected_values, actual_values;
        for (int b = 0; b < c.batch; ++b) {
            const auto column = std::size_t(b) * c.width;
            for (const int row : sampled_rows(rows, sample_count)) {
                const int channel = channel_base + row;
                const auto start  = std::size_t(slots[b]) * 3 * channels + channel;
                double h0         = bf16_to_f32(history[start]);
                double h1         = bf16_to_f32(history[start + channels]);
                double h2         = bf16_to_f32(history[start + 2 * channels]);
                for (int t = 0; t < valid[b]; ++t) {
                    const auto physical = column + t;
                    const double latest = weights.project(channel, x.data() + physical * p.hidden);
                    const double convolution =
                        bf16_to_f32(conv_weight[channel]) * h0 +
                        bf16_to_f32(conv_weight[channel + channels]) * h1 +
                        bf16_to_f32(conv_weight[channel + 2 * channels]) * h2 +
                        bf16_to_f32(conv_weight[channel + 3 * channels]) * latest;
                    expected_values.push_back(convolution / (1.0 + std::exp(-convolution)));
                    actual_values.push_back(bf16_to_f32(actual[physical * rows + row]));
                    record_expected.push_back(latest);
                    record_actual.push_back(bf16_to_f32(projected[physical * channels + channel]));
                    h0 = h1;
                    h1 = h2;
                    h2 = latest;
                }
            }
            for (int t = valid[b]; t < c.width; ++t) {
                const auto first = actual.begin() + (column + t) * rows;
                if (!std::all_of(first, first + rows, [](auto word) { return word == 0; })) {
                    std::cerr << p.name << " invalid Q/K/V tail differs from BF16 +0\n";
                    return failures + 1;
                }
            }
        }
        if (!expected_values.empty())
            failures += verify_reduction(label + (part == &query ? " query"
                                                  : part == &key ? " key"
                                                                 : " value"),
                                         actual_values, expected_values, criterion);
        channel_base += rows;
    }
    for (int b = 0; b < c.batch; ++b) {
        for (int t = 0; t < c.width; ++t) {
            const auto column = std::size_t(b) * c.width + t;
            for (const auto row : sampled_rows(value_rows, sample_count)) {
                z_expected.push_back(weights.project(channels + row, x.data() + column * p.hidden));
                z_actual.push_back(bf16_to_f32(z_bits[column * value_rows + row]));
            }
            if (t < valid[b]) {
                const auto first = projected.begin() + column * channels;
                if (!std::all_of(first, first + channels, [](std::uint16_t word) {
                        return std::isfinite(bf16_to_f32(word));
                    })) {
                    std::cerr << p.name << " non-finite valid convolution record\n";
                    return failures + 1;
                }
            }
        }
    }
    if (!record_expected.empty())
        failures += verify_reduction(label + " projected history", record_actual, record_expected,
                                     criterion);
    failures += verify_reduction(label + " z", z_actual, z_expected, criterion);
    return failures;
}

} // namespace

namespace ninfer::test::gdn_input_conv {

int run_record(Weights& weights, const RecordCase& c) {
    const auto p          = weights.profile;
    const auto projection = weights.projection(c.policy);
    const int width = c.width, batch = c.batch, columns = width * batch;
    const int value_rows = p.value_rows(), channels = 2 * p.qk_rows + value_rows;
    const bool dense           = c.valid.empty();
    auto valid_columns         = dense ? std::vector<int>(batch, width) : c.valid;
    constexpr int source_slots = 8;
    auto activation            = make_bf16_activation(p.hidden, columns, c.seed);
    auto conv_values           = std::vector<float>(4 * channels);
    fill_uniform(conv_values, c.seed + 1, -.02F, .02F);
    auto state_values = std::vector<float>(std::size_t(channels) * 3 * source_slots);
    fill_uniform(state_values, c.seed + 2, -.05F, .05F);
    const auto conv_weight_bits = bf16_bits(conv_values);
    const auto state_before     = bf16_bits(state_values);
    auto device_x               = to_device_bf16(activation);
    auto device_conv_weight     = to_device(conv_weight_bits);
    auto state                  = to_device(state_before);
    auto device_valid           = dense ? DeviceBuffer{} : to_device(valid_columns);
    const Tensor x(device_x.span(), DType::BF16, {p.hidden, width, batch});
    const Tensor conv_weight(device_conv_weight.span(), DType::BF16, {channels, 4});
    const Tensor history_tensor(state.span(), DType::BF16, {channels, 3, source_slots});
    RecordStateFixture history(history_tensor, 2);
    const Tensor valid = dense ? Tensor{} : Tensor(device_valid.span(), DType::I32, {batch});
    std::vector<int> slots(batch);
    for (int b = 0; b < batch; ++b) slots[b] = (3 * b + 7) % source_slots;
    const auto selectors = [&] {
        auto offsets = history.record_offsets(slots);
        for (int b = 0; b < batch; ++b)
            if (!valid_columns[b]) offsets[b] = -1;
        return offsets;
    };
    auto initial_offsets = selectors();
    auto device_initial  = to_device(initial_offsets);
    const Tensor initial(device_initial.span(), DType::I64, {batch});

    GuardedBf16Tensor query(p.qk_rows, columns), key(p.qk_rows, columns);
    GuardedBf16Tensor value(value_rows, columns), z(value_rows, columns), record(channels, columns);
    auto conv_record = record.tensor().view({channels, width, batch});
    const ops::GdnConvOutputs outputs{query.tensor().view({p.qk_rows, width, batch}),
                                      key.tensor().view({p.qk_rows, width, batch}),
                                      value.tensor().view({value_rows, width, batch}),
                                      z.tensor().view({value_rows, width, batch})};
    const auto capacity =
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(projection, batch, width, width);
    auto& context = cuda_environment().context;
    DeviceBuffer scratch(context.device(), std::max<std::size_t>(256, capacity));
    MemoryArena workspace(scratch.span());
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto execution = context.execution_view(stream);
    cuda_synchronize();
    const auto launch = [&] {
        ops::gdn_input_proj_conv_record(x, projection, conv_weight, history.view(initial), valid,
                                        outputs, conv_record, workspace, execution);
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto activation_bits = bf16_bits(activation);
    if (width == 2 || (width == 4 && batch == 8) || width == 9 || width == 10 || width == 16) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        activation      = make_bf16_activation(p.hidden, columns, c.seed + 19);
        activation_bits = bf16_bits(activation);
        CUDA_CHECK(cudaMemcpyAsync(device_x.data(), activation_bits.data(), x.bytes(),
                                   cudaMemcpyHostToDevice, stream));
        if (!dense) {
            if (valid_columns[0]) valid_columns[0] = 1;
            CUDA_CHECK(cudaMemcpyAsync(device_valid.data(), valid_columns.data(), valid.bytes(),
                                       cudaMemcpyHostToDevice, stream));
        }
        for (auto& slot : slots) slot = (slot + 1) % source_slots;
        initial_offsets = selectors();
        CUDA_CHECK(cudaMemcpyAsync(device_initial.data(), initial_offsets.data(), initial.bytes(),
                                   cudaMemcpyHostToDevice, stream));
        for (auto* tensor : {&query, &key, &value, &z, &record}) tensor->repaint(stream);
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    int failures = verify_projection(weights, c, activation, conv_weight_bits, state_before, slots,
                                     valid_columns, record, query, key, value, z);
    history.copy_back();
    failures += history.verify_guards("record history");
    failures += verify_preserved("record source history", state, state_before);
    failures += verify_preserved("record x", device_x, activation_bits);
    failures += verify_preserved("record conv weight", device_conv_weight, conv_weight_bits);
    failures += verify_preserved("record initial offsets", device_initial, initial_offsets);
    if (!dense) failures += verify_preserved("record valid columns", device_valid, valid_columns);
    for (const auto* tensor : {&query, &key, &value, &z, &record})
        failures += tensor->verify_guards("projection record");
    if (workspace.used() != 0 || workspace.peak_used() > capacity) {
        std::cerr << p.name << " record workspace query/execution mismatch\n";
        ++failures;
    }
    return failures;
}

} // namespace ninfer::test::gdn_input_conv
