#include "support.h"
#include "ops/state_fixture.h"
#include "core/weight.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/device.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;

std::vector<std::uint16_t> make_bf16_bits(std::size_t elements, std::uint32_t seed, float low,
                                          float high) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return bf16_bits(values);
}

int verify_equal(std::string_view label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin())) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int verify_zero_tail(std::string_view label, const std::vector<std::uint16_t>& values,
                     std::int32_t rows, std::int32_t width, std::int32_t batch,
                     const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = valid_columns[static_cast<std::size_t>(batch_row)]; token < width;
             ++token) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + token) * rows;
            for (std::int32_t row = 0; row < rows; ++row) {
                if (values[base + row] != 0) {
                    std::cerr << label << ": invalid tail is not exact zero\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_conv_record(std::string_view label, const std::vector<std::uint16_t>& snapshot_state,
                       const std::vector<std::uint16_t>& record, std::int32_t channels,
                       std::int32_t width, std::int32_t batch,
                       const std::vector<std::int32_t>& valid_columns,
                       const std::vector<std::int32_t>& snapshot_bases) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = 0; token < valid_columns[static_cast<std::size_t>(batch_row)];
             ++token) {
            const std::size_t snapshot =
                static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(batch_row)] +
                                         token) *
                    slot_stride +
                2ULL * channels;
            const std::size_t record_column =
                static_cast<std::size_t>(batch_row * width + token) * channels;
            if (!std::equal(snapshot_state.begin() + static_cast<std::ptrdiff_t>(snapshot),
                            snapshot_state.begin() +
                                static_cast<std::ptrdiff_t>(snapshot + channels),
                            record.begin() + static_cast<std::ptrdiff_t>(record_column))) {
                std::cerr << label << ": conv record differs from snapshot newest column\n";
                return 1;
            }
        }
    }
    return 0;
}

template <class SnapshotLaunch, class RecordLaunch>
int run_case(std::string_view label, std::int32_t hidden, std::int32_t value_rows,
             std::int32_t z_rows, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::size_t snapshot_workspace_bytes,
             std::size_t record_workspace_bytes, SnapshotLaunch&& snapshot_launch,
             RecordLaunch&& record_launch, std::uint32_t seed) {
    const std::int32_t channels          = kQueryRows + kKeyRows + value_rows;
    const std::int32_t aggregate_columns = width * batch;
    const std::int32_t source_slots      = 8;
    const std::int32_t slots             = aggregate_columns + source_slots;
    const bool dense                     = valid_columns.empty();
    if (valid_columns.empty()) { valid_columns.assign(static_cast<std::size_t>(batch), width); }

    const std::vector<float> activation = make_bf16_activation(hidden, aggregate_columns, seed);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(channels) * 4, seed + 1, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before = make_bf16_bits(
        static_cast<std::size_t>(channels) * 3 * source_slots, seed + 2, -0.05F, 0.05F);

    std::vector<std::uint16_t> snapshot_before(static_cast<std::size_t>(channels) * 3 * slots);
    std::copy(state_before.begin(), state_before.end(),
              snapshot_before.begin() + static_cast<std::size_t>(channels) * 3 * aggregate_columns);
    std::vector<std::int32_t> snapshot_initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        snapshot_bases[static_cast<std::size_t>(batch_row)] = batch_row * width;
        // Read-only selectors may repeat; source capacity does not depend on W.
        initial_slots[static_cast<std::size_t>(batch_row)] =
            batch_row == 7 ? 7 : (batch_row * 3 + 7) % source_slots;
        snapshot_initial_slots[static_cast<std::size_t>(batch_row)] =
            aggregate_columns + initial_slots[static_cast<std::size_t>(batch_row)];
    }

    DeviceBuffer device_x           = to_device_bf16(activation);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer snapshot_state     = to_device(snapshot_before);
    DeviceBuffer record_state       = to_device(state_before);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    DeviceBuffer device_initial          = to_device(initial_slots);
    DeviceBuffer device_snapshot_initial = to_device(snapshot_initial_slots);
    DeviceBuffer device_snapshot         = to_device(snapshot_bases);

    GuardedBf16Tensor snapshot_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor snapshot_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor snapshot_value(value_rows, aggregate_columns);
    GuardedBf16Tensor snapshot_z(z_rows, aggregate_columns);
    GuardedBf16Tensor record_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor record_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor record_value(value_rows, aggregate_columns);
    GuardedBf16Tensor record_z(z_rows, aggregate_columns);
    GuardedBf16Tensor conv_record(channels, aggregate_columns);

    Tensor x(device_x.span(), DType::BF16, {hidden, width, batch});
    Tensor conv_weight(device_conv_weight.span(), DType::BF16, {channels, 4});
    Tensor snapshot_state_view(snapshot_state.span(), DType::BF16, {channels, 3, slots});
    SlottedStateFixture snapshot_history(snapshot_state_view, 2);
    snapshot_state_view = snapshot_history.tensor;
    Tensor record_state_view(record_state.span(), DType::BF16, {channels, 3, source_slots});
    SlottedStateFixture record_history(record_state_view, 2);
    record_state_view = record_history.tensor;
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.span(), DType::I32, {batch}); }
    Tensor initial(device_initial.span(), DType::I32, {batch});
    Tensor snapshot_initial(device_snapshot_initial.span(), DType::I32, {batch});
    Tensor snapshot_base(device_snapshot.span(), DType::I32, {batch});
    Tensor snapshot_q(snapshot_query.span(), DType::BF16, {kQueryRows, width, batch});
    Tensor snapshot_k(snapshot_key.span(), DType::BF16, {kKeyRows, width, batch});
    Tensor snapshot_v(snapshot_value.span(), DType::BF16, {value_rows, width, batch});
    Tensor snapshot_z_view(snapshot_z.span(), DType::BF16, {z_rows, width, batch});
    Tensor record_q(record_query.span(), DType::BF16, {kQueryRows, width, batch});
    Tensor record_k(record_key.span(), DType::BF16, {kKeyRows, width, batch});
    Tensor record_v(record_value.span(), DType::BF16, {value_rows, width, batch});
    Tensor record_z_view(record_z.span(), DType::BF16, {z_rows, width, batch});
    Tensor conv_record_view(conv_record.span(), DType::BF16, {channels, width, batch});

    ninfer::DeviceBuffer snapshot_workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        std::max<std::size_t>(256, snapshot_workspace_bytes));

    MemoryArena snapshot_workspace(snapshot_workspace_storage.span());
    ninfer::DeviceBuffer record_workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        std::max<std::size_t>(256, record_workspace_bytes));
    MemoryArena record_workspace(record_workspace_storage.span());
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    cuda_synchronize();
    const auto launch = [&] {
        snapshot_launch(x, conv_weight, snapshot_state_view, valid, snapshot_initial, snapshot_base,
                        snapshot_q, snapshot_k, snapshot_v, snapshot_z_view, snapshot_workspace,
                        stream_execution);
        record_launch(x, conv_weight, record_state_view, valid, initial, conv_record_view, record_q,
                      record_k, record_v, record_z_view, record_workspace, stream_execution);
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto activation_bits = bf16_bits(activation);
    if (width == 2 || width == 9 || width == 10 || width == 16) {
        // 2 is the bottom of the Q5 parent's split4 band, 9 the count that used to be its top, 10
        // the count that is its top now, and 16 a grouped representative extent.
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // The same graph must consume changed inputs at the captured addresses.
        activation_bits = bf16_bits(make_bf16_activation(hidden, aggregate_columns, seed + 19));
        CUDA_CHECK(cudaMemcpyAsync(device_x.data(), activation_bits.data(),
                                   activation_bits.size() * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    int failures = 0;
    failures +=
        verify_equal(std::string(label) + " query", snapshot_query.bits(), record_query.bits());
    failures += verify_equal(std::string(label) + " key", snapshot_key.bits(), record_key.bits());
    failures +=
        verify_equal(std::string(label) + " value", snapshot_value.bits(), record_value.bits());
    failures += verify_equal(std::string(label) + " z", snapshot_z.bits(), record_z.bits());
    failures += verify_zero_tail(std::string(label) + " query", record_query.bits(), kQueryRows,
                                 width, batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " key", record_key.bits(), kKeyRows, width,
                                 batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " value", record_value.bits(), value_rows,
                                 width, batch, valid_columns);

    snapshot_history.copy_back();
    record_history.copy_back();
    failures += snapshot_history.verify_guards("snapshot history");
    failures += record_history.verify_guards("record history");
    const std::vector<std::uint16_t> snapshot_state_after =
        from_device<std::uint16_t>(snapshot_state, snapshot_before.size());
    const std::vector<std::uint16_t> record_state_after =
        from_device<std::uint16_t>(record_state, state_before.size());
    failures += verify_conv_record(label, snapshot_state_after, conv_record.bits(), channels, width,
                                   batch, valid_columns, snapshot_bases);
    failures +=
        verify_equal(std::string(label) + " source state", state_before, record_state_after);

    failures += verify_preserved(std::string(label) + " x", device_x, activation_bits);
    failures +=
        verify_preserved(std::string(label) + " conv weight", device_conv_weight, conv_weight_bits);
    failures += verify_preserved(std::string(label) + " initial", device_initial, initial_slots);
    if (!dense) {
        failures += verify_preserved(std::string(label) + " valid", device_valid, valid_columns);
    }
    failures += snapshot_query.verify_guards(std::string(label) + " snapshot query");
    failures += snapshot_key.verify_guards(std::string(label) + " snapshot key");
    failures += snapshot_value.verify_guards(std::string(label) + " snapshot value");
    failures += snapshot_z.verify_guards(std::string(label) + " snapshot z");
    failures += record_query.verify_guards(std::string(label) + " record query");
    failures += record_key.verify_guards(std::string(label) + " record key");
    failures += record_value.verify_guards(std::string(label) + " record value");
    failures += record_z.verify_guards(std::string(label) + " record z");
    failures += conv_record.verify_guards(std::string(label) + " conv record");
    if (snapshot_workspace.used() != 0 ||
        snapshot_workspace.peak_used() != snapshot_workspace_bytes) {
        std::cerr << label << ": snapshot workspace query/execution mismatch\n";
        ++failures;
    }
    if (record_workspace.used() != 0 || record_workspace.peak_used() != record_workspace_bytes) {
        std::cerr << label << ": record workspace query/execution mismatch\n";
        ++failures;
    }
    return failures;
}

} // namespace

namespace ninfer::test::gdn_input_conv {
int run_record(Weights& weights, const RecordCase& c) {
    const auto p              = weights.profile;
    const auto projection     = weights.projection(c.policy);
    const auto snapshot_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        projection, c.batch, c.width, c.width);
    const auto record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        projection, c.batch, c.width, c.width);
    return run_case(
        p.name, p.hidden, p.value_rows(), p.value_rows(), c.width, c.batch, c.valid, snapshot_bytes,
        record_bytes,
        [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid,
            const Tensor& initial, const Tensor& bases, Tensor& q, Tensor& k, Tensor& v, Tensor& z,
            MemoryArena& arena, CudaExecutionView execution) {
            ops::gdn_input_proj_conv_snapshot(x, projection, conv, {state, initial, bases}, valid,
                                              {q, k, v, z}, arena, execution);
        },
        [&](const Tensor& x, const Tensor& conv, const Tensor& state, const Tensor& valid,
            const Tensor& initial, Tensor& record, Tensor& q, Tensor& k, Tensor& v, Tensor& z,
            MemoryArena& arena, CudaExecutionView execution) {
            ops::gdn_input_proj_conv_record(x, projection, conv, {state, initial}, valid,
                                            {q, k, v, z}, record, arena, execution);
        },
        c.seed);
}
} // namespace ninfer::test::gdn_input_conv
