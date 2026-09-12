#include "ninfer/ops/prepare_ragged_prefix.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr std::array<int, 8> LaneOrder{7, 0, 4, 2, 6, 1, 5, 3};

int run_case(int rows, int width, int batch, bool padded, int mode, bool replay) {
    const std::int64_t column_stride = rows + (padded ? 8 : 0);
    const std::int64_t lane_stride   = column_stride * width + (padded ? 16 : 0);
    std::vector<std::uint16_t> source(lane_stride * 8);
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<std::uint16_t>((i * 40503) ^ (i >> 13) ^ (i >> 19));
    GuardedDeviceBuffer input(source.size() * 2), dl(batch * 4), ds(batch * 4), de(batch * 4),
        destination(static_cast<std::size_t>(rows) * width * batch * 2),
        positions(width * batch * 4), counts(batch * 4);
    Tensor x(input.data(), DType::BF16, {rows, width, 8});
    x.nb[1] = column_stride * 2;
    x.nb[2] = lane_stride * 2;
    x.nb[3] = lane_stride * 8 * 2;
    Tensor l(dl.data(), DType::I32, {batch}), s(ds.data(), DType::I32, {batch}),
        e(de.data(), DType::I32, {batch}),
        out(destination.data(), DType::BF16, {rows, width, batch}),
        pos(positions.data(), DType::I32, {width, batch}), n(counts.data(), DType::I32, {batch});
    DeviceContext device;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto launch = [&] {
        ops::prepare_ragged_prefix(x, l, s, mode == 0 ? s : e, out, pos, n, device.stream);
    };
    std::vector<int> lanes(batch), starts(batch), ends(batch), expected_counts(batch),
        expected_positions(width * batch);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows) * width * batch);
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase)
            for (auto& bits : source) bits ^= 0xa35c;
        std::fill(expected.begin(), expected.end(), 0);
        for (int b = 0; b < batch; ++b) {
            lanes[b]           = LaneOrder[(b + 3 * phase) % 8];
            starts[b]          = 1023 + 317 * b + 259001 * phase;
            const int pattern  = (b + width + phase) % 4;
            const int count    = mode == 0      ? 0
                                 : mode == 2    ? width
                                 : mode == 3    ? 1
                                 : pattern == 0 ? width
                                 : pattern == 1 ? std::max(width - 1, 0)
                                 : pattern == 2 ? 1
                                                : 0;
            ends[b]            = starts[b] + count;
            expected_counts[b] = count;
            for (int column = 0; column < width; ++column) {
                expected_positions[b * width + column] =
                    starts[b] + (column < count ? column : std::max(count - 1, 0));
                for (int d = 0; d < rows; ++d)
                    if (column < count)
                        expected[(static_cast<std::size_t>(b) * width + column) * rows + d] =
                            source[lanes[b] * lane_stride + column * column_stride + d];
            }
        }
        input.copy_from_host(source.data(), source.size() * 2);
        dl.copy_from_host(lanes.data(), batch * 4);
        ds.copy_from_host(starts.data(), batch * 4);
        de.copy_from_host(ends.data(), batch * 4);
        destination.fill(0xff);
        positions.fill(0xa7);
        counts.fill(0xa7);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(device.stream);
        else
            launch();
        cuda_synchronize(device.stream);
        const std::string label = "prepare_ragged_prefix D=" + std::to_string(rows) +
                                  " Wc=" + std::to_string(width) + " B=" + std::to_string(batch) +
                                  " mode=" + std::to_string(mode) +
                                  (padded ? " padded" : " dense") +
                                  (replay ? " graph " : " eager ") + std::to_string(phase);
        failures +=
            verify_exact(label.c_str(),
                         from_device<std::uint16_t>(destination.data(), expected.size()), expected);
        failures += verify_exact((label + " positions").c_str(),
                                 from_device<int>(positions.data(), expected_positions.size()),
                                 expected_positions);
        failures += verify_exact((label + " counts").c_str(),
                                 from_device<int>(counts.data(), batch), expected_counts);
        failures += verify_exact((label + " source unchanged").c_str(),
                                 from_device<std::uint16_t>(input.data(), source.size()), source);
        failures += verify_exact((label + " lanes unchanged").c_str(),
                                 from_device<int>(dl.data(), batch), lanes);
        failures += verify_exact((label + " starts unchanged").c_str(),
                                 from_device<int>(ds.data(), batch), starts);
        failures += verify_exact((label + " ends unchanged").c_str(),
                                 from_device<int>(de.data(), batch), ends);
        failures += input.verify_guards(label) + dl.verify_guards(label) + ds.verify_guards(label) +
                    de.verify_guards(label) + destination.verify_guards(label) +
                    positions.verify_guards(label) + counts.verify_guards(label);
    }
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        int failures = 0;
        for (int batch = 1; batch <= 8; ++batch)
            for (int width = 1; width <= 16; ++width)
                failures +=
                    run_case(25600, width, batch, (batch + width) % 2, 1, batch == 1 || batch == 8);
        for (int width : {1, 3, 8, 9, 16})
            for (int batch : {1, 8})
                for (int mode : {0, 2, 3})
                    failures += run_case(25600, width, batch, true, mode, true);
        for (int rows : {8, 24, 16384})
            for (int width : {1, 3, 16}) failures += run_case(rows, width, 8, true, 1, true);
        failures += run_case(25600, 17, 1, true, 1, true);
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " prepare_ragged_prefix\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "prepare_ragged_prefix: " << error.what() << '\n';
        return 1;
    }
}
