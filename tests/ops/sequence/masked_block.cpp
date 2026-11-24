#include "ops/suite.h"
#include "core/device.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include <algorithm>
#include "ninfer/ops/sequence.h"
#include "ops/op_tester.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kMaskId = 248077;

struct MaskedBlockExpected {
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> positions;
};

MaskedBlockExpected masked_block_oracle(std::int32_t anchor, std::int32_t length, int block_size) {
    MaskedBlockExpected expected{
        .ids       = std::vector<std::int32_t>(static_cast<std::size_t>(block_size), kMaskId),
        .positions = std::vector<std::int32_t>(static_cast<std::size_t>(block_size)),
    };
    expected.ids[0] = anchor;
    for (int i = 0; i < block_size; ++i) {
        expected.positions[static_cast<std::size_t>(i)] = length + i;
    }
    return expected;
}

int run_case(int block_size, std::int32_t anchor_value, std::int32_t length_value) {
    const auto expected = masked_block_oracle(anchor_value, length_value, block_size);
    DeviceBuffer anchor = to_device<std::int32_t>({anchor_value});
    DeviceBuffer length = to_device<std::int32_t>({length_value});
    DeviceBuffer valid  = to_device<std::int32_t>({block_size});
    GuardedDeviceBuffer ids(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    GuardedDeviceBuffer positions(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    ids.fill(0xcd);
    positions.fill(0xef);

    Tensor anchor_tensor(anchor.span(), DType::I32, {1});
    Tensor length_tensor(length.span(), DType::I32, {1});
    Tensor valid_tensor(valid.span(), DType::I32, {1});
    Tensor ids_tensor(ids.span(), DType::I32, {block_size, 1});
    Tensor positions_tensor(positions.span(), DType::I32, {block_size, 1});
    ops::prepare_masked_block(anchor_tensor, length_tensor, valid_tensor, kMaskId, ids_tensor,
                              positions_tensor, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label = "prepare_masked_block W=" + std::to_string(block_size);
    int failures            = verify_exact(
        (label + " ids").c_str(),
        from_device<std::int32_t>(ids.data(), static_cast<std::size_t>(block_size)), expected.ids);
    failures += verify_exact(
        (label + " positions").c_str(),
        from_device<std::int32_t>(positions.data(), static_cast<std::size_t>(block_size)),
        expected.positions);
    failures += verify_exact((label + " anchor unchanged").c_str(),
                             from_device<std::int32_t>(anchor, 1), {anchor_value});
    failures += verify_exact((label + " length unchanged").c_str(),
                             from_device<std::int32_t>(length, 1), {length_value});
    failures += ids.verify_guards((label + " ids guards").c_str());
    failures += positions.verify_guards((label + " positions guards").c_str());
    return failures;
}

int run_batch_case(int width, int batch, bool dense) {
    constexpr int mask = 248070;
    std::vector<std::int32_t> anchors(batch), lengths(batch), valid(batch);
    for (int b = 0; b < batch; ++b) {
        anchors[b] = 9173 + 111 * b;
        lengths[b] = 37 + 1009 * b;
        valid[b]   = dense ? width : 1 + (3 * b) % width;
    }
    DeviceBuffer device_anchors = to_device(anchors);
    DeviceBuffer device_lengths = to_device(lengths);
    DeviceBuffer device_valid   = to_device(valid);
    GuardedDeviceBuffer ids(static_cast<std::size_t>(width * batch) * sizeof(std::int32_t));
    GuardedDeviceBuffer positions(ids.bytes());
    ids.fill(0xcd);
    positions.fill(0xef);
    Tensor a(device_anchors.span(), DType::I32, {batch});
    Tensor l(device_lengths.span(), DType::I32, {batch});
    Tensor v(device_valid.span(), DType::I32, {batch});
    Tensor out(ids.span(), DType::I32, {width, batch});
    Tensor pos(positions.span(), DType::I32, {width, batch});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    cuda_synchronize();
    const auto launch = [&] {
        ops::prepare_masked_block(a, l, v, mask, out, pos, stream_execution);
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (width == 16 && batch == 8) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int b = 0; b < batch; ++b) {
            anchors[b] += 11;
            lengths[b] += 7;
            valid[b] = 1 + valid[b] % width;
        }
        CUDA_CHECK(cudaMemcpyAsync(device_anchors.data(), anchors.data(), device_anchors.size(),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_lengths.data(), lengths.data(), device_lengths.size(),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_valid.data(), valid.data(), device_valid.size(),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::vector<std::int32_t> expected_ids(width * batch), expected_positions(width * batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < width; ++j) {
            expected_ids[b * width + j]       = j == 0 ? anchors[b] : mask;
            expected_positions[b * width + j] = lengths[b] + std::min(j, valid[b] - 1);
        }
    int failures =
        verify_exact("DFlash2 masked ids",
                     from_device<std::int32_t>(ids.data(), expected_ids.size()), expected_ids);
    failures += verify_exact("DFlash2 masked positions",
                             from_device<std::int32_t>(positions.data(), expected_positions.size()),
                             expected_positions);
    failures += verify_exact("anchors unchanged", from_device<std::int32_t>(device_anchors, batch),
                             anchors);
    failures += verify_exact("lengths unchanged", from_device<std::int32_t>(device_lengths, batch),
                             lengths);
    failures +=
        verify_exact("valid unchanged", from_device<std::int32_t>(device_valid, batch), valid);
    failures += ids.verify_guards("masked ids");
    failures += positions.verify_guards("masked positions");
    return failures;
}

} // namespace

void add_masked_block_cases(ninfer::test::OpSuite& suite) {
    for (int block_size : {2, 7, 16}) {
        suite.add("masked_block/w" + std::to_string(block_size),
                  [=] { return run_case(block_size, 9173 + block_size, 37); });
        suite.add("masked_block/position_limit/w" + std::to_string(block_size), [=] {
            return run_case(block_size, 100000 + block_size,
                            std::numeric_limits<std::int32_t>::max() - (block_size - 1));
        });
    }
    for (int width = 2; width <= 16; ++width)
        for (int batch : {1, 8})
            for (bool dense : {false, true})
                suite.add("masked_block/batched/w" + std::to_string(width) + "/b" +
                              std::to_string(batch) + (dense ? "/dense" : "/ragged"),
                          [=] { return run_batch_case(width, batch, dense); },
                          {{"execution", width == 16 && batch == 8 ? "graph" : "eager"}});
    suite.add("masked_block/batched/w7/b3/ragged", [] { return run_batch_case(7, 3, false); },
              {{"execution", "eager"}});
}
