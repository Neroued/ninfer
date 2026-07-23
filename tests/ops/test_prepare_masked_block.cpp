#include "ninfer/ops/prepare_masked_block.h"

#include "ops/launcher/prepare_masked_block.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kMaskId = 248077;

std::vector<std::int32_t> read_guarded(const GuardedDBuf& buffer, int count) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(count));
    buffer.copy_to_host(host.data(), host.size() * sizeof(std::int32_t));
    return host;
}

int verify_case(int block_size, std::int32_t length_value,
                ops::detail::PrepareMaskedBlockRoute route, bool force_route) {
    const std::vector<std::int32_t> host_anchor{9173 + block_size};
    const std::vector<std::int32_t> host_length{length_value};
    DBuf anchor = to_device(host_anchor);
    DBuf length = to_device(host_length);
    GuardedDBuf ids(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    GuardedDBuf positions(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    ids.fill(0xcd);
    positions.fill(0xef);

    Tensor anchor_tensor(anchor.p, DType::I32, {1});
    Tensor length_tensor(length.p, DType::I32, {1});
    Tensor ids_tensor(ids.data(), DType::I32, {block_size});
    Tensor positions_tensor(positions.data(), DType::I32, {block_size});
    if (force_route) {
        auto plan  = ops::detail::prepare_masked_block_resolve_plan(block_size);
        plan.route = route;
        ops::detail::prepare_masked_block_launch(anchor_tensor, length_tensor, kMaskId, ids_tensor,
                                                 positions_tensor, plan, nullptr);
    } else {
        ops::prepare_masked_block(anchor_tensor, length_tensor, kMaskId, ids_tensor,
                                  positions_tensor, nullptr);
    }
    cuda_synchronize();

    std::vector<std::int32_t> expected_ids(static_cast<std::size_t>(block_size), kMaskId);
    std::vector<std::int32_t> expected_positions(static_cast<std::size_t>(block_size));
    expected_ids[0] = host_anchor[0];
    for (int i = 0; i < block_size; ++i) {
        expected_positions[static_cast<std::size_t>(i)] = length_value + i;
    }

    const std::string label =
        "prepare masked B=" + std::to_string(block_size) +
        (force_route ? " " + std::string(ops::detail::prepare_masked_block_route_name(route))
                     : " production");
    int failures =
        verify_exact((label + " ids").c_str(), read_guarded(ids, block_size), expected_ids);
    failures += verify_exact((label + " positions").c_str(), read_guarded(positions, block_size),
                             expected_positions);
    failures += verify_exact((label + " anchor unchanged").c_str(),
                             from_device<std::int32_t>(anchor, host_anchor.size()), host_anchor);
    failures += verify_exact((label + " length unchanged").c_str(),
                             from_device<std::int32_t>(length, host_length.size()), host_length);
    failures += ids.verify_guards((label + " ids guards").c_str());
    failures += positions.verify_guards((label + " positions guards").c_str());
    return failures;
}

int graph_case(int block_size) {
    std::vector<std::int32_t> host_anchor{0};
    std::vector<std::int32_t> host_length{0};
    DBuf anchor = to_device(host_anchor);
    DBuf length = to_device(host_length);
    GuardedDBuf ids(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    GuardedDBuf positions(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    Tensor anchor_tensor(anchor.p, DType::I32, {1});
    Tensor length_tensor(length.p, DType::I32, {1});
    Tensor ids_tensor(ids.data(), DType::I32, {block_size});
    Tensor positions_tensor(positions.data(), DType::I32, {block_size});

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate prepare masked graph");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCapture prepare masked");
    ops::prepare_masked_block(anchor_tensor, length_tensor, kMaskId, ids_tensor, positions_tensor,
                              stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture prepare masked");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "cudaGraphInstantiate prepare masked");

    const std::array<std::int32_t, 2> anchors{41, 123456};
    const std::array<std::int32_t, 2> lengths{7, std::numeric_limits<std::int32_t>::max() -
                                                     (block_size - 1)};
    int failures = 0;
    for (std::size_t replay = 0; replay < anchors.size(); ++replay) {
        host_anchor[0] = anchors[replay];
        host_length[0] = lengths[replay];
        cuda_check(cudaMemcpyAsync(anchor.p, host_anchor.data(), sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "copy prepare masked graph anchor");
        cuda_check(cudaMemcpyAsync(length.p, host_length.data(), sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   "copy prepare masked graph length");
        cuda_check(cudaMemsetAsync(ids.data(), 0xcd, ids.bytes(), stream),
                   "poison prepare masked graph ids");
        cuda_check(cudaMemsetAsync(positions.data(), 0xef, positions.bytes(), stream),
                   "poison prepare masked graph positions");
        cuda_check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch prepare masked");
        cuda_synchronize(stream);

        std::vector<std::int32_t> expected_ids(static_cast<std::size_t>(block_size), kMaskId);
        std::vector<std::int32_t> expected_positions(static_cast<std::size_t>(block_size));
        expected_ids[0] = anchors[replay];
        for (int i = 0; i < block_size; ++i) {
            expected_positions[static_cast<std::size_t>(i)] = lengths[replay] + i;
        }
        const std::string label = "prepare masked graph B=" + std::to_string(block_size) +
                                  " replay=" + std::to_string(replay);
        failures +=
            verify_exact((label + " ids").c_str(), read_guarded(ids, block_size), expected_ids);
        failures += verify_exact((label + " positions").c_str(),
                                 read_guarded(positions, block_size), expected_positions);
    }
    failures += ids.verify_guards("prepare masked graph ids guards");
    failures += positions.verify_guards("prepare masked graph positions guards");

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return failures;
}

template <class Call>
int expect_invalid_argument(const char* label, Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": expected std::invalid_argument\n";
    return 1;
}

int validation_cases() {
    constexpr int block_size = 4;
    DBuf anchor              = to_device<std::int32_t>({13});
    DBuf length              = to_device<std::int32_t>({29});
    DBuf shared(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    DBuf output(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    Tensor anchor_tensor(anchor.p, DType::I32, {1});
    Tensor length_tensor(length.p, DType::I32, {1});
    Tensor shared_tensor(shared.p, DType::I32, {block_size});
    Tensor output_tensor(output.p, DType::I32, {block_size});

    int failures = expect_invalid_argument("negative mask", [&] {
        ops::prepare_masked_block(anchor_tensor, length_tensor, -1, shared_tensor, output_tensor,
                                  nullptr);
    });
    failures += expect_invalid_argument("overlapping outputs", [&] {
        ops::prepare_masked_block(anchor_tensor, length_tensor, kMaskId, shared_tensor,
                                  shared_tensor, nullptr);
    });

    Tensor aliased_anchor(shared.p, DType::I32, {1});
    failures += expect_invalid_argument("overlapping input and output", [&] {
        ops::prepare_masked_block(aliased_anchor, length_tensor, kMaskId, shared_tensor,
                                  output_tensor, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "prepare_masked_block: SKIP (CUDA unavailable)\n";
        return 0;
    }

    int failures = 0;
    constexpr std::array routes{
        ops::detail::PrepareMaskedBlockRoute::Warp32,
        ops::detail::PrepareMaskedBlockRoute::Block64,
        ops::detail::PrepareMaskedBlockRoute::Block128,
        ops::detail::PrepareMaskedBlockRoute::Block256,
    };
    for (int block_size = 2; block_size <= 16; ++block_size) {
        failures += verify_case(block_size, 37, routes[0], false);
        failures +=
            verify_case(block_size, std::numeric_limits<std::int32_t>::max() - (block_size - 1),
                        routes[0], false);
        for (const auto route : routes) { failures += verify_case(block_size, 101, route, true); }
    }
    failures += graph_case(2);
    failures += graph_case(16);
    failures += validation_cases();

    if (failures != 0) {
        std::cerr << "prepare_masked_block failures=" << failures << '\n';
        return 1;
    }
    std::cout << "prepare_masked_block: PASS\n";
    return 0;
}
