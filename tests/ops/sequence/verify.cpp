#include "ops/suite.h"
#include "ninfer/ops/sequence.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include <string_view>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int prepare_verify_case(int k, int batch) {
    const int width = k + 1;
    std::vector<std::int32_t> anchors(batch), lengths(batch), extents(batch), drafts(k * batch);
    for (int b = 0; b < batch; ++b) {
        anchors[b] = 70000 + 11 * b;
        lengths[b] = 131072 + 1009 * b;
        for (int j = 0; j < k; ++j) drafts[b * k + j] = 37 + 257 * b + 7919 * j;
    }
    DeviceBuffer d_anchors = to_device(anchors);
    DeviceBuffer d_lengths = to_device(lengths);
    DeviceBuffer d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device(extents);
    GuardedDeviceBuffer d_full(width * batch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_ids(d_full.bytes()), d_positions(d_full.bytes());
    Tensor a(d_anchors.span(), DType::I32, {batch}), l(d_lengths.span(), DType::I32, {batch});
    Tensor d(d_drafts.span(), DType::I32, {k, batch}), e(d_extents.span(), DType::I32, {batch});
    Tensor full(d_full.span(), DType::I32, {width, batch}),
        ids(d_ids.span(), DType::I32, {width, batch});
    Tensor positions(d_positions.span(), DType::I32, {width, batch});
    DeviceContext context;
    cuda_synchronize();
    const auto launch = [&] {
        ops::prepare_verify_inputs(a, d, l, e, full, positions, context.execution_view());
        ops::prepare_verify_ids(a, d, e, ids, context.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (k == 15 && batch == 8) {
        definition.capture(context.stream(), launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase <= k; ++phase) {
        for (int b = 0; b < batch; ++b) extents[b] = (phase + 3 * b) % width;
        CUDA_CHECK(cudaMemcpyAsync(d_extents.data(), extents.data(), d_extents.size(),
                                   cudaMemcpyHostToDevice, context.stream()));
        if (graph.ready())
            graph.launch(context.stream());
        else
            launch();
        context.synchronize();
        std::vector<std::int32_t> expected_ids(width * batch), expected_positions(width * batch);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < width; ++j) {
                expected_ids[b * width + j] =
                    j > 0 && j <= extents[b] ? drafts[b * k + j - 1] : anchors[b];
                expected_positions[b * width + j] = lengths[b] + std::min(j, extents[b]);
            }
        failures +=
            verify_exact("verify ids", from_device<std::int32_t>(d_ids.data(), expected_ids.size()),
                         expected_ids);
        failures += verify_exact("verify full ids",
                                 from_device<std::int32_t>(d_full.data(), expected_ids.size()),
                                 expected_ids);
        failures +=
            verify_exact("verify positions",
                         from_device<std::int32_t>(d_positions.data(), expected_positions.size()),
                         expected_positions);
        failures +=
            verify_exact("extents unchanged", from_device<std::int32_t>(d_extents, batch), extents);
    }
    failures +=
        verify_exact("anchors unchanged", from_device<std::int32_t>(d_anchors, batch), anchors);
    failures +=
        verify_exact("lengths unchanged", from_device<std::int32_t>(d_lengths, batch), lengths);
    failures += verify_exact("drafts unchanged", from_device<std::int32_t>(d_drafts, drafts.size()),
                             drafts);
    failures += d_full.verify_guards("verify full ids");
    failures += d_ids.verify_guards("verify ids");
    failures += d_positions.verify_guards("verify positions");
    return failures;
}

} // namespace

void add_verify_cases(ninfer::test::OpSuite& suite) {
    for (int k = 1; k <= 15; ++k)
        for (int batch : {1, 8})
            suite.add("prepare_verify/k" + std::to_string(k) + "/b" + std::to_string(batch),
                      [=] { return prepare_verify_case(k, batch); },
                      {{"execution", k == 15 && batch == 8 ? "graph" : "eager"}});
}
