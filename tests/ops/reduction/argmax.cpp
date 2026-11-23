#include "ops/suite.h"
#include "ninfer/ops/argmax.h"
#include "cuda_fixture.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include <algorithm>
#include "ops/op_tester.h"

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::int32_t> argmax_oracle(const std::vector<std::uint16_t>& logits,
                                        std::int32_t physical_rows, std::int32_t tokens,
                                        std::int32_t valid_rows) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        std::int32_t best      = 0;
        float best_value       = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            const float value = bf16_to_f32(logits[base + row]);
            if (value > best_value) {
                best       = row;
                best_value = value;
            }
        }
        expected[static_cast<std::size_t>(token)] = best;
    }
    return expected;
}

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t tokens,
                                       std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            const float value  = -24.0f + static_cast<float>(mixed % 3072u) * (1.0f / 256.0f);
            logits[base + row] = f32_to_bf16(value);
        }

        std::int32_t first  = 17 + token * 7919;
        std::int32_t second = valid_rows - 1 - token * 65537;
        first %= valid_rows;
        second %= valid_rows;
        if (second < 0) { second += valid_rows; }
        if (first == second) { second = (second + 1) % valid_rows; }
        if (second < first) {
            const std::int32_t temporary = first;
            first                        = second;
            second                       = temporary;
        }
        logits[base + first] =
            f32_to_bf16(token % 3 == 0 ? -0.5f : 32.0f + static_cast<float>(token));
        logits[base + second] = logits[base + first];

        if (token == 0) {
            std::fill(logits.begin() + base, logits.begin() + base + valid_rows,
                      f32_to_bf16(-4.0f));
        }
        if (valid_rows < physical_rows) {
            logits[base + valid_rows]        = f32_to_bf16(32768.0f);
            logits[base + physical_rows - 1] = f32_to_bf16(65536.0f);
        }
    }
    return logits;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens,
             std::size_t offset = 0) {
    auto logits   = make_logits(physical_rows, tokens, valid_rows);
    auto expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t) + offset);
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    auto* logits_data = static_cast<std::byte*>(device_logits.data()) + offset;
    CUDA_CHECK(cudaMemcpy(logits_data, logits.data(), logits.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.span().subspan(offset), DType::BF16,
                         {physical_rows, tokens});
    Tensor output_tensor(device_output.span(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows,
                ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    if (physical_rows == 248320 && tokens == 128) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto stream_execution =
            ninfer::test::cuda_environment().context.execution_view(stream);
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::argmax(logits_tensor, output_tensor, valid_rows, stream_execution);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int t = 0; t < tokens; ++t)
            logits[static_cast<std::size_t>(t) * physical_rows + valid_rows - 1] =
                f32_to_bf16(8192.0f);
        CUDA_CHECK(cudaMemcpyAsync(logits_data, logits.data(),
                                   logits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);
    }

    const auto actual =
        from_device<std::int32_t>(device_output.data(), static_cast<std::size_t>(tokens));
    const auto logits_after = from_device<std::uint16_t>(logits_data, logits.size());
    const std::string label = "argmax rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);

    int failures = 0;
    failures += verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " preserves logits").c_str(), logits_after, logits);
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

int probability_case(int physical_rows, int valid_rows) {
    constexpr int columns = 32;
    auto& device          = cuda_environment().context;
    const auto cuda       = device.execution_view();
    auto logits           = make_logits(physical_rows, columns, valid_rows);
    // Force a cross-tile tie whose denominator would be lost by absolute FP32 LSE.
    const auto last = std::size_t(columns - 1) * physical_rows;
    logits[last] = logits[last + valid_rows - 1] = f32_to_bf16(0x1p120f);
    GuardedDeviceBuffer input(logits.size() * 2 + 2), indices(columns * 4),
        probabilities(columns * 4);
    auto staging = input.span().subspan(2);
    Tensor x(staging, DType::BF16, {physical_rows, columns});
    Tensor ids(indices.span(), DType::I32, {columns});
    Tensor p(probabilities.span(), DType::FP32, {columns});
    const auto bytes = ops::argmax_with_probability_workspace_capacity_bytes(valid_rows, columns);
    GuardedDeviceBuffer storage(std::max<std::size_t>(bytes, 1));
    MemoryArena workspace(storage.span());
    DecodeGraphDefinition definition;
    definition.capture(cuda.stream(), [&] {
        ops::argmax_with_probability(x, ids, p, valid_rows, workspace, cuda);
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);

    int failures = 0;
    for (int phase = 0; phase < 2; ++phase) {
        if (phase) {
            logits[last]           = f32_to_bf16(-0x1p120f);
            logits[valid_rows - 1] = f32_to_bf16(-3.0f);
        }
        CUDA_CHECK(cudaMemcpy(x.data, logits.data(), logits.size() * 2, cudaMemcpyHostToDevice));
        const auto expected = argmax_oracle(logits, physical_rows, columns, valid_rows);
        std::vector<double> reference(columns);
        for (int t = 0; t < columns; ++t) {
            const auto base      = std::size_t(t) * physical_rows;
            const double maximum = bf16_to_f32(logits[base + expected[t]]);
            double sum           = 0.0;
            for (int v = 0; v < valid_rows; ++v)
                sum += std::exp(double(bf16_to_f32(logits[base + v])) - maximum);
            reference[t] = 1.0 / sum;
        }
        auto check = [&](int t) {
            device.synchronize();
            const auto actual = from_device<float>(probabilities.data(), t);
            failures += verify_exact("argmax probability IDs", from_device<int>(indices.data(), t),
                                     std::vector<int>(expected.begin(), expected.begin() + t));
            for (int i = 0; i < t; ++i)
                if (!(actual[i] > 0.0f && actual[i] <= 1.0f &&
                      std::abs(actual[i] - reference[i]) <= 3e-6 * reference[i])) {
                    std::cerr << "argmax confidence at " << i << " is " << actual[i]
                              << ", expected " << reference[i] << '\n';
                    ++failures;
                }
        };
        graph.launch(cuda.stream());
        check(columns);
        if (phase == 0) {
            for (int t = 1; t < columns; ++t) {
                auto short_ids = ids.slice(0, 0, t), short_p = p.slice(0, 0, t);
                ops::argmax_with_probability(x.slice(1, 0, t), short_ids, short_p, valid_rows,
                                             workspace, cuda);
                check(t);
            }
        }
    }
    auto empty_x = x, empty_ids = ids, empty_p = p;
    empty_x.ne[1] = empty_ids.ne[0] = empty_p.ne[0] = 0;
    ops::argmax_with_probability(empty_x, empty_ids, empty_p, valid_rows, workspace, cuda);
    failures += workspace.used() != 0 || workspace.peak_used() != bytes;
    return failures + input.verify_guards("argmax probability logits") +
           indices.verify_guards("argmax probability IDs") +
           probabilities.verify_guards("argmax probabilities") +
           storage.verify_guards("argmax probability workspace");
}

} // namespace

void add_argmax_cases(ninfer::test::OpSuite& suite) {
    for (auto [physical, valid] :
         {std::pair{509, 509}, {4096, 4096}, {131072, 131072}, {248320, 248077}})
        suite.add("argmax/probability/v" + std::to_string(valid),
                  [=] { return probability_case(physical, valid); }, {{"execution", "graph"}});
    suite.add("argmax/unaligned/full", [] { return run_case(248320, 248077, 3, 2); });
    for (int tokens : {1, 2, 8, 9, 64, 15, 128})
        suite.add("argmax/padded/t" + std::to_string(tokens),
                  [=] { return run_case(248320, 248077, tokens); },
                  {{"execution", tokens == 128 ? "graph" : "eager"}});
    for (int tokens : {1, 15, 120})
        suite.add("argmax/dense/t" + std::to_string(tokens),
                  [=] { return run_case(131072, 131072, tokens); });
}
