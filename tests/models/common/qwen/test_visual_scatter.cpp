#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "models/common/vision/execution.h"
#include "models/common/vision/resources.h"

#include "core/transfer.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    constexpr std::int32_t d = 5120;
    constexpr std::int32_t t = 4;
    constexpr std::int32_t v = 3;
    std::vector<float> token_embeddings(static_cast<std::size_t>(d) * t);
    std::vector<float> visual_embeddings(static_cast<std::size_t>(d) * v);
    for (int col = 0; col < t; ++col) {
        std::fill_n(token_embeddings.data() + static_cast<std::size_t>(col) * d, d,
                    static_cast<float>(10 + col));
    }
    for (int col = 0; col < v; ++col) {
        std::fill_n(visual_embeddings.data() + static_cast<std::size_t>(col) * d, d,
                    static_cast<float>(100 + col));
    }
    round_to_bf16(token_embeddings);
    round_to_bf16(visual_embeddings);
    DeviceBuffer dvisual = to_device_bf16(visual_embeddings);
    const auto cuda      = cuda_environment().default_execution;
    models::vision::Storage storage(
        cuda_environment().context,
        {.max_features = v, .output_width = d, .output_bytes = std::size_t(d) * v * 2});
    auto visual = storage.output(0, v);
    copy_bytes_async({visual.data, visual.bytes(), visual.device}, dvisual.span(), cuda);
    const std::vector<std::int32_t> scatter_indices{2, 4, 7};
    ninfer::DeviceBuffer work_storage(ninfer::test::cuda_environment().context.device(),
                                      1ULL << 20);
    MemoryArena work(work_storage.span());
    PinnedHostBuffer host_indices(v * sizeof(std::int32_t));

    auto run = [&](std::int32_t prompt_tokens, const char* label) {
        DeviceBuffer dinput = to_device_bf16(token_embeddings);
        Tensor input(dinput.span(), DType::BF16, {d, t});
        work.reset();
        ninfer::models::vision::WindowPlan plan;
        plan.outputs = {{0, 0, {}}};
        for (auto p : scatter_indices)
            if (p < prompt_tokens) plan.outputs[0].token_positions.push_back(p);
        auto indices    = work.alloc(DType::I32, {v});
        const auto view = models::vision::prepare_embedding_view(storage, plan, 1, t, indices,
                                                                 host_indices.span(), cuda);
        if (view)
            for (const auto& slice : view->slices)
                ops::scatter(slice.features, slice.columns, input, cuda);
        cudaDeviceSynchronize();

        std::vector<double> reference(token_embeddings.begin(), token_embeddings.end());
        for (int row = 0; row < d; ++row) {
            reference[static_cast<std::size_t>(1) * d + row] = visual_embeddings[row];
            if (prompt_tokens > 4) {
                reference[static_cast<std::size_t>(3) * d + row] =
                    visual_embeddings[static_cast<std::size_t>(d) + row];
            }
        }
        return verify_exact(label, from_device_bf16(dinput, reference.size()), reference);
    };

    int failures = 0;
    failures += run(8, "MTP shifted visual composition crosses chunk boundary");
    failures += run(4, "MTP shifted visual composition keeps bonus token embedding");
    {
        models::vision::WindowPlan plan;
        plan.outputs        = {{0, 0, {4}}, {1, 1, {2, 7}}};
        DeviceBuffer dinput = to_device_bf16(token_embeddings);
        Tensor input(dinput.span(), DType::BF16, {d, t});
        work.reset();
        auto indices    = work.alloc(DType::I32, {v});
        const auto view = models::vision::prepare_embedding_view(storage, plan, 2, t, indices,
                                                                 host_indices.span(), cuda);
        for (const auto& slice : view->slices)
            ops::scatter(slice.features, slice.columns, input, cuda);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> reference(token_embeddings.begin(), token_embeddings.end());
        for (int row = 0; row < d; ++row) {
            reference[row]                      = visual_embeddings[d + row];
            reference[std::size_t(2) * d + row] = visual_embeddings[row];
        }
        failures += verify_exact("rank-major visual composition preserves text gaps",
                                 from_device_bf16(dinput, reference.size()), reference);
    }
    std::cout << (failures ? "FAIL" : "OK") << " shifted visual composition\n";
    return failures ? 1 : 0;
}
