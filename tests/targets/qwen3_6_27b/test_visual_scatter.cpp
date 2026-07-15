#include "ops/op_tester.h"
#include "targets/qwen3_6_27b_rtx5090/impl/config.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/visual_scatter.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    constexpr std::int32_t d = targets::qwen3_6_27b_rtx5090::detail::TextConfig::hidden;
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
    DBuf dvisual = to_device_bf16(visual_embeddings);
    Tensor visual(dvisual.p, DType::BF16, {d, v});
    const std::vector<std::int32_t> scatter_indices{2, 4, 7};
    WorkspaceArena work(1ULL << 20);

    auto run = [&](std::int32_t prompt_tokens, const char* label) {
        DBuf dinput = to_device_bf16(token_embeddings);
        Tensor input(dinput.p, DType::BF16, {d, t});
        work.reset();
        targets::qwen3_6_27b_rtx5090::detail::schedule::detail::scatter_shifted_visual_embeddings(
            input, visual, scatter_indices, 1, prompt_tokens, work, nullptr);
        cudaDeviceSynchronize();

        std::vector<double> reference(token_embeddings.begin(), token_embeddings.end());
        for (int row = 0; row < d; ++row) {
            reference[static_cast<std::size_t>(1) * d + row] = visual_embeddings[row];
            if (prompt_tokens > 4) {
                reference[static_cast<std::size_t>(3) * d + row] =
                    visual_embeddings[static_cast<std::size_t>(d) + row];
            }
        }
        return verify(label, from_device_bf16(dinput, reference.size()), reference,
                      Tolerance::bf16_elementwise());
    };

    int failures = 0;
    failures += run(8, "MTP shifted visual composition crosses chunk boundary");
    failures += run(4, "MTP shifted visual composition keeps bonus token embedding");
    std::cout << (failures ? "FAIL" : "OK") << " shifted visual composition\n";
    return failures ? 1 : 0;
}
