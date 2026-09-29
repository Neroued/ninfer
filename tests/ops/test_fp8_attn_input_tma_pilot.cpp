// PILOT: the TMA activation-staging FP8 variant must be bit-identical to the
// production Prefill route: TMA changes only how activation bytes arrive in
// shared memory, never the values the MMA pipeline consumes.
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

int compare_outputs(const char* label, const GuardedBf16Tensor& base,
                    const GuardedBf16Tensor& tma) {
    int failures = base.verify_guards(label);
    failures += tma.verify_guards(label);
    failures += base.verify_fully_written(label);
    failures += tma.verify_fully_written(label);
    const std::vector<std::uint16_t> host_base = base.bits();
    const std::vector<std::uint16_t> host_tma  = tma.bits();
    if (host_base.size() != host_tma.size()) {
        std::printf("%s: size mismatch %zu vs %zu\n", label, host_base.size(), host_tma.size());
        return failures + 1;
    }
    for (std::size_t i = 0; i < host_base.size(); ++i) {
        if (host_base[i] != host_tma[i]) {
            std::printf("%s: mismatch at %zu: base=%04x tma=%04x\n", label, i, host_base[i],
                        host_tma[i]);
            return failures + 1;
        }
    }
    return failures;
}

int check_clean_state(const char* label) {
    // A rejected launch must throw before any kernel runs: the quantizer faults on
    // misaligned input instead of throwing, so any CUDA error here means validation ran
    // after a launch and poisoned the context for every later free.
    if (cudaGetLastError() != cudaSuccess) {
        std::printf("%s: rejected launch left a CUDA error state\n", label);
        return 1;
    }
    return 0;
}

int compare_case(DevicePackedWeight& parent, std::int32_t tokens, std::uint32_t seed) {
    constexpr std::int32_t kHidden = 5120, kQRows = 6144, kKvRows = 1024;
    const std::vector<float> activation       = make_bf16_activation(kHidden, tokens, seed);
    const std::vector<std::uint16_t> act_bits = bf16_bits(activation);
    DeviceBuffer input                        = to_device(act_bits);
    DeviceBuffer codes(static_cast<std::size_t>(tokens) * kHidden);
    DeviceBuffer scales(static_cast<std::size_t>(tokens) * sizeof(float));
    const ops::detail::Fp8A8Workspace workspace{static_cast<std::uint8_t*>(codes.p),
                                                static_cast<float*>(scales.p)};
    Tensor x(input.p, DType::BF16, {kHidden, tokens});
    GuardedBf16Tensor query0(kQRows, tokens), gate0(kQRows, tokens), key0(kKvRows, tokens),
        value0(kKvRows, tokens);
    GuardedBf16Tensor query1(kQRows, tokens), gate1(kQRows, tokens), key1(kKvRows, tokens),
        value1(kKvRows, tokens);
    Tensor q0 = query0.tensor(), g0 = gate0.tensor(), k0 = key0.tensor(), v0 = value0.tensor();
    Tensor q1 = query1.tensor(), g1 = gate1.tensor(), k1 = key1.tensor(), v1 = value1.tensor();
    ops::detail::fp8_attn_input_a8_launch(x, parent.view(), q0, g0, k0, v0, workspace, nullptr);
    ops::detail::fp8_attn_input_a8_tma_pilot_launch(x, parent.view(), q1, g1, k1, v1, workspace,
                                                    nullptr);
    cuda_synchronize();
    int failures = 0;
    failures += compare_outputs("tma-pilot q", query0, query1);
    failures += compare_outputs("tma-pilot gate", gate0, gate1);
    failures += compare_outputs("tma-pilot k", key0, key1);
    failures += compare_outputs("tma-pilot v", value0, value1);
    failures += verify_preserved("tma-pilot x", input, act_bits);
    failures += parent.verify_preserved("tma-pilot parent");
    return failures;
}

int compare_graph_case(DevicePackedWeight& parent, std::int32_t tokens, std::uint32_t seed) {
    constexpr std::int32_t kHidden = 5120, kQRows = 6144, kKvRows = 1024;
    const std::vector<float> activation       = make_bf16_activation(kHidden, tokens, seed);
    const std::vector<std::uint16_t> act_bits = bf16_bits(activation);
    DeviceBuffer input                        = to_device(act_bits);
    DeviceBuffer codes(static_cast<std::size_t>(tokens) * kHidden);
    DeviceBuffer scales(static_cast<std::size_t>(tokens) * sizeof(float));
    const ops::detail::Fp8A8Workspace workspace{static_cast<std::uint8_t*>(codes.p),
                                                static_cast<float*>(scales.p)};
    Tensor x(input.p, DType::BF16, {kHidden, tokens});
    GuardedBf16Tensor query0(kQRows, tokens), gate0(kQRows, tokens), key0(kKvRows, tokens),
        value0(kKvRows, tokens);
    GuardedBf16Tensor query1(kQRows, tokens), gate1(kQRows, tokens), key1(kKvRows, tokens),
        value1(kKvRows, tokens);
    Tensor q0 = query0.tensor(), g0 = gate0.tensor(), k0 = key0.tensor(), v0 = value0.tensor();
    Tensor q1 = query1.tensor(), g1 = gate1.tensor(), k1 = key1.tensor(), v1 = value1.tensor();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph0;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ops::detail::fp8_attn_input_a8_launch(x, parent.view(), q0, g0, k0, v0, workspace, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph0));
    cudaGraphExec_t exec0;
    CUDA_CHECK(cudaGraphInstantiate(&exec0, graph0, nullptr, nullptr, 0));
    cudaGraph_t graph1;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ops::detail::fp8_attn_input_a8_tma_pilot_launch(x, parent.view(), q1, g1, k1, v1, workspace,
                                                    stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph1));
    cudaGraphExec_t exec1;
    CUDA_CHECK(cudaGraphInstantiate(&exec1, graph1, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec0, stream));
    CUDA_CHECK(cudaGraphLaunch(exec1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(exec0));
    CUDA_CHECK(cudaGraphExecDestroy(exec1));
    CUDA_CHECK(cudaGraphDestroy(graph0));
    CUDA_CHECK(cudaGraphDestroy(graph1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    int failures = 0;
    failures += compare_outputs("tma-pilot graph q", query0, query1);
    failures += compare_outputs("tma-pilot graph gate", gate0, gate1);
    failures += compare_outputs("tma-pilot graph k", key0, key1);
    failures += compare_outputs("tma-pilot graph v", value0, value1);
    return failures;
}

int expect_reject(DevicePackedWeight& parent, std::int32_t tokens, const char* label) {
    constexpr std::int32_t kHidden = 5120, kQRows = 6144, kKvRows = 1024;
    DeviceBuffer input(static_cast<std::size_t>(tokens) * kHidden * 2);
    DeviceBuffer codes(static_cast<std::size_t>(tokens) * kHidden);
    DeviceBuffer scales(static_cast<std::size_t>(tokens) * sizeof(float));
    const ops::detail::Fp8A8Workspace workspace{static_cast<std::uint8_t*>(codes.p),
                                                static_cast<float*>(scales.p)};
    Tensor x(input.p, DType::BF16, {kHidden, tokens});
    GuardedBf16Tensor query(kQRows, tokens), gate(kQRows, tokens), key(kKvRows, tokens),
        value(kKvRows, tokens);
    Tensor q = query.tensor(), g = gate.tensor(), k = key.tensor(), v = value.tensor();
    try {
        ops::detail::fp8_attn_input_a8_tma_pilot_launch(x, parent.view(), q, g, k, v, workspace,
                                                        nullptr);
    } catch (const std::invalid_argument&) { return check_clean_state(label); }
    std::printf("%s: T=%d was accepted\n", label, tokens);
    return 1;
}

int expect_reject_misaligned(DevicePackedWeight& parent) {
    constexpr std::int32_t kHidden = 5120, kQRows = 6144, kKvRows = 1024;
    constexpr std::int32_t kTokens = 192;
    const std::vector<float> activation = make_bf16_activation(kHidden, kTokens, 7U);
    DeviceBuffer input                  = to_device(bf16_bits(activation));
    DeviceBuffer codes(static_cast<std::size_t>(kTokens) * kHidden + 16);
    DeviceBuffer scales(static_cast<std::size_t>(kTokens) * sizeof(float));
    // Offset the view by one byte so the activation base is not 16-byte aligned.
    const ops::detail::Fp8A8Workspace workspace{static_cast<std::uint8_t*>(codes.p) + 1,
                                                static_cast<float*>(scales.p)};
    Tensor x(input.p, DType::BF16, {kHidden, kTokens});
    GuardedBf16Tensor query(kQRows, kTokens), gate(kQRows, kTokens), key(kKvRows, kTokens),
        value(kKvRows, kTokens);
    Tensor q = query.tensor(), g = gate.tensor(), k = key.tensor(), v = value.tensor();
    try {
        ops::detail::fp8_attn_input_a8_tma_pilot_launch(x, parent.view(), q, g, k, v, workspace,
                                                        nullptr);
    } catch (const std::invalid_argument&) { return check_clean_state("tma-pilot"); }
    std::printf("tma-pilot: misaligned activation was accepted\n");
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        for (std::uint32_t pseed : {349U, 937U}) {
            DevicePackedWeight parent(quantized_weight::make_patterned_weight(
                QType::FP8_E4M3FN_ROW_BF16, 14336, 5120, pseed));
            failures += compare_case(parent, 192, 101U ^ pseed);
            failures += compare_case(parent, 320, 103U ^ pseed);
            failures += compare_case(parent, 1024, 107U ^ pseed);
            failures += compare_case(parent, 2048, 109U ^ pseed);
            if (pseed == 349U) {
                // The eager launches above prime both kernels' shared-memory attributes
                // before capture below; attribute calls must not run inside capture.
                failures += compare_graph_case(parent, 1024, 113U);
                failures += expect_reject(parent, 200, "tma-pilot");
                failures += expect_reject(parent, 64, "tma-pilot");
                failures += expect_reject_misaligned(parent);
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " fp8 attn_input TMA pilot\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("tma pilot error: %s\n", error.what());
        return 1;
    }
}
