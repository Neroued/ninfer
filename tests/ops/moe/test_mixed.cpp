#include "fixture.h"
#include "ninfer/ops/sum_contributions.h"

namespace ninfer::test::moe {

int test_mixed(Fixture& fixture, CudaExecutionView execution) {
    using ops::ExpertSubsetKind;
    using ops::MoeEpilogue;
    CpuTopology topology;
    CpuExecutor executor(topology, plan_cpu_team(topology.info(), 1));
    const auto features = executor.execution_view().features();
    if (!features.has(CpuFeature::Avx2) || !features.has(CpuFeature::Fma) ||
        !features.has(CpuFeature::F16c)) {
        std::cout << "SKIP: CPU/GPU composition requires AVX2/FMA/F16C\n";
        return 0;
    }
    HostBank native(fixture);
    const auto source = native.source();
    const ops::CpuSwiGLUExperts cpu_weights{source.gate_up, source.down};
    int failures = 0;
    for (int tokens : {1, 9, 72}) {
        auto batch = fixture.batch(tokens);
        DeviceBatch device_batch(batch);
        Tensor x(batch.input.data(), Device::cpu(), DType::BF16, {kHidden, tokens});
        ops::ExpertRoutes routes{
            Tensor(batch.ids.data(), Device::cpu(), DType::I32, {kTopK, tokens}),
            Tensor(batch.coefficients.data(), Device::cpu(), DType::FP32, {kTopK, tokens})};
        const auto& profile = fixture.profile();
        HostBuffer partial(std::size_t(kHidden) * tokens * 4);
        Tensor cpu_output(partial.span(), DType::FP32, {kHidden, tokens});
        HostBuffer scratch(ops::routed_swiglu_workspace_capacity_bytes(
            profile.routed, ExpertSubsetKind::MaskOnes, tokens, tokens));
        MemoryArena cpu_arena(scratch.span());
        GuardedDeviceBuffer gpu_scratch(ops::routed_swiglu_workspace_capacity_bytes(
            profile.routed, profile.shared, ExpertSubsetKind::MaskZeros, DType::FP32,
            MoeEpilogue::Write, tokens, tokens));
        MemoryArena gpu_arena(gpu_scratch.span());
        GuardedDeviceBuffer cpu_contribution(partial.size()), gpu_contribution(partial.size()),
            output(std::size_t(kHidden) * tokens * 2);
        Tensor cpu_on_device(cpu_contribution.span(), DType::FP32, {kHidden, tokens});
        Tensor gpu_output(gpu_contribution.span(), DType::FP32, {kHidden, tokens});
        Tensor destination(output.span(), DType::BF16, {kHidden, tokens});
        for (int partition : {0, 1, 2}) {
            std::vector<int> mask(kExperts);
            for (int e = 0; e < kExperts; ++e) {
                mask[e] = partition == 2 || (partition == 1 && e % 3 != 0);
            }
            auto device_mask = to_device<int>(mask);
            ops::routed_swiglu(x, profile.routed, cpu_weights, routes,
                               {ExpertSubsetKind::MaskOnes,
                                Tensor(mask.data(), Device::cpu(), DType::I32, {kExperts})},
                               cpu_output, cpu_arena, executor.execution_view());
            cpu_contribution.copy_from_host(partial.data(), partial.size());
            ops::routed_swiglu(
                device_batch.x, profile.routed, fixture.weights().routed, device_batch.routes,
                {ExpertSubsetKind::MaskZeros, Tensor(device_mask.span(), DType::I32, {kExperts})},
                {fixture.weights().shared, device_batch.shared}, MoeEpilogue::Write, gpu_output,
                gpu_arena, execution);
            const std::array contributions{cpu_on_device, gpu_output};
            ops::sum_contributions(contributions, &device_batch.residual, destination, execution);
            cuda_synchronize(execution.stream());
            const auto reference =
                fixture.contribution(batch, ExpertSubsetKind::All, {}, true, true);
            const auto actual = from_device_bf16(destination.data, reference.size());
            const auto label  = "CPU/GPU routed + shared + residual T=" + std::to_string(tokens) +
                               " partition=" + std::to_string(partition);
            failures += verify_reduction(label, actual, reference, kContributionA16);
            failures += output.verify_guards(label) + gpu_scratch.verify_guards(label) +
                        cpu_contribution.verify_guards(label) +
                        gpu_contribution.verify_guards(label);
        }
    }
    return failures;
}

} // namespace ninfer::test::moe

int main() {
    using namespace ninfer;
    using namespace ninfer::test;
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        auto& context = cuda_environment().context;
        cudaStream_t stream;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "create mixed MoE test stream");
        const auto execution = context.execution_view(stream);
        int failures         = 0;
        for (const auto down : {QType::Q5_G64_FP16, QType::Q6_G64_FP16}) {
            moe::Fixture fixture(QType::Q4_G64_FP16, down);
            failures += moe::test_mixed(fixture, execution);
            failures += fixture.verify_preserved();
        }
        cuda_check(cudaStreamDestroy(stream), "destroy mixed MoE test stream");
        std::cout << (failures ? "FAIL" : "OK") << " mixed MoE contracts\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
