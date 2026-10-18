#include "fixture.h"

namespace ninfer::test::moe {
namespace {
int complete(Fixture& fixture, CudaExecutionView execution, int tokens, bool write) {
    const auto& profile = fixture.profile();
    const bool q5       = profile.routed.ffn.down_type == QType::Q5_G64_FP16;
    int failures        = 0;
    if (!write) {
        const auto batch = fixture.batch(tokens, tokens == 1 ? 1 : 0);
        DeviceBatch input(batch);
        GuardedDeviceBuffer storage(batch.residual.size() * 2);
        storage.copy_from_host(batch.residual.data(), storage.bytes());
        Tensor output(storage.span(), DType::BF16, {kHidden, tokens});
        const auto bytes = ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
            profile, ops::ExpertSubsetKind::All, ops::MoeEpilogue::AddResidual, tokens, tokens);
        GuardedDeviceBuffer scratch(bytes);
        MemoryArena arena(scratch.span());
        const auto run = [&] {
            cuda_check(cudaMemcpyAsync(output.data, input.residual.data, output.bytes(),
                                       cudaMemcpyDeviceToDevice, execution.stream()),
                       "seed residual");
            ops::topk_softmax_swiglu_moe(input.x, profile, fixture.weights(), {},
                                         ops::MoeEpilogue::AddResidual, output, arena, execution);
        };
        if (q5 && tokens == 2) {
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            cuda_check(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeGlobal),
                       "capture MoE");
            run();
            cuda_check(cudaStreamEndCapture(execution.stream(), &graph), "end MoE capture");
            cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                       "instantiate MoE");
            for (int i = 0; i < 2; ++i) {
                cuda_check(cudaGraphLaunch(executable, execution.stream()), "replay MoE");
            }
            cuda_synchronize(execution.stream());
            cuda_check(cudaGraphExecDestroy(executable), "destroy MoE executable");
            cuda_check(cudaGraphDestroy(graph), "destroy MoE graph");
        } else {
            run();
            cuda_synchronize(execution.stream());
        }
        const std::string label = "complete MoE T=" + std::to_string(tokens);
        failures += verify_reduction(label, from_device_bf16(output.data, batch.complete.size()),
                                     batch.complete, kCompleteA16);
        failures += storage.verify_guards(label);
        failures += scratch.verify_guards(label + " scratch");
        failures +=
            verify_exact("complete input preservation",
                         from_device<std::uint16_t>(input.x.data, batch.input.size()), batch.input);
        if (arena.used() != 0 || arena.peak_used() != bytes) {
            std::cerr << label << " workspace mismatch\n";
            ++failures;
        }
        if (tokens == 1) {
            for (const auto device :
                 {Device::cpu(), Device::cuda(execution.device().index() + 1)}) {
                auto invalid   = input.x;
                invalid.device = device;
                bool rejected  = false;
                try {
                    ops::topk_softmax_swiglu_moe(invalid, profile, fixture.weights(), {},
                                                 ops::MoeEpilogue::Write, output, arena, execution);
                } catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected) {
                    std::cerr << "accepted a mismatched device\n";
                    ++failures;
                }
            }
        }
    } else {
        auto batch = fixture.batch(tokens);
        DeviceBatch input(batch);
        DeviceBuffer buffer(execution.device(), batch.input.size() * 2);
        buffer.fill(0xff);
        Tensor output(buffer.span(), DType::BF16, {kHidden, tokens});
        DeviceBuffer scratch(
            execution.device(),
            ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
                profile, ops::ExpertSubsetKind::All, ops::MoeEpilogue::Write, tokens, tokens));
        MemoryArena arena(scratch.span());
        ops::topk_softmax_swiglu_moe(input.x, profile, fixture.weights(), {},
                                     ops::MoeEpilogue::Write, output, arena, execution);
        cuda_synchronize(execution.stream());
        auto reference = batch.complete;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            reference[i] -= bf16_to_f32(batch.residual[i]);
        }
        failures +=
            verify_reduction("complete Write", from_device_bf16(output.data, reference.size()),
                             reference, kCompleteA16);
    }
    return failures;
}

int complete_subset(Fixture& fixture, CudaExecutionView execution, int tokens,
                    ops::ExpertSubsetKind kind) {
    auto batch = fixture.batch(tokens);
    DeviceBatch input(batch);
    std::vector<int> mask(kExperts);
    auto dm = to_device(mask);
    const ops::ExpertSubset subset{kind, Tensor(dm.span(), DType::I32, {kExperts})};
    auto weights = fixture.weights();
    if (kind == ops::ExpertSubsetKind::None) { weights.routed = {}; }
    const auto& profile = fixture.profile();
    const auto mode     = tokens % 2 ? ops::MoeEpilogue::Write : ops::MoeEpilogue::AddResidual;
    const auto bytes =
        ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(profile, kind, mode, tokens, tokens);
    GuardedDeviceBuffer scratch(bytes), storage(batch.input.size() * 2);
    MemoryArena arena(scratch.span());
    Tensor output(storage.span(), DType::BF16, {kHidden, tokens});
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeThreadLocal),
               "capture");
    ops::topk_softmax_swiglu_moe(input.x, profile, weights, subset, mode, output, arena, execution);
    cuda_check(cudaStreamEndCapture(execution.stream(), &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
    int failures = 0;
    for (int replay = 0; replay < 3; ++replay) {
        for (int e = 0; e < kExperts; ++e) {
            mask[e] =
                replay == 2 ? int(kind == ops::ExpertSubsetKind::MaskZeros) : (e + replay) % 2;
        }
        dm.copy_from_host(mask.data(), dm.size());
        if (mode == ops::MoeEpilogue::AddResidual)
            storage.copy_from_host(batch.residual.data(), storage.bytes());
        else
            storage.fill(0xff);
        cuda_check(cudaGraphLaunch(executable, execution.stream()), "replay");
        cuda_synchronize(execution.stream());
        std::vector<double> reference(batch.input.size());
        for (int t = 0; t < tokens; ++t) {
            const auto& c = fixture.logical(batch.patterns[t]);
            for (int h = 0; h < kHidden; ++h) {
                double v = c.route.shared * c.shared[h];
                if (mode == ops::MoeEpilogue::AddResidual) v += c.residual[h];
                for (int j = 0; j < kTopK; ++j) {
                    const int e = c.route.ids[j];
                    if ((kind == ops::ExpertSubsetKind::MaskOnes && mask[e]) ||
                        (kind == ops::ExpertSubsetKind::MaskZeros && !mask[e])) {
                        v += c.route.coefficients[j] * c.expert(e)[h];
                    }
                }
                reference[std::size_t(t) * kHidden + h] = v;
            }
        }
        failures +=
            verify_reduction("complete subset", from_device_bf16(output.data, reference.size()),
                             reference, kCompleteA16);
        failures += storage.verify_guards("complete subset output") +
                    scratch.verify_guards("complete subset scratch");
    }
    cuda_check(cudaGraphExecDestroy(executable), "destroy exec");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    if (arena.used() || arena.peak_used() != bytes) ++failures;
    return failures;
}
} // namespace

void add_complete_cases(std::vector<FixtureCase>& cases, QType gate_up, QType down) {
    const bool q8 = gate_up == QType::Q8_G32_FP16, q5 = down == QType::Q5_G64_FP16;
    const auto widths = q8   ? std::vector{1, 2, 19, 20, 383, 384, 768}
                        : q5 ? std::vector{1, 2, 46, 47, 768, 4097}
                             : std::vector{1, 2, 46, 47, 768};
    for (int t : widths)
        cases.push_back(
            {{"complete/t" + std::to_string(t), {{"execution", q5 && t == 2 ? "graph" : "eager"}}},
             [=](Fixture& f, auto e) { return complete(f, e, t, false); }});
    for (int t : {1, q8 ? 20 : 47})
        cases.push_back({{"complete/write/t" + std::to_string(t), {{"execution", "eager"}}},
                         [=](Fixture& f, auto e) { return complete(f, e, t, true); }});
    for (auto subset : {ops::ExpertSubsetKind::MaskOnes, ops::ExpertSubsetKind::MaskZeros,
                        ops::ExpertSubsetKind::None}) {
        for (int t : {1, 2, 16, q8 ? 20 : 47, q8 ? 384 : 768}) {
            cases.push_back(
                {{"complete/subset" + std::to_string(int(subset)) + "/t" + std::to_string(t),
                  {{"execution", "graph"}}},
                 [=](Fixture& f, auto e) { return complete_subset(f, e, t, subset); }});
        }
    }
    cases.push_back({{"complete/capacity", {}}, [=](Fixture& f, auto) {
                         std::size_t peak = 0;
                         for (int t : widths)
                             peak = std::max(peak,
                                             ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
                                                 f.profile(), ops::ExpertSubsetKind::All,
                                                 ops::MoeEpilogue::AddResidual, t, t));
                         return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
                                    f.profile(), ops::ExpertSubsetKind::All,
                                    ops::MoeEpilogue::AddResidual, 1, widths.back()) != peak;
                     }});
}
} // namespace ninfer::test::moe
