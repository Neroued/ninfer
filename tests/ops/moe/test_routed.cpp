#include "fixture.h"

namespace ninfer::test::moe {
namespace {
using ops::ExpertSubsetKind;
using ops::MoeEpilogue;

class PackedBank {
public:
    PackedBank(const Fixture& fixture, bool indexed)
        : fixture_(fixture), indexed_(indexed),
          gate_up_(cuda_environment().context.device(), matrix_size(true)),
          down_(cuda_environment().context.device(), matrix_size(false)),
          gate_offsets_(cuda_environment().context.device(), indexed ? kExperts * 8 : 0),
          down_offsets_(cuda_environment().context.device(), indexed ? kExperts * 8 : 0) {
        if (indexed) {
            for (int e = 0; e < kExperts; ++e) {
                aliases.push_back(fixture.experts()[e % fixture.experts().size()].id);
            }
        } else {
            for (const auto& expert : fixture.experts()) { copy(expert, expert.id); }
        }
        view = {matrix(true), matrix(false), kExperts, gate_up_.device()};
    }

    void publish(std::span<const int> mask, ExpertSubsetKind subset, bool reverse = false) {
        if (!indexed_) { return; }
        const auto experts = fixture_.experts();
        std::vector<std::uint64_t> gu(kExperts, UINT64_MAX), dn(kExperts, UINT64_MAX);
        for (std::size_t i = 0; i < experts.size(); ++i) {
            const int slot = reverse ? experts.size() - 1 - i : i;
            copy(experts[i], slot);
        }
        for (int e = 0; e < kExperts; ++e) {
            const bool selected = subset == ExpertSubsetKind::All ||
                                  (subset == ExpertSubsetKind::MaskOnes && mask[e] == 1) ||
                                  (subset == ExpertSubsetKind::MaskZeros && mask[e] == 0);
            if (selected) {
                const auto index = std::size_t(e) % experts.size();
                const auto slot  = reverse ? experts.size() - 1 - index : index;
                gu[e]            = slot * entry_bytes(true);
                dn[e]            = slot * entry_bytes(false);
            }
        }
        gate_offsets_.copy_from_host(gu.data(), gu.size() * 8);
        down_offsets_.copy_from_host(dn.data(), dn.size() * 8);
    }

    ExpertBankDeviceView view;
    std::vector<int> aliases;

private:
    const quantized_weight::PackedWeight& sample(bool gu) const {
        return gu ? fixture_.experts()[0].gate_up : fixture_.experts()[0].down;
    }

    std::size_t entry_bytes(bool gu) const { return (sample(gu).payload.size() + 511) / 256 * 256; }

    std::size_t matrix_size(bool gu) const {
        return entry_bytes(gu) * (indexed_ ? fixture_.experts().size() : kExperts);
    }

    void copy(const HostExpert& expert, int slot) {
        gate_up_.copy_from_host(expert.gate_up.payload.data(), expert.gate_up.payload.size(),
                                slot * entry_bytes(true));
        down_.copy_from_host(expert.down.payload.data(), expert.down.payload.size(),
                             slot * entry_bytes(false));
    }

    ExpertMatrixDeviceView matrix(bool gu) const {
        const auto& buffer  = gu ? gate_up_ : down_;
        const auto& offsets = gu ? gate_offsets_ : down_offsets_;
        const auto& packed  = sample(gu);
        const auto* base    = static_cast<const std::byte*>(buffer.data());
        ExpertMatrixDeviceView out;
        out.codes  = base;
        out.scales = base + packed.scale_plane_offset;
        if (packed.high_plane_bytes) { out.high = base + packed.high_plane_offset; }
        if (indexed_) {
            out.offsets         = static_cast<const std::uint64_t*>(offsets.data());
            out.indexed_storage = buffer.span();
        } else {
            out.code_stride = out.scale_stride = entry_bytes(gu);
            if (out.high) { out.high_stride = entry_bytes(gu); }
        }
        return out;
    }

    const Fixture& fixture_;
    bool indexed_;
    DeviceBuffer gate_up_, down_, gate_offsets_, down_offsets_;
};

int run(Fixture& fixture, CudaExecutionView execution, int tokens, ExpertSubsetKind kind,
        DType type, bool shared, MoeEpilogue mode, PackedBank* bank = nullptr, bool graph = false) {
    auto batch = fixture.batch(tokens);
    if (bank && !bank->aliases.empty()) {
        for (std::size_t i = 0; i < batch.ids.size(); ++i) { batch.ids[i] = i % kExperts; }
    }
    DeviceBatch input(batch);
    std::vector<int> mask(kExperts);
    for (int e = 0; e < kExperts; ++e) { mask[e] = e % 3 != 0; }
    auto device_mask = to_device(mask);
    ops::ExpertSubset subset{kind, Tensor(device_mask.span(), DType::I32, {kExperts})};
    if (bank) { bank->publish(mask, kind); }
    const ops::CudaSwiGLUExperts experts =
        bank ? ops::CudaSwiGLUExperts{bank->view, {}} : fixture.weights().routed;
    const auto& profile = fixture.profile();
    const auto bytes =
        shared ? ops::routed_swiglu_workspace_capacity_bytes(profile.routed, profile.shared, kind,
                                                             type, mode, tokens, tokens)
               : ops::routed_swiglu_workspace_capacity_bytes(profile.routed, kind, type, mode,
                                                             tokens, tokens);
    // A zero-scratch call also accepts an exhausted arena with an unaligned cursor.
    const std::size_t prefix = bytes == 0 ? 1 : 0;
    GuardedDeviceBuffer scratch(bytes + prefix),
        output_storage(batch.input.size() * (type == DType::BF16 ? 2 : 4));
    MemoryArena arena(scratch.span());
    if (prefix) { (void)arena.alloc_bytes(prefix, 1); }
    Tensor output(output_storage.span(), type, {kHidden, tokens});
    const ops::ScaledSwiGLUTerm term{fixture.weights().shared, input.shared};
    const auto call = [&] {
        // None accepts no routed addresses or routes, including in a captured call.
        const auto& selected_experts =
            kind == ExpertSubsetKind::None ? ops::CudaSwiGLUExperts{} : experts;
        const auto& routes = kind == ExpertSubsetKind::None ? ops::ExpertRoutes{} : input.routes;
        if (shared) {
            ops::routed_swiglu(input.x, profile.routed, selected_experts, routes, subset, term,
                               mode, output, arena, execution);
        } else {
            ops::routed_swiglu(input.x, profile.routed, selected_experts, routes, subset, mode,
                               output, arena, execution);
        }
    };
    cudaGraph_t captured       = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        cuda_check(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeGlobal),
                   "capture routed");
        call();
        cuda_check(cudaStreamEndCapture(execution.stream(), &captured), "end routed capture");
        cuda_check(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0),
                   "instantiate routed");
    }
    int failures = 0;
    for (int replay = 0; replay < (graph ? 3 : 1); ++replay) {
        if (replay) {
            for (auto& bit : mask) {
                bit = replay == 2 ? int(kind == ExpertSubsetKind::MaskZeros) : 1 - bit;
            }
            device_mask.copy_from_host(mask.data(), mask.size() * 4);
            if (bank) { bank->publish(mask, kind, true); }
            // Route contents and indexed physical locations may both change with fixed nodes.
            for (int t = 0; t < tokens; ++t) {
                std::reverse(batch.ids.begin() + t * kTopK, batch.ids.begin() + (t + 1) * kTopK);
            }
            cuda_check(cudaMemcpy(input.routes.ids.data, batch.ids.data(), batch.ids.size() * 4,
                                  cudaMemcpyHostToDevice),
                       "change routes");
        }
        if (mode == MoeEpilogue::AddResidual) {
            output_storage.copy_from_host(batch.residual.data(), output.bytes());
        } else {
            output_storage.fill(0xff);
        }
        if (graph) {
            cuda_check(cudaGraphLaunch(executable, execution.stream()), "replay routed");
        } else {
            call();
        }
        cuda_synchronize(execution.stream());
        const auto label = std::string("routed T=") + std::to_string(tokens) +
                           " subset=" + std::to_string(int(kind)) +
                           " shared=" + std::to_string(shared) +
                           " dtype=" + std::to_string(int(type)) +
                           (bank ? (bank->aliases.empty() ? " strided" : " indexed") : " affine");
        const auto reference = fixture.contribution(
            batch, kind, mask, shared, mode == MoeEpilogue::AddResidual,
            bank ? std::span<const int>(bank->aliases) : std::span<const int>{});
        std::vector<double> actual;
        if (type == DType::BF16) {
            actual = from_device_bf16(output.data, reference.size());
        } else {
            const auto values = from_device<float>(output.data, reference.size());
            actual.assign(values.begin(), values.end());
        }
        failures += verify_reduction(label, actual, reference, kContributionA16);
        failures += output_storage.verify_guards(label) + scratch.verify_guards(label);
        if (arena.used() != prefix || arena.peak_used() != prefix + bytes) {
            std::cerr << label << " workspace mismatch\n";
            ++failures;
        }
    }
    if (graph) {
        cuda_check(cudaGraphExecDestroy(executable), "destroy routed executable");
        cuda_check(cudaGraphDestroy(captured), "destroy routed graph");
    }
    return failures;
}
} // namespace

void add_routed_cases(std::vector<FixtureCase>& cases, QType gate_up) {
    const bool q8  = gate_up == QType::Q8_G32_FP16;
    const auto add = [&](int t, ExpertSubsetKind kind, DType dtype, bool shared,
                         MoeEpilogue epilogue) {
        cases.push_back(
            {{"routed/t" + std::to_string(t) + "/subset" + std::to_string(int(kind)) + "/dtype" +
                  std::to_string(int(dtype)) + (shared ? "/shared" : "/no_shared") +
                  (epilogue == MoeEpilogue::Write ? "/write" : "/residual"),
              {}},
             [=](Fixture& f, auto e) { return run(f, e, t, kind, dtype, shared, epilogue); }});
    };
    for (int tokens : {1, 2, q8 ? 19 : 46, q8 ? 20 : 47, 64, 768}) {
        add(tokens, ExpertSubsetKind::All, DType::FP32, false, MoeEpilogue::Write);
        add(tokens, ExpertSubsetKind::MaskOnes, DType::BF16, true, MoeEpilogue::Write);
        add(tokens, ExpertSubsetKind::MaskZeros, DType::BF16, false, MoeEpilogue::AddResidual);
    }
    for (int tokens : {1, 15, 16, 17, 4097}) {
        add(tokens, ExpertSubsetKind::None, DType::FP32, false, MoeEpilogue::Write);
        add(tokens, ExpertSubsetKind::None, DType::BF16, false, MoeEpilogue::AddResidual);
        add(tokens, ExpertSubsetKind::None, DType::FP32, true, MoeEpilogue::Write);
    }
    cases.push_back({{"routed/strided_bank", {}}, [](Fixture& fixture, auto execution) {
                         int failures = 0;
                         PackedBank strided(fixture, false);
                         for (int tokens : {1, 64})
                             failures += run(fixture, execution, tokens, ExpertSubsetKind::All,
                                             DType::FP32, true, MoeEpilogue::Write, &strided);
                         return failures;
                     }});
    cases.push_back(
        {{"routed/indexed_bank", {{"execution", "graph"}}}, [](Fixture& fixture, auto execution) {
             int failures = 0;
             PackedBank indexed(fixture, true);
             for (int tokens : {1, 64, 768}) {
                 failures += run(fixture, execution, tokens, ExpertSubsetKind::MaskOnes,
                                 DType::FP32, true, MoeEpilogue::Write, &indexed, true);
                 failures += run(fixture, execution, tokens, ExpertSubsetKind::MaskZeros,
                                 DType::FP32, false, MoeEpilogue::Write, &indexed);
             }
             return failures;
         }});
}
} // namespace ninfer::test::moe
