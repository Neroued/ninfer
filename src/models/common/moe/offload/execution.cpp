#include "models/common/moe/offload/execution.h"

#include "core/layout.h"
#include "core/nvtx.h"
#include "core/transfer.h"
#include "ninfer/ops/sum_contributions.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer::models::moe::offload {
namespace {

using Subset = ops::ExpertSubsetKind;

Subset cpu_subset(Subset gpu) {
    switch (gpu) {
    case Subset::All:
        return Subset::None;
    case Subset::None:
        return Subset::All;
    case Subset::MaskOnes:
        return Subset::MaskZeros;
    default:
        throw std::invalid_argument("MoE GPU subset must be All, None or MaskOnes");
    }
}

struct FrameLayout {
    TensorRegion host_input, shared, gpu_partial, device_cpu_partial, host_partial;
    LayoutRegion device_routes, host_routes;
    std::int32_t top_k, tokens;
    std::size_t device_bytes, pinned_bytes;

    FrameLayout(const ops::MoeProfile& profile, std::int32_t columns)
        : top_k(profile.routed.top_k), tokens(columns) {
        const auto h = profile.routed.ffn.hidden;
        LayoutBuilder device, host;
        const auto coefficients_offset = (std::size_t(top_k) * tokens * 4 + 15) / 16 * 16;
        const auto route_bytes         = coefficients_offset + std::size_t(top_k) * tokens * 4;
        device_routes                  = device.add(route_bytes, 16, "MoE Device routes");
        host_routes                    = host.add(route_bytes, 16, "MoE Host routes");
        host_input  = host.add_tensor(DType::BF16, {h, tokens}, 16, "MoE Host input");
        shared      = device.add_tensor(DType::FP32, {tokens}, 4, "MoE shared coefficients");
        gpu_partial = device.add_tensor(DType::FP32, {h, tokens}, 16, "MoE GPU contribution");
        device_cpu_partial =
            device.add_tensor(DType::FP32, {h, tokens}, 16, "MoE Device CPU contribution");
        host_partial = host.add_tensor(DType::FP32, {h, tokens}, 16, "MoE Host contribution");
        device_bytes = device.finish(256);
        pinned_bytes = host.finish(256);
    }

    Frame bind(MemorySpan device, MemorySpan host) const {
        const auto routes = [&](MemorySpan storage) {
            const auto bytes               = std::size_t(top_k) * tokens * 4;
            const auto coefficients_offset = (bytes + 15) / 16 * 16;
            return ops::ExpertRoutes{
                Tensor(storage.subspan(0, bytes), DType::I32, {top_k, tokens}),
                Tensor(storage.subspan(coefficients_offset, bytes), DType::FP32, {top_k, tokens})};
        };
        const auto dr = device_routes.bind(device), hr = host_routes.bind(host);
        return {host_input.bind(host),
                routes(dr),
                routes(hr),
                dr,
                hr,
                shared.bind(device),
                gpu_partial.bind(device),
                device_cpu_partial.bind(device),
                host_partial.bind(host)};
    }
};

MemorySpan bytes(const Tensor& tensor, bool pinned = false) {
    return {tensor.data, tensor.bytes(), tensor.device, pinned};
}

void validate_inputs(const Tensor& hidden, const Tensor& output, const ops::MoeProfile& profile,
                     Device device) {
    const auto h = profile.routed.ffn.hidden;
    for (const auto* tensor : {&hidden, &output}) {
        if (tensor->device != device || tensor->dtype != DType::BF16 || tensor->ne[0] != h ||
            tensor->ne[1] != hidden.ne[1] || tensor->ne[1] <= 0 || tensor->ne[2] != 1 ||
            tensor->ne[3] != 1 || !tensor->is_contiguous() || tensor->data == nullptr ||
            reinterpret_cast<std::uintptr_t>(tensor->data) % 16) {
            throw std::invalid_argument(
                "MoE hidden/output must be aligned contiguous CUDA BF16 [H,T]");
        }
    }
    const auto a = reinterpret_cast<std::uintptr_t>(hidden.data);
    const auto b = reinterpret_cast<std::uintptr_t>(output.data);
    if ((a <= b ? b - a : a - b) < hidden.bytes()) {
        throw std::invalid_argument("MoE hidden and output overlap");
    }
}

void outside_capture(CudaExecutionView execution) {
    cudaStreamCaptureStatus status;
    const auto error = cudaStreamIsCapturing(execution.stream(), &status);
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("MoE capture query: ") + cudaGetErrorString(error));
    }
    if (status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("cached MoE handoff must execute outside CUDA Graph capture");
    }
}

void drain_after_failure(ExecutionView execution) noexcept {
    const auto drain = [](cudaStream_t stream) {
        const auto error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
            std::fprintf(stderr, "MoE failed execution cleanup: %s\n", cudaGetErrorString(error));
        }
    };
    if (execution.host) { drain(execution.host->transfer.stream()); }
    drain(execution.cuda.stream());
}

} // namespace

ops::ExpertSubset PartitionView::gpu() const noexcept { return {gpu_kind, device_mask}; }

ops::ExpertSubset PartitionView::cpu() const noexcept {
    const auto kind = gpu_kind == Subset::All    ? Subset::None
                      : gpu_kind == Subset::None ? Subset::All
                                                 : Subset::MaskZeros;
    return {kind, host_mask};
}

void ExecutionRequirements::include(const ExecutionRequirements& other) noexcept {
    device_frame_bytes  = std::max(device_frame_bytes, other.device_frame_bytes);
    pinned_frame_bytes  = std::max(pinned_frame_bytes, other.pinned_frame_bytes);
    cpu_workspace_bytes = std::max(cpu_workspace_bytes, other.cpu_workspace_bytes);
    gpu_workspace_bytes = std::max(gpu_workspace_bytes, other.gpu_workspace_bytes);
}

ExecutionRequirements describe_execution(const ops::MoeProfile& profile, Subset gpu_subset,
                                         ops::MoeEpilogue epilogue, std::int32_t first,
                                         std::int32_t last) {
    if (epilogue != ops::MoeEpilogue::Write && epilogue != ops::MoeEpilogue::AddResidual)
        throw std::invalid_argument("MoE output epilogue is invalid");
    const auto cpu = cpu_subset(gpu_subset);
    if (gpu_subset == Subset::All) {
        return {
            0, 0, 0,
            ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(profile, epilogue, first, last)};
    }
    const ops::RouteProfile route{profile.routed.ffn.hidden, profile.routed.experts,
                                  profile.routed.top_k};
    const auto route_bytes = ops::topk_softmax_route_workspace_capacity_bytes(route, first, last);
    const auto gpu_bytes   = ops::routed_swiglu_workspace_capacity_bytes(
        profile.routed, profile.shared, gpu_subset, DType::FP32, ops::MoeEpilogue::Write, first,
        last);
    const auto cpu_bytes =
        ops::routed_swiglu_workspace_capacity_bytes(profile.routed, cpu, first, last);
    const FrameLayout frame(profile, last);
    return {frame.device_bytes, frame.pinned_bytes, cpu_bytes, std::max(route_bytes, gpu_bytes)};
}

PreparedExecution prepare_execution(const Parameters& parameters, PartitionView partition,
                                    const ExpertBankSource& source,
                                    const ExpertBankDeviceView& bank, ops::MoeEpilogue epilogue,
                                    std::int32_t tokens, ExecutionView execution) {
    const auto& profile = parameters.profile;
    if (partition.bank != parameters.bank || partition.experts != profile.routed.experts)
        throw std::invalid_argument("MoE partition/source does not match the layer bank");
    if (bank.device != execution.cuda.device() || bank.count != std::uint32_t(partition.experts))
        throw std::invalid_argument("MoE cache bank does not match the execution device/profile");
    const auto requirements =
        describe_execution(profile, partition.gpu_kind, epilogue, tokens, tokens);
    Frame frame;
    if (partition.gpu_kind != Subset::All) {
        if (!execution.host) throw std::invalid_argument("mixed MoE requires Host resources");
        auto& host = *execution.host;
        if (host.transfer.device() != execution.cuda.device() ||
            host.transfer.stream() == execution.cuda.stream() ||
            host.device_frame.device != execution.cuda.device() ||
            host.host_frame.device != Device::cpu() || !host.host_frame.host_pinned ||
            host.workspace.device() != Device::cpu())
            throw std::invalid_argument("MoE bridge resources have incompatible locations/streams");
        if (!source.gate_up.parent || !source.down.parent ||
            source.gate_up.parent->device != Device::cpu() ||
            source.down.parent->device != Device::cpu())
            throw std::invalid_argument("mixed MoE requires native CPU expert sources");
        frame = FrameLayout(profile, tokens).bind(host.device_frame, host.host_frame);
    }
    return {parameters, partition, source, bank, epilogue, tokens, execution, frame, requirements};
}

void enqueue_gpu(const Tensor& hidden, const PreparedExecution& call, Tensor& output,
                 MemoryArena& workspace, GpuSubmission submission,
                 const ops::MoeExecutionHints& hints) {
    const auto& p        = call.parameters;
    const auto execution = call.execution.cuda;
    validate_inputs(hidden, output, p.profile, execution.device());
    if (hidden.ne[1] != call.tokens)
        throw std::invalid_argument("MoE prepared token count differs from input");
    if (call.partition.gpu_kind == Subset::All) {
        moe::execute(hidden, p, call.bank, call.epilogue, output, workspace, execution, hints);
        return;
    }
    auto scope = workspace.scope();
    MemoryArena scratch(workspace.alloc_bytes(call.requirements.gpu_workspace_bytes));
    auto frame       = call.frame;
    const auto& host = *call.execution.host;
    // A separate captured D2H branch may be scheduled behind the GPU experts. Put
    // the input copy first so CPU work can start while those experts are running.
    const bool parallel_copy = call.tokens > 8 && submission == GpuSubmission::Eager;
    const auto transfer      = parallel_copy ? host.transfer : execution;
    const ops::RouteProfile route{p.profile.routed.ffn.hidden, p.profile.routed.experts,
                                  p.profile.routed.top_k};
    ops::topk_softmax_route(hidden, route, p.control, frame.device_routes, frame.shared, scratch,
                            execution);
    if (parallel_copy) {
        host.route_ready.record(execution.stream());
        host.route_ready.wait(transfer.stream());
    }
    copy_bytes_async(bytes(frame.host_input, true), bytes(hidden), transfer);
    copy_bytes_async(frame.host_route_bytes, frame.device_route_bytes, transfer);
    if (submission == GpuSubmission::Capture)
        host.host_ready.record_external(transfer.stream());
    else
        host.host_ready.record(transfer.stream());
    ops::routed_swiglu(hidden, p.profile.routed, {call.bank, p.activation_divisors},
                       frame.device_routes, call.partition.gpu(), {p.shared, frame.shared},
                       ops::MoeEpilogue::Write, frame.gpu_partial, scratch, execution);
}

bool compute_cpu(const PreparedExecution& call) {
    const auto& frame = call.frame;
    const auto& p     = call.parameters;
    const auto* ids   = static_cast<const std::int32_t*>(frame.host_routes.ids.data);
    const auto* mask  = static_cast<const std::int32_t*>(call.partition.host_mask.data);
    const auto kind   = call.partition.gpu_kind;
    if (kind == Subset::All) return false;
    const auto assignments = std::size_t(p.profile.routed.top_k) * call.tokens;
    const auto cpu_assignments =
        kind == Subset::None
            ? assignments
            : std::count_if(ids, ids + assignments, [&](auto id) { return mask[id] == 0; });
    if (!cpu_assignments) return false;
    const auto& host = *call.execution.host;
    auto scope       = host.workspace.scope();
    MemoryArena scratch(host.workspace.alloc_bytes(call.requirements.cpu_workspace_bytes));
    auto destination = frame.host_partial;
    nvtx::ScopedRange cpu(nvtx::Name::MoeCpu, nvtx::Category::Moe, cpu_assignments);
    ops::routed_swiglu(frame.host_input, p.profile.routed, {call.source.gate_up, call.source.down},
                       frame.host_routes, call.partition.cpu(), destination, scratch, host.cpu);
    return true;
}

void enqueue_finish(const PreparedExecution& call, Tensor& output, bool cpu_work) {
    if (call.partition.gpu_kind == Subset::All) return;
    const auto& frame      = call.frame;
    const auto execution   = call.execution.cuda;
    const Tensor* residual = call.epilogue == ops::MoeEpilogue::AddResidual ? &output : nullptr;
    if (cpu_work) {
        const auto& host         = *call.execution.host;
        const bool parallel_copy = call.tokens > 8;
        const auto transfer      = parallel_copy ? host.transfer : execution;
        copy_bytes_async(bytes(frame.device_cpu_partial), bytes(frame.host_partial, true),
                         transfer);
        if (parallel_copy) {
            host.partial_ready.record(transfer.stream());
            host.partial_ready.wait(execution.stream());
        }
        const std::array contributions{frame.gpu_partial, frame.device_cpu_partial};
        ops::sum_contributions(contributions, residual, output, execution);
    } else {
        const std::array contributions{frame.gpu_partial};
        ops::sum_contributions(contributions, residual, output, execution);
    }
}

bool complete(const PreparedExecution& call, Tensor& output) {
    if (call.partition.gpu_kind == Subset::All) return false;
    {
        nvtx::ScopedRange wait(nvtx::Name::MoeInputWait, nvtx::Category::Moe);
        call.execution.host->host_ready.synchronize();
    }
    const bool cpu_work = compute_cpu(call);
    enqueue_finish(call, output, cpu_work);
    return cpu_work;
}

bool execute(const Tensor& hidden, const Parameters& parameters, const PartitionView& partition,
             const ExpertBankSource& source, const ExpertBankDeviceView& bank,
             ops::MoeEpilogue epilogue, Tensor& output, MemoryArena& workspace,
             ExecutionView execution, const ops::MoeExecutionHints& hints) {
    outside_capture(execution.cuda);
    if (execution.host) outside_capture(execution.host->transfer);
    const auto call =
        prepare_execution(parameters, partition, source, bank, epilogue, hidden.ne[1], execution);
    try {
        enqueue_gpu(hidden, call, output, workspace, GpuSubmission::Eager, hints);
        return complete(call, output);
    } catch (...) {
        drain_after_failure(execution);
        throw;
    }
}

} // namespace ninfer::models::moe::offload
