#pragma once

#include "models/common/moe/execution.h"

namespace ninfer::models::moe::offload {

struct PartitionView {
    ExpertBankId bank;
    std::int32_t experts           = 0;
    ops::ExpertSubsetKind gpu_kind = ops::ExpertSubsetKind::All;
    Tensor device_mask, host_mask;

    [[nodiscard]] ops::ExpertSubset gpu() const noexcept;
    [[nodiscard]] ops::ExpertSubset cpu() const noexcept;
};

struct ExecutionRequirements {
    std::size_t device_frame_bytes  = 0;
    std::size_t pinned_frame_bytes  = 0;
    std::size_t cpu_workspace_bytes = 0;
    std::size_t gpu_workspace_bytes = 0;

    void include(const ExecutionRequirements& other) noexcept;
};

// GPU subset is All, None or MaskOnes; the CPU always consumes its complement.
[[nodiscard]] ExecutionRequirements
describe_execution(const ops::MoeProfile& profile, ops::ExpertSubsetKind gpu_subset,
                   ops::MoeEpilogue epilogue, std::int32_t min_tokens, std::int32_t max_tokens);

struct HostExecutionView {
    CpuExecutionView cpu;
    CudaExecutionView transfer;
    MemorySpan device_frame, host_frame;
    MemoryArena& workspace;
    CudaCompletionEvent& route_ready;
    CudaCompletionEvent& host_ready;
    CudaCompletionEvent& partial_ready;
};

struct ExecutionView {
    CudaExecutionView cuda;
    std::optional<HostExecutionView> host;
};

struct Frame {
    Tensor host_input;
    ops::ExpertRoutes device_routes, host_routes;
    MemorySpan device_route_bytes, host_route_bytes;
    Tensor shared, gpu_partial, device_cpu_partial, host_partial;
};

// Borrowed operands and stable bridge storage for one layer invocation. Preparing a
// call does not acquire weights, enqueue work, or inspect route/mask contents.
struct PreparedExecution {
    const Parameters& parameters;
    PartitionView partition;
    const ExpertBankSource& source;
    const ExpertBankDeviceView& bank;
    ops::MoeEpilogue epilogue;
    std::int32_t tokens;
    ExecutionView execution;
    Frame frame;
    ExecutionRequirements requirements;
};

[[nodiscard]] PreparedExecution prepare_execution(const Parameters&, PartitionView,
                                                  const ExpertBankSource&,
                                                  const ExpertBankDeviceView&, ops::MoeEpilogue,
                                                  std::int32_t tokens, ExecutionView);

// Capture orders input D2H before GPU experts and records a Host-observable input event.
enum class GpuSubmission { Eager, Capture };
void enqueue_gpu(const Tensor& hidden, const PreparedExecution&, Tensor& output, MemoryArena&,
                 GpuSubmission, const ops::MoeExecutionHints& = {});
// Host input must be ready. Returns false without a collective when all routes hit the GPU.
[[nodiscard]] bool compute_cpu(const PreparedExecution&);
void enqueue_finish(const PreparedExecution&, Tensor& output, bool cpu_work);
// Join the already submitted input transfer into compute retirement without running CPU work.
void enqueue_discard(const PreparedExecution&);
// Empty while the D2H input is pending. An engaged result submits the output chain and
// reports whether CPU work was needed. Consume that result once; all-GPU needs no event query.
[[nodiscard]] std::optional<bool> try_complete(const PreparedExecution&, Tensor& output);

} // namespace ninfer::models::moe::offload
