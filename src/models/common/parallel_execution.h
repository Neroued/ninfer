#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/observability/execution.h"
#include <functional>
#include <span>

namespace ninfer::distributed {
class Group;
struct PeerTransfer;
enum class Collective;
enum class Reduction;
} // namespace ninfer::distributed

namespace ninfer::models {

struct ParallelBinding {
    const distributed::Group* group = nullptr;
    std::uint32_t output_rank       = 0;
};

// Borrows the execution group's communicator. Program owns the stream and dependencies;
// every collective and peer exchange follows the caller's rank-wide communication order.
class ParallelExecution {
public:
    explicit ParallelExecution(ParallelBinding);
    [[nodiscard]] std::uint32_t size() const;
    void sum(Tensor&, CudaExecutionView);
    // Each U32 word has one contributing rank; all other ranks supply zero.
    void merge_unique(Tensor&, CudaExecutionView);
    // Unsigned 64-bit ordering for packed vocabulary candidates.
    void maximum(Tensor&, CudaExecutionView);
    void gather(const Tensor&, Tensor&, CudaExecutionView);
    void broadcast(Tensor&, CudaExecutionView);
    void broadcast(Tensor&, std::uint32_t owner, CudaExecutionView);
    void exchange(std::span<const distributed::PeerTransfer>, CudaExecutionView);
    void record_graph(CommunicationWork&, const std::function<void()>&);
    void replayed(const CommunicationWork&) const;

private:
    void enqueue(distributed::Collective, distributed::Reduction, const Tensor&, Tensor&, DType,
                 std::size_t count, std::uint32_t owner, CudaExecutionView);
    ParallelBinding binding_;
};

} // namespace ninfer::models
