#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::distributed {
class Group;
enum class Collective;
enum class Reduction;
} // namespace ninfer::distributed

namespace ninfer::models::qwen4_exp {

struct ParallelBinding {
    const distributed::Group* group = nullptr;
    std::uint32_t output_rank       = 0;
};

// Borrows the execution group's communicator. Program owns the stream and dependencies;
// all calls, including PLE and vocabulary, follow the same rank-wide communication order.
class ParallelExecution {
public:
    ParallelExecution(ParallelBinding, const DeviceContext&);
    [[nodiscard]] bool output_owner() const;
    [[nodiscard]] std::uint32_t size() const;
    void sum(Tensor&, CudaExecutionView);
    void merge_embedding(Tensor&, CudaExecutionView);
    void maximum(Tensor&, CudaExecutionView);
    void gather(const Tensor&, Tensor&, CudaExecutionView);
    void broadcast(Tensor&, CudaExecutionView);
    void broadcast(Tensor&, std::uint32_t owner, CudaExecutionView);
    void synchronize() const;

private:
    void enqueue(distributed::Collective, distributed::Reduction, const Tensor&, Tensor&, DType,
                 std::size_t count, std::uint32_t owner, CudaExecutionView);
    ParallelBinding binding_;
    CudaStream stream_;
    CudaExecutionView cuda_;
    CudaCompletionEvent input_, output_;
};

} // namespace ninfer::models::qwen4_exp
