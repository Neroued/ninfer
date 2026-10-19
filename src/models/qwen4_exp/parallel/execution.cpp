#include "models/qwen4_exp/parallel/execution.h"
#include "distributed/communication.h"

namespace ninfer::models::qwen4_exp {

ParallelExecution::ParallelExecution(ParallelBinding binding, const DeviceContext& device)
    : binding_(binding), stream_(device), cuda_(device.execution_view(stream_.stream())),
      input_(device), output_(device) {
    if (!binding.group) throw std::invalid_argument("Qwen4Exp parallel execution has no group");
}

bool ParallelExecution::output_owner() const {
#if NINFER_ENABLE_DISTRIBUTED
    return binding_.group->rank() == binding_.output_rank;
#else
    return true;
#endif
}

std::uint32_t ParallelExecution::size() const {
#if NINFER_ENABLE_DISTRIBUTED
    return binding_.group->size();
#else
    return 1;
#endif
}

void ParallelExecution::enqueue(distributed::Collective operation, distributed::Reduction reduction,
                                const Tensor& source, Tensor& destination, DType type,
                                std::size_t count, std::uint32_t owner,
                                CudaExecutionView execution) {
#if NINFER_ENABLE_DISTRIBUTED
    input_.record(execution.stream());
    input_.wait(cuda_.stream());
    binding_.group->enqueue(operation, source.data, destination.data, count, type, reduction, owner,
                            cuda_);
    output_.record(cuda_.stream());
    output_.wait(execution.stream());
#else
    throw std::logic_error("distributed execution was not built");
#endif
}

void ParallelExecution::sum(Tensor& value, CudaExecutionView cuda) {
    enqueue(distributed::Collective::AllReduce, distributed::Reduction::Sum, value, value,
            value.dtype, value.numel(), 0, cuda);
}

void ParallelExecution::merge_embedding(Tensor& value, CudaExecutionView cuda) {
    enqueue(distributed::Collective::AllReduce, distributed::Reduction::Sum, value, value,
            DType::U32, value.bytes() / sizeof(std::uint32_t), 0, cuda);
}

void ParallelExecution::maximum(Tensor& value, CudaExecutionView cuda) {
    enqueue(distributed::Collective::AllReduce, distributed::Reduction::Max, value, value,
            DType::U64, value.numel(), 0, cuda);
}

void ParallelExecution::gather(const Tensor& source, Tensor& destination, CudaExecutionView cuda) {
    enqueue(distributed::Collective::AllGather, distributed::Reduction::Sum, source, destination,
            source.dtype, source.numel(), 0, cuda);
}

void ParallelExecution::broadcast(Tensor& value, CudaExecutionView cuda) {
    broadcast(value, binding_.output_rank, cuda);
}

void ParallelExecution::broadcast(Tensor& value, std::uint32_t owner, CudaExecutionView cuda) {
    enqueue(distributed::Collective::Broadcast, distributed::Reduction::Sum, value, value,
            value.dtype, value.numel(), owner, cuda);
}

void ParallelExecution::record_graph(CommunicationWork& work, const std::function<void()>& body) {
#if NINFER_ENABLE_DISTRIBUTED
    binding_.group->record_graph(work, body);
#else
    body();
#endif
}

void ParallelExecution::replayed(const CommunicationWork& work) const {
#if NINFER_ENABLE_DISTRIBUTED
    binding_.group->replayed(work);
#endif
}

void ParallelExecution::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(cuda_.stream())); }

} // namespace ninfer::models::qwen4_exp
