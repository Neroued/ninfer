#include "models/common/parallel_execution.h"
#include "distributed/communication.h"
#include "ninfer/ops/collective.h"

#include <limits>

namespace ninfer::models {

ParallelExecution::ParallelExecution(ParallelBinding binding) : binding_(binding) {
    if (!binding.group) throw std::invalid_argument("parallel execution has no group");
}

std::uint32_t ParallelExecution::size() const {
#if NINFER_ENABLE_DISTRIBUTED
    return binding_.group->size();
#else
    return 1;
#endif
}

void ParallelExecution::sum(Tensor& value, CudaExecutionView cuda) {
#if NINFER_ENABLE_DISTRIBUTED
    ops::all_reduce(value, value, distributed::Reduction::Sum, *binding_.group, cuda);
#else
    throw std::logic_error("distributed execution was not built");
#endif
}

void ParallelExecution::merge_unique(Tensor& value, CudaExecutionView cuda) {
    const auto bytes = value.bytes();
    if (!value.is_contiguous() || bytes % sizeof(std::uint32_t) ||
        bytes / sizeof(std::uint32_t) > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("unique-word merge requires contiguous U32-sized words");
    Tensor words(value.data, value.device, DType::U32,
                 {static_cast<std::int32_t>(bytes / sizeof(std::uint32_t))});
    sum(words, cuda);
}

void ParallelExecution::maximum(Tensor& value, CudaExecutionView cuda) {
    if (value.dtype != DType::I64 && value.dtype != DType::U64)
        throw std::invalid_argument("vocabulary maximum requires 64-bit packed candidates");
    auto keys  = value;
    keys.dtype = DType::U64;
#if NINFER_ENABLE_DISTRIBUTED
    ops::all_reduce(keys, keys, distributed::Reduction::Max, *binding_.group, cuda);
#else
    throw std::logic_error("distributed execution was not built");
#endif
}

void ParallelExecution::gather(const Tensor& source, Tensor& destination, CudaExecutionView cuda) {
#if NINFER_ENABLE_DISTRIBUTED
    ops::all_gather(source, destination, *binding_.group, cuda);
#else
    throw std::logic_error("distributed execution was not built");
#endif
}

void ParallelExecution::broadcast(Tensor& value, CudaExecutionView cuda) {
    broadcast(value, binding_.output_rank, cuda);
}

void ParallelExecution::broadcast(Tensor& value, std::uint32_t owner, CudaExecutionView cuda) {
#if NINFER_ENABLE_DISTRIBUTED
    ops::broadcast(value, owner, *binding_.group, cuda);
#else
    throw std::logic_error("distributed execution was not built");
#endif
}

void ParallelExecution::exchange(std::span<const distributed::PeerTransfer> transfers,
                                 CudaExecutionView execution) {
#if NINFER_ENABLE_DISTRIBUTED
    binding_.group->exchange(transfers, execution);
#else
    throw std::logic_error("distributed execution was not built");
#endif
}

void ParallelExecution::record_graph(CommunicationGraphWork& work,
                                     const std::function<void()>& body) {
#if NINFER_ENABLE_DISTRIBUTED
    binding_.group->record_graph(work, body);
#else
    body();
#endif
}

void ParallelExecution::replayed(const CommunicationGraphWork& work) const {
#if NINFER_ENABLE_DISTRIBUTED
    binding_.group->replayed(work);
#endif
}

} // namespace ninfer::models
