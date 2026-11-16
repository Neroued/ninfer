#include "models/common/parallel_execution.h"
#include "distributed/communication.h"
#include "ninfer/ops/collective.h"

#include <limits>
#include <algorithm>

namespace ninfer::models {

TokenPartition::TokenPartition(std::int32_t n, std::uint32_t size, std::uint32_t index)
    : columns(n), ranks(size), rank(index) {
    if (n < 1 || !size || index >= size) throw std::invalid_argument("invalid token partition");
    const auto width = (std::uint64_t(n) + size - 1) / size;
    const auto total = width * size;
    if (total > std::numeric_limits<std::int32_t>::max())
        throw std::overflow_error("token partition exceeds tensor extent");
    segment = static_cast<std::int32_t>(width);
    padded  = static_cast<std::int32_t>(total);
    begin   = static_cast<std::int32_t>(index * width);
    count   = std::clamp(n - begin, 0, segment);
}

Tensor TokenPartition::local(const Tensor& value, int axis) const {
    return count ? value.slice(axis, begin, count) : Tensor{};
}

Tensor TokenPartition::exchange(const Tensor& value, int axis) const {
    return value.slice(axis, begin, segment);
}

void TokenPartition::clear_padding(const Tensor& value, int axis, CudaExecutionView cuda) const {
    if (columns == padded) return;
    auto tail = value.slice(axis, columns, padded - columns);
    if (!tail.is_contiguous() || tail.device != cuda.device())
        throw std::invalid_argument("token padding requires contiguous device columns");
    CUDA_CHECK(cudaMemsetAsync(tail.data, 0, tail.bytes(), cuda.stream()));
}

TokenPartition SequenceParallelPolicy::partition(std::int32_t columns, std::uint32_t rank) const {
    return minimum_columns && columns >= minimum_columns ? TokenPartition(columns, ranks, rank)
                                                         : TokenPartition(columns);
}

std::int32_t SequenceParallelPolicy::local_capacity(std::int32_t maximum) const {
    if (!minimum_columns || maximum < minimum_columns) return TokenPartition(maximum).columns;
    return std::max(minimum_columns - 1, partition(maximum, 0).segment);
}

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

void ParallelExecution::sum_scatter(const Tensor& source, Tensor& destination,
                                    CudaExecutionView cuda) {
#if NINFER_ENABLE_DISTRIBUTED
    ops::reduce_scatter(source, destination, distributed::Reduction::Sum, *binding_.group, cuda);
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
