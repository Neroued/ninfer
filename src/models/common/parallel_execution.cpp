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

std::size_t StageBoundaryField::transfer_elements(std::size_t elements, TpLayout layout,
                                                  std::uint32_t tensor_ranks) {
    if (!tensor_ranks) throw std::invalid_argument("empty pipeline tensor group");
    return layout == TpLayout::Replicated && elements % tensor_ranks == 0 ? elements / tensor_ranks
                                                                          : elements;
}

StageBoundaryField StageBoundaryField::bind(const Tensor& local, TpLayout layout,
                                            std::uint32_t rank, std::uint32_t ranks) {
    if (rank >= ranks || !local.is_contiguous())
        throw std::invalid_argument("invalid pipeline boundary field");
    const auto elements = static_cast<std::size_t>(local.numel());
    const auto count    = transfer_elements(elements, layout, ranks);
    if (count == elements) return {local, {}};
    if (elements > INT32_MAX) throw std::overflow_error("pipeline boundary exceeds tensor extent");
    auto flat = local.view({static_cast<std::int32_t>(elements)});
    return {
        flat.slice(0, static_cast<std::int32_t>(rank * count), static_cast<std::int32_t>(count)),
        flat};
}

ParallelExecution::ParallelExecution(ParallelBinding binding)
    : binding_(binding), graph_groups_{binding.group} {
    if (!binding.group) throw std::invalid_argument("parallel execution has no group");
}

std::uint32_t ParallelExecution::size() const {
#if NINFER_ENABLE_DISTRIBUTED
    return binding_.group->size();
#else
    return 1;
#endif
}

std::uint32_t ParallelExecution::rank() const {
#if NINFER_ENABLE_DISTRIBUTED
    return binding_.group->rank();
#else
    return 0;
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

void ParallelExecution::gather(const Tensor& source, Tensor& destination, CudaExecutionView cuda) {
#if NINFER_ENABLE_DISTRIBUTED
    ops::all_gather(source, destination, *binding_.group, cuda);
#else
    throw std::logic_error("distributed execution was not built");
#endif
}

void ParallelExecution::gather_unsigned_words(const Tensor& source, Tensor& destination,
                                              CudaExecutionView cuda) {
    if (source.dtype != DType::I64 || destination.dtype != DType::I64)
        throw std::invalid_argument("packed vocabulary records require I64 storage");
    auto local    = source;
    auto gathered = destination;
    local.dtype = gathered.dtype = DType::U64;
    gather(local, gathered, cuda);
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

void ParallelExecution::track_graph_group(const distributed::Group* group) {
    if (group && std::ranges::find(graph_groups_, group) == graph_groups_.end())
        graph_groups_.push_back(group);
}

bool ParallelExecution::supports_graph_gating() const noexcept {
#if NINFER_ENABLE_DISTRIBUTED
    return std::ranges::all_of(graph_groups_,
                               [](const auto* group) { return group->supports_graph_gating(); });
#else
    return true;
#endif
}

CommunicationGraphWork ParallelExecution::captured_work() const {
#if NINFER_ENABLE_DISTRIBUTED
    return binding_.group->captured_work();
#else
    throw std::logic_error("no distributed graph capture");
#endif
}

void ParallelExecution::record_graph(CommunicationGraphWork& work,
                                     const std::function<void()>& body) {
#if NINFER_ENABLE_DISTRIBUTED
    distributed::record_communication_graph(graph_groups_, work, body);
#else
    body();
#endif
}

void ParallelExecution::replayed(const CommunicationGraphWork& work) const {
#if NINFER_ENABLE_DISTRIBUTED
    for (const auto* group : graph_groups_) group->replayed(work);
#endif
}

} // namespace ninfer::models
