#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/observability/execution.h"
#include <functional>
#include <span>

namespace ninfer::distributed {
class Group;
struct PeerTransfer;
} // namespace ninfer::distributed

namespace ninfer::models {

// Contiguous columns, with equal communication segments and an optional final padding tail.
// Mathematical views contain only real columns; the owner provides padded backing explicitly.
struct TokenPartition {
    explicit TokenPartition(std::int32_t columns, std::uint32_t ranks = 1, std::uint32_t rank = 0);

    std::int32_t columns, segment, begin, count, padded;
    std::uint32_t ranks, rank;

    [[nodiscard]] bool split() const { return ranks > 1; }

    [[nodiscard]] Tensor local(const Tensor&, int axis) const;
    [[nodiscard]] Tensor exchange(const Tensor&, int axis) const;
    void clear_padding(const Tensor&, int axis, CudaExecutionView) const;
};

enum class TpLayout { Sharded, Replicated };

// A boundary payload consists of ordered tensor fields. Already sharded fields travel
// unchanged. An equally divisible replica travels in TP-lane order and is gathered at
// the receiver. Uneven replicas can still travel intact.
struct StageBoundaryField {
    Tensor transfer;
    Tensor replica;
    static std::size_t transfer_elements(std::size_t elements, TpLayout layout,
                                         std::uint32_t tensor_ranks);
    static StageBoundaryField bind(const Tensor& local, TpLayout layout, std::uint32_t rank,
                                   std::uint32_t ranks);
};

struct SequenceParallelPolicy {
    std::uint32_t ranks          = 1;
    std::int32_t minimum_columns = 0;

    [[nodiscard]] TokenPartition partition(std::int32_t columns, std::uint32_t rank) const;
    [[nodiscard]] std::int32_t local_capacity(std::int32_t maximum_columns) const;
};

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
    void sum_scatter(const Tensor&, Tensor&, CudaExecutionView);
    // Each U32 word has one contributing rank; all other ranks supply zero.
    void merge_unique(Tensor&, CudaExecutionView);
    void gather(const Tensor&, Tensor&, CudaExecutionView);
    // Vocabulary Ops use I64 storage. Gather the same bits as unsigned records.
    void gather_unsigned_words(const Tensor&, Tensor&, CudaExecutionView);
    void broadcast(Tensor&, CudaExecutionView);
    void broadcast(Tensor&, std::uint32_t owner, CudaExecutionView);
    void exchange(std::span<const distributed::PeerTransfer>, CudaExecutionView);
    void track_graph_group(const distributed::Group*);
    [[nodiscard]] bool supports_graph_gating() const noexcept;
    [[nodiscard]] CommunicationGraphWork captured_work() const;
    void record_graph(CommunicationGraphWork&, const std::function<void()>&);
    void replayed(const CommunicationGraphWork&) const;

private:
    ParallelBinding binding_;
    std::vector<const distributed::Group*> graph_groups_;
};

} // namespace ninfer::models
