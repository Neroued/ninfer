#pragma once

#include "models/common/moe/execution.h"
#include "models/common/parallel_execution.h"
#include "models/common/moe/placement.h"
#include "models/common/moe/step.h"
#include "distributed/communication.h"
#include "distributed/wire.h"
#include "ninfer/ops/moe_exchange.h"

namespace ninfer::models::moe {

// Logical group members, per-destination send sizes and return size completely specify one
// dispatch/return pair. A committed physical column extent determines the same block count
// on every peer, independently of local active rows.
struct ExpertExchangeStage {
    distributed::GroupId group;
    ExpertBankId bank;
    std::vector<distributed::RankId> members;
    std::vector<int> slots;
    int experts, hidden, top_k, block_columns;
    std::uint64_t placement_signature;
    std::vector<IndexRange> intermediate;
    std::vector<std::size_t> dispatch_bytes;
    std::size_t return_bytes;

    [[nodiscard]] distributed::GroupRequirements requirements(unsigned peer) const;
    [[nodiscard]] int blocks(int physical_columns) const;
    [[nodiscard]] std::size_t frame_bytes(unsigned peer) const;
    [[nodiscard]] std::size_t routed_workspace_bytes(unsigned peer,
                                                     const ops::RoutedSwiGLUProfile&) const;
    [[nodiscard]] std::size_t softmax_workspace_bytes(unsigned peer, const ops::MoeProfile&) const;
    [[nodiscard]] std::size_t sigmoid_workspace_bytes(unsigned peer,
                                                      const ops::SigmoidMoeProfile&) const;
    // The rank-load digest binds weights to the installed topology. This description lets
    // participants match layer, peer order and physical envelopes before launching any exchange.
    void encode(distributed::wire::Encoder&, int physical_columns) const;
    void expect(distributed::wire::Decoder&, int physical_columns) const;
    friend bool operator==(const ExpertExchangeStage&, const ExpertExchangeStage&) = default;
};

[[nodiscard]] ExpertExchangeStage prepare_expert_exchange(const distributed::GroupDescription&,
                                                          const ExpertDomain&, ExpertBankId,
                                                          int hidden, int top_k, int block_columns);

[[nodiscard]] distributed::GroupRequirements
expert_exchange_requirements(const distributed::GroupDescription&, const ExpertDomain&,
                             unsigned peer, int hidden, int top_k, int block_columns);

// Immutable maps live with the prepared layer, independently of in-flight frame storage.
class DeviceExpertPlacement {
public:
    DeviceExpertPlacement(const ExpertDomain&, const ExpertExchangeStage&, unsigned peer, Device);

    [[nodiscard]] std::size_t bytes() const { return storage_.size(); }

    ops::ExpertDispatchView dispatch;
    Tensor logical_ids;
    const ExpertExchangeStage stage;
    const unsigned peer;
private:
    DeviceBuffer storage_;
};

class ResidentExchangePlan {
public:
    ResidentExchangePlan(std::shared_ptr<const ExpertDomain>, const distributed::GroupDescription&,
                         unsigned peer, int hidden, int top_k);

    [[nodiscard]] const ExpertExchangeStage& stage(ExpertBankId bank) const {
        return stages_.at(bank.index);
    }

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    [[nodiscard]] std::unique_ptr<ResidentExpertExchange> prepare(const distributed::Group&) const;
private:
    std::shared_ptr<const ExpertDomain> domain_;
    unsigned peer_;
    std::vector<ExpertExchangeStage> stages_;
    std::vector<std::size_t> frame_banks_;
    std::vector<std::size_t> bank_frames_;
    std::size_t bytes_ = 0;
};

// Inputs received from one source and its destination for weighted FP32 row results.
struct ReceivedExpertRows {
    ops::ExpertPacket packet;
    Tensor counts, hidden, partial;
};

class ExpertExchange {
public:
    ExpertExchange(ExpertExchangeStage, const distributed::Group&);
    [[nodiscard]] std::size_t bytes() const { return storage_.size(); }

    // One ordered stream owns the frame. The call borrows the source through combine.
    // A failed or abandoned dispatch must be drained with its execution group.
    class Call {
    public:
        Call(const Call&)            = delete;
        Call& operator=(const Call&) = delete;

        [[nodiscard]] std::span<const ReceivedExpertRows> received() const;
        void combine(const Tensor* shared, Tensor* observation) &&;
    private:
        friend class ExpertExchange;

        Call(ExpertExchange& frame, const LayerSource* source, CudaExecutionView exec)
            : frame_(&frame), source_(source), exec_(exec) {}

        ExpertExchange* frame_;
        const LayerSource* source_;
        CudaExecutionView exec_;
    };

    void validate(const DeviceExpertPlacement&, const ops::RoutedSwiGLUProfile&, int extent,
                  const LayerSource*, Tensor* observation, CudaExecutionView) const;
    [[nodiscard]] Call dispatch(const DeviceExpertPlacement&, const LayerSource*,
                                const ops::ExpertRoutes*, CudaExecutionView);
private:
    ExpertExchangeStage stage_;
    const distributed::Group& group_;
    DeviceBuffer storage_;
    ops::ExpertPeerBuffers* device_peers_ = nullptr;
    std::vector<ReceivedExpertRows> received_;
    std::vector<distributed::PeerTransfer> send_, back_;
    bool occupied_ = false;
};

// The three callers supply distinct model mathematics. They share the same dispatch,
// expert and combine path, including when a rank only services other sources.
void softmax_exchange(const ExpertExchangeWork&, const Parameters&, const LayerSource*,
                      MemoryArena&, CudaExecutionView);
void sigmoid_exchange(const ExpertExchangeWork&, const ops::SigmoidMoeProfile&,
                      const ops::SigmoidMoeWeights&, const LayerSource*, bool representative,
                      ParallelExecution*, MemoryArena&, CudaExecutionView);
void routed_exchange(const ExpertExchangeWork&, const ops::RoutedSwiGLUProfile&,
                     const ops::CudaSwiGLUExperts&, const LayerSource*, const ops::ExpertRoutes*,
                     const Tensor* shared, MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::moe
