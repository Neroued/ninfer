#pragma once

#include "models/common/moe/execution.h"
#include "models/common/moe/placement.h"
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
    // The rank-load digest binds weights to the installed topology. This description lets
    // participants match layer, peer order and physical envelopes before launching any exchange.
    void encode(distributed::wire::Encoder&, int physical_columns) const;
    void expect(distributed::wire::Decoder&, int physical_columns) const;
    friend bool operator==(const ExpertExchangeStage&, const ExpertExchangeStage&) = default;
};

[[nodiscard]] ExpertExchangeStage prepare_expert_exchange(const distributed::GroupDescription&,
                                                          const ExpertDomain&, ExpertBankId,
                                                          int hidden, int top_k, int block_columns);

inline constexpr int expert_exchange_block_columns = 64;
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

// Compatible banks share one in-flight frame, including its send/return buffers. Enqueues on the
// supplied ordered stream; backing and source views survive all consumers, including
// capture/replay. reset/destroy after retirement.
class ExpertExchange {
public:
    ExpertExchange(ExpertExchangeStage, const distributed::Group&);

    [[nodiscard]] std::size_t bytes() const { return storage_.size(); }

    [[nodiscard]] std::size_t routed_workspace_bytes(const ops::RoutedSwiGLUProfile&) const;
    [[nodiscard]] std::size_t softmax_workspace_bytes(const ops::MoeProfile&) const;
    // A null source still services every peer. Optional U64[4,peers] observation is reset
    // per call and contains tx unique rows, tx assignments, rx unique rows, rx assignments.
    void softmax_layer(const DeviceExpertPlacement&, const Parameters&, int physical_columns,
                       const LayerSource*, MemoryArena&, CudaExecutionView,
                       Tensor* observation = nullptr);
    // For a model with its own router and shared term. Source routes are global and shared is
    // its already scaled FP32 contribution. Both are ignored when the local source is absent.
    void routed_layer(const DeviceExpertPlacement&, const ops::RoutedSwiGLUProfile&,
                      const ops::CudaSwiGLUExperts&, int physical_columns, const LayerSource*,
                      const ops::ExpertRoutes*, const Tensor* shared, MemoryArena&,
                      CudaExecutionView, Tensor* observation = nullptr);

private:
    struct Buffers {
        ops::ExpertPeerBuffers view;
        Tensor counts, hidden, partial;
    };

    void block(const DeviceExpertPlacement&, const ops::RoutedSwiGLUProfile&,
               const ops::CudaSwiGLUExperts&, const LayerSource*, const ops::ExpertRoutes*,
               const Tensor*, MemoryArena&, CudaExecutionView, Tensor*);
    void check(const DeviceExpertPlacement&, const ops::RoutedSwiGLUProfile&, int,
               const LayerSource*, Tensor*, CudaExecutionView) const;
    ExpertExchangeStage stage_;
    const distributed::Group& group_;
    DeviceBuffer storage_;
    ops::ExpertPeerBuffers* device_peers_ = nullptr;
    std::vector<Buffers> buffers_;
    std::vector<distributed::PeerTransfer> send_, back_;
};

} // namespace ninfer::models::moe
