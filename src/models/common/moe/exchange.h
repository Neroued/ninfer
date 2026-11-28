#pragma once

#include "models/common/moe/execution.h"
#include "models/common/parallel_execution.h"
#include "models/common/moe/placement.h"
#include "models/common/moe/step.h"
#include "distributed/communication.h"
#include "distributed/wire.h"
#include "ninfer/ops/moe_exchange.h"
#include <utility>

namespace ninfer::models::moe {

// Logical group members, per-destination send sizes and return size completely specify one
// dispatch/return pair. A committed physical column extent determines the same block count
// on every peer, independently of local active rows.
enum class ExpertInputLayout { PacketRows, TokenRoutes };

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
    int blocks_per_window = 1;
    bool overlap                   = false;
    ExpertInputLayout input_layout = ExpertInputLayout::PacketRows;

    [[nodiscard]] int window_columns() const { return block_columns * blocks_per_window; }

    [[nodiscard]] int windows(int extent) const;
    [[nodiscard]] std::size_t layer_workspace_bytes(std::size_t per_window) const;


    [[nodiscard]] distributed::GroupRequirements requirements(unsigned peer) const;
    [[nodiscard]] int blocks(int physical_columns) const;
    [[nodiscard]] std::size_t frame_bytes(unsigned peer) const;
    [[nodiscard]] std::size_t routed_workspace_bytes(unsigned peer, const ops::RoutedSwiGLUProfile&,
                                                     std::optional<ExpertInputLayout> = {}) const;
    [[nodiscard]] std::size_t softmax_workspace_bytes(unsigned peer, const ops::MoeProfile&,
                                                      unsigned tensor_size             = 1,
                                                      std::optional<ExpertInputLayout> = {}) const;
    [[nodiscard]] std::size_t sigmoid_workspace_bytes(unsigned peer, const ops::SigmoidMoeProfile&,
                                                      unsigned tensor_size             = 1,
                                                      std::optional<ExpertInputLayout> = {}) const;
    [[nodiscard]] ExpertWindowWorkspace workspace(unsigned peer, const ops::MoeProfile&,
                                                  unsigned tensor_size, ExpertInputLayout) const;
    [[nodiscard]] ExpertWindowWorkspace workspace(unsigned peer, const ops::SigmoidMoeProfile&,
                                                  unsigned tensor_size, ExpertInputLayout) const;
    // The rank-load digest binds weights to the installed topology. This description lets
    // participants match layer, peer order and physical envelopes before launching any exchange.
    void encode(distributed::wire::Encoder&, int physical_columns) const;
    void expect(distributed::wire::Decoder&, int physical_columns) const;
    friend bool operator==(const ExpertExchangeStage&, const ExpertExchangeStage&) = default;
};

[[nodiscard]] ExpertExchangeStage prepare_expert_exchange(const distributed::GroupDescription&,
                                                          const ExpertDomain&, ExpertBankId,
                                                          int hidden, int top_k, int block_columns,
                                                          int blocks_per_window = 1);

// Immutable maps live with the prepared layer, independently of in-flight frame storage.
class DeviceExpertPlacement {
public:
    DeviceExpertPlacement(const ExpertDomain&, const ExpertExchangeStage&, unsigned peer, Device);

    [[nodiscard]] std::size_t bytes() const { return storage_.size(); }

    ops::ExpertDispatchView dispatch;
    Tensor logical_ids;
    const ExpertExchangeStage stage;
    const unsigned peer;
    const std::vector<std::uint32_t> peer_owners;
private:
    DeviceBuffer storage_;
};

class ResidentExchangePlan {
public:
    ResidentExchangePlan(std::shared_ptr<const ExpertDomain>, const distributed::GroupDescription&,
                         unsigned peer, int hidden, int top_k, ExchangeGeometry geometry = {64, 1});

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    template <class Profile>
    [[nodiscard]] std::size_t
    bind_layer(ExpertBankId, const Profile&, unsigned tensor_size,
               ExpertInputLayout preferred = ExpertInputLayout::PacketRows);
    [[nodiscard]] ExchangeGeometry geometry() const noexcept { return geometry_; }

    [[nodiscard]] distributed::GroupRequirements requirements() const;

    [[nodiscard]] std::unique_ptr<ResidentExpertExchange> prepare(const distributed::Group&,
                                                                  const DeviceContext&) const;
private:
    std::shared_ptr<const ExpertDomain> domain_;
    unsigned peer_;
    ExchangeGeometry geometry_;
    std::vector<ExpertExchangeStage> stages_;
    std::vector<std::vector<ExpertExchangeStage>> frame_shapes_;
    std::size_t payload_bytes_ = 0;
    std::vector<std::size_t> bank_frames_;
    std::vector<std::vector<ExpertWindowWorkspace>> workspaces_;
    std::size_t bytes_ = 0;
};

// One auxiliary stream per Program, two completion records for the two packet slots.
class ExpertOverlapResources {
public:
    explicit ExpertOverlapResources(const DeviceContext&);
    void join_after_failure(CudaExecutionView) noexcept;

    struct Events {
        explicit Events(const DeviceContext& device)
            : inputs(device), received(device), done(device) {}

        CudaCompletionEvent inputs, received, done;
    };

    CudaStream stream;
    CudaExecutionView compute;
    std::array<Events, 2> slots;
};

// Packet shapes share the largest payload backing. Their immutable address tables
// remain distinct so a captured graph never observes a later shape's descriptors.
struct ExpertExchangeStorage {
    ExpertExchangeStorage(Device device, std::size_t bytes) : payload(device, bytes) {
        payload.fill(0);
    }

    DeviceBuffer payload;
    bool occupied = false;
};

class ExpertExchange {
public:
    ExpertExchange(ExpertExchangeStage, const distributed::Group&);
    ExpertExchange(ExpertExchangeStage, const distributed::Group&,
                   std::shared_ptr<ExpertExchangeStorage>);

    [[nodiscard]] const ExpertExchangeStage& stage() const noexcept { return stage_; }

    [[nodiscard]] std::size_t bytes() const { return metadata_.size() + storage_->payload.size(); }

    [[nodiscard]] Tensor unit_coefficients(int columns) const { return unit_.slice(0, 0, columns); }

    // One ordered stream owns the frame. The call borrows the source through combine.
    // A failed or abandoned dispatch must be drained with its execution group.
    class Call {
    public:
        Call(Call&& other) noexcept
            : frame_(std::exchange(other.frame_, nullptr)), source_(other.source_),
              packets_(other.packets_), exec_(other.exec_) {}
        Call(const Call&)            = delete;
        Call& operator=(const Call&) = delete;

        [[nodiscard]] ops::ExpertPacketBatch received() const;
        void combine(const Tensor* shared, Tensor* observation,
                     const Tensor* shared_gate = nullptr) &&;
    private:
        friend class ExpertExchange;

        Call(ExpertExchange& frame, const LayerSource* source, int packets, CudaExecutionView exec)
            : frame_(&frame), source_(source), packets_(packets), exec_(exec) {}

        ExpertExchange* frame_;
        const LayerSource* source_;
        int packets_;
        CudaExecutionView exec_;
    };

    void validate(const DeviceExpertPlacement&, const ops::RoutedSwiGLUProfile&, int extent,
                  const LayerSource*, Tensor* observation, CudaExecutionView) const;
    [[nodiscard]] Call dispatch(const DeviceExpertPlacement&, const LayerSource*,
                                const ops::ExpertRoutes*, int matched_columns, CudaExecutionView,
                                std::span<const std::uint8_t> source_owners = {});
private:
    ExpertExchangeStage stage_;
    const distributed::Group& group_;
    std::shared_ptr<ExpertExchangeStorage> storage_;
    DeviceBuffer metadata_;
    ops::ExpertPeerBuffers* device_peers_ = nullptr;
    std::vector<distributed::PeerTransfer> send_, back_;
    std::vector<distributed::PeerTransfer> selected_send_, selected_back_;
    bool sparse_ = false;
    Tensor unit_;
};

// The three callers supply distinct model mathematics. They share the same dispatch,
// expert and combine path, including when a rank only services other sources.
void softmax_exchange(const ExpertExchangeWork&, const Parameters&, const LayerSource*,
                      MemoryArena&, CudaExecutionView, ParallelExecution* = nullptr,
                      const TokenPartition* = nullptr);
void sigmoid_exchange(const ExpertExchangeWork&, const ops::SigmoidMoeProfile&,
                      const ops::SigmoidMoeWeights&, const LayerSource*, ParallelExecution*,
                      MemoryArena&, CudaExecutionView, const TokenPartition* = nullptr);
void routed_exchange(const ExpertExchangeWork&, const ops::RoutedSwiGLUProfile&,
                     const ops::CudaSwiGLUExperts&, const LayerSource*, const ops::ExpertRoutes*,
                     const Tensor* shared, MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::moe
