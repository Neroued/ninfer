#pragma once

#include "models/common/batch.h"
#include "models/common/moe/placement.h"
#include "models/common/speculative/controller.h"
#include "models/common/token_input.h"
#include "distributed/wire.h"
#include "ninfer/types.h"

namespace ninfer::models {

void encode_batch_execution_spec(distributed::wire::Encoder&, const BatchExecutionSpec&);
[[nodiscard]] BatchExecutionSpec decode_batch_execution_spec(distributed::wire::Decoder&);

void encode_token_counts(distributed::wire::Encoder&, const TokenCounts&);
[[nodiscard]] TokenCounts decode_token_counts(distributed::wire::Decoder&);

void encode_expert_partitions(distributed::wire::Encoder&, std::span<const ExpertPartition>);
[[nodiscard]] std::vector<ExpertPartition> decode_expert_partitions(distributed::wire::Decoder&);

void encode_expert_domain(distributed::wire::Encoder&, const moe::ExpertDomain*);
[[nodiscard]] std::shared_ptr<const moe::ExpertDomain>
decode_expert_domain(distributed::wire::Decoder&);

void encode_weight_ids(distributed::wire::Encoder&, std::span<const WeightId>);
[[nodiscard]] std::vector<WeightId> decode_weight_ids(distributed::wire::Decoder&);

void encode_assemblies(distributed::wire::Encoder&, std::span<const ParameterAssembly>);
[[nodiscard]] std::vector<ParameterAssembly> decode_assemblies(distributed::wire::Decoder&);

inline constexpr std::size_t kSpeculativeStateWireBytes =
    10 * speculative::kMaxSpeculativeDrafts + 6;
inline constexpr std::size_t kSpeculativeCostCellWireBytes = 4 * 4 + 2 + 2 * 8;
inline constexpr std::size_t kSpeculativeCostModelWireBytes =
    1 + 5 * 4 + 8 * std::tuple_size_v<speculative::SpeculativeCostModel::Features> + 1 + 4 * 8 +
    speculative::kMaxOnlineSpeculativeCostCells * kSpeculativeCostCellWireBytes;

void encode_speculative_state(distributed::wire::Encoder&,
                              const speculative::SequenceSpeculativeState&);
[[nodiscard]] speculative::SequenceSpeculativeState
decode_speculative_state(distributed::wire::Decoder&);

void encode_speculative_cost_model(distributed::wire::Encoder&,
                                   const speculative::SpeculativeCostModel&);
[[nodiscard]] speculative::SpeculativeCostModel
decode_speculative_cost_model(distributed::wire::Decoder&);

// Only options consumed by rank preparation/execution cross this boundary. Frontend,
// deployment and coordinator capacity policy remain with their owners.
void encode_rank_options(distributed::wire::Encoder&, const EngineOptions&);
[[nodiscard]] EngineOptions decode_rank_options(distributed::wire::Decoder&);

} // namespace ninfer::models
