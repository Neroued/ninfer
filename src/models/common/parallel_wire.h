#pragma once

#include "models/common/parallel.h"
#include "models/common/token_input.h"
#include "distributed/wire.h"
#include "ninfer/types.h"

namespace ninfer::models {

void encode_token_counts(distributed::wire::Encoder&, const TokenCounts&);
[[nodiscard]] TokenCounts decode_token_counts(distributed::wire::Decoder&);

void encode_expert_partitions(distributed::wire::Encoder&, std::span<const ExpertPartition>);
[[nodiscard]] std::vector<ExpertPartition> decode_expert_partitions(distributed::wire::Decoder&);

void encode_assemblies(distributed::wire::Encoder&, std::span<const ParameterAssembly>);
[[nodiscard]] std::vector<ParameterAssembly> decode_assemblies(distributed::wire::Decoder&);

// Only options consumed by rank preparation/execution cross this boundary. Frontend,
// deployment and coordinator capacity policy remain with their owners.
void encode_rank_options(distributed::wire::Encoder&, const EngineOptions&);
[[nodiscard]] EngineOptions decode_rank_options(distributed::wire::Decoder&);

} // namespace ninfer::models
