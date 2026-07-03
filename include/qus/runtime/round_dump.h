#pragma once

#include "qus/core/kv_cache.h"
#include "qus/core/state_store.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <filesystem>

namespace qus {

void write_mtp_round_state_dump(const std::filesystem::path& out_dir, std::uint64_t round_index,
                                std::uint32_t committed_length, const GdnState& gdn,
                                const KVCache& mtp_kv, cudaStream_t stream);

} // namespace qus
