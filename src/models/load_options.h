#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <string_view>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose          = EnginePurpose::Generation;
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;
    // Tensor-parallel degree (1 or 2). At 2 the dense Text, MTP and output head are split
    // across both devices of an ExecutionContext; Vision is held whole by `vision_rank` and a
    // masked draft or proposal head by rank 0 (models/qwen3_5/load/sharding.h).
    int tp          = 1;
    int vision_rank = 0;
    // text/token_embedding stays in one mapped page-locked host copy that every rank reads in
    // place (ShardAxis::HostMapped) instead of a device copy per rank.
    bool embedding_host = false;
    // Qwen4Exp at tp 2: split every hyper-connection mixer by residual streams (rank r holds
    // streams [r*hc/2, (r+1)*hc/2) of the wide residual and the matching norm, Down/inject columns
    // and Up rows; models/qwen3_5/load/sharding.h) instead of replicating it on both ranks. Halves
    // the mixers' bytes per rank for two all-reduces per mixer. Ignored by other architectures and
    // at tp 1.
    bool hc_split_streams = false;
    // Most bytes of the host-mapped parents to keep page-locked (EngineOptions); 0 = no limit.
    std::uint64_t host_pinned_max_bytes = 0;

    bool operator==(const LoadOptions&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        }
        return {};
    }
};

// The rank whose device holds the Vision tower: the index of `vision_device` in `devices`, and
// rank 0 when it is unset. Engine validation rejects a `vision_device` outside `devices`.
[[nodiscard]] inline int vision_rank(const EngineOptions& options) noexcept {
    if (!options.vision_device) { return 0; }
    const auto found =
        std::find(options.devices.begin(), options.devices.end(), *options.vision_device);
    return found == options.devices.end() ? 0 : static_cast<int>(found - options.devices.begin());
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose               = options.purpose,
            .vision                = options.enable_vision,
            .speculative           = options.speculative.backend,
            .proposal_head         = options.speculative.proposal_head,
            .tp                    = options.tp,
            .vision_rank           = vision_rank(options),
            .embedding_host        = options.embedding_host,
            .hc_split_streams      = options.tp_hc_split_streams,
            .host_pinned_max_bytes = options.host_pinned_max_bytes};
}

} // namespace ninfer::models
