#pragma once

#include "models/common/batch.h"
#include "ninfer/observability/execution.h"

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::speculative {

struct RoundMember {
    std::uint32_t owner         = 0;
    std::uint32_t hard_limit    = 0;
    std::uint32_t granted_limit = 0;
};

// A missing graph entry denotes Eager. Width one denotes an ordinary decode.
struct RoundRoute {
    PhysicalBatchShape shape;
    std::uint32_t width = 1, mtp_steps = 0, head_columns = 0;
    std::optional<std::uint32_t> graph_entry;

    [[nodiscard]] static RoundRoute eager(PhysicalBatchShape, SpeculativeBackend,
                                          std::uint32_t max_drafts, std::uint32_t depth);
    [[nodiscard]] std::uint32_t verification_limit(SpeculativeBackend) const noexcept;
    [[nodiscard]] bool covers(SpeculativeBackend, std::uint32_t depth) const noexcept;
    void validate(SpeculativeBackend, std::uint32_t max_drafts) const;
    auto operator<=>(const RoundRoute&) const = default;
};

class DecodeGraphRoutes {
public:
    DecodeGraphRoutes(const BatchExecutionPlan&, SpeculativeBackend, std::uint32_t draft_window,
                      std::span<const std::uint32_t> target_widths = {});
    [[nodiscard]] std::span<const RoundRoute> at(PhysicalBatchShape shape) const;
    [[nodiscard]] std::vector<std::uint32_t> target_widths() const;
    [[nodiscard]] RoundRoute select_route(PhysicalBatchShape, SpeculativeBackend,
                                          std::uint32_t depth) const;

private:
    std::map<PhysicalBatchShape, std::vector<RoundRoute>> buckets_;
    SpeculativeBackend backend_;
};

struct MemberSlot {
    std::uint32_t global, owner, local;
    bool operator==(const MemberSlot&) const = default;
};

class DecodeRoundPlan {
public:
    [[nodiscard]] const RoundRoute& route() const noexcept { return route_; }

    [[nodiscard]] std::span<const std::uint32_t> verify_drafts() const noexcept { return widths_; }

    [[nodiscard]] std::span<const MemberSlot> slots() const noexcept { return slots_; }

    [[nodiscard]] std::span<const std::uint64_t> owner_target_tokens() const noexcept {
        return owner_tokens_;
    }

    [[nodiscard]] std::uint64_t valid_target_tokens() const noexcept { return target_tokens_; }

    [[nodiscard]] static DecodeRoundPlan freeze(SpeculativeBackend, std::uint32_t max_drafts,
                                                std::uint32_t owner_count,
                                                std::span<const RoundMember>, RoundRoute,
                                                std::vector<std::uint32_t> verify_drafts);
    [[nodiscard]] static DecodeRoundPlan
    select(SpeculativeBackend, std::uint32_t max_drafts, std::uint32_t owner_count,
           PhysicalBatchShape, ExecutionPath, const DecodeGraphRoutes*,
           std::span<const RoundMember>, std::span<const std::uint32_t> requested_drafts = {});
    [[nodiscard]] SpeculativeExecutionGeometry geometry(ExecutionPath) const;
private:
    DecodeRoundPlan() = default;
    RoundRoute route_;
    std::vector<std::uint32_t> widths_;
    std::vector<MemberSlot> slots_;
    std::vector<std::uint64_t> owner_tokens_;
    std::uint64_t target_tokens_ = 0;
};

} // namespace ninfer::models::speculative
