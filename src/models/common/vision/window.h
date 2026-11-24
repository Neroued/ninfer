#pragma once

#include "models/common/frontend/media.h"
#include "runtime/executor/contract.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::vision {

// One independently encodable unit. Patch offsets are relative to its prepared payload.
struct Unit {
    std::uint32_t item = 0, index = 0;
    std::uint64_t patch_begin = 0;
    std::uint32_t patches     = 0;
    std::uint32_t token_begin = 0, features = 0;
    std::uint64_t cost = 0;

    [[nodiscard]] std::uint32_t end() const { return token_begin + features; }
};

struct Placement {
    std::uint32_t unit, rank, feature_begin;
};

struct OutputBlock {
    std::uint32_t rank, feature_begin;
    std::vector<std::int32_t> token_positions;
};

struct WindowPlan {
    std::vector<Unit> units;
    std::vector<Placement> placements;
    std::vector<OutputBlock> outputs;
    std::uint32_t consume_end = 0;
    std::uint32_t features    = 0;
};

// Select complete units intersecting [begin,end), stopping before a unit that would exceed
// capacity. Output is rank-major, with monotonically ordered token positions within each rank.
[[nodiscard]] std::shared_ptr<const WindowPlan> plan_window(std::span<const Unit>,
                                                            std::uint32_t begin, std::uint32_t end,
                                                            std::uint32_t feature_capacity,
                                                            std::uint32_t ranks);
void validate_units(std::span<const Unit>, std::uint32_t feature_capacity,
                    std::uint32_t prompt_tokens);

struct EmbeddingSlice {
    std::uint32_t source_begin = 0;
    std::vector<std::int32_t> destination_columns;
};

[[nodiscard]] std::vector<EmbeddingSlice> embedding_slices(const WindowPlan&, std::uint32_t begin,
                                                           std::uint32_t tokens);

struct WindowUse {
    std::uint64_t id = 0;
    runtime::ExecutionMember owner;
    std::shared_ptr<const WindowPlan> plan;
    bool produce = false;
};

// Coordinator-owned single output lease. Logical completion never releases an in-flight
// reader; the native Session reports consumption and the Executor reports retirement.
class Window {
public:
    [[nodiscard]] bool available(runtime::ExecutionMember) const noexcept;
    [[nodiscard]] std::optional<runtime::ExecutionMember> owner() const noexcept;
    [[nodiscard]] const WindowUse* current() const noexcept;
    [[nodiscard]] std::uint32_t limit(runtime::ExecutionMember, std::span<const Unit>,
                                      std::uint32_t begin, std::uint32_t end,
                                      std::uint32_t capacity) const;
    [[nodiscard]] WindowUse reserve(runtime::ExecutionMember, std::shared_ptr<const WindowPlan>);
    void submitted(std::uint64_t id, runtime::ExecutionTicket);
    void consumed(std::uint64_t id, bool complete);
    void retired(runtime::ExecutionTicket);
    void revoke(runtime::ExecutionMember);

private:
    std::optional<WindowUse> active_;
    std::optional<runtime::ExecutionTicket> reader_;
    std::uint64_t generation_ = 0;
    bool produced_            = false;
    bool release_             = false;
};

} // namespace ninfer::models::vision
