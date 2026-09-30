#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::runtime {

// Engine-side ownership of one request lane.
enum class LogicalLaneState : std::uint8_t {
    Free,
    Materializing,
    Active,
    TerminalPending,
};

// Engine-side driver of the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache.md §8).
// Retention policy lives in the Program's prefix index; this class only tracks lane ownership and
// forwards admission, prefetch and terminal settlement.
template <class ModelContract>
class HybridResourceManager {
public:
    using Program                      = typename ModelContract::Program;
    using PreparedPrompt               = typename ModelContract::PreparedPrompt;
    using RequestBasePlan              = typename ModelContract::RequestBasePlan;
    using PersistentBackfillProof      = typename ModelContract::PersistentBackfillProof;
    using SequenceHandle               = typename ModelContract::SequenceHandle;
    using StartResult                  = typename ModelContract::StartResult;
    using FinishResult                 = typename ModelContract::FinishResult;
    using AbortResult                  = typename ModelContract::AbortResult;
    using Quote                        = typename ModelContract::HybridAdmissionQuote;
    using ProgramMaterializationResult = typename ModelContract::MaterializationResult;

    class Choice {
    public:
        Choice(Choice&&) noexcept        = default;
        Choice& operator=(Choice&&)      = delete;
        Choice(const Choice&)            = delete;
        Choice& operator=(const Choice&) = delete;

        [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return quote_.summary; }

        [[nodiscard]] LaneId destination() const noexcept { return quote_.destination; }

    private:
        explicit Choice(Quote&& quote) : quote_(std::move(quote)) {}

        Quote quote_;

        friend class HybridResourceManager;
    };

    class PublishedActivation {
    public:
        PublishedActivation(PublishedActivation&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), result_(std::move(other.result_)),
              destination_(other.destination_) {}

        PublishedActivation& operator=(PublishedActivation&&)      = delete;
        PublishedActivation(const PublishedActivation&)            = delete;
        PublishedActivation& operator=(const PublishedActivation&) = delete;

        [[nodiscard]] const SequenceHandle& sequence() const {
            if (!result_) { throw std::logic_error("published activation is empty"); }
            return result_->sequence;
        }

    private:
        PublishedActivation(HybridResourceManager& owner, StartResult&& result, LaneId destination)
            : owner_(&owner), result_(std::move(result)), destination_(destination) {}

        HybridResourceManager* owner_ = nullptr;
        std::optional<StartResult> result_;
        LaneId destination_{};

        friend class HybridResourceManager;
    };

    struct MaterializationOutcome {
        ContextTransactionStatus status = ContextTransactionStatus::Aborted;
        std::optional<PublishedActivation> activation;
        MaterializationDiagnostics diagnostics;
    };

    enum class MaterializationReserveResult : std::uint8_t {
        Reserved,
        Stale,
        Aborted,
    };

    using ContextTransactionOutcome =
        std::variant<ContextTransactionInProgress, MaterializationOutcome>;

    struct Inspection {
        Readiness readiness = Readiness::TemporarilyBlocked;
        std::optional<Choice> choice;
    };

    explicit HybridResourceManager(std::uint32_t lane_count) : lane_count_(lane_count) {
        if (lane_count == 0 || lane_count > kMaximumConcurrency) {
            throw std::invalid_argument("hybrid resource-manager lane count is invalid");
        }
    }

    [[nodiscard]] Inspection inspect(Program& program, const PreparedPrompt& prompt,
                                     const RequestBasePlan& base, std::uint64_t publication_order) {
        if (open_lane_ || program.has_context_transaction()) {
            return {.readiness = Readiness::TemporarilyBlocked};
        }
        if (publication_order == 0) {
            throw std::invalid_argument("request publication order is zero");
        }
        if (!program.isolated_request_feasible(base)) {
            return {.readiness = Readiness::PermanentlyInfeasible};
        }
        std::optional<LaneId> destination;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            if (lanes_[lane] == Lane::Free) {
                destination = LaneId{lane};
                break;
            }
        }
        if (!destination) { return {.readiness = Readiness::TemporarilyBlocked}; }
        Quote quote = program.hybrid_quote(prompt, base, *destination);
        if (quote.readiness != Readiness::Ready) { return {.readiness = quote.readiness}; }
        return {.readiness = Readiness::Ready, .choice = Choice(std::move(quote))};
    }

    // Backfill past a blocked FIFO head: the Program proves the head's root entitlement stays
    // admissible beside every persistent borrower and the candidate once the head's donors finish.
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(Program& program, const RequestBasePlan& blocked_head,
                              const Choice& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const {
        return program.prove_persistent_backfill(blocked_head, candidate.quote_,
                                                 persistent_borrowers);
    }

    // A blocked FIFO head's Host-only blocks are copied into spare Device cache while it waits
    // (hybrid-prefix-cache §6.6). Matching a long prompt walks its path, so a new attempt
    // needs a new head, progress on the last attempt, or more room.
    void prefetch_blocked_head(Program& program, const PreparedPrompt& prompt,
                               const RequestBasePlan& base, std::uint64_t publication_order) {
        if (open_lane_ || program.has_context_transaction()) { return; }
        if (publication_order == prefetch_order_ && !prefetch_retry_ &&
            program.hybrid_prefetch_room() <= prefetch_room_) {
            return;
        }
        const std::optional<std::uint32_t> started = program.hybrid_prefetch(prompt, base);
        prefetch_order_                            = publication_order;
        // A prefetch still in flight, or one that copied blocks, may leave more to copy.
        prefetch_retry_ = !started || *started != 0;
        prefetch_room_  = program.hybrid_prefetch_room();
    }

    [[nodiscard]] MaterializationReserveResult
    reserve_materialization(Program& program, Choice&& choice, PreparedPrompt&& prompt,
                            CancellationFlagView cancellation) {
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("hybrid admission overlaps an open resource transaction");
        }
        const LaneId lane = choice.destination();
        require_lane(lane, Lane::Free);
        const ContextTransactionReserveStatus status = program.hybrid_reserve_materialization(
            std::move(choice.quote_), std::move(prompt), cancellation);
        if (status == ContextTransactionReserveStatus::Aborted) {
            return cancellation.requested() ? MaterializationReserveResult::Aborted
                                            : MaterializationReserveResult::Stale;
        }
        lanes_[lane.value] = Lane::Materializing;
        open_lane_         = lane;
        return MaterializationReserveResult::Reserved;
    }

    [[nodiscard]] std::optional<ContextTransactionKind> context_transaction_kind() const noexcept {
        if (open_lane_) { return ContextTransactionKind::Materialization; }
        return std::nullopt;
    }

    [[nodiscard]] ContextTransactionOutcome
    progress_context_transaction(Program& program, CancellationFlagView cancellation) {
        if (!open_lane_ || !program.has_context_transaction()) {
            throw std::logic_error("hybrid manager has no progressable admission");
        }
        auto progress = program.progress_context_transaction(cancellation);
        if (std::holds_alternative<ContextTransactionInProgress>(progress)) {
            return ContextTransactionInProgress{};
        }
        auto* result = std::get_if<ProgramMaterializationResult>(&progress);
        if (result == nullptr) {
            throw std::logic_error("hybrid admission returned a non-materialization result");
        }
        const LaneId lane = *open_lane_;
        MaterializationOutcome outcome;
        outcome.status      = result->status;
        outcome.diagnostics = result->diagnostics;
        if (result->status == ContextTransactionStatus::Published) {
            if (!result->published) {
                throw std::logic_error("published hybrid admission has no sequence");
            }
            outcome.activation.emplace(
                PublishedActivation(*this, std::move(*result->published), lane));
            return outcome;
        }
        lanes_[lane.value] = Lane::Free;
        open_lane_.reset();
        return outcome;
    }

    void adopt(Program& program, PublishedActivation&& activation) noexcept {
        if (activation.owner_ != this || !activation.result_ || !open_lane_ ||
            activation.destination_.value != open_lane_->value ||
            lanes_[activation.destination_.value] != Lane::Materializing) {
            std::terminate();
        }
        lanes_[activation.destination_.value] = Lane::Active;
        activation.result_.reset();
        activation.owner_ = nullptr;
        open_lane_.reset();
        program.finalize_context_transaction();
    }

    [[nodiscard]] FinishResult finish(Program& program, LaneId lane, SequenceHandle sequence) {
        require_lane(lane, Lane::TerminalPending);
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("terminal finish overlaps an open resource transaction");
        }
        FinishResult result = program.finish(sequence);
        if (result.status == ConsumeStatus::Consumed) {
            lanes_[lane.value] = Lane::Free;
            return result;
        }
        AbortResult discarded = program.abort(sequence);
        if (discarded.status != ConsumeStatus::Consumed) {
            throw std::logic_error("Program could neither publish nor discard a sequence");
        }
        lanes_[lane.value] = Lane::Free;
        FinishResult released;
        released.status      = ConsumeStatus::Consumed;
        released.timings     = discarded.timings;
        released.speculative = std::move(discarded.speculative);
        return released;
    }

    [[nodiscard]] AbortResult abort(Program& program, LaneId lane, SequenceHandle sequence) {
        if (open_lane_ || program.has_context_transaction()) {
            throw std::logic_error("terminal abort overlaps an open resource transaction");
        }
        if (lanes_.at(lane.value) != Lane::Active &&
            lanes_.at(lane.value) != Lane::TerminalPending) {
            throw std::logic_error("aborted lane has no active owner");
        }
        AbortResult result = program.abort(sequence);
        if (result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("Program did not consume aborted sequence");
        }
        lanes_[lane.value] = Lane::Free;
        return result;
    }

    void apply_commit(std::span<const LaneId> lanes,
                      const typename ModelContract::CommitResult& result) {
        if (lanes.size() != result.row_count) {
            throw std::logic_error("commit result membership is not row aligned");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const LaneId lane = lanes[row];
            require_lane(lane, Lane::Active);
            switch (result.rows[row].disposition) {
            case CommitDisposition::Active:
                break;
            case CommitDisposition::Finishable:
                lanes_[lane.value] = Lane::TerminalPending;
                break;
            case CommitDisposition::CancelledReleased:
                lanes_[lane.value] = Lane::Free;
                break;
            }
        }
    }

    void apply_discard(std::span<const LaneId> lanes,
                       const typename ModelContract::DiscardResult& result) {
        if (lanes.size() != result.row_count || result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("pending discard did not consume its membership");
        }
        for (const LaneId lane : lanes) { release_lane(lane); }
    }

    void release_failed_commit(std::span<const LaneId> lanes) noexcept {
        for (const LaneId lane : lanes) {
            if (lane.value < lane_count_ && (lanes_[lane.value] == Lane::Active ||
                                             lanes_[lane.value] == Lane::TerminalPending)) {
                lanes_[lane.value] = Lane::Free;
            }
        }
    }

    void populate_runtime_stats(Program& program, RuntimeStats& out) const noexcept {
        const auto usage                     = program.physical_usage();
        out.device_state_occupied_slots      = usage.device_state_slots;
        out.device_main_kv_occupied_pages    = usage.device_main_kv_pages;
        out.device_backend_kv_occupied_pages = usage.device_backend_kv_pages;
        const auto cache                     = program.hybrid_stats();
        out.cached_blocks                    = cache.device_resident_blocks;
        out.evictable_blocks                 = cache.device_evictable_blocks;
        out.tree_blocks                      = cache.nodes;
        out.snapshots                        = cache.snapshots;
        out.host_cache_capacity_bytes        = cache.host_slab_bytes * cache.host_slabs;
        out.host_cache_used_bytes =
            cache.host_slab_bytes * (cache.host_slabs - cache.host_free_slabs);
        out.blocks_inserted         = cache.blocks_inserted;
        out.blocks_reattached       = cache.blocks_reattached;
        out.blocks_duplicate        = cache.blocks_duplicate;
        out.taps_created            = cache.taps_created;
        out.taps_skipped            = cache.taps_skipped;
        out.endpoints_created       = cache.endpoints_created;
        out.host_image_writes       = cache.host_image_writes;
        out.host_block_writes       = cache.host_block_writes;
        out.host_image_restores     = cache.host_image_restores;
        out.host_block_restores     = cache.host_block_restores;
        out.host_write_bytes        = cache.host_write_bytes;
        out.host_restore_bytes      = cache.host_restore_bytes;
        out.evicted_blocks          = cache.evicted_blocks;
        out.host_snapshot_evictions = cache.host_snapshot_evictions;
        out.host_dead_reclaims      = cache.host_dead_reclaims;
        out.unbacked_node_losses    = cache.unbacked_node_losses;
    }

    [[nodiscard]] LogicalLaneState lane_state(LaneId lane) const noexcept {
        return lane.value < lane_count_ ? lanes_[lane.value] : LogicalLaneState::Free;
    }

    void clear_after_program_cleanup() noexcept {
        open_lane_.reset();
        lanes_.fill(Lane::Free);
        prefetch_order_ = 0;
        prefetch_retry_ = false;
        prefetch_room_  = 0;
    }

private:
    using Lane = LogicalLaneState;

    void require_lane(LaneId lane, Lane expected) const {
        if (lane.value >= lane_count_ || lanes_[lane.value] != expected) {
            throw std::logic_error("hybrid lane is not in its expected ownership state");
        }
    }

    void release_lane(LaneId lane) {
        if (lane.value >= lane_count_ ||
            (lanes_[lane.value] != Lane::Active && lanes_[lane.value] != Lane::TerminalPending)) {
            throw std::logic_error("released lane has no active owner");
        }
        lanes_[lane.value] = Lane::Free;
    }

    std::uint32_t lane_count_ = 0;
    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::optional<LaneId> open_lane_;
    // The last prefetch attempt: the head it served, whether to try again and the room it left.
    std::uint64_t prefetch_order_ = 0;
    bool prefetch_retry_          = false;
    std::uint32_t prefetch_room_  = 0;
};

} // namespace ninfer::runtime
