#pragma once

#include "runtime/executor/collective_step.h"
#include "runtime/executor/executor.h"

namespace ninfer::runtime {

// Lives on one replica owner thread. The logical source may wait through a service step for
// another replica. Only the local physical executor can grant a frame or retire its readers.
template <class Types, class Model>
class CollectiveExecutor final : public Executor<Types> {
    using Base   = Executor<Types>;
    using Work   = typename Types::Work;
    using Commit = typename Types::Commit;
    using Event  = ExecutionEvent<Types>;
public:
    CollectiveExecutor(std::unique_ptr<Base> physical, std::shared_ptr<CollectiveStepDomain> domain,
                       std::size_t member, Model model, std::uint32_t max_decode_rows)
        : Base(max_decode_rows), physical_(std::move(physical)), domain_(std::move(domain)),
          member_(member), model_(std::move(model)), admissions_(physical_->capacity()),
          contexts_(physical_->capacity().context_frames) {
        if (physical_->capacity().compute_frames != 1 || member_ >= domain_->size())
            throw std::invalid_argument(
                "collective owner requires one compute frame and domain membership");
    }

    ~CollectiveExecutor() override {
        set_peer_wakeup({});
        if (!closed_) domain_->fail("collective executor released before drain");
    }

    void bind_to_current_thread() override { physical_->bind_to_current_thread(); }

    void set_peer_wakeup(std::function<void()> wake) override {
        domain_->set_wakeup(member_, std::move(wake));
    }

    [[nodiscard]] SessionEpoch epoch() const noexcept override { return physical_->epoch(); }

    [[nodiscard]] ExecutionCapacity capacity() const noexcept override {
        return admissions_.capacity();
    }

    [[nodiscard]] std::optional<ExecutionReservation> try_reserve(FrameKind kind) override {
        require_open();
        return admissions_.try_reserve(kind);
    }

    [[nodiscard]] ExecutionTicket submit(ExecutionReservation reservation, Work work) override {
        require_open();
        if (!reservation) throw std::logic_error("empty collective executor reservation");
        const ExecutionTicket logical{epoch(), ++next_execution_, reservation.frame()};
        if (logical.frame.kind == FrameKind::Context) {
            auto frame = physical_->try_reserve(FrameKind::Context);
            if (!frame) throw std::logic_error("admitted context has no physical frame");
            const auto ticket = physical_->submit(std::move(*frame), std::move(work));
            (void)admissions_.accept(std::move(reservation));
            contexts_.at(ticket.frame.index) = Mapping{logical, ticket};
        } else {
            auto source = model_.shape(work);
            if (source) domain_->validate(*source);
            (void)admissions_.accept(std::move(reservation));
            pending_.emplace(Pending{logical, std::move(work), std::move(source)});
            progress_or_fail();
        }
        return logical;
    }

    bool try_withdraw(ExecutionTicket ticket) override {
        if (!pending_ || pending_->ticket != ticket) return false;
        if (offered_source_ && !withdraw_offer()) {
            // A peer may already be executing our source's matching phase.
            const auto d = domain_->decision(member_, offer_);
            if (!d || d->source) return false;
        }
        admissions_.retire(ticket.frame);
        pending_.reset();
        return true;
    }

    void supply(ExecutionTicket ticket, DecisionInput input) override {
        physical_->supply(source_mapping(ticket).physical, std::move(input));
    }

    [[nodiscard]] CommitReceipt commit(ExecutionTicket ticket, Commit commit) override {
        auto receipt = physical_->commit(source_mapping(ticket).physical, std::move(commit));
        return {ticket, receipt.command};
    }

    void discard(ExecutionTicket ticket) override {
        if (pending_ && pending_->ticket == ticket) {
            pending_->discard = true;
        } else {
            physical_->discard(source_mapping(ticket).physical);
        }
    }

    [[nodiscard]] std::optional<Event> poll() override {
        if (closed_) return {};
        try {
            for (;;) {
                progress();
                auto event = physical_->poll();
                if (!event) return {};
                if (auto* error = std::get_if<ExecutionFailed>(&*event)) {
                    domain_->fail(error->message);
                    error->message = *domain_->failure();
                    return event;
                }
                const auto ticket  = std::visit([](const auto& e) { return e.ticket; }, *event);
                const bool retired = std::holds_alternative<DeviceRetired>(*event);
                if (ticket.frame.kind == FrameKind::Context) {
                    auto& mapping = contexts_.at(ticket.frame.index);
                    if (!mapping || mapping->physical != ticket)
                        throw std::logic_error(
                            "context event has no collective executor admission");
                    set_ticket(*event, mapping->logical);
                    if (retired) {
                        admissions_.retire(mapping->logical.frame);
                        mapping.reset();
                    }
                    return event;
                }
                if (!running_ || running_->physical != ticket)
                    throw std::logic_error("compute event has no physical collective frame");
                const auto source = running_->logical;
                if (retired) {
                    if (running_->step) domain_->retire(member_, running_->step);
                    running_.reset();
                    if (source) admissions_.retire(source->frame);
                } else if (!source) {
                    const auto* result = std::get_if<ResultReady<Types>>(&*event);
                    if (!result)
                        throw std::logic_error("collective service requested a logical decision");
                    (void)physical_->commit(ticket, model_.service_commit(result->result));
                }
                if (source) {
                    set_ticket(*event, *source);
                    return event;
                }
            }
        } catch (const std::exception& e) {
            fail(e.what());
            return physical_->poll();
        }
    }

    void flush() override {
        progress_or_fail();
        physical_->flush();
    }

    void wait_for_activity(std::chrono::milliseconds timeout) const override {
        physical_->wait_for_activity(timeout);
    }

    void fail(std::string reason) override {
        domain_->fail(std::move(reason));
        physical_->fail(*domain_->failure());
    }

    [[nodiscard]] bool failed() const noexcept override { return physical_->failed(); }

    [[nodiscard]] bool has_pending_work() const noexcept override {
        return !closed_ && (admissions_.busy() || running_.has_value() || domain_->ready(member_));
    }

    void finish_published_work() override {
        if (closed_) return;
        if (pending_ && !try_withdraw(pending_->ticket)) pending_->discard = true;
        if (running_ && running_->logical) physical_->discard(running_->physical);
        domain_->stop_sources(member_);
        while (!domain_->stop_drained() || running_ || pending_) {
            if (auto event = poll()) {
                if (const auto* error = std::get_if<ExecutionFailed>(&*event))
                    throw std::runtime_error(error->message);
            } else {
                physical_->flush();
                physical_->wait_for_activity(std::chrono::milliseconds(1));
            }
        }
        (void)withdraw_offer();
        reserved_.reset();
    }

    void drain() override {
        if (closed_) return;
        domain_->close(member_);
        // Peers still progressing a published step must either finish it or see the root
        // failure before their workers are shut down. Native drain alone cannot join it.
        if (domain_->ready(member_) || pending_ || running_)
            fail("collective owner stopped with outstanding work");
        (void)withdraw_offer();
        reserved_.reset();
        closed_ = true;
        try {
            physical_->drain();
        } catch (const std::exception& e) {
            domain_->fail(e.what());
            throw;
        }
        pending_.reset();
        running_.reset();
        admissions_.retire_submitted();
    }

    void measure(MeasurementCommand command) override {
        if (has_pending_work() || !withdraw_offer())
            throw std::logic_error("collective measurement requires an idle owner");
        reserved_.reset();
        physical_->measure(command);
    }

    void finish_startup(bool calibrated) override {
        domain_->finish_startup_sources(member_, calibrated);
        auto measured = domain_->startup_drained();
        while (!measured) {
            if (auto event = poll()) {
                if (const auto* error = std::get_if<ExecutionFailed>(&*event))
                    throw std::runtime_error(error->message);
                throw std::logic_error("collective calibration has an unconsumed source event");
            }
            physical_->flush();
            measured = domain_->startup_drained();
            if (!measured) physical_->wait_for_activity(std::chrono::milliseconds(1));
        }
        if (admissions_.busy() || running_ || !withdraw_offer())
            throw std::logic_error("collective startup finished with an owned frame");
        reserved_.reset();
        physical_->finish_startup(*measured);
    }

    void observe_in_flight(std::vector<ExecutionProgress>& out) const override {
        physical_->observe_in_flight(out);
    }

    [[nodiscard]] std::span<const RankSnapshot> observations() const noexcept override {
        return physical_->observations();
    }

    [[nodiscard]] std::span<const RankDescriptor> rank_descriptions() const noexcept override {
        return physical_->rank_descriptions();
    }

    [[nodiscard]] const ExecutionStatistics& execution_statistics() const noexcept override {
        return physical_->execution_statistics();
    }

    [[nodiscard]] std::span<const ExecutionCounters> decode_statistics() const noexcept override {
        return physical_->decode_statistics();
    }

private:
    struct Pending {
        ExecutionTicket ticket;
        Work work;
        std::optional<CollectiveShape> source;
        bool discard = false;
    };

    struct Mapping {
        ExecutionTicket logical, physical;
    };

    struct Flight {
        ExecutionTicket physical;
        std::optional<ExecutionTicket> logical;
        std::uint64_t step;
    };

    void require_open() const {
        if (const auto failure = domain_->failure()) throw std::runtime_error(*failure);
        if (closed_ || domain_->closed()) throw std::logic_error("collective executor is closed");
    }

    Mapping source_mapping(ExecutionTicket ticket) const {
        if (ticket.frame.kind == FrameKind::Context) {
            for (const auto& mapping : contexts_)
                if (mapping && mapping->logical == ticket) return *mapping;
        } else if (running_ && running_->logical == ticket) {
            return {ticket, running_->physical};
        }
        throw std::logic_error("collective executor ticket has no physical source");
    }

    static void set_ticket(Event& event, ExecutionTicket ticket) {
        std::visit([&](auto& e) { e.ticket = ticket; }, event);
    }

    bool withdraw_offer() {
        if (!offer_) return true;
        if (!domain_->withdraw(member_, offer_)) return false;
        offer_          = 0;
        offered_source_ = false;
        return true;
    }

    void submit_source(std::uint64_t step) {
        const bool discard = pending_->discard;
        const auto ticket  = physical_->submit(std::move(*reserved_), std::move(pending_->work));
        running_.emplace(Flight{ticket, pending_->ticket, step});
        pending_.reset();
        if (discard) physical_->discard(ticket);
    }

    void progress() {
        if (closed_ || physical_->failed()) return;
        if (const auto error = domain_->failure()) {
            physical_->fail(*error);
            return;
        }
        if (running_) return;
        if (offer_ && !offered_source_ && pending_) (void)withdraw_offer();
        if (!offer_) {
            if (domain_->closed()) {
                if (pending_) fail("collective peer stopped before source submission");
                return;
            }
            if (!reserved_) reserved_ = physical_->try_reserve(FrameKind::Compute);
            if (!reserved_) return;
            if (pending_ && !pending_->source) {
                submit_source(0);
                reserved_.reset();
                return;
            }
            offered_source_ = pending_.has_value();
            offer_          = domain_->offer(member_, reserved_->frame(),
                                    pending_ ? pending_->source : std::nullopt);
        }
        auto decision = domain_->decision(member_, offer_);
        if (!decision) {
            if (domain_->closed() && pending_)
                fail("collective peer stopped before source submission");
            return;
        }
        if (decision->reserved_frame != reserved_->frame())
            throw std::logic_error("collective publication refers to a replaced physical frame");
        if (decision->source) {
            model_.bind(pending_->work, *decision);
            submit_source(decision->step);
        } else {
            auto work         = model_.service(*decision);
            const auto ticket = physical_->submit(std::move(*reserved_), std::move(work));
            running_.emplace(Flight{ticket, {}, decision->step});
        }
        reserved_.reset();
        offer_          = 0;
        offered_source_ = false;
    }

    void progress_or_fail() {
        try {
            progress();
        } catch (const std::exception& e) { fail(e.what()); }
    }

    std::unique_ptr<Base> physical_;
    std::shared_ptr<CollectiveStepDomain> domain_;
    std::size_t member_;
    Model model_;
    ExecutionPool admissions_;
    std::vector<std::optional<Mapping>> contexts_;
    std::optional<Pending> pending_;
    std::optional<Flight> running_;
    std::optional<ExecutionReservation> reserved_;
    std::uint64_t next_execution_ = 0, offer_ = 0;
    bool offered_source_ = false, closed_ = false;
};

} // namespace ninfer::runtime
