#pragma once

#include "runtime/executor/executor.h"
#include "runtime/executor/rank_worker.h"

#include <exception>
#include <type_traits>

namespace ninfer::runtime {

template <class Types>
class LocalExecutor final : public Executor<typename Types::ExecutionTypes> {
public:
    using Protocol = typename Types::ExecutionTypes;
    using Event    = ExecutionEvent<Protocol>;

    LocalExecutor(std::unique_ptr<RankWorker<Types>> worker, SessionEpoch epoch)
        : worker_(std::move(worker)), epoch_(epoch), pool_(worker_->program().capacity()),
          compute_(pool_.capacity().compute_frames), context_(pool_.capacity().context_frames) {}

    ~LocalExecutor() override {
        try {
            worker_->bind_to_current_thread();
            worker_->program().drain();
        } catch (...) {
            // RankProgram's teardown retains its allocation owners until outstanding work ends.
        }
    }

    void bind_to_current_thread() override { worker_->bind_to_current_thread(); }

    [[nodiscard]] SessionEpoch epoch() const noexcept override { return epoch_; }

    [[nodiscard]] ExecutionCapacity capacity() const noexcept override { return pool_.capacity(); }

    [[nodiscard]] std::optional<ExecutionReservation> try_reserve(FrameKind kind) override {
        require_available();
        return pool_.try_reserve(kind);
    }

    [[nodiscard]] ExecutionTicket submit(ExecutionReservation reservation,
                                         typename Protocol::Work work) override {
        require_available();
        if (!reservation) throw std::logic_error("empty execution reservation");
        worker_->program().validate_work(work, reservation.frame().kind);
        const ExecutionTicket ticket{epoch_, ++next_execution_,
                                     pool_.accept(std::move(reservation))};
        auto& flight = slot(ticket.frame);
        flight       = {ticket, next_command(), Phase::Running, {}};
        try {
            worker_->program().start({ticket, flight.command, std::move(work)});
        } catch (...) { record_failure(ticket); }
        return ticket;
    }

    void supply(ExecutionTicket ticket, DecisionInput input) override {
        require_available();
        auto& flight = find(ticket);
        if (flight.phase != Phase::Decision || flight.decision_stage != input.stage) {
            throw std::logic_error("decision input does not match the pending execution stage");
        }
        worker_->program().validate_decision(ticket, input);
        flight.command = next_command();
        flight.phase   = Phase::Running;
        try {
            worker_->program().supply(ticket, flight.command, std::move(input));
        } catch (...) { record_failure(ticket); }
    }

    [[nodiscard]] CommitReceipt commit(ExecutionTicket ticket,
                                       typename Protocol::Commit commit) override {
        require_available();
        auto& flight = find(ticket);
        if (flight.phase != Phase::Result) {
            throw std::logic_error("execution has no result awaiting commit");
        }
        worker_->program().prepare_commit(ticket, commit);
        flight.command = next_command();
        flight.phase   = Phase::Retiring;
        try {
            worker_->program().commit(ticket, flight.command, std::move(commit));
        } catch (...) {
            record_failure(ticket);
            throw;
        }
        return {ticket, flight.command};
    }

    void discard(ExecutionTicket ticket) override {
        require_available();
        auto& flight = find(ticket);
        if (flight.phase == Phase::Retiring || flight.phase == Phase::Discarding) return;
        flight.command = next_command();
        flight.phase   = Phase::Discarding;
        try {
            worker_->program().discard(ticket, flight.command);
        } catch (...) { record_failure(ticket); }
    }

    [[nodiscard]] std::optional<Event> poll() override {
        if (failure_) return Event{std::exchange(failure_, {}).value()};
        if (failed_ || closed_) return {};
        std::optional<Event> event;
        try {
            for (;;) {
                event = worker_->program().poll();
                if (!event) return {};
                if (std::holds_alternative<ExecutionFailed>(*event)) {
                    failed_ = true;
                    return event;
                }
                const auto ticket =
                    std::visit([](const auto& value) { return value.ticket; }, *event);
                const auto& flight = find(ticket);
                if (flight.phase != Phase::Discarding ||
                    (!std::holds_alternative<DecisionNeeded>(*event) &&
                     !std::holds_alternative<ResultReady<Protocol>>(*event)))
                    break;
            }
            std::visit(
                [&](const auto& value) {
                    using Value  = std::decay_t<decltype(value)>;
                    auto& flight = find(value.ticket);
                    if constexpr (std::is_same_v<Value, DecisionNeeded>) {
                        if (flight.phase != Phase::Running) {
                            throw std::logic_error("execution requested a decision out of order");
                        }
                        flight.phase          = Phase::Decision;
                        flight.decision_stage = value.request.stage;
                    } else if constexpr (std::is_same_v<Value, DecisionConsumed>) {
                        if (value.stage != flight.decision_stage) {
                            throw std::logic_error("decision upload refers to another stage");
                        }
                    } else if constexpr (std::is_same_v<Value, ResultReady<Protocol>>) {
                        if (flight.phase != Phase::Running) {
                            throw std::logic_error("execution returned a result out of order");
                        }
                        flight.phase = value.ticket.frame.kind == FrameKind::Compute
                                           ? Phase::Result
                                           : Phase::Retiring;
                    } else if constexpr (std::is_same_v<Value, DeviceRetired>) {
                        if ((flight.phase != Phase::Retiring &&
                             flight.phase != Phase::Discarding) ||
                            value.command != flight.command) {
                            throw std::logic_error("execution retired before its final command");
                        }
                        pool_.retire(value.ticket.frame);
                        flight = {};
                    } else {
                        failed_ = true;
                    }
                },
                *event);
        } catch (...) {
            record_failure(event
                               ? std::visit([](const auto& value) { return value.ticket; }, *event)
                               : ExecutionTicket{});
            return Event{std::exchange(failure_, {}).value()};
        }
        return event;
    }

    void flush() override {}

    void drain() override {
        closed_ = true;
        std::exception_ptr error;
        try {
            worker_->program().drain();
        } catch (...) { error = std::current_exception(); }
        pool_.retire_submitted();
        std::fill(compute_.begin(), compute_.end(), Flight{});
        std::fill(context_.begin(), context_.end(), Flight{});
        if (error) std::rethrow_exception(error);
    }

    void fail(std::string message) override {
        if (failed_) return;
        failed_  = true;
        failure_ = ExecutionFailed{{}, std::move(message)};
    }

    [[nodiscard]] bool failed() const noexcept override { return failed_; }

    [[nodiscard]] bool has_pending_work() const noexcept override { return pool_.busy(); }

    [[nodiscard]] MemorySummary memory_summary() const override {
        return worker_->program().memory_summary();
    }

    void reset_memory_peaks() noexcept override { worker_->program().reset_memory_peaks(); }

private:
    enum class Phase : std::uint8_t { Empty, Running, Decision, Result, Retiring, Discarding };

    struct Flight {
        ExecutionTicket ticket;
        CommandSequence command;
        Phase phase                  = Phase::Empty;
        std::uint32_t decision_stage = 0;
    };

    [[nodiscard]] Flight& slot(FrameId frame) {
        return (frame.kind == FrameKind::Compute ? compute_ : context_).at(frame.index);
    }

    [[nodiscard]] Flight& find(ExecutionTicket ticket) {
        auto& flight = slot(ticket.frame);
        if (ticket.epoch != epoch_ || flight.ticket != ticket || !pool_.submitted(ticket.frame)) {
            throw std::logic_error("execution ticket is stale");
        }
        return flight;
    }

    [[nodiscard]] CommandSequence next_command() noexcept { return {++next_command_}; }

    void require_available() const {
        if (closed_) throw std::runtime_error("executor is closed");
        if (failed_) throw std::runtime_error("executor is unavailable after an execution failure");
    }

    void record_failure(ExecutionTicket ticket) {
        std::string message;
        try {
            throw;
        } catch (const std::exception& error) { message = error.what(); } catch (...) {
            message = "unknown rank execution failure";
        }
        failed_ = true;
        if (!failure_) failure_ = ExecutionFailed{ticket, std::move(message)};
    }

    std::unique_ptr<RankWorker<Types>> worker_;
    SessionEpoch epoch_;
    ExecutionPool pool_;
    std::vector<Flight> compute_, context_;
    std::uint64_t next_execution_ = 0, next_command_ = 0;
    bool failed_ = false, closed_ = false;
    std::optional<ExecutionFailed> failure_;
};

} // namespace ninfer::runtime
