#pragma once

#include "distributed/bootstrap.h"
#include "runtime/executor/executor.h"
#include "runtime/executor/protocol.h"
#include "runtime/executor/input.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>
#include <deque>
#include <thread>

namespace ninfer::runtime {

// Codec owns the model's address-free execution contract and rank-local encoding recipes.
// It validates values before acceptance; workers validate actual physical bindings separately.
template <class Types, class Codec>
class DistributedExecutor final : public Executor<Types> {
public:
    using Event  = ExecutionEvent<Types>;
    using Work   = typename Types::Work;
    using Result = typename Types::Result;
    using Commit = typename Types::Commit;

    DistributedExecutor(std::unique_ptr<distributed::ExecutionGroup> group,
                        ExecutionCapacity capacity, Codec codec,
                        std::vector<RankSnapshot> initial_observations = {})
        : group_(std::move(group)), pool_(capacity), codec_(std::move(codec)),
          compute_(capacity.compute_frames), context_(capacity.context_frames),
          progress_(group_->topology().ranks.size()), measurement_reports_(progress_.size()),
          measurement_errors_(progress_.size()) {
        validate_control_capacity();
        this->observations_.resize(progress_.size());
        if (!initial_observations.empty()) this->observations_ = std::move(initial_observations);
        for (const auto& rank : group_->topology().ranks) {
            this->observations_[rank.rank].rank = rank.rank;
            this->observations_[rank.rank].node = rank.node;
            // Runtime communication counters start at activation, after preparation and capture.
            for (auto& counters : this->observations_[rank.rank].communication) {
                counters.eager = {};
                counters.graph = {};
            }
        }
    }

    ~DistributedExecutor() override {
        if (!closed_) group_->fail("distributed executor released with live execution");
    }

    // Coordinator calls are serialized by the Engine. CUDA affinity belongs to remote workers.
    void bind_to_current_thread() override {}

    [[nodiscard]] SessionEpoch epoch() const noexcept override { return {group_->epoch()}; }

    [[nodiscard]] ExecutionCapacity capacity() const noexcept override { return pool_.capacity(); }

    [[nodiscard]] std::optional<ExecutionReservation> try_reserve(FrameKind kind) override {
        require_available();
        return pool_.try_reserve(kind);
    }

    [[nodiscard]] ExecutionTicket submit(ExecutionReservation reservation, Work work) override {
        require_available();
        if (!reservation) throw std::logic_error("empty execution reservation");
        codec_.validate_work(work, reservation.frame().kind);
        auto scope = codec_.scope(work, group_->topology());
        scope.validate(group_->topology());
        ExecutionTicket id{epoch(), next_execution_ + 1, reservation.frame()};
        Flight prepared;
        prepared.ticket = id;
        prepared.scope  = std::move(scope);
        if constexpr (requires { codec_.prepare_inputs(work, group_->topology()); })
            prepared.inputs = inputs_.begin(codec_.prepare_inputs(work, group_->topology()));
        prepared.work.emplace(std::move(work));
        prepared.phase = Phase::Inputs;
        prepared.retired.resize(group_->topology().ranks.size());
        prepared.context_results.resize(group_->topology().ranks.size());
        prepared.consumed.resize(group_->topology().ranks.size());
        (void)pool_.accept(std::move(reservation));
        prepared.clock.start(Types::describe_work(*prepared.work), observation_time_ns());
        slot(id.frame) = std::move(prepared);
        ++next_execution_;
        try {
            progress_inputs();
        } catch (const std::exception& e) { fail(e.what()); }
        return id;
    }

    void supply(ExecutionTicket id, DecisionInput input) override {
        require_available();
        auto& flight = find(id);
        if (flight.phase != Phase::Decision || !flight.decision ||
            input.stage != flight.decision->stage)
            throw std::logic_error("decision input does not match pending execution");
        codec_.validate_decision(*flight.work, *flight.decision, input);
        protocol::Envelope envelope{id, {next_command_ + 1}, group_->topology().plan, flight.scope};
        auto messages   = encode(envelope, distributed::wire::Kind::Supply,
                                 [&](auto& out, auto) { protocol::encode(out, input); });
        flight.command  = envelope.command;
        flight.phase    = Phase::Running;
        flight.supplied = true;
        ++next_command_;
        try {
            send(messages, true);
        } catch (const std::exception& e) { fail(e.what()); }
    }

    [[nodiscard]] CommitReceipt commit(ExecutionTicket id, Commit decisions) override {
        require_available();
        auto& flight = find(id);
        if (flight.phase != Phase::Result || !flight.result)
            throw std::logic_error("commit has no pending distributed result");
        codec_.validate_commit(*flight.work, *flight.result, decisions);
        protocol::Envelope envelope{id, {next_command_ + 1}, group_->topology().plan, flight.scope};
        auto messages =
            encode(envelope, distributed::wire::Kind::Commit,
                   [&](auto& out, auto rank) { codec_.encode_commit(out, decisions, rank); });
        // Receipt is produced only after the complete fanout has entered the ordered queue.
        try {
            send(messages, true);
        } catch (const std::exception& e) {
            fail(e.what());
            throw;
        }
        flight.command = envelope.command;
        flight.phase   = Phase::Retiring;
        flight.clock.resolve(observation_time_ns());
        ++next_command_;
        return {id, flight.command};
    }

    void discard(ExecutionTicket id) override {
        require_available();
        auto& flight = find(id);
        if (flight.phase == Phase::Discarding || flight.phase == Phase::Retiring) return;
        if (flight.phase == Phase::Inputs) {
            flight.cancel_requested = true;
            return;
        }
        protocol::Envelope envelope{id, {next_command_ + 1}, group_->topology().plan, flight.scope};
        auto messages  = encode(envelope, distributed::wire::Kind::Discard, [](auto&, auto) {});
        flight.command = envelope.command;
        flight.phase   = Phase::Discarding;
        flight.clock.resolve(observation_time_ns());
        ++next_command_;
        std::erase_if(events_, [&](const Event& event) {
            return std::visit(
                [&](const auto& value) {
                    using T = std::decay_t<decltype(value)>;
                    return value.ticket == id && (std::is_same_v<T, DecisionNeeded> ||
                                                  std::is_same_v<T, ResultReady<Types>>);
                },
                event);
        });
        try {
            send(messages, true);
        } catch (const std::exception& e) { fail(e.what()); }
    }

    [[nodiscard]] std::optional<Event> poll() override {
        if (failure_) return Event{std::exchange(failure_, {}).value()};
        if (failed_ || closed_) return {};
        try {
            while (auto incoming = group_->transport().poll()) {
                auto& message = incoming->message;
                if (message.epoch != epoch().value)
                    throw std::logic_error("stale rank event epoch");
                distributed::wire::Decoder in(message.payload);
                if (message.kind == distributed::wire::Kind::Observation) {
                    auto snapshot = protocol::rank_snapshot(in);
                    in.finish();
                    const auto& identity = group_->topology().rank(incoming->peer);
                    if (snapshot.rank != identity.rank || snapshot.node != identity.node)
                        throw std::logic_error("resource observation has an invalid rank");
                    auto& previous = this->observations_.at(incoming->peer);
                    if (snapshot.stamp.sequence <= previous.stamp.sequence) continue;
                    snapshot.received_ns         = observation_time_ns();
                    snapshot.gpu_intervals       = previous.gpu_intervals;
                    snapshot.communication_eager = previous.communication_eager;
                    snapshot.communication_graph = previous.communication_graph;
                    snapshot.compute_paths       = previous.compute_paths;
                    merge_host_observations(snapshot.host, previous.host);
                    auto& memory                         = snapshot.memory;
                    const auto& old                      = previous.memory;
                    memory.workspace.required_peak_bytes = std::max(
                        memory.workspace.required_peak_bytes, old.workspace.required_peak_bytes);
                    memory.workspace.measurement_peak_bytes =
                        std::max(memory.workspace.measurement_peak_bytes,
                                 old.workspace.measurement_peak_bytes);
                    if (memory.ple && old.ple)
                        memory.ple->uploaded_bytes =
                            std::max(memory.ple->uploaded_bytes, old.ple->uploaded_bytes);
                    if (memory.experts && old.experts)
                        memory.experts->uploaded_bytes =
                            std::max(memory.experts->uploaded_bytes, old.experts->uploaded_bytes);

                    previous = std::move(snapshot);
                    continue;
                }
                if (message.kind == distributed::wire::Kind::InputReady) {
                    const auto id = in.u64();
                    in.finish();
                    inputs_.complete(incoming->peer, id);
                    continue;
                }
                if (message.kind == distributed::wire::Kind::Failed) {
                    auto detail = in.string();
                    in.finish();
                    throw std::runtime_error("rank " + std::to_string(incoming->peer) + ": " +
                                             detail);
                }
                if (message.kind == distributed::wire::Kind::Progress) {
                    const auto applied = in.u64(), retired = in.u64();
                    in.finish();
                    if (applied > next_command_ || retired > applied)
                        throw std::logic_error("rank progressed past accepted commands");
                    auto& progress = progress_.at(incoming->peer);
                    if (applied < progress.applied || retired < progress.retired)
                        throw std::logic_error("rank progress moved backwards");
                    progress = {applied, retired};
                    continue;
                }
                if (message.kind == distributed::wire::Kind::MeasurementComplete) {
                    if (message.payload.size() >
                        protocol::measurement_report_capacity(codec_.control_budget().report))
                        throw std::logic_error(
                            "rank resource report exceeded its prepared capacity");
                    if (!measurement_query_ || in.u64() != measurement_query_ ||
                        measurement_reports_.at(incoming->peer))
                        throw std::logic_error("unexpected measurement acknowledgement");
                    measurement_errors_.at(incoming->peer) = in.string();
                    auto updated                           = protocol::rank_snapshot(in);
                    const auto& identity = group_->topology().rank(incoming->peer);
                    if (updated.rank != identity.rank || updated.node != identity.node)
                        throw std::logic_error("measurement report has an invalid rank");
                    auto& previous              = this->observations_[incoming->peer];
                    updated.gpu_intervals       = previous.gpu_intervals;
                    updated.communication_eager = previous.communication_eager;
                    updated.communication_graph = previous.communication_graph;
                    updated.compute_paths       = previous.compute_paths;
                    merge_host_observations(updated.host, previous.host);
                    previous                                        = std::move(updated);
                    this->observations_[incoming->peer].received_ns = observation_time_ns();
                    measurement_reports_[incoming->peer]            = true;
                    in.finish();
                    continue;
                }
                if (message.kind != distributed::wire::Kind::Event)
                    throw std::logic_error("unexpected runtime control message");
                if (message.payload.size() > codec_.control_budget().event)
                    throw std::logic_error("rank event exceeded its prepared capacity");
                auto event = protocol::decode_event<Types>(in, codec_);
                in.finish();
                consume(incoming->peer, std::move(event));
            }
            progress_inputs();
            group_->check_processes();
        } catch (const std::exception& e) {
            fail(e.what());
            return Event{std::exchange(failure_, {}).value()};
        }
        if (events_.empty()) return {};
        auto event = std::move(events_.front());
        events_.pop_front();
        return event;
    }

    void flush() override {
        if (!failed_ && !closed_) {
            try {
                progress_inputs();
                group_->check_processes();
            } catch (const std::exception& e) { fail(e.what()); }
        }
    }

    void drain() override {
        if (closed_) return;
        if (failed_) {
            group_->fail("draining failed execution group");
            pool_.retire_submitted();
            closed_ = true;
            return;
        }
        for (auto* flights : {&compute_, &context_})
            for (const auto& flight : *flights)
                if (flight.phase != Phase::Empty && flight.ticket.frame.kind == FrameKind::Compute)
                    discard(flight.ticket);
        const auto until = std::chrono::steady_clock::now() + group_->shutdown_timeout();
        while (pool_.in_flight() && !failed_) {
            if (std::chrono::steady_clock::now() >= until) {
                fail("distributed execution drain timed out");
                break;
            }
            (void)poll();
            std::this_thread::yield();
        }
        if (failed_)
            group_->fail("execution failed while draining");
        else
            group_->shutdown();
        pool_.retire_submitted();
        closed_ = true;
    }

    void fail(std::string message) override {
        if (failed_) return;
        failed_  = true;
        failure_ = ExecutionFailed{{}, message};
        group_->fail(message);
    }

    [[nodiscard]] bool failed() const noexcept override { return failed_; }

    [[nodiscard]] bool has_pending_work() const noexcept override { return pool_.busy(); }

    void measure(MeasurementCommand command) override {
        require_available();
        if (pool_.busy()) throw std::logic_error("measurement boundary requires an idle executor");
        try {
            measurement_boundary(command);
        } catch (...) {
            const auto error = std::current_exception();
            if (!failed_ && command.action != MeasurementAction::Cancel) {
                try {
                    measurement_boundary({MeasurementAction::Cancel});
                } catch (...) { fail("failed to close rank measurement windows"); }
            }
            std::rethrow_exception(error);
        }
    }

private:
    enum class Phase { Empty, Inputs, Running, Decision, Result, Retiring, Discarding };

    struct Flight {
        ExecutionTicket ticket;
        CommandSequence command;
        protocol::Scope scope;
        Phase phase = Phase::Empty;
        std::optional<Work> work;
        std::optional<Result> result;
        std::optional<DecisionRequest> decision;
        std::vector<std::optional<Result>> context_results;
        std::vector<std::optional<DecisionConsumed>> consumed;
        std::vector<bool> retired;
        std::vector<InputTransfers::Key> inputs;
        bool supplied = false, consumed_published = false, result_published = false;
        bool cancel_requested = false;
        ExecutionClock clock;
        GpuExecutionObservation gpu;
    };

    [[nodiscard]] Flight& slot(FrameId frame) {
        return (frame.kind == FrameKind::Compute ? compute_ : context_).at(frame.index);
    }

    [[nodiscard]] Flight& find(ExecutionTicket id) {
        auto& flight = slot(id.frame);
        if (id.epoch != epoch() || flight.ticket != id || !pool_.submitted(id.frame))
            throw std::logic_error("stale distributed execution ticket");
        return flight;
    }

    void require_available() const {
        if (closed_ || failed_) throw std::runtime_error("distributed executor is unavailable");
    }

    void progress_inputs() {
        inputs_.progress(group_->transport(), epoch().value);
        for (auto* flights : {&compute_, &context_})
            for (auto& flight : *flights) {
                if (flight.phase != Phase::Inputs || !inputs_.ready(flight.inputs)) continue;
                protocol::Envelope envelope{
                    flight.ticket, {next_command_ + 1}, group_->topology().plan, flight.scope};
                auto messages =
                    encode(envelope, distributed::wire::Kind::Submit, [&](auto& out, auto rank) {
                        codec_.encode_work(out, *flight.work, rank);
                    });
                send(messages, false);
                flight.command = envelope.command;
                flight.phase   = Phase::Running;
                ++next_command_;
                if (flight.cancel_requested) discard(flight.ticket);
            }
    }

    void retire_inputs(const Work& work) {
        if constexpr (requires { codec_.retire_inputs(work); }) {
            for (const auto& release : codec_.retire_inputs(work)) {
                distributed::wire::Encoder out;
                out.u64(release.id);
                auto message = std::make_shared<const distributed::wire::Message>(
                    distributed::wire::Message{distributed::wire::Kind::InputRelease, epoch().value,
                                               std::move(out).take()});
                std::vector<distributed::OutgoingMessage> requests;
                for (const auto& rank : group_->topology().ranks)
                    requests.push_back({rank.rank, message});
                send(requests, true);
                inputs_.release(release.id);
            }
        }
    }

    void validate_control_capacity() const {
        protocol::validate_control_capacity(codec_.control_budget(), pool_.capacity(),
                                            group_->transport().limits(),
                                            group_->topology().ranks.size());
    }

    void measurement_boundary(MeasurementCommand command) {
        require_available();
        std::fill(measurement_reports_.begin(), measurement_reports_.end(), false);
        std::fill(measurement_errors_.begin(), measurement_errors_.end(), std::string{});
        distributed::wire::Encoder out;
        out.u64(++measurement_query_);
        out.u64(next_command_);
        out.u8(static_cast<std::uint8_t>(command.action));
        out.boolean(command.profile);
        auto message = std::make_shared<distributed::wire::Message>(distributed::wire::Message{
            distributed::wire::Kind::Measurement, epoch().value, std::move(out).take()});
        std::vector<distributed::OutgoingMessage> messages;
        for (const auto& rank : group_->topology().ranks) messages.push_back({rank.rank, message});
        send(messages, true);
        std::deque<Event> deferred;
        const auto until = std::chrono::steady_clock::now() + group_->shutdown_timeout();
        while (std::any_of(measurement_reports_.begin(), measurement_reports_.end(),
                           [](const auto& value) { return !value; })) {
            if (std::chrono::steady_clock::now() >= until) {
                std::string message = "rank measurement boundary timed out at command " +
                                      std::to_string(next_command_);
                for (std::size_t rank = 0; rank < progress_.size(); ++rank)
                    if (!measurement_reports_[rank])
                        message += "; rank " + std::to_string(rank) +
                                   " applied=" + std::to_string(progress_[rank].applied) +
                                   " retired=" + std::to_string(progress_[rank].retired);
                fail(message);
                throw std::runtime_error(message);
            }
            if (auto event = poll()) {
                if (const auto* error = std::get_if<ExecutionFailed>(&*event))
                    throw std::runtime_error(error->message);
                deferred.push_back(std::move(*event));
            } else
                std::this_thread::yield();
        }
        events_.insert(events_.begin(), std::make_move_iterator(deferred.begin()),
                       std::make_move_iterator(deferred.end()));
        for (std::size_t rank = 0; rank < measurement_errors_.size(); ++rank)
            if (!measurement_errors_[rank].empty())
                throw std::runtime_error("rank " + std::to_string(rank) + ": " +
                                         measurement_errors_[rank]);
    }

    template <class Encode>
    std::vector<distributed::OutgoingMessage>
    encode(const protocol::Envelope& envelope, distributed::wire::Kind kind, Encode encode_body) {
        std::vector<distributed::OutgoingMessage> result;
        result.reserve(group_->topology().ranks.size());
        for (const auto& rank : group_->topology().ranks) {
            distributed::wire::Encoder out;
            protocol::encode(out, envelope);
            if (envelope.scope.contains(rank.rank)) encode_body(out, rank.rank);
            auto payload      = std::move(out).take();
            const auto budget = codec_.control_budget();
            const auto limit =
                kind == distributed::wire::Kind::Submit
                    ? budget.work(envelope.ticket.frame.kind)
                    : (kind == distributed::wire::Kind::Supply ? budget.decision : budget.commit);
            if (payload.size() > limit)
                throw std::logic_error("rank " + std::to_string(rank.rank) + " message " +
                                       std::to_string(static_cast<unsigned>(kind)) + " encoded " +
                                       std::to_string(payload.size()) + " bytes; Codec bound is " +
                                       std::to_string(limit));
            auto message = std::make_shared<distributed::wire::Message>(
                distributed::wire::Message{kind, epoch().value, std::move(payload)});
            result.push_back({rank.rank, std::move(message)});
        }
        return result;
    }

    void send(std::span<const distributed::OutgoingMessage> messages, bool continuation) {
        if (!group_->transport().try_send(messages, continuation
                                                        ? distributed::Traffic::Continuation
                                                        : distributed::Traffic::Work))
            throw std::logic_error("execution exceeded its reserved control capacity");
    }

    void publish_result(Flight& f) {
        if (!f.result || f.result_published || f.phase == Phase::Discarding) return;
        if (f.supplied && !f.consumed_published) return;
        f.result_published = true;
        f.phase = f.ticket.frame.kind == FrameKind::Compute ? Phase::Result : Phase::Retiring;
        f.clock.result(observation_time_ns());
        events_.emplace_back(
            ResultReady<Types>{f.ticket, *f.result, f.clock.finish(observation_time_ns())});
    }

    void consume(distributed::RankId rank, Event event) {
        std::visit(
            [&](auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, ExecutionFailed>) {
                    throw std::runtime_error("rank " + std::to_string(rank) + ": " + value.message);
                } else {
                    auto& f = find(value.ticket);
                    if (!f.scope.contains(rank))
                        throw std::logic_error("event came from an unrelated rank");
                    if constexpr (std::is_same_v<T, DecisionNeeded>) {
                        if (f.phase == Phase::Discarding) return;
                        if (rank != f.scope.result_owner || f.phase != Phase::Running || f.decision)
                            throw std::logic_error("decision request arrived out of order");
                        f.decision = value.request;
                        f.phase    = Phase::Decision;
                        events_.emplace_back(std::move(value));
                    } else if constexpr (std::is_same_v<T, DecisionConsumed>) {
                        if (!f.decision || value.stage != f.decision->stage || f.consumed.at(rank))
                            throw std::logic_error("decision consumption arrived out of order");
                        f.consumed[rank] = std::move(value);
                        if (std::all_of(f.scope.owners.begin(), f.scope.owners.end(),
                                        [&](auto r) { return f.consumed[r].has_value(); })) {
                            DecisionConsumed combined{f.ticket, f.decision->stage, {}};
                            for (auto r : f.scope.owners)
                                for (const auto& row : f.consumed[r]->rows) {
                                    auto found = std::find_if(
                                        combined.rows.begin(), combined.rows.end(),
                                        [&](const auto& v) { return v.member == row.member; });
                                    if (found == combined.rows.end())
                                        combined.rows.push_back(row);
                                    else
                                        found->bytes += row.bytes;
                                }
                            f.consumed_published = true;
                            if (f.phase != Phase::Discarding)
                                events_.emplace_back(std::move(combined));
                            publish_result(f);
                        }
                    } else if constexpr (std::is_same_v<T, ResultReady<Types>>) {
                        if (f.phase == Phase::Discarding) return;
                        if (f.ticket.frame.kind == FrameKind::Context) {
                            if (f.context_results.at(rank))
                                throw std::logic_error("duplicate context result");
                            f.context_results[rank] = std::move(value.result);
                            if (std::all_of(
                                    f.scope.owners.begin(), f.scope.owners.end(),
                                    [&](auto r) { return f.context_results[r].has_value(); }))
                                f.result = codec_.merge_context_results(f.scope, f.context_results);
                        } else {
                            if (rank != f.scope.result_owner || f.result)
                                throw std::logic_error("unexpected compute result owner");
                            f.result = std::move(value.result);
                        }
                        publish_result(f);
                    } else if constexpr (std::is_same_v<T, DeviceRetired>) {
                        this->observe_retirement(rank, value.observed,
                                                 value.ticket.frame.kind == FrameKind::Compute);
                        merge_gpu_intervals(f.gpu, value.observed.gpu);
                        if (f.ticket.frame.kind == FrameKind::Compute &&
                            f.phase != Phase::Retiring && f.phase != Phase::Discarding)
                            throw std::logic_error("compute frame retired before resolution");
                        if (value.command != f.command || f.retired.at(rank))
                            throw std::logic_error("stale or duplicate rank retirement");
                        f.retired[rank] = true;
                        if (std::all_of(f.scope.owners.begin(), f.scope.owners.end(),
                                        [&](auto r) { return f.retired[r]; })) {
                            if (f.ticket.frame.kind == FrameKind::Context &&
                                f.phase != Phase::Discarding && !f.result_published)
                                throw std::logic_error("context retired before publication");
                            auto observed         = f.clock.finish(observation_time_ns());
                            observed.max_rank_gpu = f.gpu;
                            accumulate(this->execution_statistics_, observed);
                            events_.emplace_back(DeviceRetired{f.ticket, f.command, {}, observed});
                            retire_inputs(*f.work);
                            pool_.retire(f.ticket.frame);
                            f = {};
                        }
                    }
                }
            },
            event);
    }

    std::unique_ptr<distributed::ExecutionGroup> group_;
    ExecutionPool pool_;
    Codec codec_;
    InputTransfers inputs_;
    std::vector<Flight> compute_, context_;

    struct Progress {
        std::uint64_t applied = 0, retired = 0;
    };

    std::vector<Progress> progress_;
    std::vector<bool> measurement_reports_;
    std::vector<std::string> measurement_errors_;
    std::deque<Event> events_;
    std::optional<ExecutionFailed> failure_;
    std::uint64_t next_execution_ = 0, next_command_ = 0;
    std::uint64_t measurement_query_ = 0;
    bool failed_ = false, closed_ = false;
};

} // namespace ninfer::runtime
