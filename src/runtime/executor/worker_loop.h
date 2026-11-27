#pragma once

#include "distributed/bootstrap.h"
#include "runtime/executor/measurement.h"
#include "runtime/executor/protocol.h"
#include "runtime/observability/snapshot.h"

#include <chrono>
#include <deque>
#include <optional>
#include <thread>
#include <vector>

namespace ninfer::runtime {

// The caller destroys Program before closing the connection. Commands are received independently
// of device progress and applied in peer sequence order at each ticket's safe boundary.
template <class Types, class Program, class Codec, class Connection = distributed::WorkerConnection>
class WorkerLoop {
public:
    WorkerLoop(Connection& connection, Program& program, Codec codec)
        : connection_(connection), program_(program), codec_(std::move(codec)),
          budget_(protocol::with_communication_capacity(codec_.control_budget(),
                                                        connection.communication_statistics())),
          compute_(program.capacity().compute_frames), context_(program.capacity().context_frames) {
        const auto& identity = connection_.identity();
        const auto& replica  = connection_.topology().replica_for_rank(identity.rank);
        connection_.reserve_control_capacity(protocol::control_transport_limits(
            budget_, program.capacity(), replica.ranks.size(), false));
        if constexpr (requires { codec_.bind_input_storage(program_.input_storage()); })
            codec_.bind_input_storage(program_.input_storage());
        connection_.reset_communication_statistics();
        observation_.rank     = identity.rank;
        observation_.node     = identity.node;
        observation_.hardware = connection_.hardware_name();
        connection_.set_communication_active(false);
    }

    // False means orderly shutdown with every frame retired.
    [[nodiscard]] bool advance() {
        try {
            return advance_impl();
        } catch (const std::exception& error) {
            // Abort communicators before Program unwinding waits on device work.
            connection_.fail(error.what());
            throw;
        }
    }

private:
    static constexpr std::size_t kMaxPollBatch = 32;

    enum class Phase { Empty, Running, Decision, Result, Retiring, Discarding };

    struct PendingSupply {
        protocol::Envelope envelope;
        DecisionInput input;
    };

    struct PendingCommit {
        protocol::Envelope envelope;
        typename Types::Commit commit;
    };

    struct PendingDiscard {
        protocol::Envelope envelope;
        protocol::DiscardMode mode = protocol::DiscardMode::BeforeSupply;
    };

    struct QueuedSubmit {
        protocol::Envelope envelope;
        typename Types::Work work;
        std::optional<PendingSupply> pending_supply;
        std::optional<PendingCommit> pending_commit;
        std::optional<PendingDiscard> pending_discard;
    };

    struct Flight {
        protocol::Envelope envelope;
        Phase phase                  = Phase::Empty;
        std::uint32_t decision_stage = 0;
        bool supply_applied          = false;
        std::uint64_t first_sequence = 0;
        std::optional<PendingSupply> pending_supply;
        std::optional<PendingCommit> pending_commit;
        std::optional<PendingDiscard> pending_discard;
    };

    bool advance_impl() {
        connection_.check_communication();
        drain_program_and_continuations();

        std::size_t polled = 0;
        for (; polled < kMaxPollBatch; ++polled) {
            auto message = connection_.poll();
            if (!message) break;
            if (message->epoch != connection_.epoch())
                throw std::logic_error("worker received a stale epoch");
            if (message->kind == distributed::wire::Kind::InputBegin ||
                message->kind == distributed::wire::Kind::InputChunk ||
                message->kind == distributed::wire::Kind::InputEnd ||
                message->kind == distributed::wire::Kind::InputRelease) {
                if constexpr (requires { codec_.receive_input(message->kind, message->payload); }) {
                    auto ready = codec_.receive_input(message->kind, message->payload);
                    if (message->kind == distributed::wire::Kind::InputChunk) {
                        distributed::wire::Encoder credit;
                        credit.u64(1);
                        connection_.send(distributed::wire::Kind::InputCredit,
                                         std::move(credit).take(), true);
                    }
                    if (ready) {
                        distributed::wire::Encoder out;
                        out.u64(*ready);
                        connection_.send(distributed::wire::Kind::InputReady, std::move(out).take(),
                                         true);
                    }
                } else {
                    throw std::logic_error("execution codec has no input resources");
                }
                continue;
            }
            if (message->kind == distributed::wire::Kind::Measurement) {
                distributed::wire::Decoder query(message->payload);
                if (measurement_boundary_)
                    throw std::logic_error("overlapping rank measurement boundaries");
                MeasurementBoundary boundary;
                boundary.id              = query.u64();
                boundary.through_command = query.u64();
                boundary.command.action  = static_cast<MeasurementAction>(query.u8());
                boundary.command.profile = query.boolean();
                query.finish();
                measurement_boundary_ = boundary;
                continue;
            }
            if (message->kind == distributed::wire::Kind::Shutdown) {
                drain_program_and_continuations();
                if (!submit_queue_.empty() || active() || measurement_boundary_)
                    throw std::logic_error("shutdown arrived before execution retirement");
                program_.drain();
                measurement_.cancel();
                return false;
            }
            receive_command(std::move(*message));
        }

        drain_program_and_continuations();
        progress_measurement();
        sample_resources();
        if (communication_active_ && !active()) {
            connection_.set_communication_active(false);
            communication_active_ = false;
        }
        if (polled < kMaxPollBatch) {
            if (!needs_device_poll() && submit_queue_.empty() && !measurement_boundary_) {
                connection_.wait(std::chrono::milliseconds(active() ? 5 : 50));
            } else {
                std::this_thread::yield();
            }
        }
        return true;
    }

    void receive_command(distributed::wire::Message message) {
        switch (message.kind) {
        case distributed::wire::Kind::Submit:
        case distributed::wire::Kind::Supply:
        case distributed::wire::Kind::Commit:
        case distributed::wire::Kind::Discard:
            break;
        default:
            throw std::logic_error("unexpected worker runtime command");
        }
        distributed::wire::Decoder in(message.payload);
        auto envelope   = protocol::envelope(in);
        const auto rank = connection_.identity().rank;
        if (envelope.ticket.epoch.value != connection_.epoch() ||
            envelope.plan != connection_.topology().plan ||
            envelope.peer_sequence != received_ + 1)
            throw std::logic_error("worker command identity or order differs");
        envelope.scope.validate(connection_.topology());
        if (!envelope.scope.contains(rank))
            throw std::logic_error("worker received command outside scope");
        if (message.kind == distributed::wire::Kind::Supply &&
            !envelope.scope.is_decision_owner(rank))
            throw std::logic_error("non-decision-owner received supply");

        ++received_;

        if (message.kind == distributed::wire::Kind::Submit) {
            auto work = codec_.decode_work(in);
            in.finish();
            submit_queue_.push_back(
                QueuedSubmit{std::move(envelope), std::move(work), {}, {}, {}});
            return;
        }

        const auto check_scope = [&](const protocol::Scope& expected) {
            if (expected.participants != envelope.scope.participants ||
                expected.decision_owners != envelope.scope.decision_owners ||
                expected.result_owner != envelope.scope.result_owner)
                throw std::logic_error("execution participants changed during a ticket");
        };

        auto queued = std::find_if(
            submit_queue_.begin(), submit_queue_.end(),
            [&](const QueuedSubmit& q) { return q.envelope.ticket == envelope.ticket; });

        if (message.kind == distributed::wire::Kind::Supply) {
            auto input = protocol::decision_input(in);
            in.finish();
            if (queued != submit_queue_.end()) {
                check_scope(queued->envelope.scope);
                if (queued->pending_supply)
                    throw std::logic_error("duplicate supply on queued execution");
                queued->pending_supply = PendingSupply{std::move(envelope), std::move(input)};
            } else {
                auto& flight = find(envelope.ticket);
                check_scope(flight.envelope.scope);
                if (flight.pending_supply || flight.supply_applied)
                    throw std::logic_error("duplicate supply on active execution");
                flight.pending_supply = PendingSupply{std::move(envelope), std::move(input)};
            }
            return;
        }

        if (message.kind == distributed::wire::Kind::Commit) {
            auto commit = codec_.decode_commit(in);
            in.finish();
            if (queued != submit_queue_.end()) {
                check_scope(queued->envelope.scope);
                if (queued->pending_commit || queued->pending_discard)
                    throw std::logic_error("conflicting continuation on queued execution");
                queued->pending_commit = PendingCommit{std::move(envelope), std::move(commit)};
            } else {
                auto& flight = find(envelope.ticket);
                check_scope(flight.envelope.scope);
                if (flight.pending_commit || flight.pending_discard ||
                    flight.phase == Phase::Retiring || flight.phase == Phase::Discarding)
                    throw std::logic_error("conflicting continuation on active execution");
                flight.pending_commit = PendingCommit{std::move(envelope), std::move(commit)};
            }
            return;
        }

        const auto mode = static_cast<protocol::DiscardMode>(in.u8());
        in.finish();
        if (queued != submit_queue_.end()) {
            check_scope(queued->envelope.scope);
            if (queued->pending_commit || queued->pending_discard)
                throw std::logic_error("conflicting discard on queued execution");
            queued->pending_discard = PendingDiscard{std::move(envelope), mode};
        } else {
            auto& flight = find(envelope.ticket);
            check_scope(flight.envelope.scope);
            if (flight.pending_commit || flight.pending_discard ||
                flight.phase == Phase::Retiring || flight.phase == Phase::Discarding)
                throw std::logic_error("conflicting discard on active execution");
            flight.pending_discard = PendingDiscard{std::move(envelope), mode};
        }
    }

    void drain_program_and_continuations() {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            while (auto event =
                       observe_host_call(host_.progress_ns, [&] { return program_.poll(); })) {
                consume(std::move(*event));
                progressed = true;
            }
            if (step_tickets()) progressed = true;
        }
    }

    [[nodiscard]] bool step_tickets() {
        if (!submit_queue_.empty() && next_to_apply(submit_queue_.front().envelope)) {
            auto& queued = submit_queue_.front();
            auto& flight = slot(queued.envelope.ticket.frame);
            if (flight.phase != Phase::Empty) return false;
            program_.validate_work(queued.work, queued.envelope.ticket.frame.kind);
            flight                 = {};
            flight.envelope        = std::move(queued.envelope);
            flight.phase           = Phase::Running;
            flight.first_sequence  = flight.envelope.peer_sequence;
            flight.pending_supply  = std::move(queued.pending_supply);
            flight.pending_commit  = std::move(queued.pending_commit);
            flight.pending_discard = std::move(queued.pending_discard);
            flight.supply_applied  = submitted_decision_stage<Types>(queued.work).has_value();
            auto work              = std::move(queued.work);
            submit_queue_.pop_front();

            ++applied_prefix_;
            if (!communication_active_) {
                connection_.set_communication_active(true);
                communication_active_ = true;
            }
            observe_host_call(host_.submit_ns, [&] {
                program_.start(ExecutionRequest<Types>{
                    flight.envelope.ticket, flight.envelope.command, std::move(work)});
            });
            return true;
        }

        const auto my_rank = connection_.identity().rank;
        for (auto* flights : {&compute_, &context_}) {
            for (auto& flight : *flights) {
                if (flight.phase == Phase::Empty) continue;

                if (flight.pending_supply && flight.phase == Phase::Decision &&
                    next_to_apply(flight.pending_supply->envelope)) {
                    if (flight.pending_discard &&
                        flight.pending_discard->mode == protocol::DiscardMode::BeforeSupply)
                        throw std::logic_error("discard before supply conflicts with supply");
                    auto supply = std::exchange(flight.pending_supply, std::nullopt).value();
                    if (supply.input.stage != flight.decision_stage)
                        throw std::logic_error("worker decision stage differs");
                    program_.validate_decision(supply.envelope.ticket, supply.input);
                    observe_host_call(host_.supply_ns, [&] {
                        program_.supply(supply.envelope.ticket, supply.envelope.command,
                                        std::move(supply.input));
                    });
                    flight.envelope.command = supply.envelope.command;
                    flight.phase            = Phase::Running;
                    flight.supply_applied   = true;
                    ++applied_prefix_;
                    return true;
                }

                if (flight.pending_commit && flight.phase == Phase::Result &&
                    next_to_apply(flight.pending_commit->envelope)) {
                    auto commit = std::exchange(flight.pending_commit, std::nullopt).value();
                    program_.prepare_commit(commit.envelope.ticket, commit.commit);
                    observe_host_call(host_.commit_ns, [&] {
                        program_.commit(commit.envelope.ticket, commit.envelope.command,
                                        std::move(commit.commit));
                    });
                    flight.envelope.command = commit.envelope.command;
                    flight.phase            = Phase::Retiring;
                    ++applied_prefix_;
                    return true;
                }

                if (flight.pending_discard && next_to_apply(flight.pending_discard->envelope)) {
                    bool ready_to_discard = false;
                    if (flight.envelope.ticket.frame.kind == FrameKind::Context) {
                        throw std::logic_error("accepted Context work must retire naturally");
                    } else if (flight.pending_discard->mode ==
                               protocol::DiscardMode::BeforeSupply) {
                        ready_to_discard =
                            !flight.supply_applied &&
                            (flight.phase == Phase::Decision || flight.phase == Phase::Result) &&
                            program_.can_discard(flight.envelope.ticket);
                    } else {
                        const bool need_supply = flight.envelope.scope.is_decision_owner(my_rank);
                        ready_to_discard       = (!need_supply || flight.supply_applied) &&
                                           flight.phase == Phase::Result &&
                                           program_.can_discard(flight.envelope.ticket);
                    }
                    if (ready_to_discard) {
                        auto discard = std::exchange(flight.pending_discard, std::nullopt).value();
                        observe_host_call(host_.discard_ns, [&] {
                            program_.discard(discard.envelope.ticket, discard.envelope.command);
                        });
                        flight.envelope.command = discard.envelope.command;
                        flight.phase            = Phase::Discarding;
                        ++applied_prefix_;
                        return true;
                    }
                }
            }
        }
        return false;
    }

    [[nodiscard]] bool next_to_apply(const protocol::Envelope& command) const {
        return command.peer_sequence == applied_prefix_ + 1 &&
               (!measurement_boundary_ ||
                command.peer_sequence <= measurement_boundary_->through_command);
    }

    [[nodiscard]] std::uint64_t retired_prefix() const noexcept {
        // Every applied command belongs to a live or retired ticket. The first live Submit
        // is the only hole that can limit the contiguous retirement watermark.
        auto retired = applied_prefix_;
        for (const auto* flights : {&compute_, &context_})
            for (const auto& flight : *flights)
                if (flight.phase != Phase::Empty)
                    retired = std::min(retired, flight.first_sequence - 1);
        return retired;
    }

    void publish_progress() {
        distributed::wire::Encoder out;
        out.u64(applied_prefix_);
        out.u64(retired_prefix());
        (void)connection_.try_send_traffic(distributed::wire::Kind::Progress, std::move(out).take(),
                                           distributed::Traffic::Telemetry);
    }

    void observe_resources(bool idle) {
        observation_.memory        = program_.observe_resources();
        observation_.host          = host_;
        observation_.communication = connection_.communication_statistics();
        if (idle) observation_.physical_memory = connection_.sample_idle_physical_memory();
        observation_.stamp = {observation_.stamp.sequence + 1, observation_time_ns()};
    }

    void progress_measurement() {
        if (!measurement_boundary_) return;
        const auto& boundary = *measurement_boundary_;
        if (applied_prefix_ < boundary.through_command ||
            retired_prefix() < boundary.through_command || active())
            return;
        std::string error;
        try {
            measurement_.apply(boundary.command, program_);
            if (boundary.command.action == MeasurementAction::ResetStartup) {
                host_ = {};
                connection_.reset_communication_statistics();
            }
        } catch (const std::exception& e) { error = e.what(); }
        observe_resources(true);
        distributed::wire::Encoder report;
        protocol::encode_measurement_report(report, boundary.id, error, observation_);
        auto payload = std::move(report).take();
        if (payload.size() > protocol::measurement_report_capacity(budget_.report))
            throw std::logic_error("rank measurement report exceeds its prepared capacity");
        connection_.send(distributed::wire::Kind::MeasurementComplete, std::move(payload), true);
        measurement_boundary_.reset();
    }

    void sample_resources() {
        if (!received_) return;
        const auto now = observation_time_ns();
        if (now - observation_.stamp.sampled_ns < kResourceSamplePeriodNs) return;
        observe_resources(!active());
        distributed::wire::Encoder out;
        protocol::encode(out, observation_);
        (void)connection_.try_send_traffic(distributed::wire::Kind::Observation,
                                           std::move(out).take(), distributed::Traffic::Telemetry);
    }

    Flight& slot(FrameId frame) {
        return (frame.kind == FrameKind::Compute ? compute_ : context_).at(frame.index);
    }

    Flight& find(ExecutionTicket ticket) {
        auto& flight = slot(ticket.frame);
        if (flight.phase == Phase::Empty || flight.envelope.ticket != ticket)
            throw std::logic_error("worker command refers to a stale frame");
        return flight;
    }

    [[nodiscard]] bool active() const noexcept {
        for (const auto* flights : {&compute_, &context_})
            for (const auto& f : *flights)
                if (f.phase != Phase::Empty) return true;
        return false;
    }

    [[nodiscard]] bool needs_device_poll() const noexcept {
        for (const auto* flights : {&compute_, &context_})
            for (const auto& f : *flights)
                // DecisionNeeded may be published before the forward device work completes.
                if (f.phase == Phase::Running || f.phase == Phase::Decision ||
                    f.phase == Phase::Retiring || f.phase == Phase::Discarding)
                    return true;
        return false;
    }

    void consume(ExecutionEvent<Types> event) {
        bool publish = true;
        std::visit(
            [&](auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, ExecutionFailed>) {
                    throw std::runtime_error(value.message);
                } else {
                    auto& flight      = find(value.ticket);
                    const bool result_owner =
                        connection_.identity().rank == flight.envelope.scope.result_owner;
                    const bool suppressing =
                        flight.phase == Phase::Discarding || flight.pending_discard.has_value();
                    if constexpr (std::is_same_v<T, DecisionNeeded>) {
                        flight.phase          = Phase::Decision;
                        flight.decision_stage = value.request.stage;
                        publish               = result_owner && !suppressing;
                    } else if constexpr (std::is_same_v<T, DecisionConsumed>) {
                        publish = flight.envelope.scope.is_decision_owner(
                            connection_.identity().rank);
                    } else if constexpr (std::is_same_v<T, ResultReady<Types>>) {
                        if (flight.phase == Phase::Discarding) {
                            publish = false;
                            return;
                        }
                        flight.phase = value.ticket.frame.kind == FrameKind::Compute
                                           ? Phase::Result
                                           : Phase::Retiring;
                        publish      = !suppressing &&
                                  (result_owner || value.ticket.frame.kind == FrameKind::Context);
                    } else if constexpr (std::is_same_v<T, DeviceRetired>) {
                        value.observed.host = host_;
                        connection_.observe_communication_totals(
                            value.observed.communication_eager, value.observed.communication_graph);
                        if (flight.envelope.command != value.command ||
                            (flight.phase != Phase::Retiring && flight.phase != Phase::Discarding))
                            throw std::logic_error("worker retired an unresolved frame");
                        flight = {};
                    }
                }
            },
            event);
        if (publish) {
            auto payload = protocol::encode_event<Types>(event, codec_);
            if (payload.size() > budget_.event)
                throw std::logic_error("execution event exceeds its prepared capacity");
            connection_.send(distributed::wire::Kind::Event, std::move(payload), true);
        }
        if (std::holds_alternative<DeviceRetired>(event)) publish_progress();
    }

    Connection& connection_;
    Program& program_;
    Codec codec_;
    protocol::ControlBudget budget_;
    bool communication_active_ = false;
    RankSnapshot observation_;
    RankHostStatistics host_;
    std::vector<Flight> compute_, context_;
    std::deque<QueuedSubmit> submit_queue_;

    struct MeasurementBoundary {
        std::uint64_t id = 0, through_command = 0;
        MeasurementCommand command{};
    };

    RankMeasurement measurement_;
    std::optional<MeasurementBoundary> measurement_boundary_;
    std::uint64_t received_ = 0, applied_prefix_ = 0;
};

} // namespace ninfer::runtime
