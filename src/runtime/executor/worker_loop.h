#pragma once

#include "distributed/bootstrap.h"
#include "runtime/executor/measurement.h"
#include "runtime/executor/protocol.h"
#include "runtime/observability/snapshot.h"

#include <deque>
#include <map>

namespace ninfer::runtime {

// The caller destroys Program before closing the connection. Received continuations wait for
// their local stage; a commit from the coordinator does not imply local ResultReady yet.
template <class Types, class Program, class Codec>
class WorkerLoop {
public:
    WorkerLoop(distributed::WorkerConnection& connection, Program& program, Codec codec)
        : connection_(connection), program_(program), codec_(std::move(codec)),
          compute_(program.capacity().compute_frames), context_(program.capacity().context_frames) {
        if constexpr (requires { codec_.bind_input_storage(program_.input_storage()); })
            codec_.bind_input_storage(program_.input_storage());
        connection_.communication().reset_statistics();
        const auto& identity  = connection_.bootstrap().identity;
        observation_.rank     = identity.rank;
        observation_.node     = identity.node;
        observation_.hardware = connection_.device().properties().name;
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
    bool advance_impl() {
        connection_.communication().check();
        while (auto event = observe_host_call(host_.progress_ns, [&] { return program_.poll(); }))
            consume(std::move(*event));
        while (commands_.size() < 64) {
            auto message = connection_.poll();
            if (!message) break;
            if (message->epoch != connection_.bootstrap().epoch)
                throw std::logic_error("worker received a stale epoch");
            if (message->kind == distributed::wire::Kind::InputBegin ||
                message->kind == distributed::wire::Kind::InputChunk ||
                message->kind == distributed::wire::Kind::InputEnd ||
                message->kind == distributed::wire::Kind::InputRelease) {
                if constexpr (requires { codec_.receive_input(message->kind, message->payload); }) {
                    if (auto ready = codec_.receive_input(message->kind, message->payload)) {
                        distributed::wire::Encoder out;
                        out.u64(*ready);
                        connection_.send(distributed::wire::Kind::InputReady,
                                         std::move(out).take());
                    }
                } else
                    throw std::logic_error("execution codec has no input resources");
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
                if (!commands_.empty() || active() || measurement_boundary_)
                    throw std::logic_error("shutdown arrived before execution retirement");
                program_.drain();
                measurement_.cancel();
                return false;
            }
            switch (message->kind) {
            case distributed::wire::Kind::Submit:
            case distributed::wire::Kind::Supply:
            case distributed::wire::Kind::Commit:
            case distributed::wire::Kind::Discard:
                break;
            default:
                throw std::logic_error("unexpected worker runtime command");
            }
            distributed::wire::Decoder in(message->payload);
            auto envelope = protocol::envelope(in);
            if (envelope.ticket.epoch.value != connection_.bootstrap().epoch ||
                envelope.plan != connection_.topology().plan ||
                envelope.command.value != received_ + 1)
                throw std::logic_error("worker command identity or order differs");
            envelope.scope.validate(connection_.topology());
            ++received_;
            commands_.push_back({std::move(envelope), std::move(*message)});
        }
        if (!commands_.empty()) apply();
        progress_measurement();
        sample_resources();
        return true;
    }

    void observe_resources(bool idle) {
        observation_.memory        = program_.observe_resources();
        observation_.host          = host_;
        observation_.communication = connection_.communication().statistics();
        if (idle)
            observation_.physical_memory = sample_physical_memory(connection_.device().device());
        observation_.stamp = {observation_.stamp.sequence + 1, observation_time_ns()};
    }

    void progress_measurement() {
        if (!measurement_boundary_) return;
        const auto& boundary = *measurement_boundary_;
        if (applied_ < boundary.through_command || retired_prefix_ < boundary.through_command ||
            active())
            return;
        std::string error;
        try {
            measurement_.apply(boundary.command, program_);
        } catch (const std::exception& e) { error = e.what(); }
        observe_resources(true);
        distributed::wire::Encoder report;
        protocol::encode_measurement_report(report, boundary.id, error, observation_);
        auto payload = std::move(report).take();
        if (payload.size() > protocol::measurement_report_capacity(codec_.control_budget().report))
            throw std::logic_error("rank measurement report exceeds its prepared capacity");
        connection_.send(distributed::wire::Kind::MeasurementComplete, std::move(payload));
        measurement_boundary_.reset();
    }

    void sample_resources() {
        if (!received_) return;
        const auto now = observation_time_ns();
        if (now - observation_.stamp.sampled_ns < kResourceSamplePeriodNs) return;
        observe_resources(!active());
        distributed::wire::Encoder out;
        protocol::encode(out, observation_);
        // Cumulative samples may be skipped under pressure; execution continuations keep their
        // reserved capacity and the next sample still contains all completed work.
        (void)connection_.try_send(distributed::wire::Kind::Observation, std::move(out).take(),
                                   false);
    }

    enum class Phase { Empty, Running, Decision, Result, Retiring, Discarding };

    struct Flight {
        protocol::Envelope envelope;
        Phase phase                  = Phase::Empty;
        std::uint32_t decision_stage = 0;
        std::vector<std::uint64_t> commands;
    };

    struct Command {
        protocol::Envelope envelope;
        distributed::wire::Message message;
    };

    Flight& slot(FrameId frame) {
        return (frame.kind == FrameKind::Compute ? compute_ : context_).at(frame.index);
    }

    Flight& find(ExecutionTicket ticket) {
        auto& flight = slot(ticket.frame);
        if (flight.phase == Phase::Empty || flight.envelope.ticket != ticket)
            throw std::logic_error("worker command refers to a stale frame");
        return flight;
    }

    bool active() const {
        for (const auto* flights : {&compute_, &context_})
            for (const auto& f : *flights)
                if (f.phase != Phase::Empty) return true;
        return false;
    }

    void progress() {
        while (!completed_.empty() && completed_.begin()->first == retired_prefix_ + 1 &&
               completed_.begin()->second) {
            ++retired_prefix_;
            completed_.erase(completed_.begin());
        }
        distributed::wire::Encoder out;
        out.u64(applied_);
        out.u64(retired_prefix_);
        connection_.send(distributed::wire::Kind::Progress, std::move(out).take());
    }

    void apply() {
        auto& command        = commands_.front();
        const auto& envelope = command.envelope;
        if (measurement_boundary_ &&
            envelope.command.value > measurement_boundary_->through_command)
            return;
        distributed::wire::Decoder in(command.message.payload);
        (void)protocol::envelope(in);
        const auto rank = connection_.bootstrap().identity.rank;
        if (!envelope.scope.contains(rank)) {
            in.finish();
            applied_ = envelope.command.value;
            completed_.emplace(applied_, true);
            commands_.pop_front();
            progress();
            return;
        }
        if (command.message.kind == distributed::wire::Kind::Submit) {
            auto& flight = slot(envelope.ticket.frame);
            if (flight.phase != Phase::Empty) return;
            auto work = codec_.decode_work(in);
            in.finish();
            program_.validate_work(work, envelope.ticket.frame.kind);
            flight = {envelope, Phase::Running, 0, {envelope.command.value}};
            observe_host_call(host_.submit_ns, [&] {
                program_.start(
                    ExecutionRequest<Types>{envelope.ticket, envelope.command, std::move(work)});
            });
        } else {
            auto& flight = find(envelope.ticket);
            if (flight.envelope.scope.owners != envelope.scope.owners ||
                flight.envelope.scope.result_owner != envelope.scope.result_owner)
                throw std::logic_error("execution participants changed during a ticket");
            switch (command.message.kind) {
            case distributed::wire::Kind::Supply: {
                if (flight.phase != Phase::Decision) return;
                auto input = protocol::decision_input(in);
                in.finish();
                if (input.stage != flight.decision_stage)
                    throw std::logic_error("worker decision stage differs");
                program_.validate_decision(envelope.ticket, input);
                observe_host_call(host_.supply_ns, [&] {
                    program_.supply(envelope.ticket, envelope.command, std::move(input));
                });
                flight.phase = Phase::Running;
                break;
            }
            case distributed::wire::Kind::Commit: {
                if (flight.phase != Phase::Result) return;
                auto commit = codec_.decode_commit(in);
                in.finish();
                program_.prepare_commit(envelope.ticket, commit);
                observe_host_call(host_.commit_ns, [&] {
                    program_.commit(envelope.ticket, envelope.command, std::move(commit));
                });
                flight.phase = Phase::Retiring;
                break;
            }
            case distributed::wire::Kind::Discard:
                if (!program_.can_discard(envelope.ticket)) return;
                in.finish();
                observe_host_call(host_.discard_ns,
                                  [&] { program_.discard(envelope.ticket, envelope.command); });
                flight.phase = Phase::Discarding;
                break;
            default:
                throw std::logic_error("invalid worker continuation");
            }
            flight.envelope.command = envelope.command;
            flight.commands.push_back(envelope.command.value);
        }
        applied_ = envelope.command.value;
        completed_.emplace(applied_, false);
        commands_.pop_front();
    }

    void consume(ExecutionEvent<Types> event) {
        bool publish = true;
        std::visit(
            [&](auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, ExecutionFailed>) {
                    throw std::runtime_error(value.message);
                } else {
                    auto& flight = find(value.ticket);
                    const bool result_owner =
                        connection_.bootstrap().identity.rank == flight.envelope.scope.result_owner;
                    if constexpr (std::is_same_v<T, DecisionNeeded>) {
                        flight.phase          = Phase::Decision;
                        flight.decision_stage = value.request.stage;
                        publish               = result_owner;
                    } else if constexpr (std::is_same_v<T, ResultReady<Types>>) {
                        if (flight.phase == Phase::Discarding) {
                            publish = false;
                            return;
                        }
                        flight.phase = value.ticket.frame.kind == FrameKind::Compute
                                           ? Phase::Result
                                           : Phase::Retiring;
                        publish = result_owner || value.ticket.frame.kind == FrameKind::Context;
                    } else if constexpr (std::is_same_v<T, DeviceRetired>) {
                        value.observed.host = host_;
                        connection_.communication().observe_totals(
                            value.observed.communication_eager, value.observed.communication_graph);
                        if (flight.envelope.command != value.command ||
                            (flight.phase != Phase::Retiring && flight.phase != Phase::Discarding))
                            throw std::logic_error("worker retired an unresolved frame");
                        for (auto sequence : flight.commands) completed_.at(sequence) = true;
                        flight = {};
                    }
                }
            },
            event);
        if (publish) {
            auto payload = protocol::encode_event<Types>(event, codec_);
            if (payload.size() > codec_.control_budget().event)
                throw std::logic_error("execution event exceeds its prepared capacity");
            connection_.send(distributed::wire::Kind::Event, std::move(payload));
        }
        if (std::holds_alternative<DeviceRetired>(event)) progress();
    }

    distributed::WorkerConnection& connection_;
    Program& program_;
    Codec codec_;
    RankSnapshot observation_;
    RankHostStatistics host_;
    std::vector<Flight> compute_, context_;
    std::deque<Command> commands_;
    std::map<std::uint64_t, bool> completed_;

    struct MeasurementBoundary {
        std::uint64_t id = 0, through_command = 0;
        MeasurementCommand command{};
    };

    RankMeasurement measurement_;
    std::optional<MeasurementBoundary> measurement_boundary_;
    std::uint64_t received_ = 0, applied_ = 0, retired_prefix_ = 0;
};

} // namespace ninfer::runtime
