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

    DistributedExecutor(std::unique_ptr<distributed::ReplicaGroup> group,
                        ExecutionCapacity capacity, Codec codec, std::uint32_t max_decode_rows,
                        std::vector<RankSnapshot> initial_observations = {})
        : Executor<Types>(max_decode_rows), group_(std::move(group)), replica_(group_->replica()),
          pool_(capacity), codec_(std::move(codec)), budget_(codec_.control_budget()),
          compute_(capacity.compute_frames), context_(capacity.context_frames),
          progress_(replica_.ranks.size()), measurement_reports_(progress_.size()),
          measurement_errors_(progress_.size()), peer_commands_(progress_.size(), 0) {
        const auto local_count = replica_.ranks.size();
        this->observations_.resize(local_count);
        if (initial_observations.size() == local_count) {
            this->observations_ = std::move(initial_observations);
        } else if (initial_observations.size() == group_->topology().ranks.size()) {
            for (std::size_t i = 0; i < local_count; ++i)
                this->observations_[i] = std::move(initial_observations.at(replica_.ranks[i]));
        } else if (!initial_observations.empty()) {
            throw std::invalid_argument(
                "initial observations do not match replica or cluster rank count");
        }
        const auto model_budget = budget_;
        for (std::size_t i = 0; i < local_count; ++i) {
            const auto& rank            = group_->topology().rank(replica_.ranks[i]);
            this->observations_[i].rank = rank.rank;
            this->observations_[i].node = rank.node;
            const auto& observed        = this->observations_[i];
            budget_.report              = std::max(
                budget_.report,
                protocol::with_communication_capacity(model_budget, observed.communication).report);
            this->rank_descriptions_.push_back(
                {rank.rank, rank.node, rank.device, observed.hardware,
                 observed.physical_memory && observed.physical_memory->shared_physical_memory,
                 rank.tp, rank.ep, rank.dp, rank.pp, rank.adp});
            // Runtime communication counters start at activation, after preparation and capture.
            for (auto& comm_group : this->observations_[i].communication)
                for (auto& backend : comm_group.backends) {
                    backend.eager = {};
                    backend.graph = {};
                }
        }
        group_->reserve_control_capacity(
            protocol::control_transport_limits(budget_, pool_.capacity(), local_count, true));
    }

    ~DistributedExecutor() override {
        if (!closed_) group_->fail("distributed executor released with live execution");
    }

    [[nodiscard]] int poll_fd() const noexcept { return group_->poll_fd(); }

    void wait(std::chrono::milliseconds timeout) const { group_->wait(timeout); }

    void wait_for_activity(std::chrono::milliseconds timeout) const override {
        group_->wait(timeout);
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
        for (const auto rank : scope.participants) (void)replica_.local_index(rank);
        ExecutionTicket id{epoch(), next_execution_ + 1, reservation.frame()};
        Flight prepared;
        prepared.ticket = id;
        prepared.scope  = std::move(scope);
        if (const auto stage = submitted_decision_stage<Types>(work)) {
            if (prepared.scope.decision_owners.empty())
                throw std::logic_error("submitted decision has no consuming rank");
            prepared.decision.emplace();
            prepared.decision->stage = *stage;
            prepared.supplied        = true;
        }
        prepared.work.emplace(std::move(work));
        prepared.phase = Phase::Inputs;
        prepared.retired.resize(replica_.ranks.size());
        prepared.context_results.resize(replica_.ranks.size());
        prepared.consumed.resize(replica_.ranks.size());
        prepared.clock.start(Types::describe_work(*prepared.work), observation_time_ns());

        // Empty keys mark work that can submit without using the single media transfer slot.
        if constexpr (requires { codec_.prepare_inputs(*prepared.work, replica_); }) {
            if (!codec_.has_inputs(*prepared.work)) prepared.inputs.emplace();
        } else {
            prepared.inputs.emplace();
        }
        const bool prepare_now =
            !prepared.inputs && !input_owner_ && !inputs_.has_pending_releases() &&
            (pending_submits_.empty() || slot(pending_submits_.back()).inputs.has_value());
        pending_submits_.push_back(id.frame);
        bool accepted = false;
        try {
            (void)pool_.accept(std::move(reservation));
            accepted     = true;
            auto& flight = slot(id.frame);
            flight       = std::move(prepared);
            if (prepare_now) prepare_inputs(flight);
        } catch (...) {
            pending_submits_.pop_back();
            if (accepted) {
                slot(id.frame) = {};
                pool_.retire(id.frame);
            }
            throw;
        }
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
        if (flight.scope.decision_owners.empty())
            throw std::logic_error("decision supply has no decision owners in scope");
        codec_.validate_decision(*flight.work, *flight.decision, input);
        protocol::Envelope envelope{
            id, {next_command_ + 1}, 0, group_->topology().plan, flight.scope};
        try {
            (void)send_to_peers(flight.scope.decision_owners, envelope,
                                distributed::wire::Kind::Supply, distributed::Traffic::Continuation,
                                [&](auto& out, auto) { protocol::encode(out, input); });
        } catch (const std::exception& e) {
            fail(e.what());
            return;
        }
        flight.command  = envelope.command;
        flight.phase    = Phase::Running;
        flight.supplied = true;
        ++next_command_;
    }

    [[nodiscard]] CommitReceipt commit(ExecutionTicket id, Commit decisions) override {
        require_available();
        auto& flight = find(id);
        if (flight.phase != Phase::Result || !flight.result)
            throw std::logic_error("commit has no pending distributed result");
        codec_.validate_commit(*flight.work, *flight.result, decisions);
        protocol::Envelope envelope{
            id, {next_command_ + 1}, 0, group_->topology().plan, flight.scope};
        // Receipt is produced only after the complete fanout has entered the ordered queue.
        try {
            (void)send_to_peers(
                flight.scope.participants, envelope, distributed::wire::Kind::Commit,
                distributed::Traffic::Continuation,
                [&](auto& out, auto rank) { codec_.encode_commit(out, decisions, rank); });
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
        const auto suppress_result = [&] {
            std::erase_if(events_, [&](const Event& event) {
                return std::visit(
                    [&](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        return value.ticket == id && (std::is_same_v<T, DecisionNeeded> ||
                                                      std::is_same_v<T, DecisionConsumed> ||
                                                      std::is_same_v<T, ResultReady<Types>>);
                    },
                    event);
            });
        };
        if (id.frame.kind == FrameKind::Context) {
            if (std::any_of(events_.begin(), events_.end(), [&](const auto& event) {
                    const auto* retired = std::get_if<DeviceRetired>(&event);
                    return retired && retired->ticket == id;
                })) {
                suppress_result();
                return;
            }
            auto& flight = find(id);
            if (flight.phase == Phase::Discarding) return;
            suppress_result();
            flight.clock.resolve(observation_time_ns());
            if (flight.phase == Phase::Inputs) {
                events_.emplace_back(DeviceRetired{
                    id, flight.command, {}, flight.clock.finish(observation_time_ns())});
                retire_inputs(flight);
                std::erase(pending_submits_, id.frame);
                pool_.retire(id.frame);
                flight = {};
            } else {
                // Accepted Context work retires naturally with its Submit command.
                flight.phase = Phase::Discarding;
            }
            return;
        }

        auto& flight = find(id);
        if (flight.phase == Phase::Discarding || flight.phase == Phase::Retiring) return;
        if (flight.phase == Phase::Inputs) {
            flight.cancel_requested = true;
            return;
        }
        const auto mode = flight.supplied ? protocol::DiscardMode::AfterSupply
                                          : protocol::DiscardMode::BeforeSupply;
        protocol::Envelope envelope{
            id, {next_command_ + 1}, 0, group_->topology().plan, flight.scope};
        suppress_result();
        try {
            (void)send_to_peers(flight.scope.participants, envelope,
                                distributed::wire::Kind::Discard,
                                distributed::Traffic::Continuation,
                                [&](auto& out, auto) { out.u8(static_cast<std::uint8_t>(mode)); });
        } catch (const std::exception& e) {
            fail(e.what());
            return;
        }
        flight.command = envelope.command;
        flight.phase   = Phase::Discarding;
        flight.clock.resolve(observation_time_ns());
        ++next_command_;
    }

    [[nodiscard]] std::optional<Event> poll() override {
        if (failure_) return Event{std::exchange(failure_, {}).value()};
        if (failed_ || closed_) return {};
        try {
            while (auto incoming = group_->poll()) {
                auto& message = incoming->message;
                distributed::wire::Decoder in(message.payload);
                const auto local_idx = replica_.local_index(incoming->peer);
                if (message.kind == distributed::wire::Kind::Observation) {
                    auto snapshot = protocol::rank_snapshot(in);
                    in.finish();
                    const auto& identity = group_->topology().rank(incoming->peer);
                    if (snapshot.rank != identity.rank || snapshot.node != identity.node)
                        throw std::logic_error("resource observation has an invalid rank");
                    auto& previous = this->observations_.at(local_idx);
                    if (snapshot.stamp.sequence <= previous.stamp.sequence) continue;
                    snapshot.received_ns         = observation_time_ns();
                    snapshot.gpu_intervals       = previous.gpu_intervals;
                    snapshot.communication_eager = previous.communication_eager;
                    snapshot.communication_graph = previous.communication_graph;
                    snapshot.compute_paths       = previous.compute_paths;
                    snapshot.attention           = previous.attention;
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
                if (message.kind == distributed::wire::Kind::InputCredit) {
                    const auto count = in.u64();
                    in.finish();
                    inputs_.acknowledge_chunks(incoming->peer, count);
                    continue;
                }
                if (message.kind == distributed::wire::Kind::InputReady) {
                    const auto id = in.u64();
                    in.finish();
                    inputs_.complete(incoming->peer, id);
                    continue;
                }
                if (message.kind == distributed::wire::Kind::Progress) {
                    const auto applied = in.u64(), retired = in.u64();
                    in.finish();
                    if (applied > peer_commands_.at(local_idx) || retired > applied)
                        throw std::logic_error("rank progressed past accepted commands");
                    auto& progress = progress_.at(local_idx);
                    if (applied < progress.applied || retired < progress.retired)
                        throw std::logic_error("rank progress moved backwards");
                    progress = {applied, retired};
                    continue;
                }
                if (message.kind == distributed::wire::Kind::MeasurementComplete) {
                    if (message.payload.size() >
                        protocol::measurement_report_capacity(budget_.report))
                        throw std::logic_error(
                            "rank resource report exceeded its prepared capacity");
                    if (!measurement_query_ || in.u64() != measurement_query_ ||
                        measurement_reports_.at(local_idx))
                        throw std::logic_error("unexpected measurement acknowledgement");
                    measurement_errors_.at(local_idx) = in.string();
                    auto updated                      = protocol::rank_snapshot(in);
                    const auto& identity              = group_->topology().rank(incoming->peer);
                    if (updated.rank != identity.rank || updated.node != identity.node)
                        throw std::logic_error("measurement report has an invalid rank");
                    auto& previous              = this->observations_[local_idx];
                    updated.gpu_intervals       = previous.gpu_intervals;
                    updated.communication_eager = previous.communication_eager;
                    updated.communication_graph = previous.communication_graph;
                    updated.compute_paths       = previous.compute_paths;
                    updated.attention           = previous.attention;
                    merge_host_observations(updated.host, previous.host);
                    previous                                   = std::move(updated);
                    this->observations_[local_idx].received_ns = observation_time_ns();
                    measurement_reports_[local_idx]            = true;
                    in.finish();
                    continue;
                }
                if (message.kind != distributed::wire::Kind::Event)
                    throw std::logic_error("unexpected runtime control message");
                if (message.payload.size() > budget_.event)
                    throw std::logic_error("rank event exceeded its prepared capacity");
                auto event = protocol::decode_event<Types>(in, codec_);
                in.finish();
                consume(incoming->peer, std::move(event));
            }
            progress_inputs();
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
                group_->progress();
            } catch (const std::exception& e) { fail(e.what()); }
        }
    }

    void drain() override {
        if (closed_) return;
        for (const auto& flight : compute_) {
            if (failed_) break;
            if (flight.phase != Phase::Empty) discard(flight.ticket);
        }
        const auto until = std::chrono::steady_clock::now() + group_->shutdown_timeout();
        while ((pool_.in_flight() || inputs_.has_pending_releases()) && !failed_) {
            if (std::chrono::steady_clock::now() >= until) {
                fail("distributed execution drain timed out");
                break;
            }
            if (!poll()) group_->wait(std::chrono::milliseconds(1));
        }

        std::exception_ptr shutdown_error;
        if (!failed_) {
            try {
                group_->shutdown();
            } catch (...) { shutdown_error = std::current_exception(); }
        }
        // The failed group will never consume further input messages. Release both the codec
        // binding and any uploads that had yet to hand off their payload to a worker.
        if (input_owner_) {
            try {
                retire_codec_inputs(*slot(input_owner_->frame).work);
            } catch (...) {
                if (!shutdown_error) shutdown_error = std::current_exception();
            }
        }
        input_owner_.reset();
        inputs_ = {};
        compute_.clear();
        context_.clear();
        pending_submits_.clear();
        events_.clear();
        pool_.retire_submitted();
        closed_ = true;
        if (failed_ || shutdown_error) {
            try {
                group_->wait_for_workers();
            } catch (...) {
                if (!shutdown_error) throw;
            }
            if (shutdown_error) std::rethrow_exception(shutdown_error);
        }
    }

    void fail(std::string message) override {
        if (failed_) return;
        failed_  = true;
        failure_ = ExecutionFailed{{}, message};
        group_->fail(message);
    }

    [[nodiscard]] bool failed() const noexcept override { return failed_; }

    [[nodiscard]] bool has_pending_work() const noexcept override { return pool_.busy(); }

    void observe_in_flight(std::vector<ExecutionProgress>& out) const override {
        for (const auto* flights : {&compute_, &context_})
            for (const auto& flight : *flights) {
                if (flight.phase == Phase::Empty) continue;
                const auto stage = flight.phase == Phase::Decision     ? ExecutionStage::Decision
                                   : flight.phase == Phase::Result     ? ExecutionStage::ResultReady
                                   : flight.phase == Phase::Retiring   ? ExecutionStage::Retiring
                                   : flight.phase == Phase::Discarding ? ExecutionStage::Discarding
                                                                       : ExecutionStage::Submitted;
                out.push_back(flight.clock.progress(flight.ticket.id, stage));
            }
    }

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
        std::optional<std::vector<InputTransfers::Key>> inputs;
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

    void retire_codec_inputs(const Work& work) {
        if constexpr (requires { codec_.retire_inputs(work); }) (void)codec_.retire_inputs(work);
    }

    void prepare_inputs(Flight& flight) {
        auto& keys = flight.inputs.emplace();
        if constexpr (requires { codec_.prepare_inputs(*flight.work, replica_); }) {
            auto resources = codec_.prepare_inputs(*flight.work, replica_);
            if (resources.empty()) return;

            // Preparing media installs a single coordinator-side codec binding. Validate the
            // entire transfer before publishing any frames or assigning ownership to the ticket.
            try {
                for (const auto& resource : resources)
                    if (!flight.scope.contains(resource.rank))
                        throw std::invalid_argument("execution input peer is outside scope");
                InputTransfers::validate_frames(resources, group_->limits());
                keys = inputs_.begin(std::move(resources));
            } catch (...) {
                retire_codec_inputs(*flight.work);
                throw;
            }
            input_owner_ = flight.ticket;
        }
    }

    void progress_inputs() {
        inputs_.progress(*group_, epoch().value);
        while (!pending_submits_.empty()) {
            const auto frame = pending_submits_.front();
            auto& flight     = slot(frame);
            if (flight.phase != Phase::Inputs) {
                pending_submits_.pop_front();
                continue;
            }
            if (!flight.inputs) {
                if (input_owner_ || inputs_.has_pending_releases()) break;
                prepare_inputs(flight);
                if (input_owner_) inputs_.progress(*group_, epoch().value);
            }
            if (!inputs_.ready(*flight.inputs)) break;
            protocol::Envelope envelope{
                flight.ticket, {next_command_ + 1}, 0, group_->topology().plan, flight.scope};
            const bool sent =
                send_to_peers(flight.scope.participants, envelope, distributed::wire::Kind::Submit,
                              distributed::Traffic::Work, [&](auto& out, auto rank) {
                                  codec_.encode_work(out, *flight.work, rank);
                              });
            if (!sent) break;
            pending_submits_.pop_front();
            flight.command = envelope.command;
            flight.phase   = Phase::Running;
            ++next_command_;
            if (flight.cancel_requested) discard(flight.ticket);
        }
    }

    void retire_inputs(const Flight& flight) {
        if (input_owner_ != flight.ticket) return;
        retire_codec_inputs(*flight.work);
        inputs_.release(*flight.inputs);
        input_owner_.reset();
    }

    void measurement_boundary(MeasurementCommand command) {
        require_available();
        const auto until = std::chrono::steady_clock::now() + group_->shutdown_timeout();
        while (inputs_.has_pending_releases() && !failed_) {
            if (std::chrono::steady_clock::now() >= until) {
                fail("rank input release drain timed out before measurement");
                throw std::runtime_error("rank input release drain timed out before measurement");
            }
            if (auto event = poll()) {
                if (const auto* error = std::get_if<ExecutionFailed>(&*event))
                    throw std::runtime_error(error->message);
            } else {
                group_->wait(std::chrono::milliseconds(1));
            }
        }
        require_available();
        std::fill(measurement_reports_.begin(), measurement_reports_.end(), false);
        std::fill(measurement_errors_.begin(), measurement_errors_.end(), std::string{});
        const auto query = ++measurement_query_;
        std::vector<distributed::OutgoingMessage> messages;
        messages.reserve(replica_.ranks.size());
        for (std::size_t i = 0; i < replica_.ranks.size(); ++i) {
            distributed::wire::Encoder out;
            out.u64(query);
            out.u64(peer_commands_[i]);
            out.u8(static_cast<std::uint8_t>(command.action));
            out.boolean(command.profile);
            auto message = std::make_shared<distributed::wire::Message>(distributed::wire::Message{
                distributed::wire::Kind::Measurement, epoch().value, std::move(out).take()});
            messages.push_back({replica_.ranks[i], std::move(message)});
        }
        if (!group_->try_send(messages, distributed::Traffic::Continuation))
            throw std::logic_error("measurement exceeded its reserved control capacity");
        std::deque<Event> deferred;
        while (std::any_of(measurement_reports_.begin(), measurement_reports_.end(),
                           [](const auto& value) { return !value; })) {
            if (std::chrono::steady_clock::now() >= until) {
                std::string message = "rank measurement boundary timed out at command " +
                                      std::to_string(next_command_);
                for (std::size_t i = 0; i < progress_.size(); ++i)
                    if (!measurement_reports_[i])
                        message += "; rank " + std::to_string(replica_.ranks[i]) +
                                   " expected=" + std::to_string(peer_commands_[i]) +
                                   " applied=" + std::to_string(progress_[i].applied) +
                                   " retired=" + std::to_string(progress_[i].retired);
                fail(message);
                throw std::runtime_error(message);
            }
            if (auto event = poll()) {
                if (const auto* error = std::get_if<ExecutionFailed>(&*event))
                    throw std::runtime_error(error->message);
                deferred.push_back(std::move(*event));
            } else {
                group_->wait(std::chrono::milliseconds(1));
            }
        }
        events_.insert(events_.begin(), std::make_move_iterator(deferred.begin()),
                       std::make_move_iterator(deferred.end()));
        for (std::size_t i = 0; i < measurement_errors_.size(); ++i)
            if (!measurement_errors_[i].empty())
                throw std::runtime_error("rank " + std::to_string(replica_.ranks[i]) + ": " +
                                         measurement_errors_[i]);
    }

    template <class Encode>
    [[nodiscard]] bool send_to_peers(std::span<const distributed::RankId> recipients,
                                     const protocol::Envelope& envelope,
                                     distributed::wire::Kind kind, distributed::Traffic traffic,
                                     Encode encode_body) {
        std::vector<distributed::OutgoingMessage> messages;
        messages.reserve(recipients.size());
        for (const auto rank_id : recipients) {
            const auto local_idx        = replica_.local_index(rank_id);
            auto rank_envelope          = envelope;
            rank_envelope.peer_sequence = peer_commands_.at(local_idx) + 1;
            distributed::wire::Encoder out;
            protocol::encode(out, rank_envelope);
            encode_body(out, rank_id);
            auto payload = std::move(out).take();
            const auto limit =
                kind == distributed::wire::Kind::Submit
                    ? budget_.work(envelope.ticket.frame.kind)
                    : (kind == distributed::wire::Kind::Supply ? budget_.decision : budget_.commit);
            if (payload.size() > limit)
                throw std::logic_error("rank " + std::to_string(rank_id) + " message " +
                                       std::to_string(static_cast<unsigned>(kind)) + " encoded " +
                                       std::to_string(payload.size()) + " bytes; Codec bound is " +
                                       std::to_string(limit));
            auto message = std::make_shared<distributed::wire::Message>(
                distributed::wire::Message{kind, epoch().value, std::move(payload)});
            messages.push_back({rank_id, std::move(message)});
        }
        if (!group_->try_send(messages, traffic)) {
            if (traffic == distributed::Traffic::Work) return false;
            throw std::logic_error("execution exceeded its reserved control capacity");
        }
        for (const auto rank_id : recipients) ++peer_commands_.at(replica_.local_index(rank_id));
        return true;
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
                    auto& f              = find(value.ticket);
                    const auto local_idx = replica_.local_index(rank);
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
                        if (f.phase == Phase::Discarding) return;
                        if (!f.scope.is_decision_owner(rank) || !f.decision ||
                            value.stage != f.decision->stage || f.consumed.at(local_idx))
                            throw std::logic_error("decision consumption arrived out of order");
                        f.consumed[local_idx] = std::move(value);
                        if (std::all_of(f.scope.decision_owners.begin(),
                                        f.scope.decision_owners.end(), [&](auto r) {
                                            return f.consumed[replica_.local_index(r)].has_value();
                                        })) {
                            DecisionConsumed combined{f.ticket, f.decision->stage, {}};
                            for (auto r : f.scope.decision_owners)
                                for (const auto& row : f.consumed[replica_.local_index(r)]->rows) {
                                    auto found = std::find_if(
                                        combined.rows.begin(), combined.rows.end(),
                                        [&](const auto& v) { return v.member == row.member; });
                                    if (found == combined.rows.end())
                                        combined.rows.push_back(row);
                                    else
                                        found->bytes += row.bytes;
                                }
                            f.consumed_published = true;
                            events_.emplace_back(std::move(combined));
                            publish_result(f);
                        }
                    } else if constexpr (std::is_same_v<T, ResultReady<Types>>) {
                        if (f.phase == Phase::Discarding) return;
                        if (f.ticket.frame.kind == FrameKind::Context) {
                            if (f.context_results.at(local_idx))
                                throw std::logic_error("duplicate context result");
                            f.context_results[local_idx] = std::move(value.result);
                            if (std::all_of(f.scope.participants.begin(),
                                            f.scope.participants.end(), [&](auto r) {
                                                return f.context_results[replica_.local_index(r)]
                                                    .has_value();
                                            }))
                                f.result = codec_.merge_context_results(f.scope, replica_,
                                                                        f.context_results);
                        } else {
                            if (rank != f.scope.result_owner || f.result)
                                throw std::logic_error("unexpected compute result owner");
                            f.result = std::move(value.result);
                        }
                        publish_result(f);
                    } else if constexpr (std::is_same_v<T, DeviceRetired>) {
                        this->observe_retirement(static_cast<std::uint32_t>(local_idx),
                                                 value.observed,
                                                 value.ticket.frame.kind == FrameKind::Compute);
                        merge_gpu_intervals(f.gpu, value.observed.gpu);
                        if (f.ticket.frame.kind == FrameKind::Compute &&
                            f.phase != Phase::Retiring && f.phase != Phase::Discarding)
                            throw std::logic_error("compute frame retired before resolution");
                        if (value.command != f.command || f.retired.at(local_idx))
                            throw std::logic_error("stale or duplicate rank retirement");
                        f.retired[local_idx] = true;
                        if (std::all_of(
                                f.scope.participants.begin(), f.scope.participants.end(),
                                [&](auto r) { return f.retired[replica_.local_index(r)]; })) {
                            if (f.ticket.frame.kind == FrameKind::Context &&
                                f.phase != Phase::Discarding && !f.result_published)
                                throw std::logic_error("context retired before publication");
                            auto observed         = f.clock.finish(observation_time_ns());
                            observed.max_rank_gpu = f.gpu;
                            this->observe_execution(observed);
                            events_.emplace_back(DeviceRetired{f.ticket, f.command, {}, observed});
                            retire_inputs(f);
                            pool_.retire(f.ticket.frame);
                            f = {};
                        }
                    }
                }
            },
            event);
    }

    std::unique_ptr<distributed::ReplicaGroup> group_;
    distributed::ReplicaTopology replica_;
    ExecutionPool pool_;
    Codec codec_;
    protocol::ControlBudget budget_;
    InputTransfers inputs_;
    std::optional<ExecutionTicket> input_owner_;
    std::vector<Flight> compute_, context_;
    std::deque<FrameId> pending_submits_;

    struct Progress {
        std::uint64_t applied = 0, retired = 0;
    };

    std::vector<Progress> progress_;
    std::vector<bool> measurement_reports_;
    std::vector<std::string> measurement_errors_;
    std::vector<std::uint64_t> peer_commands_;
    std::deque<Event> events_;
    std::optional<ExecutionFailed> failure_;
    std::uint64_t next_execution_ = 0, next_command_ = 0;
    std::uint64_t measurement_query_ = 0;
    bool failed_ = false, closed_ = false;
};

} // namespace ninfer::runtime
