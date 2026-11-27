#pragma once

#include "ninfer/types.h"
#include "runtime/engine/context_capacity.h"
#include "runtime/engine/instance.h"
#include "runtime/executor/measurement.h"
#include "runtime/observability/snapshot.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Per-replica scoring worker. CausalScoreCore below coordinates 1..dp replicas under a single
// engine-wide serial call_mutex_ until each accepted scoring job delivers its result.
template <class Replica>
class ReplicaCausalScoreWorker {
public:
    using ModelContract  = typename Replica::ModelContract;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;

    explicit ReplicaCausalScoreWorker(Replica& replica)
        : replica_(replica),
          initial_token_capacity_(estimate_context_token_capacity(replica.context_capacities)) {
        for (const auto& pool : replica_.session->kv_pools())
            published_.runtime.kv_pools.push_back({pool.id.index, std::string(pool.label), 0});
        published_.runtime.context_transfers.kv.resize(replica_.session->kv_pools().size());
        published_.replica_id = replica_.replica_id;
        published_.ranks.assign(replica_.executor->observations().begin(),
                                replica_.executor->observations().end());
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                replica_.executor->bind_to_current_thread();
                replica_.executor->set_peer_wakeup([this] {
                    {
                        std::lock_guard lock(queue_mutex_);
                        peer_wakeup_ = true;
                    }
                    queue_cv_.notify_one();
                });
                publish();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            replica_.executor->set_peer_wakeup({});
            throw;
        }
    }

    ~ReplicaCausalScoreWorker() noexcept {
        request_stop();
        if (worker_.joinable()) { worker_.join(); }
        replica_.executor->set_peer_wakeup({});
    }

    ReplicaCausalScoreWorker(const ReplicaCausalScoreWorker&)            = delete;
    ReplicaCausalScoreWorker& operator=(const ReplicaCausalScoreWorker&) = delete;

    void request_stop() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
    }

    [[nodiscard]] bool is_healthy() const noexcept {
        std::lock_guard lock(queue_mutex_);
        return !failed_;
    }

    [[nodiscard]] bool is_available() const noexcept {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    [[nodiscard]] std::optional<std::future<std::vector<float>>>
    try_start_score(const PreparedPrompt& prompt, std::uint32_t first_target) {
        std::future<std::vector<float>> result;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) return std::nullopt;
            if (job_ || measurement_)
                throw std::logic_error("causal scoring replica already has an in-flight call");
            auto job          = std::make_unique<Job>();
            job->prompt       = prompt;
            job->first_target = first_target;
            result            = job->promise.get_future();
            job_              = std::move(job);
        }
        queue_cv_.notify_one();
        return result;
    }

    [[nodiscard]] std::future<void> start_measure(MeasurementCommand command) {
        auto request     = std::make_unique<MeasurementRequest>();
        request->command = command;
        auto result      = request->completion.get_future();
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw std::runtime_error("causal scoring replica is unavailable");
            }
            if (job_ || measurement_)
                throw std::logic_error("causal scoring replica already has an in-flight call");
            measurement_ = std::move(request);
        }
        queue_cv_.notify_one();
        return result;
    }

    [[nodiscard]] ReplicaSnapshot snapshot(ScoringStatistics& out_scoring) const {
        std::lock_guard lock(stats_mutex_);
        out_scoring = published_scoring_;
        return published_;
    }

    [[nodiscard]] std::span<const RankDescriptor> rank_descriptions() const noexcept {
        return replica_.executor->rank_descriptions();
    }

private:
    struct Job {
        PreparedPrompt prompt;
        std::uint32_t first_target = 0;
        std::promise<std::vector<float>> promise;
    };

    struct MeasurementRequest {
        MeasurementCommand command{};
        std::promise<void> completion;
    };

    void publish() {
        bool healthy = true;
        EngineStatus status;
        {
            std::lock_guard lock(queue_mutex_);
            healthy = !failed_;
            status  = failed_     ? EngineStatus::Failed
                      : stopping_ ? EngineStatus::Stopping
                                  : EngineStatus::Ready;
        }
        std::lock_guard lock(stats_mutex_);
        published_.replica_id = replica_.replica_id;
        published_.healthy    = healthy;
        published_.status     = status;
        published_scoring_    = scoring_;
        if (!healthy) {
            published_.available_context_tokens_estimate = 0;
            published_.in_flight.clear();
            published_.runtime.context_pools.clear();
            published_.runtime.context_resources.clear();
            published_.runtime.context_metadata = {};
            published_.runtime.state_objects    = 0;
            published_.runtime.history_objects  = 0;
            for (auto& pool : published_.runtime.kv_pools) pool.occupied_page_groups = 0;
            return;
        }
        const auto usage      = replica_.session->physical_usage();
        published_.executions = replica_.executor->execution_statistics();
        published_.in_flight.clear();
        replica_.executor->observe_in_flight(published_.in_flight);
        for (auto& unit : published_.in_flight) { unit.replica_id = replica_.replica_id; }
        published_.runtime.context_pools     = usage.pools;
        published_.runtime.context_resources = usage.resources;
        const auto descriptions              = rank_descriptions();
        for (auto& pool : published_.runtime.context_pools)
            pool.rank = descriptions[pool.rank].rank;
        for (auto& resource : published_.runtime.context_resources)
            resource.rank = descriptions[resource.rank].rank;
        published_.runtime.context_metadata = usage.metadata;
        published_.runtime.state_objects    = usage.state_objects;
        published_.runtime.history_objects  = usage.history_objects;
        for (auto& pool : published_.runtime.kv_pools)
            pool.occupied_page_groups = usage.kv_page_groups.at(pool.id);
        const auto ranks = replica_.executor->observations();
        published_.ranks.assign(ranks.begin(), ranks.end());

        double min_ratio     = 1.0;
        bool has_device_pool = false;
        for (const auto& pool : usage.pools) {
            if (pool.domain != ContextMemoryDomain::Device || pool.capacity_bytes == 0) continue;
            has_device_pool = true;
            const double avail =
                static_cast<double>(pool.assigned_free_bytes + pool.unassigned_free_bytes +
                                    pool.tail_bytes + pool.cached_bytes);
            min_ratio = std::min(min_ratio, avail / static_cast<double>(pool.capacity_bytes));
        }
        published_.available_context_tokens_estimate =
            has_device_pool
                ? static_cast<std::uint64_t>(static_cast<double>(initial_token_capacity_) *
                                             std::clamp(min_ratio, 0.0, 1.0))
                : initial_token_capacity_;
    }

    void mark_failed_cleanup(std::exception_ptr error) noexcept {
        std::unique_ptr<Job> queued_job;
        std::unique_ptr<MeasurementRequest> queued_measurement;
        {
            std::lock_guard lock(queue_mutex_);
            failed_            = true;
            queued_job         = std::move(job_);
            queued_measurement = std::move(measurement_);
        }
        try {
            try {
                std::rethrow_exception(error);
            } catch (const std::exception& cause) {
                replica_.executor->fail(cause.what());
            } catch (...) { replica_.executor->fail("causal scoring replica failed"); }
        } catch (...) {}
        replica_.session->fail_all_cleanup();
        if (queued_job) ++scoring_.failed_jobs;
        publish();
        if (queued_job) queued_job->promise.set_exception(error);
        if (queued_measurement) queued_measurement->completion.set_exception(error);
    }

    void process_measurement(MeasurementRequest& request) noexcept {
        try {
            while (replica_.session->has_context_operation() ||
                   replica_.executor->has_pending_work()) {
                if (replica_.session->poll()) {
                    replica_.executor->fail("scoring measurement received an unsettled result");
                    throw std::logic_error("scoring measurement received an unsettled result");
                }
                replica_.executor->flush();
                replica_.executor->wait_for_activity(std::chrono::milliseconds(1));
            }
            replica_.executor->measure(request.command);
            publish();
            request.completion.set_value();
        } catch (...) {
            const auto error = std::current_exception();
            if (!replica_.executor->failed()) {
                try {
                    replica_.executor->measure({MeasurementAction::Cancel});
                } catch (...) { replica_.executor->fail("failed to close scoring measurement"); }
            }
            if (replica_.executor->failed()) mark_failed_cleanup(error);
            request.completion.set_exception(error);
        }
    }

    void worker_loop() noexcept {
        for (;;) {
            std::unique_ptr<Job> job;
            std::unique_ptr<MeasurementRequest> measurement;
            {
                std::unique_lock lock(queue_mutex_);
                if (!replica_.executor->has_pending_work())
                    queue_cv_.wait_for(lock, std::chrono::milliseconds(25), [&] {
                        return stopping_ || job_ || measurement_ || peer_wakeup_;
                    });
                peer_wakeup_ = false;
                if (stopping_ && !job_) {
                    measurement = std::move(measurement_);
                    lock.unlock();
                    replica_.session->fail_all_cleanup();
                    publish();
                    if (measurement)
                        measurement->completion.set_exception(std::make_exception_ptr(
                            std::runtime_error("causal scoring engine is shutting down")));
                    return;
                }
                job         = std::move(job_);
                measurement = std::move(measurement_);
            }
            if (!job && !measurement) {
                try {
                    (void)replica_.session->poll();
                    if (replica_.executor->has_pending_work())
                        replica_.executor->wait_for_activity(std::chrono::milliseconds(1));
                    if (replica_.executor->failed())
                        throw std::runtime_error("scoring worker disconnected while idle");
                } catch (...) {
                    mark_failed_cleanup(std::current_exception());
                    return;
                }
                continue;
            }
            if (measurement) {
                process_measurement(*measurement);
                if (!is_healthy()) return;
                continue;
            }
            bool native_admission = false;
            const auto started    = observation_time_ns();
            scoring_.active_jobs  = 1;
            try {
                publish();
                native_admission = true;
                std::vector<float> result;
                {
                    const auto ticket =
                        replica_.session->start_score(std::move(job->prompt), job->first_target);
                    native_admission = false;
                    if (!ticket) {
                        replica_.executor->fail("scoring execution slot is occupied");
                        throw std::logic_error("scoring execution slot is occupied");
                    }
                    publish();
                    replica_.executor->flush();
                    auto next_observation = observation_time_ns() + 100'000'000;
                    for (;;) {
                        auto event = replica_.session->poll();
                        if (event) {
                            if (event->ticket != *ticket ||
                                !std::holds_alternative<typename ModelContract::ScoreComplete>(
                                    event->result)) {
                                throw std::logic_error(
                                    "scoring received an unrelated execution result");
                            }
                            result = std::move(
                                std::get<typename ModelContract::ScoreComplete>(event->result)
                                    .logprobs);
                            break;
                        }
                        if (replica_.executor->failed())
                            throw std::runtime_error("causal scoring executor disconnected");
                        replica_.executor->flush();
                        const auto now = observation_time_ns();
                        if (now >= next_observation) {
                            publish();
                            next_observation = now + 100'000'000;
                        }
                        replica_.executor->wait_for_activity(std::chrono::milliseconds(1));
                    }
                }
                ++scoring_.completed_jobs;
                scoring_.scored_tokens += result.size();
                scoring_.active_jobs = 0;
                scoring_.wall.observe(observation_time_ns() - started);
                publish();
                job->promise.set_value(std::move(result));
            } catch (...) {
                const auto error = std::current_exception();
                ++scoring_.failed_jobs;
                scoring_.active_jobs = 0;
                bool fatal           = !native_admission || replica_.executor->failed();
                if (!fatal) {
                    try {
                        publish();
                    } catch (...) { fatal = true; }
                }
                if (fatal) mark_failed_cleanup(error);
                job->promise.set_exception(error);
                if (fatal) return;
            }
        }
    }

    Replica& replica_;
    std::uint64_t initial_token_capacity_ = 0;
    mutable std::mutex stats_mutex_;
    ReplicaSnapshot published_;
    ScoringStatistics published_scoring_;
    ScoringStatistics scoring_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::unique_ptr<Job> job_;
    std::unique_ptr<MeasurementRequest> measurement_;
    bool peer_wakeup_ = false;
    bool stopping_    = false;
    bool failed_      = false;
    std::thread worker_;
};

// Multi-replica causal scoring coordinator. Preserves a single engine-wide serial call boundary
// until each accepted score_tokens() call finishes, while isolating any failed replica domain.
template <class Instance>
class CausalScoreCore {
public:
    static constexpr EnginePurpose purpose = EnginePurpose::CausalScoring;

    using ModelContract  = typename Instance::ModelContract;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;
    using Replica        = typename Instance::Replica;

    explicit CausalScoreCore(Instance& instance) : instance_(instance) {
        if (instance_.replicas.empty()) {
            throw std::invalid_argument("CausalScoreCore requires at least one replica");
        }
        workers_.reserve(instance_.replicas.size());
        for (auto& rep : instance_.replicas) {
            workers_.push_back(std::make_unique<ReplicaCausalScoreWorker<Replica>>(*rep));
        }
    }

    ~CausalScoreCore() noexcept { request_stop(); }

    CausalScoreCore(const CausalScoreCore&)            = delete;
    CausalScoreCore& operator=(const CausalScoreCore&) = delete;

    void request_stop() noexcept {
        stopping_.store(true, std::memory_order_release);
        for (auto& w : workers_) { w->request_stop(); }
    }

    [[nodiscard]] std::vector<float> score(PreparedPrompt prompt, std::uint32_t first_target) {
        std::scoped_lock call_lock(call_mutex_);
        if (stopping_.load(std::memory_order_acquire)) {
            throw std::runtime_error("causal scoring engine is unavailable");
        }
        for (std::size_t offset = 0; offset < workers_.size(); ++offset) {
            const std::size_t idx = (rr_cursor_ + offset) % workers_.size();
            if (auto result = workers_[idx]->try_start_score(prompt, first_target)) {
                rr_cursor_ = (idx + 1) % workers_.size();
                return result->get();
            }
        }
        throw std::runtime_error("causal scoring engine is unavailable");
    }

    [[nodiscard]] bool is_available() const {
        if (stopping_.load(std::memory_order_acquire)) return false;
        return std::any_of(workers_.begin(), workers_.end(),
                           [](const auto& w) { return w->is_available(); });
    }

    [[nodiscard]] EngineSnapshot measure(MeasurementCommand command) {
        std::scoped_lock call_lock(call_mutex_);
        if (stopping_.load(std::memory_order_acquire)) {
            throw std::runtime_error("causal scoring engine is unavailable");
        }
        const bool begin = command.action == MeasurementAction::Begin;
        const bool end   = command.action == MeasurementAction::End;
        if ((begin || command.action == MeasurementAction::Reset) &&
            !measurement_participants_.empty())
            throw std::logic_error("a measurement window is already active");
        if (end && measurement_participants_.empty())
            throw std::logic_error("no measurement window is active");

        std::vector<std::size_t> healthy;
        if (!end) {
            healthy.reserve(workers_.size());
            for (std::size_t i = 0; i < workers_.size(); ++i)
                if (workers_[i]->is_available()) healthy.push_back(i);
            if (begin) measurement_participants_ = std::move(healthy);
        }
        const auto& participants = begin || end ? measurement_participants_ : healthy;
        try {
            if (participants.empty())
                throw std::runtime_error("causal scoring engine is unavailable");
            for (const auto idx : participants)
                if (!workers_[idx]->is_available())
                    throw std::runtime_error("causal scoring replica failed before measurement");
            std::vector<std::future<void>> futures;
            futures.reserve(participants.size());
            std::exception_ptr error;
            for (const auto idx : participants) {
                try {
                    futures.push_back(workers_[idx]->start_measure(command));
                } catch (...) {
                    if (!error) error = std::current_exception();
                }
            }
            for (auto& future : futures) {
                try {
                    future.get();
                } catch (...) {
                    if (!error) error = std::current_exception();
                }
            }
            if (error) std::rethrow_exception(error);
            for (const auto idx : participants)
                if (!workers_[idx]->is_available())
                    throw std::runtime_error("causal scoring replica failed during measurement");
            auto result = snapshot();
            if (end) measurement_participants_.clear();
            return result;
        } catch (...) {
            for (const auto idx : participants) {
                if (!workers_[idx]->is_available()) continue;
                try {
                    workers_[idx]->start_measure({MeasurementAction::Cancel}).get();
                } catch (...) {}
            }
            if (begin || end) measurement_participants_.clear();
            throw;
        }
    }

    [[nodiscard]] EngineSnapshot snapshot(SnapshotOptions options = {}) const {
        EngineSnapshot out;
        out.stamp = {snapshot_sequence_.fetch_add(1, std::memory_order_relaxed) + 1,
                     observation_time_ns()};
        if (options.include_requests) out.requests.emplace();

        bool any_healthy = false;
        out.replicas.reserve(workers_.size());
        std::map<std::pair<std::uint32_t, int>, PhysicalDeviceMemorySample> devices;
        std::map<std::uint32_t, PhysicalNodeMemorySample> nodes;

        for (std::size_t i = 0; i < workers_.size(); ++i) {
            ScoringStatistics rep_scoring{};
            auto rep_snap = workers_[i]->snapshot(rep_scoring);
            if (rep_snap.healthy) {
                any_healthy = true;
                out.available_context_tokens_estimate += rep_snap.available_context_tokens_estimate;
            }
            out.scoring.completed_jobs += rep_scoring.completed_jobs;
            out.scoring.failed_jobs += rep_scoring.failed_jobs;
            out.scoring.scored_tokens += rep_scoring.scored_tokens;
            out.scoring.active_jobs += rep_scoring.active_jobs;
            out.scoring.wall.count += rep_scoring.wall.count;
            out.scoring.wall.total_ns += rep_scoring.wall.total_ns;
            out.scoring.wall.maximum_ns =
                std::max(out.scoring.wall.maximum_ns, rep_scoring.wall.maximum_ns);

            if (rep_snap.healthy) {
                if (out.runtime.kv_pools.empty()) {
                    out.runtime.kv_pools = rep_snap.runtime.kv_pools;
                    out.runtime.context_transfers.kv.resize(out.runtime.kv_pools.size());
                } else {
                    for (std::size_t p = 0; p < std::min(out.runtime.kv_pools.size(),
                                                         rep_snap.runtime.kv_pools.size());
                         ++p) {
                        out.runtime.kv_pools[p].occupied_page_groups +=
                            rep_snap.runtime.kv_pools[p].occupied_page_groups;
                    }
                }
                out.runtime.context_pools.insert(out.runtime.context_pools.end(),
                                                 rep_snap.runtime.context_pools.begin(),
                                                 rep_snap.runtime.context_pools.end());
                out.runtime.context_resources.insert(out.runtime.context_resources.end(),
                                                     rep_snap.runtime.context_resources.begin(),
                                                     rep_snap.runtime.context_resources.end());
                out.runtime.context_metadata.budget_bytes +=
                    rep_snap.runtime.context_metadata.budget_bytes;
                out.runtime.context_metadata.allocated_bytes +=
                    rep_snap.runtime.context_metadata.allocated_bytes;
                out.runtime.context_metadata.reserved_bytes +=
                    rep_snap.runtime.context_metadata.reserved_bytes;
                out.runtime.state_objects += rep_snap.runtime.state_objects;
                out.runtime.history_objects += rep_snap.runtime.history_objects;
            }

            for (std::size_t k = 0; k < out.executions.size(); ++k) {
                out.executions[k].units += rep_snap.executions[k].units;
                out.executions[k].rows += rep_snap.executions[k].rows;
                out.executions[k].input_tokens += rep_snap.executions[k].input_tokens;
                out.executions[k].scheduled_positions += rep_snap.executions[k].scheduled_positions;
                for (auto field : {&ExecutionExposure::result, &ExecutionExposure::resolution,
                                   &ExecutionExposure::retirement, &ExecutionExposure::total}) {
                    auto& dst       = out.executions[k].exposure.*field;
                    const auto& src = rep_snap.executions[k].exposure.*field;
                    dst.count += src.count;
                    dst.total_ns += src.total_ns;
                    dst.maximum_ns = std::max(dst.maximum_ns, src.maximum_ns);
                }
                for (std::size_t b = 0; b < out.executions[k].latency.bins.size(); ++b) {
                    out.executions[k].latency.bins[b] += rep_snap.executions[k].latency.bins[b];
                }
            }
            out.in_flight.insert(out.in_flight.end(), rep_snap.in_flight.begin(),
                                 rep_snap.in_flight.end());
            const auto descs = workers_[i]->rank_descriptions();
            for (std::size_t r = 0; r < rep_snap.ranks.size(); ++r) {
                const auto& rank_snap = rep_snap.ranks[r];
                out.ranks.push_back(rank_snap);
                if (rep_snap.healthy && rank_snap.physical_memory && r < descs.size()) {
                    const auto key = std::pair{descs[r].node, descs[r].device};
                    auto it        = devices.find(key);
                    if (it == devices.end() ||
                        rank_snap.physical_memory->sampled_ns >= it->second.sample.sampled_ns) {
                        devices[key] = {descs[r].node, descs[r].device, *rank_snap.physical_memory};
                    }
                    auto nit = nodes.find(descs[r].node);
                    if (nit == nodes.end() ||
                        rank_snap.physical_memory->sampled_ns >= nit->second.sampled_ns) {
                        nodes[descs[r].node] = {descs[r].node,
                                                rank_snap.physical_memory->system_total_bytes,
                                                rank_snap.physical_memory->system_available_bytes,
                                                rank_snap.physical_memory->shared_physical_memory,
                                                rank_snap.physical_memory->sampled_ns};
                    }
                }
            }
            out.replicas.push_back(std::move(rep_snap));
        }
        std::sort(out.ranks.begin(), out.ranks.end(),
                  [](const auto& a, const auto& b) { return a.rank < b.rank; });
        for (auto& [k, v] : devices) out.physical_devices.push_back(std::move(v));
        for (auto& [k, v] : nodes) out.physical_nodes.push_back(std::move(v));
        out.status = stopping_.load(std::memory_order_acquire) ? EngineStatus::Stopping
                     : !any_healthy                            ? EngineStatus::Failed
                                                               : EngineStatus::Ready;
        return out;
    }

private:
    Instance& instance_;
    std::vector<std::unique_ptr<ReplicaCausalScoreWorker<Replica>>> workers_;
    std::mutex call_mutex_;
    std::vector<std::size_t> measurement_participants_;
    std::size_t rr_cursor_ = 0;
    std::atomic<bool> stopping_{false};
    mutable std::atomic<std::uint64_t> snapshot_sequence_{0};
};

} // namespace ninfer::runtime
