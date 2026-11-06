#pragma once

#include "ninfer/types.h"
#include "runtime/engine/context_capacity.h"
#include "runtime/executor/measurement.h"
#include "runtime/observability/snapshot.h"

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// One scoring job owns the execution slot until its result is delivered.
template <class Instance>
class CausalScoreCore {
public:
    static constexpr EnginePurpose purpose = EnginePurpose::CausalScoring;

    using ModelContract  = typename Instance::ModelContract;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;

    explicit CausalScoreCore(Instance& instance) : instance_(instance) {
        for (const auto& pool : instance_.session->kv_pools())
            published_snapshot_.runtime.kv_pools.push_back(
                {pool.id.index, std::string(pool.label), 0});
        published_snapshot_.runtime.context_transfers.kv.resize(
            instance_.session->kv_pools().size());
        published_snapshot_.ranks.assign(instance_.executor->observations().begin(),
                                         instance_.executor->observations().end());
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                instance_.executor->bind_to_current_thread();
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
            throw;
        }
    }

    ~CausalScoreCore() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    CausalScoreCore(const CausalScoreCore&)            = delete;
    CausalScoreCore& operator=(const CausalScoreCore&) = delete;

    [[nodiscard]] std::vector<float> score(PreparedPrompt prompt, std::uint32_t first_target) {
        // One synchronous public call owns the sole job slot until its result is delivered.
        std::scoped_lock call_lock(call_mutex_);
        auto job                               = std::make_unique<Job>();
        job->prompt                            = std::move(prompt);
        job->first_target                      = first_target;
        std::future<std::vector<float>> result = job->promise.get_future();
        {
            std::lock_guard queue_lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw std::runtime_error("causal scoring engine is unavailable");
            }
            if (job_ != nullptr) {
                throw std::logic_error("causal scoring core already has an in-flight job");
            }
            job_ = std::move(job);
        }
        queue_cv_.notify_one();
        return result.get();
    }

    [[nodiscard]] EngineSnapshot snapshot(SnapshotOptions options = {}) const {
        std::lock_guard lock(stats_mutex_);
        auto result = published_snapshot_;
        if (options.include_requests) result.requests.emplace();
        return result;
    }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    [[nodiscard]] EngineSnapshot measure(MeasurementCommand command) {
        std::scoped_lock call_lock(call_mutex_);
        auto request     = std::make_unique<MeasurementRequest>();
        request->command = command;
        auto result      = request->completion.get_future();
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_)
                throw std::runtime_error("causal scoring engine is unavailable");
            measurement_ = std::move(request);
        }
        queue_cv_.notify_one();
        return result.get();
    }

private:
    void publish() {
        EngineStatus status;
        {
            std::lock_guard lock(queue_mutex_);
            status = stopping_ ? EngineStatus::Stopping
                     : failed_ ? EngineStatus::Failed
                               : EngineStatus::Ready;
        }
        const auto usage = instance_.session->physical_usage();
        std::lock_guard lock(stats_mutex_);
        published_snapshot_.stamp = {published_snapshot_.stamp.sequence + 1, observation_time_ns()};
        published_snapshot_.scoring    = scoring_;
        published_snapshot_.executions = instance_.executor->execution_statistics();
        published_snapshot_.status     = status;
        published_snapshot_.in_flight.clear();
        instance_.executor->observe_in_flight(published_snapshot_.in_flight);
        published_snapshot_.runtime.context_pools     = usage.pools;
        published_snapshot_.runtime.context_resources = usage.resources;
        published_snapshot_.runtime.context_metadata  = usage.metadata;
        published_snapshot_.runtime.state_objects     = usage.state_objects;
        published_snapshot_.runtime.history_objects   = usage.history_objects;
        for (auto& pool : published_snapshot_.runtime.kv_pools)
            pool.occupied_page_groups = usage.kv_page_groups.at(pool.id);
        const auto ranks = instance_.executor->observations();
        published_snapshot_.ranks.assign(ranks.begin(), ranks.end());
    }

    struct Job {
        PreparedPrompt prompt;
        std::uint32_t first_target = 0;
        std::promise<std::vector<float>> promise;
    };

    struct MeasurementRequest {
        MeasurementCommand command{};
        std::promise<EngineSnapshot> completion;
    };

    void process_measurement(MeasurementRequest& request) noexcept {
        try {
            while (instance_.session->has_context_operation() ||
                   instance_.executor->has_pending_work()) {
                if (instance_.session->poll()) {
                    instance_.executor->fail("scoring measurement received an unsettled result");
                    throw std::logic_error("scoring measurement received an unsettled result");
                }
                instance_.executor->flush();
                std::this_thread::yield();
            }
            instance_.executor->measure(request.command);
            publish();
            request.completion.set_value(snapshot());
        } catch (...) {
            const auto error = std::current_exception();
            if (!instance_.executor->failed()) {
                try {
                    instance_.executor->measure({MeasurementAction::Cancel});
                } catch (...) { instance_.executor->fail("failed to close scoring measurement"); }
            }
            if (instance_.executor->failed()) {
                instance_.session->fail_all_cleanup();
                std::lock_guard lock(queue_mutex_);
                failed_ = true;
            }
            request.completion.set_exception(error);
        }
    }

    void worker_loop() noexcept {
        for (;;) {
            std::unique_ptr<Job> job;
            std::unique_ptr<MeasurementRequest> measurement;
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [&] { return stopping_ || job_ || measurement_; });
                if (stopping_ && !job_) {
                    measurement = std::move(measurement_);
                    lock.unlock();
                    instance_.session->fail_all_cleanup();
                    publish();
                    if (measurement)
                        measurement->completion.set_exception(std::make_exception_ptr(
                            std::runtime_error("causal scoring engine is shutting down")));
                    return;
                }
                job         = std::move(job_);
                measurement = std::move(measurement_);
            }
            if (measurement) {
                process_measurement(*measurement);
                continue;
            }
            bool execution_started = false;
            const auto started     = observation_time_ns();
            scoring_.active_jobs   = 1;
            publish();
            try {
                std::vector<float> result;
                {
                    const auto ticket =
                        instance_.session->start_score(std::move(job->prompt), job->first_target);
                    if (!ticket) {
                        instance_.executor->fail("scoring execution slot is occupied");
                        throw std::logic_error("scoring execution slot is occupied");
                    }
                    execution_started = true;
                    publish();
                    instance_.executor->flush();
                    auto next_observation = observation_time_ns() + 100'000'000;
                    for (;;) {
                        auto event = instance_.session->poll();
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
                        instance_.executor->flush();
                        const auto now = observation_time_ns();
                        if (now >= next_observation) {
                            publish();
                            next_observation = now + 100'000'000;
                        }
                        std::this_thread::yield();
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
                if (execution_started || instance_.executor->failed()) {
                    instance_.session->fail_all_cleanup();
                    {
                        std::lock_guard lock(queue_mutex_);
                        failed_ = true;
                    }
                }
                ++scoring_.failed_jobs;
                scoring_.active_jobs = 0;
                publish();
                try {
                    job->promise.set_exception(error);
                } catch (...) {}
            }
        }
    }

    Instance& instance_;
    mutable std::mutex stats_mutex_;
    EngineSnapshot published_snapshot_;
    ScoringStatistics scoring_;
    std::mutex call_mutex_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::unique_ptr<Job> job_;
    std::unique_ptr<MeasurementRequest> measurement_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
