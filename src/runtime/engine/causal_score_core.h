#pragma once

#include "ninfer/types.h"
#include "runtime/engine/context_capacity.h"
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
        published_snapshot_.ranks.assign(instance_.executor->observations().begin(),
                                         instance_.executor->observations().end());
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                instance_.executor->bind_to_current_thread();
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

    [[nodiscard]] EngineSnapshot snapshot() const {
        std::lock_guard lock(stats_mutex_);
        return published_snapshot_;
    }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_ && !failed_;
    }

    void reset_measurement_peaks() {
        std::scoped_lock execution_lock(execution_mutex_);
        if (instance_.executor->has_pending_work())
            throw std::logic_error("measurement window requires an idle engine");
        instance_.session->reset_measurement_peaks();
        std::lock_guard stats_lock(stats_mutex_);
        const auto ranks = instance_.executor->observations();
        published_snapshot_.ranks.assign(ranks.begin(), ranks.end());
    }

private:
    void publish() {
        std::lock_guard lock(stats_mutex_);
        published_snapshot_.stamp = {published_snapshot_.stamp.sequence + 1, observation_time_ns()};
        published_snapshot_.scoring    = scoring_;
        published_snapshot_.executions = instance_.executor->execution_statistics();
        const auto ranks               = instance_.executor->observations();
        published_snapshot_.ranks.assign(ranks.begin(), ranks.end());
    }

    struct Job {
        PreparedPrompt prompt;
        std::uint32_t first_target = 0;
        std::promise<std::vector<float>> promise;
    };

    void worker_loop() noexcept {
        for (;;) {
            std::unique_ptr<Job> job;
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [&] { return stopping_ || job_ != nullptr; });
                if (job_ == nullptr) {
                    if (stopping_) { return; }
                    continue;
                }
                job = std::move(job_);
            }
            bool execution_started = false;
            const auto started     = observation_time_ns();
            scoring_.active_jobs   = 1;
            publish();
            try {
                std::vector<float> result;
                {
                    std::scoped_lock lock(execution_mutex_);
                    const auto ticket =
                        instance_.session->start_score(std::move(job->prompt), job->first_target);
                    if (!ticket) {
                        instance_.executor->fail("scoring execution slot is occupied");
                        throw std::logic_error("scoring execution slot is occupied");
                    }
                    execution_started = true;
                    instance_.executor->flush();
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
                    {
                        std::scoped_lock lock(execution_mutex_);
                        instance_.session->fail_all_cleanup();
                    }
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
    mutable std::mutex execution_mutex_;
    mutable std::mutex stats_mutex_;
    EngineSnapshot published_snapshot_;
    ScoringStatistics scoring_;
    std::mutex call_mutex_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::unique_ptr<Job> job_;
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
