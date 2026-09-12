#pragma once

#include "core/device.h"
#include "ninfer/types.h"
#include "runtime/engine/kv_capacity.h"

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

// Single-owner execution core for offline causal scoring. It deliberately has no request queue,
// Scheduler, ResourceManager, continuation catalog, or batching policy.
template <class Instance>
class CausalScoreCore {
public:
    using Package        = typename Instance::Package;
    using PreparedPrompt = typename Package::PreparedPrompt;

    CausalScoreCore(Instance& instance, DeviceContext& device)
        : instance_(instance), device_(device) {
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                device_.bind_to_current_thread();
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
            if (stopping_) { throw std::runtime_error("causal scoring engine is stopping"); }
            if (job_ != nullptr) {
                throw std::logic_error("causal scoring core already has an in-flight job");
            }
            job_ = std::move(job);
        }
        queue_cv_.notify_one();
        return result.get();
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const noexcept { return {}; }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
        } catch (...) {}
    }

private:
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
            try {
                std::vector<float> result;
                {
                    std::scoped_lock lock(execution_mutex_);
                    result =
                        instance_.program->causal_score(std::move(job->prompt), job->first_target);
                }
                job->promise.set_value(std::move(result));
            } catch (...) {
                try {
                    job->promise.set_exception(std::current_exception());
                } catch (...) {}
            }
        }
    }

    Instance& instance_;
    DeviceContext& device_;
    mutable std::mutex execution_mutex_;
    std::mutex call_mutex_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::unique_ptr<Job> job_;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
