#include "core/cpu/task_pool.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer {

struct TaskPool::Impl {
    explicit Impl(TaskPoolOptions options) : options(std::move(options)) {
        if (this->options.threads == 0 || this->options.queue_capacity == 0)
            throw std::invalid_argument("task pool requires nonzero threads and queue capacity");
        queue.resize(this->options.queue_capacity);
        std::optional<CpuTopology> topology;
        if (!this->options.affinity.empty()) {
            topology.emplace();
            validate_cpu_affinity(topology->info(), this->options.affinity);
        }
        workers = cpu_detail::start_workers(
            this->options.threads,
            [&](std::uint32_t) {
                if (topology) bind_current_thread(*topology, this->options.affinity);
            },
            [](void* self, std::uint32_t) noexcept { static_cast<Impl*>(self)->worker_loop(); },
            this);
    }

    ~Impl() { shutdown(); }

    void shutdown() noexcept {
        {
            std::lock_guard lock(mutex);
            closed = true;
        }
        available.notify_all();
        space.notify_all();
        cpu_detail::join_workers(workers);
    }

    void push(Task& task) {
        queue[(head + queued) % queue.size()] = std::exchange(task, {});
        ++queued;
    }

    void worker_loop() noexcept {
        for (;;) {
            Task task;
            {
                std::unique_lock lock(mutex);
                available.wait(lock, [&] { return closed || queued != 0; });
                if (queued == 0) return;
                task = std::exchange(queue[head], {});
                head = (head + 1) % queue.size();
                --queued;
                ++active;
            }
            space.notify_one();
            task();
            {
                std::lock_guard lock(mutex);
                --active;
            }
        }
    }

    TaskPoolOptions options;
    mutable std::mutex mutex;
    std::condition_variable available, space;
    std::vector<Task> queue;
    std::size_t head = 0, queued = 0, active = 0;
    bool closed = false;
    std::vector<std::thread> workers;
};

TaskPool::TaskPool(TaskPoolOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

TaskPool::~TaskPool() = default;

EnqueueResult TaskPool::try_enqueue(Task& task) {
    if (!task) throw std::invalid_argument("task must not be empty");
    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->closed) return EnqueueResult::Closed;
        if (impl_->queued == impl_->queue.size()) return EnqueueResult::Full;
        impl_->push(task);
    }
    impl_->available.notify_one();
    return EnqueueResult::Accepted;
}

void TaskPool::enqueue(Task task, WaitCheck check) {
    if (!task) throw std::invalid_argument("task must not be empty");
    for (;;) {
        if (check) check();
        std::unique_lock lock(impl_->mutex);
        if (impl_->closed) throw std::runtime_error("task pool is closed");
        if (impl_->queued != impl_->queue.size()) {
            impl_->push(task);
            lock.unlock();
            impl_->available.notify_one();
            return;
        }
        const auto ready = [&] { return impl_->closed || impl_->queued < impl_->queue.size(); };
        if (check)
            impl_->space.wait_for(lock, std::chrono::milliseconds(10), ready);
        else
            impl_->space.wait(lock, ready);
    }
}

TaskPool::Snapshot TaskPool::snapshot() const {
    std::lock_guard lock(impl_->mutex);
    return {impl_->options.threads, impl_->queued, impl_->active};
}

void TaskPool::shutdown() noexcept { impl_->shutdown(); }

} // namespace ninfer
