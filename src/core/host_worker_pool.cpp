#include "core/host_worker_pool.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer {

struct HostWorkerPool::Impl {
    Impl(std::uint32_t thread_count, std::size_t capacity)
        : threads_count(thread_count), queue_capacity(capacity) {
        if (threads_count == 0) {
            throw std::invalid_argument("host worker count must be nonzero");
        }
        if (queue_capacity == 0) {
            throw std::invalid_argument("host worker queue capacity must be nonzero");
        }
        workers.reserve(threads_count);
        try {
            for (std::uint32_t index = 0; index < threads_count; ++index) {
                workers.emplace_back([this] { worker_loop(); });
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex);
                stopping = true;
            }
            work_available.notify_all();
            for (std::thread& worker : workers) {
                if (worker.joinable()) { worker.join(); }
            }
            throw;
        }
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        work_available.notify_all();
        queue_space.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) { worker.join(); }
        }
    }

    void worker_loop() noexcept {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex);
                work_available.wait(lock, [this] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) { return; }
                task = std::move(queue.front());
                queue.pop_front();
                ++active;
            }
            queue_space.notify_one();
            try {
                task();
            } catch (...) {
                // submit() wraps work in packaged_task, so an escaping exception means the
                // executor contract itself was violated.
                std::terminate();
            }
            {
                std::lock_guard lock(mutex);
                if (active == 0) { std::terminate(); }
                --active;
            }
        }
    }

    const std::uint32_t threads_count;
    const std::size_t queue_capacity;
    mutable std::mutex mutex;
    std::condition_variable work_available;
    std::condition_variable queue_space;
    std::deque<std::function<void()>> queue;
    std::vector<std::thread> workers;
    std::size_t active = 0;
    bool stopping      = false;
};

HostWorkerPool::HostWorkerPool(std::uint32_t threads, std::size_t queue_capacity)
    : impl_(std::make_unique<Impl>(threads, queue_capacity)) {}

HostWorkerPool::~HostWorkerPool() = default;

void HostWorkerPool::enqueue(std::function<void()> task, Checkpoint checkpoint) {
    if (!task) { throw std::invalid_argument("host worker task must not be empty"); }
    for (;;) {
        if (checkpoint) { checkpoint(); }
        std::unique_lock lock(impl_->mutex);
        if (impl_->stopping) { throw std::runtime_error("host worker pool is stopping"); }
        if (impl_->queue.size() < impl_->queue_capacity) {
            impl_->queue.push_back(std::move(task));
            lock.unlock();
            impl_->work_available.notify_one();
            return;
        }
        impl_->queue_space.wait_for(lock, std::chrono::milliseconds(10));
    }
}

HostWorkerPool::Snapshot HostWorkerPool::snapshot() const {
    std::lock_guard lock(impl_->mutex);
    return Snapshot{
        .threads = impl_->threads_count, .queued = impl_->queue.size(), .active = impl_->active};
}

} // namespace ninfer
