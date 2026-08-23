#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <utility>

namespace ninfer {

// Process-local bounded FIFO for ordinary host work. Callers own task semantics, cancellation,
// and result lifetime; this primitive owns only threads and queue backpressure.
class HostWorkerPool {
public:
    using Checkpoint = std::function<void()>;

    struct Snapshot {
        std::uint32_t threads = 0;
        std::size_t queued    = 0;
        std::size_t active    = 0;
    };

    HostWorkerPool(std::uint32_t threads, std::size_t queue_capacity);
    ~HostWorkerPool();

    HostWorkerPool(const HostWorkerPool&)            = delete;
    HostWorkerPool& operator=(const HostWorkerPool&) = delete;

    template <class Function>
    [[nodiscard]] auto submit(Function&& function, Checkpoint checkpoint = {})
        -> std::future<std::invoke_result_t<std::decay_t<Function>&>> {
        using Result = std::invoke_result_t<std::decay_t<Function>&>;
        auto task =
            std::make_shared<std::packaged_task<Result()>>(std::forward<Function>(function));
        std::future<Result> future = task->get_future();
        enqueue([task = std::move(task)] { (*task)(); }, std::move(checkpoint));
        return future;
    }

    [[nodiscard]] Snapshot snapshot() const;

private:
    void enqueue(std::function<void()> task, Checkpoint checkpoint);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
