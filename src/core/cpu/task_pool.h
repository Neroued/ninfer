#pragma once

#include "core/cpu/system.h"

#include <cstddef>
#include <functional>
#include <memory>

namespace ninfer {

struct TaskPoolOptions {
    std::uint32_t threads      = 0;
    std::size_t queue_capacity = 0;
    CpuAffinity affinity; // Empty inherits the creator's affinity.
};

enum class EnqueueResult { Accepted, Full, Closed };

// Independent tasks with bounded FIFO admission. Tasks own their results and exceptions;
// their bodies must not let an exception escape. Cancellation belongs to the consumer.
class TaskPool {
public:
    using Task      = std::function<void()>;
    using WaitCheck = std::function<void()>;

    struct Snapshot {
        std::uint32_t threads;
        std::size_t queued;
        std::size_t active;
    };

    explicit TaskPool(TaskPoolOptions);
    ~TaskPool();
    TaskPool(const TaskPool&)            = delete;
    TaskPool& operator=(const TaskPool&) = delete;

    // Only Accepted consumes task. Queue capacity excludes active tasks.
    [[nodiscard]] EnqueueResult try_enqueue(Task& task);
    // Checks before admission and while waiting, outside the queue lock. A thrown check
    // leaves the task unsubmitted. Blocking admission must not depend on this pool's workers.
    void enqueue(Task task, WaitCheck check = {});
    [[nodiscard]] Snapshot snapshot() const;
    // Owner-only, idempotent: close admission, drain accepted tasks, join. Not called by a task.
    // Callers must end their enqueue/snapshot calls before destroying the pool.
    void shutdown() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
