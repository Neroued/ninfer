#pragma once

#include "core/cpu/system.h"
#include "core/memory.h"

#include <cstddef>
#include <memory>

namespace ninfer {

struct CpuWorkRange {
    std::size_t begin, end;
};

namespace cpu_detail {
class TeamBarrier;
}
class CpuExecutor;

class CpuWorkerContext {
public:
    [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }

    [[nodiscard]] std::uint32_t size() const noexcept { return size_; }

    [[nodiscard]] CpuWorkRange range(std::size_t items) const noexcept;
    // Every worker reaches the same sequence of barriers, including workers with empty ranges.
    void barrier() const noexcept;

private:
    friend class CpuExecutor;
    CpuWorkerContext(std::uint32_t, std::uint32_t, cpu_detail::TeamBarrier&) noexcept;
    std::uint32_t rank_, size_;
    cpu_detail::TeamBarrier* barrier_;
};

using CpuWork = void (*)(void*, const CpuWorkerContext&) noexcept;

class CpuExecutionView {
public:
    class ActiveScope {
    public:
        ~ActiveScope();
        ActiveScope(const ActiveScope&)            = delete;
        ActiveScope& operator=(const ActiveScope&) = delete;
        ActiveScope(ActiveScope&&) noexcept;
        ActiveScope& operator=(ActiveScope&&) = delete;

    private:
        friend class CpuExecutionView;
        explicit ActiveScope(CpuExecutor&);
        CpuExecutor* owner_;
    };

    [[nodiscard]] Device device() const noexcept { return Device::cpu(); }

    [[nodiscard]] std::uint32_t worker_count() const noexcept;
    [[nodiscard]] CpuFeatures features() const noexcept;
    // Wake the team and keep it polling between collectives until the scope ends.
    // Does not submit work or wait for workers. One scope per executor; it borrows the owner.
    [[nodiscard]] ActiveScope keep_active() const;
    // Synchronous collective: all writes are visible on return. Arguments and storage are
    // borrowed until then. No nested/concurrent run, worker CUDA calls or escaping exceptions.
    void run(CpuWork, void* arguments) const;

private:
    friend class CpuExecutor;

    explicit CpuExecutionView(CpuExecutor& owner) noexcept : owner_(&owner) {}

    CpuExecutor* owner_;
};

// Owns fixed, bound workers, never Op scratch. The submitting thread is not a worker.
// Workers use nearest-even, masked exceptions and gradual underflow; work preserves controls.
// Topology is borrowed only during construction. End all calls/views before owner destruction.
class CpuExecutor {
public:
    CpuExecutor(const CpuTopology&, CpuTeamPlan);
    ~CpuExecutor();
    CpuExecutor(const CpuExecutor&)            = delete;
    CpuExecutor& operator=(const CpuExecutor&) = delete;

    [[nodiscard]] CpuExecutionView execution_view() noexcept { return CpuExecutionView(*this); }

    [[nodiscard]] const CpuTeamPlan& plan() const noexcept;

private:
    friend class CpuExecutionView;
    friend class CpuExecutionView::ActiveScope;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
