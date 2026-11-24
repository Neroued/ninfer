#include "core/cpu/executor.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer {
namespace cpu_detail {
namespace {

constexpr std::uint32_t work_spin = 32768;
// A cold team wakes unevenly. Keep early arrivals active while the other members wake,
// instead of paying another park/wakeup cycle at each dependent computation stage.
constexpr std::uint32_t barrier_spin = 32768;

void wait_changed(const std::atomic<std::uint32_t>& value, std::uint32_t previous,
                  std::uint32_t spins, bool active_epoch = false) noexcept {
    while (value.load(std::memory_order_acquire) == previous) {
        for (std::uint32_t i = 0; i < spins; ++i) {
            if (value.load(std::memory_order_acquire) != previous) return;
            _mm_pause();
        }
        if (!active_epoch || !(previous & 1U)) value.wait(previous, std::memory_order_acquire);
    }
}

} // namespace

class alignas(64) TeamBarrier {
public:
    explicit TeamBarrier(std::uint32_t size) : size_(size), remaining_(size) {}

    void arrive() noexcept {
        if (size_ == 1) return;
        const auto generation = generation_.load(std::memory_order_relaxed);
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // The RMW chain collects all preceding writes; generation publishes the reset.
            remaining_.store(size_, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
            generation_.notify_all();
        } else {
            wait_changed(generation_, generation, barrier_spin);
        }
    }

private:
    const std::uint32_t size_;
    std::atomic<std::uint32_t> remaining_;
    std::atomic<std::uint32_t> generation_{0};
};
} // namespace cpu_detail

CpuWorkerContext::CpuWorkerContext(std::uint32_t rank, std::uint32_t size,
                                   cpu_detail::TeamBarrier& barrier) noexcept
    : rank_(rank), size_(size), barrier_(&barrier) {}

CpuWorkRange CpuWorkerContext::range(std::size_t items) const noexcept {
    const auto width = items / size_;
    const auto tail  = items % size_;
    const auto begin = rank_ * width + std::min<std::size_t>(rank_, tail);
    return {begin, begin + width + (rank_ < tail)};
}

void CpuWorkerContext::barrier() const noexcept { barrier_->arrive(); }

struct CpuExecutor::Impl {
    struct alignas(64) Worker {
        std::atomic<std::uint32_t> completed{0};
        CpuFeatures features;
    };

    Impl(const CpuTopology& topology, CpuTeamPlan selected)
        : plan(std::move(selected)), count(checked_count(plan.worker_cpus.size())),
          states(std::make_unique<Worker[]>(count)), barrier(count) {
        validate_cpu_affinity(topology.info(), plan.worker_cpus);
        threads = cpu_detail::start_workers(
            count,
            [&](std::uint32_t rank) {
                bind_current_thread(topology, std::span(&plan.worker_cpus[rank], 1));
                if (std::fesetenv(FE_DFL_ENV) != 0)
                    throw std::runtime_error("initialize CPU worker floating-point environment");
                _mm_setcsr(0x1f80);
                states[rank].features = current_cpu_features();
            },
            [](void* self, std::uint32_t rank) noexcept {
                static_cast<Impl*>(self)->worker_loop(rank);
            },
            this);
        features = states[0].features;
        for (std::uint32_t rank = 1; rank < count; ++rank)
            features = features.intersect(states[rank].features);
    }

    ~Impl() {
        stopping.store(true, std::memory_order_relaxed);
        epoch.fetch_add(2, std::memory_order_release);
        epoch.notify_all();
        cpu_detail::join_workers(threads);
    }

    static std::uint32_t checked_count(std::size_t count) {
        if (count == 0 || count > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("invalid CPU team size");
        return static_cast<std::uint32_t>(count);
    }

    void worker_loop(std::uint32_t rank) noexcept {
        const CpuWorkerContext worker(rank, count, barrier);
        std::uint32_t seen = 0;
        for (;;) {
            cpu_detail::wait_changed(epoch, seen, cpu_detail::work_spin, true);
            const auto next = epoch.load(std::memory_order_acquire);
            if (stopping.load(std::memory_order_relaxed)) return;
            if ((next >> 1) != (seen >> 1)) {
                work(arguments, worker);
                states[rank].completed.store(next >> 1, std::memory_order_release);
                states[rank].completed.notify_one();
            }
            seen = next;
        }
    }

    void run(CpuWork function, void* argument) {
        if (!function) throw std::invalid_argument("CPU work must not be null");
        if (running.test_and_set(std::memory_order_acquire))
            throw std::logic_error("CPU collective is already running");
        work            = function;
        arguments       = argument;
        const auto next = (epoch.fetch_add(2, std::memory_order_release) + 2U) >> 1;
        epoch.notify_all();
        for (std::uint32_t rank = 0; rank < count; ++rank) {
            auto& done          = states[rank].completed;
            const auto previous = done.load(std::memory_order_acquire);
            if (previous != next) cpu_detail::wait_changed(done, previous, cpu_detail::work_spin);
        }
        running.clear(std::memory_order_release);
    }

    CpuTeamPlan plan;
    const std::uint32_t count;
    std::unique_ptr<Worker[]> states;
    cpu_detail::TeamBarrier barrier;
    CpuFeatures features;
    // Bit 0 controls active waiting. Higher bits identify submitted collectives, so waking
    // or releasing an idle team never replays work or acknowledges an unexecuted call.
    alignas(64) std::atomic<std::uint32_t> epoch{0};
    std::atomic<bool> stopping{false};
    std::atomic_flag running = ATOMIC_FLAG_INIT;
    CpuWork work             = nullptr;
    void* arguments          = nullptr;
    std::vector<std::thread> threads;
};

CpuExecutor::CpuExecutor(const CpuTopology& topology, CpuTeamPlan plan)
    : impl_(std::make_unique<Impl>(topology, std::move(plan))) {}

CpuExecutor::~CpuExecutor() = default;

const CpuTeamPlan& CpuExecutor::plan() const noexcept { return impl_->plan; }

std::uint32_t CpuExecutionView::worker_count() const noexcept { return owner_->impl_->count; }

CpuFeatures CpuExecutionView::features() const noexcept { return owner_->impl_->features; }

CpuExecutionView::ActiveScope::ActiveScope(CpuExecutor& owner) : owner_(&owner) {
    auto& epoch = owner_->impl_->epoch;
    if (epoch.fetch_or(1U, std::memory_order_acq_rel) & 1U)
        throw std::logic_error("CPU team already has an active scope");
    epoch.notify_all();
}

CpuExecutionView::ActiveScope::~ActiveScope() {
    if (owner_) {
        auto& epoch = owner_->impl_->epoch;
        epoch.fetch_and(~1U, std::memory_order_release);
        epoch.notify_all();
    }
}

CpuExecutionView::ActiveScope::ActiveScope(ActiveScope&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)) {}

CpuExecutionView::ActiveScope CpuExecutionView::keep_active() const { return ActiveScope(*owner_); }

void CpuExecutionView::run(CpuWork work, void* arguments) const {
    owner_->impl_->run(work, arguments);
}

} // namespace ninfer
