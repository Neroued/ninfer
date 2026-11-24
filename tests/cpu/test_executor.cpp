#include "core/cpu/executor.h"

#include <immintrin.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace ninfer;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Environment {
    Environment() { std::fegetenv(&saved); }

    ~Environment() { std::fesetenv(&saved); }

    fenv_t saved;
};

struct Work {
    const CpuTeamPlan* plan;
    std::vector<std::uint64_t> values, stages;
    std::vector<CpuFeatures> features;
    std::atomic<bool> failed{false};
    std::uint32_t iteration = 0;
    std::size_t items       = 0;
};

void compute(void* pointer, const CpuWorkerContext& worker) noexcept {
    auto& work       = *static_cast<Work*>(pointer);
    const auto check = [&](bool value) {
        if (!value) work.failed.store(true, std::memory_order_relaxed);
    };
    check(static_cast<unsigned>(sched_getcpu()) == work.plan->worker_cpus[worker.rank()]);
    check((_mm_getcsr() & 0xffc0U) == 0x1f80U && std::fegetround() == FE_TONEAREST);
    work.features[worker.rank()] = current_cpu_features();
    const auto range             = worker.range(work.items);
    for (auto i = range.begin; i < range.end; ++i) work.values[i] = (work.iteration + 1) * (i + 1);
    const auto large = worker.range(std::numeric_limits<std::size_t>::max());
    if (worker.rank() == 0) check(large.begin == 0);
    if (worker.rank() + 1 == worker.size())
        check(large.end == std::numeric_limits<std::size_t>::max());
    for (unsigned phase = 0; phase < 4; ++phase) {
        work.stages[worker.rank()] = work.iteration * 100 + phase * 10 + worker.rank();
        worker.barrier();
        for (unsigned rank = 0; rank < worker.size(); ++rank)
            check(work.stages[rank] == work.iteration * 100 + phase * 10 + rank);
        worker.barrier();
    }
}

void run_case(std::uint32_t workers) {
    Environment restore;
    std::fesetround(FE_DOWNWARD);
    _mm_setcsr(_mm_getcsr() | 0x8040U);
    const auto caller_mxcsr = _mm_getcsr();
    std::unique_ptr<CpuExecutor> executor;
    {
        CpuTopology topology;
        executor = std::make_unique<CpuExecutor>(topology, plan_cpu_team(topology.info(), workers));
    }
    // Execution does not borrow the topology or inherit the submitter's numeric controls.
    const auto execution = executor->execution_view();
    require(execution.device() == Device::cpu() && execution.worker_count() == workers,
            "execution facts disagree with owner");
    Work work{.plan     = &executor->plan(),
              .values   = std::vector<std::uint64_t>(257),
              .stages   = std::vector<std::uint64_t>(workers),
              .features = std::vector<CpuFeatures>(workers)};
    for (unsigned iteration = 0; iteration < 150; ++iteration) {
        work.iteration = iteration;
        work.items     = iteration % 3 == 0 ? 0 : iteration % 3 == 1 ? 1 : work.values.size();
        std::fill(work.values.begin(), work.values.end(), 0);
        if (iteration == 75) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        execution.run(compute, &work);
        require(!work.failed, "binding, FP controls or barrier publication failed");
        for (std::size_t i = 0; i < work.values.size(); ++i)
            require(work.values[i] == (i < work.items ? (iteration + 1) * (i + 1) : 0),
                    "collective coverage or final publication failed");
        auto common = work.features.front();
        for (auto features : work.features) common = common.intersect(features);
        require(common.bits == execution.features().bits,
                "team ISA is not the worker intersection");
    }
    require(_mm_getcsr() == caller_mxcsr && std::fegetround() == FE_DOWNWARD,
            "CPU executor changed submitting thread controls");
}

void reject_nested() {
    CpuTopology topology;
    CpuExecutor executor(topology, plan_cpu_team(topology.info(), 1));

    struct Call {
        CpuExecutionView execution;
        bool rejected = false;
    } call{executor.execution_view()};

    call.execution.run(
        [](void* pointer, const CpuWorkerContext&) noexcept {
            auto& call = *static_cast<Call*>(pointer);
            try {
                call.execution.run([](void*, const CpuWorkerContext&) noexcept {}, nullptr);
            } catch (const std::logic_error&) { call.rejected = true; }
        },
        &call);
    require(call.rejected, "nested collective was not rejected");
    call.execution.run([](void*, const CpuWorkerContext&) noexcept {}, nullptr);
}

void active_scope() {
    CpuTopology topology;
    const auto workers = std::min<std::size_t>(4, cpu_worker_order(topology.info()).size());
    CpuExecutor executor(topology, plan_cpu_team(topology.info(), workers));
    const auto execution = executor.execution_view();
    std::array<std::atomic<unsigned>, 4> calls{};
    const auto count = [](void* pointer, const CpuWorkerContext& worker) noexcept {
        auto* calls = static_cast<std::atomic<unsigned>*>(pointer);
        calls[worker.rank()].fetch_add(1, std::memory_order_relaxed);
        worker.barrier();
    };
    for (unsigned i = 0; i < 100; ++i) {
        {
            auto active = execution.keep_active();
            if (i % 20 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            for (std::size_t rank = 0; rank < workers; ++rank)
                require(calls[rank].load() == i, "waking the team replayed a collective");
            execution.run(count, calls.data());
            auto moved    = std::move(active);
            bool rejected = false;
            try {
                const auto nested = execution.keep_active();
            } catch (const std::logic_error&) { rejected = true; }
            require(rejected, "overlapping active scopes were accepted");
        }
        for (std::size_t rank = 0; rank < workers; ++rank)
            require(calls[rank].load() == i + 1, "active collective completion was lost");
    }
    try {
        const auto active = execution.keep_active();
        throw std::runtime_error("unwind activity");
    } catch (const std::runtime_error&) {}
    {
        const auto active = execution.keep_active();
        execution.run(count, calls.data());
    }
    execution.run(count, calls.data());
    for (std::size_t rank = 0; rank < workers; ++rank)
        require(calls[rank].load() == 102, "scope teardown changed collective execution");
}

void reject_concurrent() {
    CpuTopology topology;
    CpuExecutor executor(topology, plan_cpu_team(topology.info(), 1));
    const auto execution = executor.execution_view();

    struct Work {
        std::atomic<bool> started{false}, finish{false};
    } work;

    std::thread submitter([&] {
        execution.run(
            [](void* pointer, const CpuWorkerContext&) noexcept {
                auto& work = *static_cast<Work*>(pointer);
                work.started.store(true, std::memory_order_release);
                work.started.notify_one();
                work.finish.wait(false, std::memory_order_acquire);
            },
            &work);
    });
    work.started.wait(false, std::memory_order_acquire);
    bool rejected = false;
    try {
        execution.run([](void*, const CpuWorkerContext&) noexcept {}, nullptr);
    } catch (const std::logic_error&) { rejected = true; }
    work.finish.store(true, std::memory_order_release);
    work.finish.notify_one();
    submitter.join();
    require(rejected, "concurrent collective was not rejected");
    execution.run([](void*, const CpuWorkerContext&) noexcept {}, nullptr);
}
} // namespace

int main() {
    try {
        CpuTopology topology;
        run_case(1);
        run_case(std::min<std::size_t>(4, cpu_worker_order(topology.info()).size()));
        reject_nested();
        reject_concurrent();
        active_scope();
        std::cout << "OK CPU executor\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
