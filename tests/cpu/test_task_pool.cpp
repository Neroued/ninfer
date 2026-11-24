#include "core/cpu/task_pool.h"

#include <array>
#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void admission() {
    std::vector<int> order;
    TaskPool pool({.threads = 1, .queue_capacity = 2});
    // Broken-promise completion also releases the worker if an assertion unwinds this scope.
    std::promise<void> release, started;
    auto gate = release.get_future().share();
    pool.enqueue([&, gate] {
        started.set_value();
        gate.wait();
    });
    started.get_future().get();
    pool.enqueue([&] { order.push_back(1); });
    pool.enqueue([&] { order.push_back(2); });
    TaskPool::Task third = [&] { order.push_back(3); };
    require(pool.try_enqueue(third) == EnqueueResult::Full && third, "full queue consumed a task");
    auto snapshot = pool.snapshot();
    require(snapshot.active == 1 && snapshot.queued == 2, "active tasks counted as queued");
    unsigned checks = 0;
    bool cancelled  = false;
    try {
        pool.enqueue(third, [&] {
            (void)pool.snapshot(); // A wait check must not run under the queue lock.
            if (++checks == 2) throw std::runtime_error("cancel admission");
        });
    } catch (const std::runtime_error&) { cancelled = true; }
    require(cancelled && checks == 2, "blocked admission ignored cancellation");
    release.set_value();
    pool.shutdown();
    require(order == std::vector<int>({1, 2}), "FIFO/drain executed rejected work");
    require(pool.try_enqueue(third) == EnqueueResult::Closed && third,
            "closed queue consumed a task");
    pool.shutdown();
    snapshot = pool.snapshot();
    require(snapshot.active == 0 && snapshot.queued == 0, "shutdown returned before completion");
}

void shutdown_wakes_producer() {
    std::atomic<unsigned> executed{0};
    TaskPool pool({.threads = 1, .queue_capacity = 1});
    std::promise<void> release, started;
    auto gate = release.get_future().share();
    pool.enqueue([&, gate] {
        started.set_value();
        gate.wait();
        ++executed;
    });
    started.get_future().get();
    pool.enqueue([&] { ++executed; });
    std::promise<void> submitting;
    auto producer = std::async(std::launch::async, [&] {
        bool signalled = false;
        try {
            pool.enqueue([&] { ++executed; },
                         [&] {
                             if (!signalled) {
                                 submitting.set_value();
                                 signalled = true;
                             }
                         });
            return false;
        } catch (const std::runtime_error&) { return true; }
    });
    submitting.get_future().get();
    std::thread closer([&] { pool.shutdown(); });
    const bool rejected = producer.get();
    release.set_value();
    closer.join();
    require(rejected && executed == 2, "shutdown lost accepted work or admitted waiting work");
}

void concurrent_producers() {
    std::array<std::atomic<unsigned>, 600> seen{};
    TaskPool pool({.threads = 3, .queue_capacity = 5});
    std::vector<std::thread> producers;
    for (unsigned p = 0; p < 4; ++p) {
        producers.emplace_back([&, p] {
            for (unsigned i = p; i < seen.size(); i += 4) pool.enqueue([&, i] { ++seen[i]; });
        });
    }
    for (auto& producer : producers) producer.join();
    pool.shutdown();
    for (auto& count : seen) require(count == 1, "task lost or executed twice");
}

void affinity() {
    CpuTopology topology;
    const auto cpu = topology.info().processors.front().cpu;
    std::atomic<bool> correct{false};
    TaskPool pool({.threads = 1, .queue_capacity = 1, .affinity = {cpu}});
    pool.enqueue([&] {
        try {
            const auto affinity = current_thread_affinity(topology);
            correct             = affinity.size() == 1 && affinity.front() == cpu;
        } catch (...) {}
    });
    pool.shutdown();
    require(correct, "task pool ignored explicit affinity");
}
} // namespace

int main() {
    try {
        admission();
        shutdown_wakes_producer();
        concurrent_producers();
        affinity();
        std::cout << "OK task pool\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
