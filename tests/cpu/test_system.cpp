#include "core/cpu/system.h"

#include <array>
#include <atomic>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void selection() {
    CpuTopologyInfo info{
        .processors = {
            {201, 38, 7}, {11, 17, 0}, {40, 9, 0}, {3, 9, 0}, {100, 38, 7}, {10, 17, 0}}};
    require(cpu_worker_order(info) == CpuAffinity({3, 10, 100}), "physical core ordering");
    require(plan_cpu_team(info, 4, true).worker_cpus == CpuAffinity({3, 10, 100, 40}),
            "SMT displaced unused cores");
    require(cpu_worker_order({.processors = {{8, {}, {}}, {31, {}, {}}}}) == CpuAffinity({8, 31}),
            "unknown core IDs treated as siblings");
    for (auto count : {0U, 4U}) {
        bool rejected = false;
        try {
            (void)plan_cpu_team(info, count);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid team size accepted");
    }
}

void affinity() {
    CpuTopology topology;
    const auto original = current_thread_affinity(topology);
    std::exception_ptr error;
    std::thread thread([&] {
        try {
            const std::array one{topology.info().processors.back().cpu};
            bind_current_thread(topology, one);
            CpuTopology restricted;
            require(restricted.info().processors.size() == 1 &&
                        restricted.info().processors[0].cpu == one[0],
                    "discovery ignored inherited affinity");
            require(current_thread_affinity(topology) == CpuAffinity(one.begin(), one.end()),
                    "binding did not take effect");
            bind_current_thread(topology, original);
        } catch (...) { error = std::current_exception(); }
    });
    thread.join();
    if (error) std::rethrow_exception(error);
    require(current_thread_affinity(topology) == original, "discovery/worker changed caller");
}

void startup() {
    std::atomic<unsigned> calls{0};
    const auto body = [](void* pointer, std::uint32_t) noexcept {
        ++*static_cast<std::atomic<unsigned>*>(pointer);
    };
    bool rejected = false;
    try {
        auto threads = cpu_detail::start_workers(
            4,
            [](std::uint32_t rank) {
                if (rank == 1) throw std::runtime_error("initializer failure");
            },
            body, &calls);
        cpu_detail::join_workers(threads);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && calls == 0, "failed startup published worker bodies");
    auto threads = cpu_detail::start_workers(4, [](std::uint32_t) {}, body, &calls);
    cpu_detail::join_workers(threads);
    require(calls == 4, "successful startup lost workers");
}
} // namespace

int main() {
    try {
        selection();
        affinity();
        startup();
        std::cout << "OK CPU system\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
