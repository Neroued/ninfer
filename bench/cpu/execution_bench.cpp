#include "core/cpu/executor.h"

#include <sys/resource.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {
using namespace ninfer;
using Clock = std::chrono::steady_clock;

std::uint32_t number(std::string_view text) {
    std::uint32_t value;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("expected an unsigned integer");
    return value;
}

double process_cpu_us() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("read process CPU time");
    return (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1e6 + usage.ru_utime.tv_usec +
           usage.ru_stime.tv_usec;
}

void measure(std::string_view name, CpuExecutionView execution, CpuWork work, std::uint32_t samples,
             std::uint32_t idle_us) {
    std::vector<double> times(samples);
    for (unsigned i = 0; i < 20; ++i) execution.run(work, nullptr);
    for (auto& time : times) {
        const auto start = Clock::now();
        execution.run(work, nullptr);
        time = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        if (idle_us) std::this_thread::sleep_for(std::chrono::microseconds(idle_us));
    }
    std::sort(times.begin(), times.end());
    // Short latency samples can finish before OS CPU accounting updates on every worker.
    // Measure aggregate CPU cost over a sustained interval at the same call cadence.
    const auto cpu_start  = process_cpu_us();
    const auto wall_start = Clock::now();
    std::size_t cpu_calls = 0;
    Clock::time_point finish;
    do {
        execution.run(work, nullptr);
        if (idle_us) std::this_thread::sleep_for(std::chrono::microseconds(idle_us));
        ++cpu_calls;
        finish = Clock::now();
    } while (finish - wall_start < std::chrono::milliseconds(250));
    const auto cpu_us     = process_cpu_us() - cpu_start;
    const auto wall_us    = std::chrono::duration<double, std::micro>(finish - wall_start).count();
    const auto percentile = [&](std::size_t percent) {
        return times[(times.size() * percent + 99) / 100 - 1];
    };
    std::cout << name << ',' << execution.worker_count() << ',' << samples << ',' << idle_us << ','
              << percentile(50) << ',' << percentile(95) << ',' << times.back() << ',' << cpu_calls
              << ',' << cpu_us / cpu_calls << ',' << wall_us / cpu_calls << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::uint32_t workers = 1, samples = 500, idle_us = 0;
        std::optional<std::uint32_t> caller;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option(argv[i]);
            if (option == "--help") {
                std::cout << "CPU collective scheduling benchmark\n"
                             "  --workers N --samples N --idle-us N --caller-cpu ID\n"
                             "Workers use one CPU per physical core. The caller inherits affinity\n"
                             "unless specified. CPU/wall cost uses a separate >=250ms interval\n"
                             "at the same cadence, including each post-call idle interval.\n";
                return 0;
            }
            if (++i == argc) throw std::invalid_argument("missing option value");
            const auto value = number(argv[i]);
            if (option == "--workers")
                workers = value;
            else if (option == "--samples")
                samples = value;
            else if (option == "--idle-us")
                idle_us = value;
            else if (option == "--caller-cpu")
                caller = value;
            else
                throw std::invalid_argument("unknown option");
        }
        if (!samples) throw std::invalid_argument("samples must be positive");
        CpuTopology topology;
        CpuExecutor executor(topology, plan_cpu_team(topology.info(), workers));
        if (caller) bind_current_thread(topology, std::span(&*caller, 1));
        const auto identity  = current_cpu_identity();
        const auto execution = executor.execution_view();
        std::cerr << "cpu=" << identity.vendor.data() << " family=" << identity.family
                  << " model=" << identity.model << " virtualized=" << identity.virtualized
                  << " features=" << execution.features().bits << " workers=";
        for (auto cpu : executor.plan().worker_cpus) std::cerr << cpu << ' ';
        std::cerr << "caller=";
        for (auto cpu : current_thread_affinity(topology)) std::cerr << cpu << ' ';
        std::cerr << '\n';
        std::cout << std::fixed << std::setprecision(3)
                  << "operation,workers,samples,idle_us,p50_us,p95_us,max_us,cpu_calls,"
                     "cpu_us_per_call,wall_us_per_call\n";
        measure(
            "dispatch", execution, [](void*, const CpuWorkerContext&) noexcept {}, samples,
            idle_us);
        measure(
            "barrier8", execution,
            [](void*, const CpuWorkerContext& worker) noexcept {
                for (unsigned i = 0; i < 8; ++i) worker.barrier();
            },
            samples, idle_us);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
