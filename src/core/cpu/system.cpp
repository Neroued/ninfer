#include "core/cpu/system.h"

#include <hwloc.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <utility>

namespace ninfer {
namespace {

struct Bitmap {
    Bitmap() : value(hwloc_bitmap_alloc()) {
        if (!value) throw std::bad_alloc();
    }

    ~Bitmap() { hwloc_bitmap_free(value); }

    Bitmap(const Bitmap&)            = delete;
    Bitmap& operator=(const Bitmap&) = delete;
    hwloc_bitmap_t value;
};

void check(int result, const char* operation) {
    if (result < 0) throw std::system_error(errno, std::generic_category(), operation);
}

CpuAffinity cpu_list(hwloc_const_bitmap_t bitmap) {
    CpuAffinity result;
    int cpu;
    hwloc_bitmap_foreach_begin(cpu, bitmap) { result.push_back(cpu); }
    hwloc_bitmap_foreach_end();
    return result;
}

bool data_cache(hwloc_obj_type_t type) {
    return type == HWLOC_OBJ_L1CACHE || type == HWLOC_OBJ_L2CACHE || type == HWLOC_OBJ_L3CACHE ||
           type == HWLOC_OBJ_L4CACHE || type == HWLOC_OBJ_L5CACHE;
}

} // namespace

struct CpuTopology::Impl {
    hwloc_topology_t topology = nullptr;
    CpuTopologyInfo info;

    ~Impl() {
        if (topology) hwloc_topology_destroy(topology);
    }
};

CpuTopology::CpuTopology() : impl_(std::make_unique<Impl>()) {
    auto& state = *impl_;
    check(hwloc_topology_init(&state.topology), "initialize CPU topology");
    check(hwloc_topology_set_flags(state.topology, HWLOC_TOPOLOGY_FLAG_DONT_CHANGE_BINDING),
          "configure CPU topology");
    check(hwloc_topology_set_io_types_filter(state.topology, HWLOC_TYPE_FILTER_KEEP_NONE),
          "disable CPU topology I/O discovery");
    check(hwloc_topology_load(state.topology), "discover CPU topology");
    Bitmap allowed;
    check(hwloc_get_cpubind(state.topology, allowed.value, HWLOC_CPUBIND_THREAD),
          "read CPU affinity");
    hwloc_bitmap_and(allowed.value, allowed.value,
                     hwloc_topology_get_allowed_cpuset(state.topology));
    std::map<hwloc_obj_t, std::uint32_t> caches;
    hwloc_obj_t pu = nullptr;
    while ((pu = hwloc_get_next_obj_by_type(state.topology, HWLOC_OBJ_PU, pu))) {
        if (!hwloc_bitmap_isset(allowed.value, pu->os_index)) continue;
        CpuProcessor processor{.cpu = pu->os_index};
        hwloc_obj_t cache = nullptr;
        for (auto parent = pu->parent; parent; parent = parent->parent) {
            if (parent->type == HWLOC_OBJ_CORE) processor.core = parent->logical_index;
            if (data_cache(parent->type)) cache = parent;
        }
        if (cache) {
            const auto [it, inserted] = caches.emplace(cache, state.info.caches.size());
            processor.cache           = it->second;
            if (inserted) {
                state.info.caches.push_back({it->second, cache->attr->cache.depth,
                                             cache->attr->cache.size, cache->attr->cache.linesize});
            }
        }
        state.info.processors.push_back(processor);
    }
    if (state.info.processors.empty()) throw std::runtime_error("no available CPUs");
}

CpuTopology::~CpuTopology() = default;

const CpuTopologyInfo& CpuTopology::info() const noexcept { return impl_->info; }

CpuAffinity current_thread_affinity(const CpuTopology& topology) {
    Bitmap bitmap;
    check(hwloc_get_cpubind(topology.impl_->topology, bitmap.value, HWLOC_CPUBIND_THREAD),
          "read CPU affinity");
    return cpu_list(bitmap.value);
}

void validate_cpu_affinity(const CpuTopologyInfo& topology, std::span<const std::uint32_t> cpus) {
    if (cpus.empty()) throw std::invalid_argument("CPU affinity must not be empty");
    std::set<std::uint32_t> seen;
    for (const auto cpu : cpus) {
        const auto found = std::find_if(topology.processors.begin(), topology.processors.end(),
                                        [=](const auto& p) { return p.cpu == cpu; });
        if (found == topology.processors.end() || !seen.insert(cpu).second)
            throw std::invalid_argument("CPU affinity is unavailable or duplicated");
    }
}

void bind_current_thread(const CpuTopology& topology, std::span<const std::uint32_t> cpus) {
    validate_cpu_affinity(topology.info(), cpus);
    Bitmap requested;
    for (const auto cpu : cpus) hwloc_bitmap_set(requested.value, cpu);
    check(hwloc_set_cpubind(topology.impl_->topology, requested.value,
                            HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT),
          "bind CPU thread");
    Bitmap actual;
    check(hwloc_get_cpubind(topology.impl_->topology, actual.value, HWLOC_CPUBIND_THREAD),
          "verify CPU affinity");
    if (!hwloc_bitmap_isequal(actual.value, requested.value))
        throw std::runtime_error("CPU affinity did not match the requested set");
}

CpuAffinity cpu_worker_order(const CpuTopologyInfo& topology, bool use_smt) {
    auto processors = topology.processors;
    std::sort(processors.begin(), processors.end(), [](const auto& a, const auto& b) {
        return std::tie(a.cache, a.core, a.cpu) < std::tie(b.cache, b.core, b.cpu);
    });
    CpuAffinity result, siblings;
    std::set<std::uint32_t> cores;
    for (const auto& p : processors) {
        if (!p.core || cores.insert(*p.core).second)
            result.push_back(p.cpu);
        else
            siblings.push_back(p.cpu);
    }
    if (use_smt) result.insert(result.end(), siblings.begin(), siblings.end());
    return result;
}

CpuTeamPlan plan_cpu_team(const CpuTopologyInfo& topology, std::uint32_t workers, bool use_smt) {
    auto cpus = cpu_worker_order(topology, use_smt);
    if (workers == 0 || workers > cpus.size())
        throw std::invalid_argument("CPU team size exceeds the selected processors");
    cpus.resize(workers);
    return {std::move(cpus)};
}

std::vector<std::thread> cpu_detail::start_workers(std::uint32_t count,
                                                   std::function<void(std::uint32_t)> initialize,
                                                   WorkerBody body, void* argument) {
    if (count == 0 || !initialize || !body)
        throw std::invalid_argument("invalid CPU worker startup");

    struct Startup {
        std::mutex mutex;
        std::condition_variable changed;
        std::uint32_t ready                                   = 0;
        enum class Decision { Pending, Run, Cancel } decision = Decision::Pending;
        std::exception_ptr error;
        std::function<void(std::uint32_t)> initialize;
    };

    auto startup        = std::make_shared<Startup>();
    startup->initialize = std::move(initialize);
    std::vector<std::thread> threads;
    threads.reserve(count);
    try {
        for (std::uint32_t rank = 0; rank < count; ++rank) {
            threads.emplace_back([startup, body, argument, rank]() mutable {
                std::exception_ptr error;
                try {
                    startup->initialize(rank);
                } catch (...) { error = std::current_exception(); }
                bool run;
                {
                    std::unique_lock lock(startup->mutex);
                    if (error && !startup->error) startup->error = error;
                    ++startup->ready;
                    startup->changed.notify_all();
                    startup->changed.wait(
                        lock, [&] { return startup->decision != Startup::Decision::Pending; });
                    run = startup->decision == Startup::Decision::Run;
                }
                startup.reset();
                if (run) body(argument, rank);
            });
        }
        {
            std::unique_lock lock(startup->mutex);
            startup->changed.wait(lock, [&] { return startup->ready == count; });
            if (startup->error) std::rethrow_exception(startup->error);
            startup->decision = Startup::Decision::Run;
        }
        startup->changed.notify_all();
    } catch (...) {
        {
            std::lock_guard lock(startup->mutex);
            startup->decision = Startup::Decision::Cancel;
        }
        startup->changed.notify_all();
        join_workers(threads);
        throw;
    }
    return threads;
}

void cpu_detail::join_workers(std::span<std::thread> workers) noexcept {
    for (auto& worker : workers)
        if (worker.joinable()) worker.join();
}

} // namespace ninfer
