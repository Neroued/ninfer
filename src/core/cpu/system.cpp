#include "core/cpu/system.h"

#include <hwloc.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <fstream>
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
    Bitmap allowed;
    CpuTopologyInfo info;

    ~Impl() {
        if (topology) hwloc_topology_destroy(topology);
    }
};

CpuTopology::CpuTopology(CpuDiscoveryScope scope) : impl_(std::make_unique<Impl>()) {
    auto& state = *impl_;
    check(hwloc_topology_init(&state.topology), "initialize CPU topology");
    check(hwloc_topology_set_flags(state.topology, HWLOC_TOPOLOGY_FLAG_DONT_CHANGE_BINDING),
          "configure CPU topology");
    check(hwloc_topology_set_io_types_filter(state.topology, scope == CpuDiscoveryScope::AllowedNode
                                                                 ? HWLOC_TYPE_FILTER_KEEP_IMPORTANT
                                                                 : HWLOC_TYPE_FILTER_KEEP_NONE),
          "configure CPU topology I/O discovery");
    if (scope == CpuDiscoveryScope::AllowedNode)
        check(hwloc_topology_set_type_filter(state.topology, HWLOC_OBJ_PCI_DEVICE,
                                             HWLOC_TYPE_FILTER_KEEP_ALL),
              "configure PCI topology discovery");
    check(hwloc_topology_load(state.topology), "discover CPU topology");
    if (scope == CpuDiscoveryScope::CurrentThread) {
        check(hwloc_get_cpubind(state.topology, state.allowed.value, HWLOC_CPUBIND_THREAD),
              "read CPU affinity");
        hwloc_bitmap_and(state.allowed.value, state.allowed.value,
                         hwloc_topology_get_allowed_cpuset(state.topology));
    } else {
        hwloc_bitmap_copy(state.allowed.value, hwloc_topology_get_allowed_cpuset(state.topology));
    }
    std::map<hwloc_obj_t, std::uint32_t> caches;
    hwloc_obj_t pu = nullptr;
    while ((pu = hwloc_get_next_obj_by_type(state.topology, HWLOC_OBJ_PU, pu))) {
        if (!hwloc_bitmap_isset(state.allowed.value, pu->os_index)) continue;
        CpuProcessor processor{.cpu = pu->os_index};
        hwloc_obj_t cache = nullptr;
        for (auto parent = pu->parent; parent; parent = parent->parent) {
            if (parent->type == HWLOC_OBJ_CORE) processor.core = parent->logical_index;
            if (data_cache(parent->type)) cache = parent;
        }
        hwloc_obj_t numa = nullptr;
        while ((numa = hwloc_get_next_obj_by_type(state.topology, HWLOC_OBJ_NUMANODE, numa))) {
            if (numa->cpuset && hwloc_bitmap_isset(numa->cpuset, pu->os_index)) {
                processor.numa = numa->logical_index;
                break;
            }
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

std::optional<std::string> cpu_detail::canonical_pci_bus_id(std::string_view id) {
    const auto parse = [](std::string_view field, std::size_t width, unsigned max,
                          unsigned& value) {
        if (field.empty() || field.size() > width) return false;
        const auto end          = field.data() + field.size();
        const auto [ptr, error] = std::from_chars(field.data(), end, value, 16);
        return error == std::errc{} && ptr == end && value <= max;
    };
    const auto first = id.find(':');
    if (first == std::string_view::npos) return {};
    unsigned domain = 0, bus = 0, device = 0, function = 0;
    if (id.find(':', first + 1) != std::string_view::npos) {
        if (!parse(id.substr(0, first), 8, UINT32_MAX, domain)) return {};
        id.remove_prefix(first + 1);
    }
    const auto colon = id.find(':');
    const auto dot   = id.find('.');
    if (dot == std::string_view::npos || colon >= dot ||
        !parse(id.substr(0, colon), 2, 0xff, bus) ||
        !parse(id.substr(colon + 1, dot - colon - 1), 2, 0x1f, device) ||
        !parse(id.substr(dot + 1), 1, 7, function))
        return {};

    char canonical[sizeof("00000000:00:00.0")];
    std::snprintf(canonical, sizeof(canonical), "%0*x:%02x:%02x.%x", domain <= 0xffff ? 4 : 8,
                  domain, bus, device, function);
    return canonical;
}

CpuAffinity CpuTopology::device_locality_cpus(std::string_view pci_bus_id) const {
    const auto bus_id = cpu_detail::canonical_pci_bus_id(pci_bus_id);
    if (!bus_id) return {};
    const auto& state = *impl_;
    if (const auto pci = hwloc_get_pcidev_by_busidstring(state.topology, bus_id->c_str())) {
        const auto ancestor = hwloc_get_non_io_ancestor_obj(state.topology, pci);
        if (ancestor && ancestor->cpuset) {
            Bitmap local;
            hwloc_bitmap_and(local.value, ancestor->cpuset, state.allowed.value);
            if (!hwloc_bitmap_iszero(local.value)) return cpu_list(local.value);
        }
    }

    // Some kernels expose a PCI NUMA node in sysfs even when hwloc has no PCI object.
    int numa_id = -1;
    std::ifstream numa_file("/sys/bus/pci/devices/" + *bus_id + "/numa_node");
    if (!(numa_file >> numa_id) || numa_id < 0) return {};
    hwloc_obj_t node = nullptr;
    while ((node = hwloc_get_next_obj_by_type(state.topology, HWLOC_OBJ_NUMANODE, node))) {
        if (node->os_index != static_cast<unsigned>(numa_id) || !node->cpuset) continue;
        Bitmap local;
        hwloc_bitmap_and(local.value, node->cpuset, state.allowed.value);
        return cpu_list(local.value);
    }
    return {};
}

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
          "sched_setaffinity for CPU thread");
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

std::vector<CpuAffinity> plan_node_worker_affinity(const CpuTopologyInfo& topology,
                                                   std::span<const CpuAffinity> localities) {
    if (localities.empty() || topology.processors.empty())
        throw std::invalid_argument("node CPU affinity requires workers and processors");

    const auto worker_count = localities.size();
    auto ordered            = cpu_worker_order(topology);
    const bool split_cores  = ordered.size() < worker_count;
    if (split_cores) ordered = cpu_worker_order(topology, true);

    std::vector<CpuAffinity> result(worker_count);
    if (worker_count > ordered.size()) {
        for (std::size_t rank = 0; rank < worker_count; ++rank)
            result[rank].push_back(ordered[rank % ordered.size()]);
        return result;
    }

    std::vector<CpuAffinity> candidates;
    candidates.reserve(ordered.size());
    for (const auto cpu : ordered) {
        auto& cpus = candidates.emplace_back(1, cpu);
        if (split_cores) continue;
        const auto processor = std::find_if(topology.processors.begin(), topology.processors.end(),
                                            [cpu](const auto& p) { return p.cpu == cpu; });
        if (!processor->core) continue;
        for (const auto& sibling : topology.processors)
            if (sibling.cpu != cpu && sibling.core == processor->core) cpus.push_back(sibling.cpu);
    }

    std::vector<std::size_t> assigned(worker_count);
    const auto allocate = [&](bool prefer_local) {
        for (std::size_t round = 0; round * worker_count < candidates.size(); ++round) {
            for (std::size_t rank = 0; rank < worker_count; ++rank) {
                const auto quota =
                    candidates.size() / worker_count + (rank < candidates.size() % worker_count);
                if (assigned[rank] == quota || assigned[rank] > round) continue;
                const auto& locality = localities[rank];
                const auto it =
                    std::find_if(candidates.begin(), candidates.end(), [&](const auto& cpus) {
                        return !cpus.empty() &&
                               (!prefer_local ||
                                std::any_of(cpus.begin(), cpus.end(), [&](auto cpu) {
                                    return std::find(locality.begin(), locality.end(), cpu) !=
                                           locality.end();
                                }));
                    });
                if (it == candidates.end()) continue;
                result[rank].insert(result[rank].end(), it->begin(), it->end());
                it->clear();
                ++assigned[rank];
            }
        }
    };
    allocate(true);
    allocate(false);
    for (auto& cpus : result) std::sort(cpus.begin(), cpus.end());
    return result;
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
