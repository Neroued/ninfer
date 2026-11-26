#include "core/cpu/system.h"

#include <algorithm>
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
            CpuTopology allowed_node(CpuDiscoveryScope::AllowedNode);
            require(allowed_node.info().processors.size() >= topology.info().processors.size(),
                    "AllowedNode discovery shrank to thread-pinned single core");
            require(current_thread_affinity(topology) == CpuAffinity(one.begin(), one.end()),
                    "binding did not take effect");
            bind_current_thread(topology, original);
        } catch (...) { error = std::current_exception(); }
    });
    thread.join();
    if (error) std::rethrow_exception(error);
    require(current_thread_affinity(topology) == original, "discovery/worker changed caller");
}

void pci_bus_ids() {
    using cpu_detail::canonical_pci_bus_id;
    require(canonical_pci_bus_id("0000:3B:00.0") == "0000:3b:00.0", "uppercase PCI bus");
    require(canonical_pci_bus_id("528E:00:00.0") == "528e:00:00.0", "uppercase PCI domain");
    require(canonical_pci_bus_id("00000000:AF:00.0") == "0000:af:00.0", "padded PCI domain");
    require(canonical_pci_bus_id("1ABCD:FF:1F.7") == "0001abcd:ff:1f.7", "wide PCI domain");
    require(canonical_pci_bus_id("AF:00.0") == "0000:af:00.0", "omitted PCI domain");
    for (const auto bad :
         {"", "0000::00.0", "000000000:af:00.0", "100:00.0", "0000:ff:20.0", "ff:00.8", "ff:00.00",
          "../ff:00.0", "ff:00.0/..", "+ff:00.0", "ff:00.0 "}) {
        require(!canonical_pci_bus_id(bad), "malformed PCI bus ID accepted");
    }
    require(!canonical_pci_bus_id(std::string_view("ff:00.0\0/..", 11)),
            "embedded NUL in PCI bus ID accepted");
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

void node_worker_partitioning() {
    CpuTopologyInfo info{.processors = {{0, 0, 0, 0},
                                        {16, 0, 0, 0},
                                        {1, 1, 0, 0},
                                        {17, 1, 0, 0},
                                        {2, 2, 0, 0},
                                        {18, 2, 0, 0},
                                        {3, 3, 0, 0},
                                        {19, 3, 0, 0},
                                        {8, 4, 1, 1},
                                        {24, 4, 1, 1},
                                        {9, 5, 1, 1},
                                        {25, 5, 1, 1},
                                        {10, 6, 1, 1},
                                        {26, 6, 1, 1},
                                        {11, 7, 1, 1},
                                        {27, 7, 1, 1}}};
    const CpuAffinity socket0{0, 1, 2, 3, 16, 17, 18, 19};
    const CpuAffinity socket1{8, 9, 10, 11, 24, 25, 26, 27};
    const std::array localities{socket1, socket0};
    const auto local = plan_node_worker_affinity(info, localities);
    require(local[0] == socket1 && local[1] == socket0,
            "workers with different GPU localities did not retain their local cores");

    const auto verify_partition = [&](const auto& slices, bool keep_siblings) {
        CpuAffinity used;
        for (const auto& slice : slices) {
            require(!slice.empty(), "worker has an empty CPU affinity slice");
            for (const auto cpu : slice) {
                require(std::find(used.begin(), used.end(), cpu) == used.end(),
                        "node worker CPU slices overlap");
                used.push_back(cpu);
                if (keep_siblings) {
                    const auto sibling = cpu < 16 ? cpu + 16 : cpu - 16;
                    require(std::find(slice.begin(), slice.end(), sibling) != slice.end(),
                            "physical core siblings assigned to different workers");
                }
            }
        }
        require(used.size() == info.processors.size(), "node partition left CPUs idle");
    };
    // Overlapping locality preferences must still use the whole node disjointly.
    verify_partition(plan_node_worker_affinity(info, std::array{socket1, socket1}), true);
    // SMT gives every worker its own logical CPU even with more workers than cores.
    verify_partition(plan_node_worker_affinity(info, std::vector<CpuAffinity>(12)), false);
}
} // namespace

int main() {
    try {
        selection();
        affinity();
        pci_bus_ids();
        startup();
        node_worker_partitioning();
        std::cout << "OK CPU system\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
