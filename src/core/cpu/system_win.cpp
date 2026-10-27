// Windows implementation of core/cpu/system.h, mirroring the hwloc-backed Linux build.
// Topology discovery uses GetLogicalProcessorInformationEx; affinity uses group-affinity
// aware thread APIs. The pure helper logic (worker ordering, team planning, feature
// decoding, worker startup) matches system.cpp.

#include "core/cpu/system.h"

#include <intrin.h>
#include <windows.h>

#include <algorithm>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

namespace ninfer {
namespace {

// The engine numbers CPUs group-major: cpu = group * 64 + bit position. Single-group
// machines collapse to plain processor indices, matching hwloc's os_index there.
[[nodiscard]] std::uint32_t cpu_id(const GROUP_AFFINITY& affinity, DWORD bit) {
    return static_cast<std::uint32_t>(affinity.Group) * 64U + bit;
}

[[noreturn]] void fail(const char* operation) {
    throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                           operation);
}

struct ProcessorInfoDeleter {
    void operator()(void* pointer) const noexcept { ::HeapFree(::GetProcessHeap(), 0, pointer); }
};

struct SystemArray {
    std::unique_ptr<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX[], ProcessorInfoDeleter> data;
    std::size_t count = 0;

    static SystemArray query(LOGICAL_PROCESSOR_RELATIONSHIP relationship) {
        DWORD needed = 0;
        if (::GetLogicalProcessorInformationEx(relationship, nullptr, &needed) == 0 &&
            ::GetLastError() != ERROR_INSUFFICIENT_BUFFER)
            fail("GetLogicalProcessorInformationEx");
        auto* block = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            ::HeapAlloc(::GetProcessHeap(), 0, needed));
        if (!block) throw std::bad_alloc();
        SystemArray array{std::unique_ptr<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX[],
                                          ProcessorInfoDeleter>(block)};
        if (::GetLogicalProcessorInformationEx(relationship, block, &needed) == 0)
            fail("GetLogicalProcessorInformationEx");
        array.count = needed / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
        return array;
    }

    const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* begin() const noexcept { return data.get(); }
    const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* end() const noexcept {
        return data.get() + count;
    }
};

void mask_foreach(KAFFINITY mask, const auto& visit) {
    for (DWORD bit = 0; bit < 64; ++bit)
        if (mask & (static_cast<KAFFINITY>(1) << bit)) visit(bit);
}

} // namespace

struct CpuTopology::Impl {
    CpuTopologyInfo info;
};

CpuTopology::CpuTopology() : impl_(std::make_unique<Impl>()) {
    auto& state = *impl_;
    // Discovery is limited to the process's active mask, matching the Linux snapshot's
    // allowed-cpuset intersection. An empty set (a multi-group process) keeps all CPUs.
    std::set<std::uint32_t> allowed;
    {
        DWORD_PTR process_mask = 0, system_mask = 0;
        if (::GetProcessAffinityMask(::GetCurrentProcess(), &process_mask, &system_mask) != 0 &&
            process_mask != 0) {
            mask_foreach(static_cast<KAFFINITY>(process_mask),
                         [&](DWORD bit) { allowed.insert(bit); });
        }
    }

    // Map every processor to the compact id of its innermost unified/data cache.
    std::map<std::uint32_t, std::uint32_t> cache_by_processor;
    {
        const auto caches = SystemArray::query(RelationCache);
        for (const auto* entry = caches.begin(); entry != caches.end();
             entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                 reinterpret_cast<const BYTE*>(entry) + entry->Size)) {
            const auto& cache = entry->Cache;
            if (cache.Type != CacheUnified && cache.Type != CacheData) continue;
            const auto cache_id_index = static_cast<std::uint32_t>(state.info.caches.size());
            mask_foreach(cache.GroupMask.Mask,
                         [&](DWORD bit) {
                             cache_by_processor[cpu_id(cache.GroupMask, bit)] = cache_id_index;
                         });
            state.info.caches.push_back({cache_id_index, cache.Level,
                                         static_cast<std::uint64_t>(cache.CacheSize),
                                         cache.LineSize});
        }
    }

    {
        const auto cores = SystemArray::query(RelationProcessorCore);
        std::uint32_t core_index = 0;
        for (const auto* entry = cores.begin(); entry != cores.end();
             entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                 reinterpret_cast<const BYTE*>(entry) + entry->Size)) {
            const auto& core = entry->Processor;
            for (WORD group = 0; group < core.GroupCount; ++group) {
                const auto& member = core.GroupMask[group];
                mask_foreach(member.Mask, [&](DWORD bit) {
                    const auto cpu = cpu_id(member, bit);
                    if (!allowed.empty() && !allowed.contains(cpu)) return;
                    CpuProcessor processor{.cpu = cpu, .core = core_index};
                    const auto cache = cache_by_processor.find(cpu);
                    if (cache != cache_by_processor.end()) processor.cache = cache->second;
                    state.info.processors.push_back(processor);
                });
            }
            ++core_index;
        }
    }
    if (state.info.processors.empty()) throw std::runtime_error("no available CPUs");
    std::sort(state.info.processors.begin(), state.info.processors.end(),
              [](const CpuProcessor& a, const CpuProcessor& b) { return a.cpu < b.cpu; });
}

CpuTopology::~CpuTopology() = default;

const CpuTopologyInfo& CpuTopology::info() const noexcept { return impl_->info; }

namespace {

GROUP_AFFINITY requested_affinity(std::span<const std::uint32_t> cpus) {
    GROUP_AFFINITY result{};
    std::set<std::uint32_t> seen;
    WORD group = USHRT_MAX;
    for (const auto cpu : cpus) {
        if (!seen.insert(cpu).second)
            throw std::invalid_argument("CPU affinity is unavailable or duplicated");
        const auto cpu_group = static_cast<WORD>(cpu / 64U);
        if (group == USHRT_MAX) group = cpu_group;
        if (group != cpu_group)
            throw std::invalid_argument("Windows binds one thread to a single processor group");
        result.Mask |= static_cast<KAFFINITY>(1) << (cpu % 64U);
    }
    if (!result.Mask) throw std::invalid_argument("CPU affinity must not be empty");
    result.Group = group;
    return result;
}

} // namespace

CpuAffinity current_thread_affinity(const CpuTopology&) {
    GROUP_AFFINITY current{};
    if (::GetThreadGroupAffinity(::GetCurrentThread(), &current) == 0)
        fail("GetThreadGroupAffinity");
    CpuAffinity result;
    mask_foreach(current.Mask, [&](DWORD bit) { result.push_back(cpu_id(current, bit)); });
    return result;
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
    const auto requested = requested_affinity(cpus);
    if (::SetThreadGroupAffinity(::GetCurrentThread(), &requested, nullptr) == 0)
        fail("SetThreadGroupAffinity");
    GROUP_AFFINITY actual{};
    if (::GetThreadGroupAffinity(::GetCurrentThread(), &actual) == 0)
        fail("GetThreadGroupAffinity");
    if (actual.Group != requested.Group || actual.Mask != requested.Mask)
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

CpuFeatures cpu_detail::usable_features(const FeatureRegisters& r) noexcept {
    CpuFeatures result;
    constexpr auto avx_state = (1U << 26) | (1U << 27) | (1U << 28);
    if ((r.leaf1_ecx & avx_state) != avx_state || (r.xcr0 & 0x6U) != 0x6U) return result;
    const auto add = [&](bool available, CpuFeature feature) {
        if (available) result.bits |= static_cast<std::uint32_t>(feature);
    };
    add(r.leaf1_ecx & (1U << 12), CpuFeature::Fma);
    add(r.leaf1_ecx & (1U << 29), CpuFeature::F16c);
    add(r.leaf7_0_ebx & (1U << 5), CpuFeature::Avx2);
    add(r.leaf7_1_eax & (1U << 4), CpuFeature::AvxVnni);
    if ((r.xcr0 & 0xe6U) != 0xe6U || !(r.leaf7_0_ebx & (1U << 16))) return result;
    add(true, CpuFeature::Avx512F);
    add(r.leaf7_0_ebx & (1U << 17), CpuFeature::Avx512Dq);
    add(r.leaf7_0_ebx & (1U << 30), CpuFeature::Avx512Bw);
    add(r.leaf7_0_ebx & (1U << 31), CpuFeature::Avx512Vl);
    add(r.leaf7_0_ecx & (1U << 11), CpuFeature::Avx512Vnni);
    add(r.leaf7_1_eax & (1U << 5), CpuFeature::Avx512Bf16);
    return result;
}

CpuFeatures current_cpu_features() noexcept {
    cpu_detail::FeatureRegisters state;
    int registers[4]{};
    __cpuid(registers, 0);
    const auto leaves = static_cast<std::uint32_t>(registers[0]);
    if (leaves >= 1) {
        __cpuid(registers, 1);
        state.leaf1_ecx = static_cast<std::uint32_t>(registers[2]);
        if ((state.leaf1_ecx & ((1U << 26) | (1U << 27))) == ((1U << 26) | (1U << 27)))
            state.xcr0 = static_cast<std::uint64_t>(_xgetbv(0));
    }
    if (leaves >= 7) {
        __cpuidex(registers, 7, 0);
        const auto max_subleaf = static_cast<std::uint32_t>(registers[0]);
        state.leaf7_0_ebx      = static_cast<std::uint32_t>(registers[1]);
        state.leaf7_0_ecx      = static_cast<std::uint32_t>(registers[2]);
        if (max_subleaf >= 1) {
            __cpuidex(registers, 7, 1);
            state.leaf7_1_eax = static_cast<std::uint32_t>(registers[0]);
        }
    }
    return cpu_detail::usable_features(state);
}

CpuIdentity current_cpu_identity() noexcept {
    CpuIdentity result;
    int registers[4]{};
    __cpuid(registers, 0);
    std::memcpy(result.vendor.data(), &registers[1], 4);
    std::memcpy(result.vendor.data() + 4, &registers[3], 4);
    std::memcpy(result.vendor.data() + 8, &registers[2], 4);
    if (static_cast<std::uint32_t>(registers[0]) < 1) return result;
    __cpuid(registers, 1);
    const auto eax      = static_cast<std::uint32_t>(registers[0]);
    const auto ecx      = static_cast<std::uint32_t>(registers[2]);
    const auto family   = (eax >> 8) & 15U;
    result.family       = family + (family == 15 ? (eax >> 20) & 255U : 0);
    result.model        = ((eax >> 4) & 15U) | ((family == 6 || family == 15) ? (eax >> 12) & 240U : 0);
    result.stepping     = eax & 15U;
    result.virtualized  = (ecx & (1U << 31)) != 0;
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