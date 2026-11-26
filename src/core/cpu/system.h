#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace ninfer {

using CpuAffinity = std::vector<std::uint32_t>; // OS logical CPU IDs.

struct CpuProcessor {
    std::uint32_t cpu = 0;
    std::optional<std::uint32_t> core;
    std::optional<std::uint32_t> cache;
    std::optional<std::uint32_t> numa;
};

struct CpuCache {
    std::uint32_t id         = 0;
    std::uint32_t level      = 0;
    std::uint64_t bytes      = 0;
    std::uint32_t line_bytes = 0;
};

struct CpuTopologyInfo {
    std::vector<CpuProcessor> processors;
    std::vector<CpuCache> caches;
};

enum class CpuDiscoveryScope : std::uint8_t {
    CurrentThread, // Limited to the discovering thread's inherited affinity.
    AllowedNode    // Whole OS-allowed machine topology, including PCIe/NUMA locality.
};

// Discovery does not change the calling thread's CPU binding.
class CpuTopology {
public:
    explicit CpuTopology(CpuDiscoveryScope scope = CpuDiscoveryScope::CurrentThread);
    ~CpuTopology();
    CpuTopology(const CpuTopology&)            = delete;
    CpuTopology& operator=(const CpuTopology&) = delete;

    [[nodiscard]] const CpuTopologyInfo& info() const noexcept;
    [[nodiscard]] CpuAffinity device_locality_cpus(std::string_view pci_bus_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend CpuAffinity current_thread_affinity(const CpuTopology&);
    friend void bind_current_thread(const CpuTopology&, std::span<const std::uint32_t>);
};

[[nodiscard]] CpuAffinity current_thread_affinity(const CpuTopology&);
// Only changes the calling thread; requested CPUs must belong to the snapshot.
void bind_current_thread(const CpuTopology&, std::span<const std::uint32_t> cpus);

// Assigns each available core once, preferring workers whose GPU is local to the core.
// Locality lists are indexed by node-local rank. When physical cores are oversubscribed,
// SMT siblings may be assigned separately before any logical CPUs are shared.
[[nodiscard]] std::vector<CpuAffinity>
plan_node_worker_affinity(const CpuTopologyInfo&, std::span<const CpuAffinity> localities);

struct CpuTeamPlan {
    CpuAffinity worker_cpus; // One fixed CPU per worker; order defines rank.
};

// Physical cores first, ordered by visible cache/core; optional SMT siblings follow.
// Unknown core identities remain independent CPUs. These helpers do not create threads.
[[nodiscard]] CpuAffinity cpu_worker_order(const CpuTopologyInfo&, bool use_smt = false);
[[nodiscard]] CpuTeamPlan plan_cpu_team(const CpuTopologyInfo&, std::uint32_t workers,
                                        bool use_smt = false);
void validate_cpu_affinity(const CpuTopologyInfo&, std::span<const std::uint32_t> cpus);

enum class CpuFeature : std::uint32_t {
    Avx2       = 1U << 0,
    Fma        = 1U << 1,
    F16c       = 1U << 2,
    Avx512F    = 1U << 3,
    Avx512Bw   = 1U << 4,
    Avx512Dq   = 1U << 5,
    Avx512Vl   = 1U << 6,
    Avx512Bf16 = 1U << 7,
    Avx512Vnni = 1U << 8,
    AvxVnni    = 1U << 9,
};

struct CpuFeatures {
    std::uint32_t bits = 0;

    [[nodiscard]] constexpr bool has(CpuFeature feature) const noexcept {
        return (bits & static_cast<std::uint32_t>(feature)) != 0;
    }

    [[nodiscard]] constexpr bool contains(CpuFeatures required) const noexcept {
        return (bits & required.bits) == required.bits;
    }

    [[nodiscard]] constexpr CpuFeatures intersect(CpuFeatures other) const noexcept {
        return {bits & other.bits};
    }
};

struct CpuIdentity {
    std::array<char, 13> vendor{};
    std::uint32_t family = 0, model = 0, stepping = 0;
    bool virtualized = false;
};

[[nodiscard]] CpuIdentity current_cpu_identity() noexcept;
[[nodiscard]] CpuFeatures current_cpu_features() noexcept;

namespace cpu_detail {

[[nodiscard]] std::optional<std::string> canonical_pci_bus_id(std::string_view);

struct FeatureRegisters {
    std::uint32_t leaf1_ecx = 0, leaf7_0_ebx = 0, leaf7_0_ecx = 0, leaf7_1_eax = 0;
    std::uint64_t xcr0 = 0;
};

[[nodiscard]] CpuFeatures usable_features(const FeatureRegisters&) noexcept;

// No body starts until every initializer succeeds. Failure joins all started threads.
// The returned threads belong to the caller, which supplies their normal stop/wakeup protocol.
using WorkerBody = void (*)(void*, std::uint32_t) noexcept;
[[nodiscard]] std::vector<std::thread> start_workers(std::uint32_t count,
                                                     std::function<void(std::uint32_t)> initialize,
                                                     WorkerBody body, void* argument);
void join_workers(std::span<std::thread>) noexcept;

} // namespace cpu_detail
} // namespace ninfer
