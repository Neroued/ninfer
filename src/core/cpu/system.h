#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace ninfer {

using CpuAffinity = std::vector<std::uint32_t>; // OS logical CPU IDs.

struct CpuProcessor {
    std::uint32_t cpu = 0;
    std::optional<std::uint32_t> core;
    std::optional<std::uint32_t> cache;
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

// A snapshot limited to the discovering thread's allowed affinity. Discovery does not bind it.
class CpuTopology {
public:
    CpuTopology();
    ~CpuTopology();
    CpuTopology(const CpuTopology&)            = delete;
    CpuTopology& operator=(const CpuTopology&) = delete;

    [[nodiscard]] const CpuTopologyInfo& info() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend CpuAffinity current_thread_affinity(const CpuTopology&);
    friend void bind_current_thread(const CpuTopology&, std::span<const std::uint32_t>);
};

[[nodiscard]] CpuAffinity current_thread_affinity(const CpuTopology&);
// Only changes the calling thread; requested CPUs must belong to the snapshot.
void bind_current_thread(const CpuTopology&, std::span<const std::uint32_t> cpus);

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
