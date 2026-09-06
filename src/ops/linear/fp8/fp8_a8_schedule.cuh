#pragma once

#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_a8_tma.cuh"
#include "ops/linear/fp8/fp8_config.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ninfer::ops::detail {

template <class Geometry>
struct Fp8LinearA8ProductionSchedule;

template <>
struct Fp8LinearA8ProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

template <>
struct Fp8LinearA8ProductionSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg,
                                Fp8MmaFragmentPipeline::PingPong, Fp8MmaRaster::TokenFast>;
};

// TMA route. One CTA per SM with a 256-token tile. Four stages put the tensor pipeline at
// 4 * (256 + 128) * 64 = 96 KiB, which is the largest depth that fits under the 99 KiB per-CTA
// cap; five would need 120 KiB and fails to build. The epilogue shares this storage through a
// union, so it adds nothing to the budget.
// Declared only, like Fp8LinearA8ProductionSchedule above it: a geometry that has never been
// measured on this route must be a build error, not a silent opt-in to another problem's schedule.
template <class Geometry>
struct Fp8LinearA8TmaSchedule;

// 256-token tile, four stages. Four is the deepest that fits: 4 * (256 + 128) * 64 = 96 KiB against
// the 99 KiB per-CTA cap, and five would need 120 KiB. One CTA per SM follows from that budget.
using Fp8A8TmaMeasuredSchedule = Fp8A8TmaSchedule<256, 4, 1>;

template <>
struct Fp8LinearA8TmaSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

template <>
struct Fp8LinearA8TmaSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

template <>
struct Fp8LinearA8TmaSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

template <>
struct Fp8LinearA8TmaSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

template <>
struct Fp8LinearA8TmaSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8A8TmaMeasuredSchedule;
};

// Multiprocessor count of the device this thread will launch on - the current device, which is
// where the launch that follows goes. Cached per device ordinal: the count steers which kernel
// runs, and a process that touches two different GPUs must not steer the second one with the
// first one's number. Returning zero declines the route, which is always a safe answer.
inline std::int32_t fp8_a8_multiprocessor_count() {
    static std::array<std::atomic<std::int32_t>, kFp8A8MaxDevices> cache{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kFp8A8MaxDevices) {
        return 0;
    }
    const std::int32_t cached = cache[device].load(std::memory_order_acquire);
    if (cached != 0) { return cached; }
    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return 0;
    }
    cache[device].store(value, std::memory_order_release);
    return value;
}

// Which of the two routes is cheaper at this width.
//
// Both kernels tile the same problem and both leave part of a wave idle at the end, but they
// quantise differently: the TMA route runs one CTA per SM over a 256-token tile, the cp.async route
// two CTAs per SM over a 64-token tile. So the decision is a comparison of two quantised costs, not
// a score for one of them - which also means the multiprocessor count enters the model on both
// sides instead of being frozen into a fitted constant.
//
// Cost of a route is (waves it needs) x (work one CTA does). Work per CTA is proportional to its
// token tile, so the tile widths carry it and cancel into kFp8A8TmaWorkRatio below.

// Time the TMA route takes per token of work, relative to the route it replaces. An empirical
// constant for one part, solved from the widest measured point on an RTX 5090 (sm_120a, 170 SMs);
// it is not portable and not a claim about either kernel in general. What the model represents is
// wave quantisation and nothing else - it has no term for pipeline fill, for K, or for the cost of
// an epilogue, and the two width bounds below exist because of shapes it therefore cannot tell
// apart. The sweeps behind the constant and both bounds are in
// docs/maintainer/fp8-a8-tma-route.md.
inline constexpr double kFp8A8TmaWorkRatio = 0.936;

// How much cheaper the model must find the TMA route before the route is taken. This is a margin on
// modelled cost, not on measured time: the model is a wave count times a tile width, so it is
// coarse, and a decision it calls within two percent is a decision it has not really made. Widths
// whose measured gain is smaller than this are still admitted - the margin buys confidence in the
// comparison, not a floor on the payoff.
inline constexpr double kFp8A8TmaMargin = 0.02;

// Which of the two routes the model calls cheaper, as a pure function of shape, width and part.
//
// This is the whole decision and it is written once. The runtime guard below calls it with the
// multiprocessor count it reads; the coverage checks call it with the count of the device the
// sweeps were taken on. Both therefore ask the same arithmetic about the same instantiated tiles,
// which a second copy written out in terms of loose integers would not.
template <class Geometry, class TmaSchedule, class MmaSchedule>
constexpr bool fp8_a8_tma_cheaper(std::int32_t tokens, std::int64_t multiprocessors) {
    const std::int64_t tma_blocks = fp8_a8_tma_blocks<Geometry, TmaSchedule>(tokens);
    const std::int64_t mma_tiles =
        (static_cast<std::int64_t>(tokens) + MmaSchedule::kBlockTokens - 1) /
        MmaSchedule::kBlockTokens;
    const std::int64_t mma_blocks =
        static_cast<std::int64_t>(Geometry::kOutputRows / MmaSchedule::kBlockRows) * mma_tiles;
    const std::int64_t mma_slots =
        static_cast<std::int64_t>(MmaSchedule::kMinBlocksPerSm) * multiprocessors;
    const std::int64_t tma_slots =
        static_cast<std::int64_t>(TmaSchedule::kMinBlocksPerSm) * multiprocessors;
    const std::int64_t tma_waves = (tma_blocks + tma_slots - 1) / tma_slots;
    const std::int64_t mma_waves = (mma_blocks + mma_slots - 1) / mma_slots;
    // Cost is waves times the work one SM carries through a wave - not waves alone. The two routes
    // put different amounts of work on an SM at once, and each route's occupancy comes from its own
    // schedule rather than from a number written here: the shipped TMA schedule places one CTA of
    // BlockM tokens, the cp.async schedule two CTAs of BlockTokens each. Comparing wave counts
    // without that weight makes the wider tile look free, which is exactly backwards - and reading
    // one route's occupancy from its schedule while fixing the other's in the model is how a future
    // schedule change silently stops being modelled.
    const double tma =
        static_cast<double>(tma_waves * TmaSchedule::kMinBlocksPerSm * TmaSchedule::kBlockM) *
        kFp8A8TmaWorkRatio;
    const double mma =
        static_cast<double>(mma_waves * MmaSchedule::kMinBlocksPerSm * MmaSchedule::kBlockTokens);
    return tma < mma * (1.0 - kFp8A8TmaMargin);
}

// The widest token count the route is allowed to take, per geometry.
//
// The model counts blocks along output rows and token tiles and has no term for K, so it gives one
// verdict for the two 5120-row residual shapes. Measured, they diverge above 4096. The bound is a
// measurement, stated per geometry so that an unswept shape cannot inherit it; the sweep is in the
// document above.
template <class Geometry>
inline constexpr std::int32_t kFp8A8TmaMaxTokens = std::numeric_limits<std::int32_t>::max();
template <>
inline constexpr std::int32_t kFp8A8TmaMaxTokens<Fp8Residual6144Geometry> = 4096;

// The narrowest width the route is offered at, for all geometries.
//
// This is a bound on the constant, not on the model. kFp8A8TmaWorkRatio was solved at and above the
// widths the product runs and has no validation below them, so the floor goes where the calibration
// starts. The model itself does better here than that suggests - at six sampled points below 1024
// it agrees with the measurement at five - and the floor accordingly gives up measured gains of 8
// to 12 percent to avoid one measured loss of 2. Whether it should move is a question for a wider
// sweep than the one behind it; both are in the document above.
inline constexpr std::int32_t kFp8A8TmaMinTokens = 1024;

// The same question asked of a geometry, with both of its registered schedules filled in, and the
// measured width bound applied on top of the modelled cost.
template <class Geometry>
constexpr bool fp8_a8_tma_admits(std::int32_t tokens, std::int64_t multiprocessors) {
    if (tokens < kFp8A8TmaMinTokens) { return false; }
    if (tokens > kFp8A8TmaMaxTokens<Geometry>) { return false; }
    return fp8_a8_tma_cheaper<Geometry, typename Fp8LinearA8TmaSchedule<Geometry>::Type,
                              typename Fp8LinearA8ProductionSchedule<Geometry>::Type>(
        tokens, multiprocessors);
}

// The device the width bounds and the ratio were measured on. It is here so the coverage checks
// below ask their question against a fixed machine rather than against whatever built the tree.
inline constexpr std::int64_t kFp8A8CalibrationSms = 170;

// A width is covered if the runtime would take it. The runtime adds representability, the address
// test and the grid.y limit on top of this, and none of those depend on the width.
template <class Geometry>
constexpr bool fp8_a8_tma_covers(std::int32_t tokens) {
    return fp8_a8_tma_admits<Geometry>(tokens, kFp8A8CalibrationSms);
}

// The widths the numerical tests run at, pinned so they cannot quietly stop covering the route.
// These assert coverage and nothing else: that the predicate still sends these widths down the TMA
// path, so the tests that run them are testing the kernel they were extended for. They are not a
// statement that the route is faster - a predicate answer cannot carry that - and re-tuning the
// constant is expected to trip one of them. The fix then is to re-pick the test width from the new
// frontier and re-run the tests on it, not to widen the constant until the assertion passes.
//
// tests/ops/linear/test_fp8_a8.cpp and tests/ops/linear_add/test_fp8.cpp check values against a
// host reference and cannot see which kernel produced them, so without these the suite would stay
// green over a route that had disappeared. One aligned width and one leaving a partial trailing
// tile per geometry, the trailing tile being the majority of the admitted set.
static_assert(fp8_a8_tma_covers<Fp8AttnInputGeometry>(4096) &&
                  fp8_a8_tma_covers<Fp8AttnInputGeometry>(4288) &&
                  fp8_a8_tma_covers<Fp8AttnInputGeometry>(1345),
              "test_fp8_a8.cpp covers attn_input at 4096, 4288 and 1345");
static_assert(fp8_a8_tma_covers<Fp8GdnInputGeometry>(4096) &&
                  fp8_a8_tma_covers<Fp8GdnInputGeometry>(4160) &&
                  fp8_a8_tma_covers<Fp8GdnInputGeometry>(1153),
              "test_fp8_a8.cpp covers gdn_input at 4096, 4160 and 1153");
static_assert(fp8_a8_tma_covers<Fp8MlpGateUpGeometry>(4096) &&
                  fp8_a8_tma_covers<Fp8MlpGateUpGeometry>(4288) &&
                  fp8_a8_tma_covers<Fp8MlpGateUpGeometry>(1153),
              "test_fp8_a8.cpp covers mlp_gate_up at 4096, 4288 and 1153");
static_assert(fp8_a8_tma_covers<Fp8Residual6144Geometry>(1664) &&
                  fp8_a8_tma_covers<Fp8Residual6144Geometry>(4001),
              "both fp8 tests cover residual 5120x6144 at 1664 and 4001");
// attn_input reaches the route at the shipped default chunk, which is also a width its test
// runs. Every other route-taking test width is pinned below; without this one a retune could
// drop 1024 off the route and leave that test green over the kernel it was added to exercise.
static_assert(fp8_a8_tma_covers<Fp8AttnInputGeometry>(1024),
              "attn_input T=1024 is a route-taking test width");
static_assert(fp8_a8_tma_covers<Fp8Residual17408Geometry>(4160) &&
                  fp8_a8_tma_covers<Fp8Residual17408Geometry>(4001),
              "both fp8 tests cover residual 5120x17408 at 4160 and 4001");

// The two residual geometries differ only in K and the model has no K term, so the pair of
// coverage checks above is one decision written twice. They are written out separately anyway: the
// day a schedule is registered per geometry is the day they stop agreeing.

// Guard for the TMA route.
//
// Seven conditions, in the order fp8_a8_tma_applies runs them, which is the order of what they
// protect: first what the hardware cannot describe, then what the calibration cannot speak
// for, then what the launch geometry cannot carry, and only last a question of speed.
//
// TMA descriptors carry a global address, and cuTensorMapEncodeTiled rejects one it cannot
// describe. The activation codes are a workspace this op allocated, so they are aligned by
// construction. The weight codes are a byte offset into the loaded artifact, so their alignment is
// a property of the layout and not something this patch establishes. Declining is the whole
// remedy - the route being replaced reads the same bytes with no such requirement - and it keeps a
// new failure mode out of a patch that is supposed to change only speed.
//
// This is a runtime address test and lives only here. fp8_a8_tma_admits and fp8_a8_tma_covers stay
// as they are: they are compile-time statements about widths, and an address is not a width.
inline constexpr std::uintptr_t kFp8A8TmaAddressAlignment = 16;

inline bool fp8_a8_tma_addresses_admit(const void* activation_codes, const void* weight_codes) {
    const auto a = reinterpret_cast<std::uintptr_t>(activation_codes);
    const auto b = reinterpret_cast<std::uintptr_t>(weight_codes);
    if (a == 0 || b == 0) { return false; }
    return (a % kFp8A8TmaAddressAlignment) == 0 && (b % kFp8A8TmaAddressAlignment) == 0;
}

// Representability is a property of the geometry. The branch below is documentation, not a
// fallback: the call sites test this predicate at runtime, so the kernel template is instantiated
// whatever it answers, and a geometry the kernel cannot tile fails its own static_assert first. A
// geometry is admitted to this route by being registered in Fp8LinearA8TmaSchedule above, and that
// list is declared-only for the same reason.
//
// There is no condition on the width being a whole number of cp.async token tiles, and there used
// to be: it bought bit-identity with the route being replaced, and it was removed because it
// blocked widths the cost model takes and the route measures faster at. It could not do otherwise.
// The model's only width-dependent inputs are the two tile counts, both constant between adjacent
// multiples of the smaller tile, so its verdict is already fixed across a band and a condition on
// the width within that band can only subtract from a decision already made. The sweep is in
// docs/maintainer/fp8-a8-tma-route.md.
//
// So it is a question of speed, and of nothing else.
template <class Geometry, class TmaSchedule,
          class MmaSchedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type>
bool fp8_a8_tma_applies(std::int32_t tokens, const void* activation_codes,
                        const void* weight_codes) {
    // The cost model below compares this route against the one that would otherwise run, so the
    // schedule it names has to be that one. Pinning it here means a future per-op schedule
    // override cannot silently make the comparison quote a tile nobody falls back to.
    static_assert(
        std::is_same_v<MmaSchedule, typename Fp8LinearA8ProductionSchedule<Geometry>::Type>,
        "the cost model must name the schedule this route actually falls back to");
    if constexpr (!kFp8A8TmaRepresentable<Geometry, TmaSchedule>) {
        return false;
    } else {
        // Not a kernel requirement - the copy zero-fills past the extent and the store drops
        // those rows, so one partial tile is fine. It is a statement about the calibration: the
        // ratio was solved at and above kBlockM-sized widths, and a floor below the tile would put
        // the model somewhere it was never fitted.
        static_assert(kFp8A8TmaMinTokens >= TmaSchedule::kBlockM,
                      "the floor must stay inside the range the ratio was solved in");
        // Addresses before widths: a pointer the descriptor cannot describe is a hard failure in
        // the launcher, where a width that does not suit is only a slower route.
        if (!fp8_a8_tma_addresses_admit(activation_codes, weight_codes)) { return false; }
        if (tokens < kFp8A8TmaMinTokens) { return false; }
        if (tokens > kFp8A8TmaMaxTokens<Geometry>) { return false; }
        // The launcher puts token tiles on grid.y, which tops out at 65535 where the cp.async route
        // linearises into grid.x and does not. No real width comes near - it would take 16.7 M
        // tokens in one chunk - but the limit is the new route's alone, and exceeding it is a
        // launch failure rather than a slow answer, so it is cheaper to decline than to explain
        // later.
        if (fp8_a8_tma_token_tiles<Geometry, TmaSchedule>(tokens) > 65535) { return false; }
        const std::int32_t multiprocessors = fp8_a8_multiprocessor_count();
        if (multiprocessors <= 0) { return false; }
        return fp8_a8_tma_cheaper<Geometry, TmaSchedule, MmaSchedule>(tokens, multiprocessors);
    }
}

} // namespace ninfer::ops::detail
