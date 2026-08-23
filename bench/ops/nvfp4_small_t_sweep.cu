// TEMPORARY private-launcher sweep for the NVFP4 A16 small-T schedules on sm_70.
//
// docs/maintainer/op-development.md 7.1 asks for exactly this shape of tool: a private sweep over a
// small overlapping candidate set across the hot interval, qualified before timing, used to find
// the pointwise envelope and the crossovers, then deleted once production dispatch encodes the
// winners. It is not a public benchmark and is not wired into the suite.
//
// The live decision it exists to answer: the shipped schedules are commented "RTX 5090 cold-cache
// winners", and on the V100 they collapse past T=9 -- 150 GB/s at T=8 against 18 GB/s at T=16 on
// the MLP gate_up shape. The candidates below vary the three knobs that could plausibly cause
// that: CTA warp count (occupancy and register pressure), values per lane (unrolling), and
// accumulator chains (dependency-chain length, which matters on a machine whose SIMT kernels are
// issue-limited rather than bandwidth-limited).
//
// Usage: ninfer_nvfp4_small_t_sweep [attn|gdn|mlp|r6144|r17408]

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops::detail;

// One candidate schedule. WarpsPerRow and RowsPerWarp stay at the production values so
// kRowsPerCta == 2 * Warps, which keeps the kernel's 128 % kRowsPerCta assertion satisfied for
// every warp count sampled here.
template <int Tokens, int Warps, int Values, int Chains>
using Candidate =
    Nvfp4SmallTSchedule<Warps, 1, 2, Values, Tokens, Chains,
                        Nvfp4SmallTActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct,
                        Nvfp4CodeCache::Default, 1, Nvfp4SmallTBlockOrder::RowsContiguous, 1>;

struct Row {
    int tokens;
    const char* candidate;
    double median_us;
    double gbs;
    double max_abs_diff;
};

std::vector<Row> g_rows;

template <class Geometry, int Tokens, class Schedule>
void launch(const bench::PackedQuantizedWeight& packed, const void* activation, void* out,
            cudaStream_t stream) {
    constexpr int kTokenTiles = (Tokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out), Geometry::kOutputRows};
    nvfp4_small_t_kernel<Geometry, Tokens, Schedule, Nvfp4IdentityEpilogue, Nvfp4ContiguousOutput>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(activation),
            static_cast<const std::uint8_t*>(packed.weight.qdata),
            static_cast<const std::uint8_t*>(packed.weight.scales),
            1.0F / packed.weight.weight_scale_divisor, Nvfp4IdentityEpilogue{}, output);
}

// Qualification: the production schedule is the one the op tests already hold against the FP64
// oracle, so agreeing with it bit-for-bit is the property a candidate has to earn before its
// timing means anything. Reported rather than asserted, because a reordered accumulation can
// differ in the last bit without being wrong.
double max_abs_diff(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
    const auto to_float = [](std::uint16_t bits) {
        const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
        float value                 = 0.0F;
        std::memcpy(&value, &widened, sizeof(value));
        return value;
    };
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, static_cast<double>(std::abs(to_float(a[i]) - to_float(b[i]))));
    }
    return worst;
}

template <class Geometry, int Tokens, int Warps, int Values, int Chains>
void measure(const bench::PackedQuantizedWeight& packed, const DeviceBuffer& activation,
             DeviceBuffer& out, const std::vector<std::uint16_t>& reference, const char* name) {
    using Schedule = Candidate<Tokens, Warps, Values, Chains>;

    out.fill(0);
    launch<Geometry, Tokens, Schedule>(packed, activation.p, out.p, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("  T=%-3d %-16s LAUNCH FAILED (%s)\n", Tokens, name,
                    cudaGetErrorString(cudaGetLastError()));
        return;
    }
    std::vector<std::uint16_t> host(reference.size());
    out.copy_to_host(host.data(), host.size() * sizeof(std::uint16_t));

    const double bytes = static_cast<double>(Geometry::kOutputRows) *
                             static_cast<double>(Geometry::kInputRows) * 0.5 +
                         static_cast<double>(Geometry::kOutputRows) *
                             static_cast<double>(Geometry::kInputRows) / 16.0;
    const bench::Result result = bench::bench_loop(
        [&](cudaStream_t stream) { launch<Geometry, Tokens, Schedule>(packed, activation.p, out.p, stream); },
        bytes, 10, 30, 200);

    g_rows.push_back({Tokens, name, result.median_us, result.gbs, max_abs_diff(host, reference)});
    std::printf("  T=%-3d %-16s %8.1f us  %7.1f GB/s  maxdiff=%.3g\n", Tokens, name,
                result.median_us, result.gbs, g_rows.back().max_abs_diff);
}

template <class Geometry, int Tokens>
void sweep_tokens(const bench::PackedQuantizedWeight& packed, const DeviceBuffer& activation,
                  DeviceBuffer& out) {
    // Reference: the schedule production dispatch would pick for this T.
    using Production = typename Nvfp4LinearSmallTProductionSchedule<Geometry, Tokens>::Type;
    out.fill(0);
    launch<Geometry, Tokens, Production>(packed, activation.p, out.p, nullptr);
    cudaDeviceSynchronize();
    std::vector<std::uint16_t> reference(static_cast<std::size_t>(Geometry::kOutputRows) * Tokens);
    out.copy_to_host(reference.data(), reference.size() * sizeof(std::uint16_t));

    std::printf("T=%d (production: %d warps, %d values/lane)\n", Tokens, Production::kWarpsPerCta,
                Production::kValuesPerLane);
    // Round 1 established two things and killed one knob: accumulator chains never win (the
    // kernel is not dependency-bound the way the tensor-core routes are), and the whole envelope
    // is set by per-thread register footprint -- 16 warps costs 3.5x at T=16 and 6.8x at T=24,
    // while at T=32 even two warps collapse unless values-per-lane drops to 8. Round 2 therefore
    // sweeps the warp axis against the values axis, which round 1 only sampled at w8.
    measure<Geometry, Tokens, 2, 16, 1>(packed, activation, out, reference, "w2 v16");
    measure<Geometry, Tokens, 4, 16, 1>(packed, activation, out, reference, "w4 v16");
    measure<Geometry, Tokens, 8, 16, 1>(packed, activation, out, reference, "w8 v16");
    measure<Geometry, Tokens, 16, 16, 1>(packed, activation, out, reference, "w16 v16");
    measure<Geometry, Tokens, 2, 8, 1>(packed, activation, out, reference, "w2 v8");
    measure<Geometry, Tokens, 4, 8, 1>(packed, activation, out, reference, "w4 v8");
    measure<Geometry, Tokens, 8, 8, 1>(packed, activation, out, reference, "w8 v8");
    measure<Geometry, Tokens, 16, 8, 1>(packed, activation, out, reference, "w16 v8");
}

// ---------------------------------------------------------------------------
// Decode (T=1). Same exercise for the GEMV family, which the small-T retune did not touch and
// which is what plain decode actually runs: 129.3 GB/s against the Q4 route's 467.8 on the same
// shape.
template <int Warps, int Values, int Chains, Nvfp4ScaleAccess Access>
using GemvCandidate = Nvfp4GemvSchedule<Warps, 2, Values, Chains, Access, Nvfp4CodeCache::Default, 2>;

template <class Geometry, class Schedule>
void launch_gemv(const bench::PackedQuantizedWeight& packed, const void* activation, void* out,
                 cudaStream_t stream) {
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out), Geometry::kOutputRows};
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation),
        static_cast<const std::uint8_t*>(packed.weight.qdata),
        static_cast<const std::uint8_t*>(packed.weight.scales),
        1.0F / packed.weight.weight_scale_divisor, Nvfp4IdentityEpilogue{}, output);
}

template <class Geometry, int Warps, int Values, int Chains, Nvfp4ScaleAccess Access>
void measure_gemv(const bench::PackedQuantizedWeight& packed, const DeviceBuffer& activation,
                  DeviceBuffer& out, const std::vector<std::uint16_t>& reference,
                  const char* name) {
    using Schedule = GemvCandidate<Warps, Values, Chains, Access>;
    out.fill(0);
    launch_gemv<Geometry, Schedule>(packed, activation.p, out.p, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("  %-20s LAUNCH FAILED (%s)\n", name, cudaGetErrorString(cudaGetLastError()));
        return;
    }
    std::vector<std::uint16_t> host(reference.size());
    out.copy_to_host(host.data(), host.size() * sizeof(std::uint16_t));

    const double bytes = static_cast<double>(Geometry::kOutputRows) *
                         static_cast<double>(Geometry::kInputRows) * (0.5 + 1.0 / 16.0);
    const bench::Result result = bench::bench_loop(
        [&](cudaStream_t stream) { launch_gemv<Geometry, Schedule>(packed, activation.p, out.p, stream); },
        bytes, 10, 30, 200);
    std::printf("  %-20s %8.1f us  %7.1f GB/s  maxdiff=%.3g\n", name, result.median_us,
                result.gbs, max_abs_diff(host, reference));
}

template <class Geometry>
void sweep_gemv(const char* label) {
    std::printf("=== %s GEMV (T=1)  N=%d K=%d\n", label, Geometry::kOutputRows,
                Geometry::kInputRows);
    bench::PackedQuantizedWeight packed =
        bench::make_nvfp4_weight(Geometry::kOutputRows, Geometry::kInputRows);
    DeviceBuffer activation = bench::make_bf16(static_cast<std::size_t>(Geometry::kInputRows));
    DeviceBuffer out(static_cast<std::size_t>(Geometry::kOutputRows) * sizeof(std::uint16_t));

    using Production = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    out.fill(0);
    launch_gemv<Geometry, Production>(packed, activation.p, out.p, nullptr);
    cudaDeviceSynchronize();
    std::vector<std::uint16_t> reference(static_cast<std::size_t>(Geometry::kOutputRows));
    out.copy_to_host(reference.data(), reference.size() * sizeof(std::uint16_t));

    measure_gemv<Geometry, 8, 16, 4, Nvfp4ScaleAccess::StagedRaw>(packed, activation, out, reference, "w8  v16 c4 staged");
    measure_gemv<Geometry, 8, 8, 4, Nvfp4ScaleAccess::StagedRaw>(packed, activation, out, reference, "w8  v8  c4 staged");
    measure_gemv<Geometry, 4, 8, 4, Nvfp4ScaleAccess::StagedRaw>(packed, activation, out, reference, "w4  v8  c4 staged");
    measure_gemv<Geometry, 16, 8, 4, Nvfp4ScaleAccess::StagedRaw>(packed, activation, out, reference, "w16 v8  c4 staged");
    measure_gemv<Geometry, 8, 32, 4, Nvfp4ScaleAccess::StagedRaw>(packed, activation, out, reference, "w8  v32 c4 staged");
    measure_gemv<Geometry, 8, 16, 8, Nvfp4ScaleAccess::StagedRaw>(packed, activation, out, reference, "w8  v16 c8 staged");
    measure_gemv<Geometry, 8, 16, 4, Nvfp4ScaleAccess::Direct>(packed, activation, out, reference, "w8  v16 c4 direct");
    measure_gemv<Geometry, 8, 8, 4, Nvfp4ScaleAccess::Direct>(packed, activation, out, reference, "w8  v8  c4 direct");
}

template <class Geometry>
void sweep(const char* label) {
    std::printf("=== %s  N=%d K=%d\n", label, Geometry::kOutputRows, Geometry::kInputRows);
    bench::PackedQuantizedWeight packed =
        bench::make_nvfp4_weight(Geometry::kOutputRows, Geometry::kInputRows);
    DeviceBuffer activation = bench::make_bf16(static_cast<std::size_t>(Geometry::kInputRows) * 32);
    DeviceBuffer out(static_cast<std::size_t>(Geometry::kOutputRows) * 32 * sizeof(std::uint16_t));

    // The hot interval is the MTP verify band: draft-3 rounds give T = 4 per sequence, so a C8
    // batch reaches 32, and the measured collapse sits between 9 and 16. Sampled finely there and
    // sparsely outside.
    sweep_tokens<Geometry, 2>(packed, activation, out);
    sweep_tokens<Geometry, 4>(packed, activation, out);
    sweep_tokens<Geometry, 6>(packed, activation, out);
    sweep_tokens<Geometry, 8>(packed, activation, out);
    sweep_tokens<Geometry, 9>(packed, activation, out);
    sweep_tokens<Geometry, 10>(packed, activation, out);
    sweep_tokens<Geometry, 12>(packed, activation, out);
    sweep_tokens<Geometry, 13>(packed, activation, out);
    sweep_tokens<Geometry, 16>(packed, activation, out);
    sweep_tokens<Geometry, 17>(packed, activation, out);
    sweep_tokens<Geometry, 20>(packed, activation, out);
    sweep_tokens<Geometry, 24>(packed, activation, out);
    sweep_tokens<Geometry, 32>(packed, activation, out);
}

void print_envelope() {
    std::printf("\n=== per-T winner\n");
    int current = -1;
    const Row* best = nullptr;
    const auto flush_row = [&]() {
        if (best != nullptr) {
            std::printf("T=%-3d %-16s %8.1f us  %7.1f GB/s\n", best->tokens, best->candidate,
                        best->median_us, best->gbs);
        }
    };
    for (const Row& row : g_rows) {
        if (row.tokens != current) {
            flush_row();
            current = row.tokens;
            best    = &row;
        } else if (row.median_us < best->median_us) {
            best = &row;
        }
    }
    flush_row();
}

} // namespace

int main(int argc, char** argv) {
    const std::string shape = argc > 1 ? argv[1] : "mlp";
    if (shape == "mlp") {
        sweep<Nvfp4MlpGateUpGeometry>("mlp gate_up");
    } else if (shape == "attn") {
        sweep<Nvfp4AttnInputGeometry>("attn input");
    } else if (shape == "gdn") {
        sweep<Nvfp4GdnInputGeometry>("gdn input");
    } else if (shape == "r6144") {
        sweep<Nvfp4Residual6144Geometry>("residual 6144");
    } else if (shape == "r17408") {
        sweep<Nvfp4Residual17408Geometry>("residual 17408");
    } else if (shape == "gemv") {
        sweep_gemv<Nvfp4MlpGateUpGeometry>("mlp gate_up");
        sweep_gemv<Nvfp4AttnInputGeometry>("attn input");
        sweep_gemv<Nvfp4Residual17408Geometry>("residual 17408");
        return 0;
    } else {
        std::printf("usage: %s [mlp|attn|gdn|r6144|r17408|gemv]\n", argv[0]);
        return 2;
    }
    print_envelope();
    return 0;
}
