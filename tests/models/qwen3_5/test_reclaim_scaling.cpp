#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/transactions/reclaim.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {

using Clock  = std::chrono::steady_clock;
using Handle = ninfer::models::qwen3_5::CheckpointHandle;

// A lightweight checkpoint view for the pure complexity core of recovery_loss. It exposes only
// what the template needs (valid_checkpoint + checkpoint().frontier()), so the test runs on the
// CPU with synthetic checkpoints and no live ProgramImpl / CUDA device.
struct SynthState {
    std::uint32_t value = 0;
    std::uint32_t frontier() const { return value; }
};

struct SynthView {
    std::vector<SynthState> states;

    bool valid_checkpoint(Handle h) const { return h.index < states.size(); }
    const SynthState& checkpoint(Handle h) const { return states[h.index]; }
};

// Trivial prefix: always true. Isolates the loop complexity from the O(frontier) cost of the
// real checkpoint_prefix.
struct AlwaysPrefix {
    bool operator()(Handle, Handle) const { return true; }
};

// Worst case: N higher-frontier checkpoints reclaimed against N lower-frontier survivors
// (disjoint), so every survivor membership test fails and every prefix check runs.
double measure_call_ms(std::size_t n, int iters) {
    SynthView view;
    view.states.resize(2 * n);
    const auto* sentinel = &view.states;
    std::vector<Handle> pool(2 * n);
    for (std::size_t i = 0; i < pool.size(); ++i) {
        pool[i] = Handle{sentinel, static_cast<std::uint32_t>(i), 0};
        view.states[i].value = static_cast<std::uint32_t>(2 * n - i); // index 0 deepest
    }
    const std::span removed{pool.begin(), n};
    const std::span surviving{pool.begin() + n, n};
    AlwaysPrefix prefix;
    (void)ninfer::models::qwen3_5::detail::recovery_loss(view, removed, surviving, prefix);
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) {
        (void)ninfer::models::qwen3_5::detail::recovery_loss(view, removed, surviving, prefix);
    }
    const auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

std::size_t scale_max() {
    if (const char* env = std::getenv("NINFER_RECLAIM_SCALE_MAX")) {
        const auto v = std::stoul(env);
        return v ? v : 512;
    }
    return 512;
}

double exponent_ceiling() {
    if (const char* env = std::getenv("NINFER_RECLAIM_MAX_EXPONENT")) { return std::stod(env); }
    // Quadratic guard: the loop is O(n*m) after the hash-set membership fix. A cubic
    // (O(n^2*m)) regression measures ~3.0 and fails this; the fixed code measures ~2.0-2.3.
    return 2.5;
}

} // namespace

int main() {
    const std::size_t max_n = scale_max();
    std::cout << "recovery_loss scaling (CPU, synthetic checkpoints, trivial prefix)\n";
    std::cout << "  N = |removed| = |surviving| (disjoint worst case); ms per call\n";
    std::cout << "  ------------------------------------------------------------\n";

    struct Point {
        std::size_t n;
        double ms;
    };
    std::vector<Point> points;
    std::size_t n = 64;
    while (n <= max_n) {
        const int iters = n <= 128 ? 20 : (n <= 256 ? 8 : 3);
        const double ms = measure_call_ms(n, iters);
        points.push_back({n, ms});
        std::cout << "  N=" << n << "  " << ms << " ms/call\n";
        n *= 2;
    }

    // Empirical scaling exponent over the last doubling: t(2N)/t(N) = 2^e  ->  e = log2(ratio).
    double exponent = 0.0;
    if (points.size() >= 2) {
        const auto& a = points[points.size() - 2];
        const auto& b = points[points.size() - 1];
        if (a.ms > 0.0 && b.ms > 0.0) {
            exponent = std::log2(b.ms / a.ms) / std::log2(2.0);
        }
    }
    std::cout << "  ------------------------------------------------------------\n";
    std::cout << "  empirical exponent (last 2x step): " << exponent << "\n";
    std::cout << "  (linear=1, quadratic=2, cubic=3; guard ceiling=" << exponent_ceiling() << ")\n";

    if (!(exponent < exponent_ceiling())) {
        std::cerr << "FAIL: recovery_loss scaling exponent " << exponent
                  << " exceeds ceiling " << exponent_ceiling() << " (cubic regression?)\n";
        return 1;
    }
    std::cout << "OK: scaling within quadratic guard ceiling\n";
    return 0;
}
