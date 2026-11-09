// moe_grouped_nvfp4 (prefill-width routed experts over a slot pool, NVFP4 tensor cores) against
// FP64 oracles at the Qwen3.8-Flash-Next geometry (hidden 2560, expert intermediate 640, top-10).
//
// Oracles (FP64 from the represented inputs; the helpers are test_moe_slots.cpp's):
//   * mathematical: unquantized activations (W4A4 vs FP64, the moe_slots_nvfp4 bound 0.18);
//   * recipe: the documented W4A4 activation profile (5e-3 relative L2, as moe_slots).
// Cross-check: the same calls through moe_slots_nvfp4 in 8-column slices (the decode route the
// prefill route replaces): same recipe, so they differ only by FP32 accumulation order and rare
// h-quantization ties (kSameRecipe, 1e-3).
// Exact checks: the output does not depend on the slot that holds an expert (garbage in every
// other slot); routes without a slot equal zero-weight routes; a call wider than one piece equals
// its pieces called separately; a captured CUDA graph replays the eager result after the table,
// x and routing change.
#include "ninfer/ops/moe_grouped.h"
#include "ninfer/ops/moe_slots.h"

#include "core/arena.h"
#include "ops/op_tester.h"
#include "ops/metadata_timing.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kH = 2560, kI = 640, kK = 10, kE = 512, kRows13 = 2 * kI;
constexpr int kHB = kH / 16, kIB = kI / 16;
constexpr std::int64_t kSlotBytes = ops::kMoeSlotsGateUpCodeBytes + ops::kMoeSlotsGateUpScaleBytes +
                                    ops::kMoeSlotsDownCodeBytes +
                                    ops::kMoeSlotsDownScaleBytes; // 2 764 800

// W4A4 vs the unquantized FP64 oracle. Two NVFP4 activation quantizations in series (x before
// gemm1, h before gemm2); the single-quantization A4 allowance of the Linear suites is 0.16 and
// F0-A measured 0.157-0.164 for this recipe (author's kernel, same data model). 0.18 covers the
// two stages with ~10 % margin over F0-A. Gross cap: the largest error of a 2560*T Gaussian-like
// error field relative to the largest reference value is ~ the relative L2 (both extremes
// ~4 sigma), so 0.30 is ~2x.
constexpr ReductionCriterion kW4A4{0.18, 0.0, 0.30};
// Recipe oracle: same quantization formula evaluated in FP64/FP32 on the host. Differences are
// FP32 accumulation order, rcp.approx/__expf vs exact, which flip a BF16 rounding of a gemm1
// output or h with probability ~1e-4 and an E2M1 code only rarely; one flipped h code moves one
// token's output by < 1e-2 relative. 5e-3 relative L2 and 5e-2 gross are well above that and
// ~30x below the W4A4 error, so any scale, divisor or slot fault fails.
constexpr ReductionCriterion kRecipe{5.0e-3, 0.0, 5.0e-2};
// Two kernels of the same recipe: FP32 order and an occasional h code tie (one code moves one
// token by < 1e-2 relative, ~1e-4 of the routes): 1e-3 relative L2, 1e-2 gross.
constexpr ReductionCriterion kSameRecipe{1.0e-3, 0.0, 1.0e-2};
constexpr float kDivisorJitter = 0.98742F; // see reference()

const float kE2M1[16] = {0.F,  0.5F,  1.F,  1.5F,  2.F,  3.F,  4.F,  6.F,
                         -0.F, -0.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};

float e4m3_decode(std::uint8_t c) {
    const int sign = c >> 7, e = (c >> 3) & 15, m = c & 7;
    float v = 0.F;
    if (e == 0)
        v = static_cast<float>(m) * std::ldexp(1.0F, -9);
    else
        v = (1.0F + static_cast<float>(m) / 8.0F) * std::ldexp(1.0F, e - 7);
    return sign ? -v : v;
}

// E4M3FN round-to-nearest-even, satfinite, of a non-negative value (the activation scale).
float e4m3_rn(float v) {
    if (!(v > 0.0F)) return 0.0F;
    if (v >= 448.0F) return 448.0F;
    int ex = 0;
    std::frexp(v, &ex);                        // v = f * 2^ex, f in [0.5,1)
    const int exponent = std::max(ex - 1, -6); // subnormal step below 2^-6
    const float step   = std::ldexp(1.0F, exponent - 3);
    return std::min(std::nearbyint(v / step) * step, 448.0F);
}

// E2M1 round-to-nearest-even (ties to the even code), satfinite.
float e2m1_rn(float v) {
    static const float grid[8] = {0.F, 0.5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
    const float a              = std::fabs(v);
    float r                    = 6.0F;
    for (int i = 0; i < 7; ++i) {
        if (a > grid[i + 1]) continue;
        const float mid = 0.5F * (grid[i] + grid[i + 1]);
        r = a < mid ? grid[i] : a > mid ? grid[i + 1] : (i % 2 == 0 ? grid[i] : grid[i + 1]);
        break;
    }
    return std::copysign(r, v);
}

// The documented activation profile for 16 values: codes (as values) and the E4M3 scale.
void quant16(const float* v, float gs, float* codes, float& sf) {
    float amax = 0.F;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(v[i]));
    sf                    = e4m3_rn(gs * (amax * (1.0F / 6.0F)));
    const float out_scale = (amax != 0.F && sf != 0.F) ? 1.0F / (sf * (1.0F / gs)) : 0.F;
    for (int i = 0; i < 16; ++i) codes[i] = e2m1_rn(v[i] * out_scale);
}

float bf16_round(float v) { return bf16_to_f32(f32_to_bf16(v)); }

std::size_t swizzled_offset(int row, int group, int k_tiles) {
    const int inner = row % 128;
    return (static_cast<std::size_t>(row / 128) * k_tiles + group / 4) * 512 +
           static_cast<std::size_t>(inner % 32) * 16 + static_cast<std::size_t>(inner / 32) * 4 +
           group % 4;
}

void parallel_for(int n, const std::function<void(int)>& fn) {
    const int cores   = static_cast<int>(std::min(12U, std::thread::hardware_concurrency()));
    const int workers = std::max(1, std::min(n, cores));
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> errors(workers);
    for (int w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
            try {
                for (int i = w; i < n; i += workers) fn(i);
            } catch (...) { errors[w] = std::current_exception(); }
        });
    }
    for (auto& t : pool) t.join();
    for (auto& e : errors)
        if (e) std::rethrow_exception(e);
}

// One routed expert in the bank's row order: gate|up [1280,2560] (gate rows first), down
// [2560,640]; natural (row-major) scale words.
struct Expert {
    std::vector<std::uint8_t> w13, s13, w2, s2;
};

Expert make_expert(std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255), scale(0x30, 0x3F), zero(0, 63);
    Expert x;
    x.w13.resize(static_cast<std::size_t>(kRows13) * kH / 2);
    x.s13.resize(static_cast<std::size_t>(kRows13) * kHB);
    x.w2.resize(static_cast<std::size_t>(kH) * kI / 2);
    x.s2.resize(static_cast<std::size_t>(kH) * kIB);
    for (auto& b : x.w13) b = static_cast<std::uint8_t>(byte(rng));
    for (auto& b : x.w2) b = static_cast<std::uint8_t>(byte(rng));
    // valid stored scales: positive finite words, a few zero blocks
    for (auto& b : x.s13) b = static_cast<std::uint8_t>(zero(rng) == 0 ? 0 : scale(rng));
    for (auto& b : x.s2) b = static_cast<std::uint8_t>(zero(rng) == 0 ? 0 : scale(rng));
    return x;
}

// code * block scale, without the divisor (exact in FP32)
void decode(const std::vector<std::uint8_t>& codes, const std::vector<std::uint8_t>& scales,
            int rows, int cols, std::vector<float>& out) {
    out.resize(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const std::uint8_t byte = codes[(static_cast<std::size_t>(r) * cols + c) / 2];
            const int code          = (c % 2 == 0) ? (byte & 15) : (byte >> 4);
            out[static_cast<std::size_t>(r) * cols + c] =
                kE2M1[code] *
                e4m3_decode(scales[static_cast<std::size_t>(r) * (cols / 16) + c / 16]);
        }
}

struct Model {
    std::vector<int> ids;      // materialized expert ids
    std::vector<int> index_of; // [512] -> index in experts or -1
    std::vector<Expert> experts;
    std::vector<float> div13, div2; // [2E], [E]
};

Model make_model(int count, std::uint32_t seed) {
    Model m;
    std::mt19937 rng(seed);
    std::vector<int> all(kE);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    m.ids.assign(all.begin(), all.begin() + count);
    m.index_of.assign(kE, -1);
    m.experts.resize(count);
    for (int i = 0; i < count; ++i) m.index_of[m.ids[i]] = i;
    parallel_for(count, [&](int i) { m.experts[i] = make_expert(seed * 7919U + 17U * i + 1U); });
    // distinct gate/up/down divisors per expert (ModelOpt: 1/weight_scale_2), 250..1000
    std::uniform_real_distribution<float> div(250.0F, 1000.0F);
    m.div13.resize(2 * kE);
    m.div2.resize(kE);
    for (auto& d : m.div13) d = div(rng);
    for (auto& d : m.div2) d = div(rng);
    return m;
}

enum class Layout { Planar, Contiguous };

// A device slot pool; placement[i] = slot of materialized expert i, or -1. Unused slots hold
// random bytes (including NaN scale words) that must never be read.
struct DevicePool {
    DeviceBuffer memory;
    ops::MoeSlotPool pool;
};

DevicePool make_pool(const Model& m, const std::vector<std::vector<int>>& placement, int slots,
                     Layout layout, std::uint32_t seed) {
    const std::int64_t plane_bytes[4] = {ops::kMoeSlotsGateUpCodeBytes,
                                         ops::kMoeSlotsGateUpScaleBytes,
                                         ops::kMoeSlotsDownCodeBytes, ops::kMoeSlotsDownScaleBytes};
    std::int64_t base[4], stride[4];
    if (layout == Layout::Planar) {
        std::int64_t at = 0;
        for (int q = 0; q < 4; ++q) {
            base[q]   = at;
            stride[q] = plane_bytes[q];
            at += plane_bytes[q] * slots;
        }
    } else {
        std::int64_t at = 0;
        for (int q = 0; q < 4; ++q) {
            base[q]   = at;
            stride[q] = kSlotBytes;
            at += plane_bytes[q];
        }
    }
    const std::size_t total = static_cast<std::size_t>(kSlotBytes) * slots;
    std::vector<std::uint8_t> host(total);
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < total; i += 4) {
        const std::uint32_t r = rng();
        std::memcpy(host.data() + i, &r, std::min<std::size_t>(4, total - i));
    }
    for (std::size_t i = 0; i < placement.size(); ++i) {
        const Expert& x = m.experts[i];
        for (const int s : placement[i]) {
            std::uint8_t* p[4];
            for (int q = 0; q < 4; ++q) p[q] = host.data() + base[q] + stride[q] * s;
            std::memcpy(p[0], x.w13.data(), x.w13.size());
            std::memcpy(p[2], x.w2.data(), x.w2.size());
            for (int r = 0; r < kRows13; ++r)
                for (int g = 0; g < kHB; ++g)
                    p[1][swizzled_offset(r, g, kHB / 4)] =
                        x.s13[static_cast<std::size_t>(r) * kHB + g];
            for (int r = 0; r < kH; ++r)
                for (int g = 0; g < kIB; ++g)
                    p[3][swizzled_offset(r, g, kIB / 4)] =
                        x.s2[static_cast<std::size_t>(r) * kIB + g];
        }
    }
    DevicePool d;
    d.memory = DeviceBuffer(total);
    d.memory.copy_from_host(host.data(), total);
    auto* dev             = static_cast<std::uint8_t*>(d.memory.p);
    d.pool.gate_up_codes  = {dev + base[0], stride[0]};
    d.pool.gate_up_scales = {dev + base[1], stride[1]};
    d.pool.down_codes     = {dev + base[2], stride[2]};
    d.pool.down_scales    = {dev + base[3], stride[3]};
    d.pool.slots          = slots;
    return d;
}

std::vector<int> table_from(const Model& m, const std::vector<std::vector<int>>& placement,
                            int which = 0) {
    std::vector<int> table(kE, -1);
    for (std::size_t i = 0; i < placement.size(); ++i)
        if (static_cast<int>(placement[i].size()) > which) table[m.ids[i]] = placement[i][which];
    return table;
}

struct Call {
    int tokens = 0;
    std::vector<float> x;   // [T][H], BF16-representable
    std::vector<int> ids;   // [T][K]
    std::vector<float> w;   // [T][K]
    std::vector<int> table; // [E]
    float a1 = 0.F, a2 = 0.F;
};

// Each row: 10 distinct ids from the materialized set; weights = softmax of random logits.
Call make_call(const Model& m, int tokens, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Call c;
    c.tokens = tokens;
    c.x.resize(static_cast<std::size_t>(tokens) * kH);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : c.x) v = bf16_round(normal(rng));
    // an all-zero block and a tiny block (scale underflow) in token 0
    for (int i = 0; i < 16; ++i) c.x[32 + i] = 0.0F;
    for (int i = 0; i < 16; ++i) c.x[64 + i] = bf16_round(1.0e-6F * normal(rng));
    c.ids.resize(static_cast<std::size_t>(tokens) * kK);
    c.w.resize(c.ids.size());
    std::vector<int> pick = m.ids;
    for (int t = 0; t < tokens; ++t) {
        std::shuffle(pick.begin(), pick.end(), rng);
        float sum = 0.F;
        for (int k = 0; k < kK; ++k) {
            c.ids[t * kK + k] = pick[k];
            c.w[t * kK + k]   = std::exp(normal(rng));
            sum += c.w[t * kK + k];
        }
        for (int k = 0; k < kK; ++k) c.w[t * kK + k] /= sum;
    }
    return c;
}

bool resident(const Call& c, int id) { return id >= 0 && id < kE && c.table[id] >= 0; }

struct Reference {
    std::vector<double> exact, recipe; // [T][H]
    double hmax = 0.0;
};

// Calibrates the call's input divisors (x: 2688 / max|x|, h: 2688 / max|h| over the resident
// (expert, token) pairs, 2688 = 6 * 448, the F0-A data model) unless `calibrate` is false, then
// evaluates both oracles. The x divisor carries a non-dyadic factor kDivisorJitter: 2688 / max|x|
// with a BF16 max|x| makes v * divisor / scale exactly an E2M1 rounding midpoint for many BF16 v
// (e.g. 1.6015625 * 2688 / (3.84375 * 320) = 3.5), and the kernel resolves those exact ties
// through rcp.approx (the FlashInfer fast-math recipe) where the host uses exact division, so a
// tie can round to the other code. The factor moves every product off the midpoints; tie
// breaking itself is a property of the recipe that this oracle does not reproduce.
Reference reference(const Model& m, Call& c, bool calibrate) {
    const int T = c.tokens;
    // per resident expert: tokens and summed weights
    std::vector<int> experts;
    for (int i = 0; i < T * kK; ++i) {
        const int id = c.ids[i];
        if (resident(c, id) && m.index_of[id] >= 0 &&
            std::find(experts.begin(), experts.end(), id) == experts.end())
            experts.push_back(id);
    }
    if (calibrate) {
        float xmax = 0.F;
        for (const float v : c.x) xmax = std::max(xmax, std::fabs(v));
        c.a1 = 2688.0F / xmax * kDivisorJitter;
    }
    // x quantized once per token (the input divisor is per layer)
    std::vector<float> xc(c.x.size()), xsf(static_cast<std::size_t>(T) * kHB);
    for (int t = 0; t < T; ++t)
        for (int b = 0; b < kHB; ++b)
            quant16(&c.x[static_cast<std::size_t>(t) * kH + b * 16], c.a1,
                    &xc[static_cast<std::size_t>(t) * kH + b * 16], xsf[t * kHB + b]);

    const int n = static_cast<int>(experts.size());
    std::vector<std::vector<double>> h_exact(n), act(n); // [T][I] per expert
    std::vector<std::vector<float>> d2(n);
    parallel_for(n, [&](int j) {
        const int id     = experts[j];
        const Expert& ex = m.experts[m.index_of[id]];
        std::vector<float> d13;
        decode(ex.w13, ex.s13, kRows13, kH, d13);
        decode(ex.w2, ex.s2, kH, kI, d2[j]);
        h_exact[j].assign(static_cast<std::size_t>(T) * kI, 0.0);
        act[j].assign(static_cast<std::size_t>(T) * kI, 0.0);
        const double dg = m.div13[2 * id], du = m.div13[2 * id + 1];
        for (int t = 0; t < T; ++t) {
            bool used = false;
            for (int k = 0; k < kK; ++k) used |= c.ids[t * kK + k] == id;
            if (!used) continue;
            const float* xt  = &c.x[static_cast<std::size_t>(t) * kH];
            const float* xqt = &xc[static_cast<std::size_t>(t) * kH];
            for (int r = 0; r < kI; ++r) {
                double g = 0, u = 0, gq = 0, uq = 0;
                const float* wg = &d13[static_cast<std::size_t>(r) * kH];
                const float* wu = &d13[static_cast<std::size_t>(kI + r) * kH];
                for (int b = 0; b < kHB; ++b) {
                    double gb = 0, ub = 0;
                    for (int i = b * 16; i < b * 16 + 16; ++i) {
                        g += double(wg[i]) * xt[i];
                        u += double(wu[i]) * xt[i];
                        gb += double(wg[i]) * xqt[i];
                        ub += double(wu[i]) * xqt[i];
                    }
                    gq += gb * xsf[t * kHB + b];
                    uq += ub * xsf[t * kHB + b];
                }
                g /= dg;
                u /= du;
                h_exact[j][static_cast<std::size_t>(t) * kI + r] = g / (1.0 + std::exp(-g)) * u;
                const float gr = bf16_round(static_cast<float>(gq / (dg * c.a1)));
                const float ur = bf16_round(static_cast<float>(uq / (du * c.a1)));
                act[j][static_cast<std::size_t>(t) * kI + r] =
                    bf16_round(gr * (1.0F / (1.0F + std::exp(-gr))) * ur);
            }
        }
    });
    Reference ref;
    for (int j = 0; j < n; ++j)
        for (const double v : h_exact[j]) ref.hmax = std::max(ref.hmax, std::fabs(v));
    if (calibrate) c.a2 = static_cast<float>(2688.0 / ref.hmax);

    ref.exact.assign(static_cast<std::size_t>(T) * kH, 0.0);
    ref.recipe.assign(static_cast<std::size_t>(T) * kH, 0.0);
    std::vector<std::vector<double>> oe(n), orc(n); // per expert [T][H], weighted
    parallel_for(n, [&](int j) {
        const int id    = experts[j];
        const double dd = m.div2[id];
        oe[j].assign(static_cast<std::size_t>(T) * kH, 0.0);
        orc[j].assign(static_cast<std::size_t>(T) * kH, 0.0);
        for (int t = 0; t < T; ++t) {
            double w = 0.0;
            for (int k = 0; k < kK; ++k)
                if (c.ids[t * kK + k] == id) w += c.w[t * kK + k];
            if (w == 0.0) continue;
            std::vector<float> hc(kI), hsf(kIB);
            std::vector<float> a(kI);
            for (int r = 0; r < kI; ++r)
                a[r] = static_cast<float>(act[j][static_cast<std::size_t>(t) * kI + r]);
            for (int b = 0; b < kIB; ++b) quant16(&a[b * 16], c.a2, &hc[b * 16], hsf[b]);
            const double* he = &h_exact[j][static_cast<std::size_t>(t) * kI];
            for (int r = 0; r < kH; ++r) {
                const float* wd = &d2[j][static_cast<std::size_t>(r) * kI];
                double e = 0, q = 0;
                for (int b = 0; b < kIB; ++b) {
                    double qb = 0;
                    for (int i = b * 16; i < b * 16 + 16; ++i) {
                        e += double(wd[i]) * he[i];
                        qb += double(wd[i]) * hc[i];
                    }
                    q += qb * hsf[b];
                }
                oe[j][static_cast<std::size_t>(t) * kH + r]  = w * e / dd;
                orc[j][static_cast<std::size_t>(t) * kH + r] = w * q / (dd * c.a2);
            }
        }
    });
    for (int j = 0; j < n; ++j)
        for (std::size_t i = 0; i < ref.exact.size(); ++i) {
            ref.exact[i] += oe[j][i];
            ref.recipe[i] += orc[j][i];
        }
    return ref;
}

double relative_l2(const std::vector<double>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Every expert in `copies` random slots of a planar pool of `slots` slots.
std::vector<std::vector<int>> random_placement(const Model& m, int slots, std::uint32_t seed,
                                               int copies = 1) {
    std::vector<int> s(slots);
    std::iota(s.begin(), s.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(s.begin(), s.end(), rng);
    std::vector<std::vector<int>> p(m.ids.size());
    for (std::size_t i = 0; i < p.size(); ++i)
        for (int c = 0; c < copies; ++c) p[i].push_back(s[i * copies + c]);
    return p;
}

std::vector<double> as_double(const std::vector<float>& v) { return {v.begin(), v.end()}; }

class Runner {
public:
    Runner(const Model& m, int max_tokens)
        : workspace_(ops::moe_grouped_nvfp4_workspace_bytes(max_tokens) +
                     ops::moe_slots_nvfp4_workspace_bytes(ops::kMoeSlotsMaxTokens) + 1024),
          div13_(to_device(m.div13)), div2_(to_device(m.div2)), table_(kE * sizeof(int)) {
        cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
    }

    ~Runner() { cudaStreamDestroy(stream_); }

    cudaStream_t stream() const { return stream_; }

    WorkspaceArena& workspace() { return workspace_; }

    ops::MoeSlotsLayer layer(const Call& c, void* table_device) const {
        ops::MoeSlotsLayer l;
        l.slot_table            = Tensor(table_device, DType::I32, {kE});
        l.gate_up_divisors      = Tensor(div13_.p, DType::FP32, {2 * kE});
        l.down_divisors         = Tensor(div2_.p, DType::FP32, {kE});
        l.gate_up_input_divisor = c.a1;
        l.down_input_divisor    = c.a2;
        return l;
    }

    // grouped = true: moe_grouped_nvfp4 over the whole call; false: moe_slots_nvfp4 in slices.
    std::vector<float> run(const Call& c, const ops::MoeSlotPool& pool, bool grouped, int& failures,
                           const std::string& label, bool benchmark = false) {
        const int T = c.tokens;
        table_.copy_from_host(c.table.data(), c.table.size() * sizeof(int));
        std::vector<std::uint16_t> xb(c.x.size());
        for (std::size_t i = 0; i < xb.size(); ++i) xb[i] = f32_to_bf16(c.x[i]);
        DeviceBuffer dx = to_device(xb), di = to_device(c.ids), dw = to_device(c.w);
        const std::size_t count = static_cast<std::size_t>(T) * kH;
        GuardedDeviceBuffer out(count * 4);
        out.fill(0xFF); // NaN: every element must be written
        cuda_synchronize();
        const auto l = layer(c, table_.p);
        if (grouped) {
            Tensor tx(dx.p, DType::BF16, {kH, T}), ti(di.p, DType::I32, {kK, T}),
                tw(dw.p, DType::FP32, {kK, T}), to(out.data(), DType::FP32, {kH, T});
            auto launch = [&] {
                ops::moe_grouped_nvfp4(tx, ti, tw, l, pool, to, workspace_, stream_);
            };
            launch();
            if (benchmark) {
                cuda_synchronize(stream_);
                time_captured(launch, stream_, true, label.c_str(), T, 1, 1);
            }
        } else {
            for (int b = 0; b < T; b += ops::kMoeSlotsMaxTokens) {
                const int n = std::min(ops::kMoeSlotsMaxTokens, T - b);
                Tensor tx(static_cast<std::uint16_t*>(dx.p) + static_cast<std::size_t>(b) * kH,
                          DType::BF16, {kH, n}),
                    ti(static_cast<int*>(di.p) + b * kK, DType::I32, {kK, n}),
                    tw(static_cast<float*>(dw.p) + b * kK, DType::FP32, {kK, n}),
                    to(static_cast<float*>(out.data()) + static_cast<std::size_t>(b) * kH,
                       DType::FP32, {kH, n});
                ops::moe_slots_nvfp4(tx, ti, tw, l, pool, ops::MoeSlotsOutput::Store, to,
                                     workspace_, stream_);
            }
        }
        cuda_check_last_launch(grouped ? "moe_grouped_nvfp4" : "moe_slots_nvfp4");
        cuda_synchronize(stream_);
        failures += out.verify_guards(label);
        std::vector<float> v(count);
        out.copy_to_host(v.data(), count * 4);
        return v;
    }

private:
    cudaStream_t stream_ = nullptr;
    WorkspaceArena workspace_;
    DeviceBuffer div13_, div2_, table_;
};

int check_oracles(const std::string& label, const std::vector<float>& got_f, const Reference& ref) {
    const auto got = as_double(got_f);
    std::cout << label << ": rel_l2 vs FP64 (unquantized activations) "
              << relative_l2(got, ref.exact) << ", vs recipe " << relative_l2(got, ref.recipe)
              << '\n';
    return verify_reduction(label + " W4A4 vs FP64", got, ref.exact, kW4A4) +
           verify_reduction(label + " recipe", got, ref.recipe, kRecipe);
}

// (1) Oracle at prefill widths (partial tiles, several tiles per expert), every expert in a
// random slot of a pool three times larger (garbage elsewhere), and the cross-check against
// moe_slots_nvfp4 on the same pool.
int oracle_cases(const Model& m, Runner& runner, const DevicePool& pool,
                 const std::vector<std::vector<int>>& place) {
    int failures = 0;
    for (const int T : {1, 9, 64, 200, 777}) {
        Call c                  = make_call(m, T, 100U + T);
        c.table                 = table_from(m, place);
        const Reference ref     = reference(m, c, true);
        const std::string label = "oracle T=" + std::to_string(T);
        const auto got          = runner.run(c, pool.pool, true, failures, label);
        failures += check_oracles(label, got, ref);
        const auto slots = runner.run(c, pool.pool, false, failures, label + " moe_slots");
        const double d   = relative_l2(as_double(got), as_double(slots));
        std::cout << label << ": rel_l2 vs moe_slots_nvfp4 " << d << '\n';
        failures += verify_reduction(label + " vs moe_slots_nvfp4", as_double(got),
                                     as_double(slots), kSameRecipe);
    }
    return failures;
}

// (2) Slot indirection: another placement (other pool) gives bit-identical outputs.
int permutation_case(const Model& m, Runner& runner, const DevicePool& pool,
                     const std::vector<std::vector<int>>& place) {
    int failures       = 0;
    Call c             = make_call(m, 300, 200U);
    c.table            = table_from(m, place);
    c.a1               = 2688.0F / 4.5F * kDivisorJitter;
    c.a2               = 2688.0F / 60.0F;
    const auto base    = runner.run(c, pool.pool, true, failures, "perm base");
    const int slots    = static_cast<int>(m.ids.size()) * 2 + 3;
    const auto other   = random_placement(m, slots, 201U);
    const DevicePool d = make_pool(m, other, slots, Layout::Planar, 202U);
    Call cp            = c;
    cp.table           = table_from(m, other);
    failures +=
        verify_exact("permutation == base", runner.run(cp, d.pool, true, failures, "perm"), base);
    return failures;
}

// (3) Routes without a slot (not resident, -1, out of range) are skipped: oracle on the resident
// experts and bit-equal to zero-weight resident routes.
int missing_case(const Model& m, Runner& runner, const DevicePool& pool,
                 const std::vector<std::vector<int>>& place) {
    int failures = 0;
    Call c       = make_call(m, 150, 500U);
    std::mt19937 rng(501U);
    auto partial = place;
    for (std::size_t i = 0; i < partial.size(); ++i)
        if (rng() % 10 < 3) partial[i].clear();
    c.table              = table_from(m, partial);
    const int invalid[4] = {-1, kE, 7777, -5};
    for (int t = 0; t < c.tokens; t += 3)
        for (int k = 0; k < 2; ++k) c.ids[t * kK + k] = invalid[(t + k) % 4];
    std::vector<int> zero_positions;
    for (int i = 0; i < c.tokens * kK; ++i)
        if (!resident(c, c.ids[i])) zero_positions.push_back(i);
    // a token whose routes are all skipped writes zeros
    for (int k = 0; k < kK; ++k) c.ids[7 * kK + k] = -1;
    for (int k = 0; k < kK; ++k) zero_positions.push_back(7 * kK + k);
    const Reference ref = reference(m, c, true);
    const auto got      = runner.run(c, pool.pool, true, failures, "missing");
    failures += check_oracles("missing", got, ref);
    failures += verify_exact("token without experts == 0",
                             std::vector<float>(got.begin() + 7 * kH, got.begin() + 8 * kH),
                             std::vector<float>(kH, 0.0F));
    Call z  = c;
    z.table = table_from(m, place);
    for (const int i : zero_positions) {
        if (z.ids[i] < 0 || z.ids[i] >= kE) z.ids[i] = m.ids[0];
        z.w[i] = 0.0F;
    }
    failures += verify_exact("missing == zero-weight residency",
                             runner.run(z, pool.pool, true, failures, "missing zw"), got);
    return failures;
}

// (4) A call wider than one piece (T56: gate/up grouped over the whole call, down and merge per
// piece) equals its pieces called separately, bit for bit: two full pieces and a tail, and an
// exact multiple of the piece width.
int piece_case(const Model& m, Runner& runner, const DevicePool& pool,
               const std::vector<std::vector<int>>& place, int tail, unsigned seed) {
    int failures       = 0;
    const int P        = ops::kMoeGroupedChunkColumns;
    const int T        = 2 * P + tail;
    Call c             = make_call(m, T, seed);
    c.table            = table_from(m, place);
    c.a1               = 2688.0F / 4.5F * kDivisorJitter;
    c.a2               = 2688.0F / 60.0F;
    const auto whole   = runner.run(c, pool.pool, true, failures, "pieces whole");
    const int split[3] = {P, P, tail};
    const int pieces   = tail > 0 ? 3 : 2;
    std::size_t at     = 0;
    for (int p = 0, b = 0; p < pieces; b += split[p], ++p) {
        Call s   = c;
        s.tokens = split[p];
        s.x.assign(c.x.begin() + static_cast<std::ptrdiff_t>(b) * kH,
                   c.x.begin() + static_cast<std::ptrdiff_t>(b + split[p]) * kH);
        s.ids.assign(c.ids.begin() + b * kK, c.ids.begin() + (b + split[p]) * kK);
        s.w.assign(c.w.begin() + b * kK, c.w.begin() + (b + split[p]) * kK);
        const auto part = runner.run(s, pool.pool, true, failures, "piece");
        failures += verify_exact(
            ("piece " + std::to_string(p) + " == whole").c_str(), part,
            std::vector<float>(whole.begin() + static_cast<std::ptrdiff_t>(at),
                               whole.begin() + static_cast<std::ptrdiff_t>(at + part.size())));
        at += part.size();
    }
    return failures;
}

// (5) CUDA graph: one call captured; replays after the table, x and routing change reproduce the
// eager results bit for bit.
int graph_case(const Model& m, Runner& runner, const DevicePool& pool,
               const std::vector<std::vector<int>>& place) {
    int failures = 0;
    const int T  = 130;
    Call v1 = make_call(m, T, 900U), v2 = make_call(m, T, 901U);
    v1.table = table_from(m, place);
    reference(m, v1, true);
    v2.table = v1.table;
    for (int e = 0; e < kE; e += 3) v2.table[e] = -1;
    v2.a1             = v1.a1;
    v2.a2             = v1.a2;
    const auto eager1 = runner.run(v1, pool.pool, true, failures, "graph eager v1");
    const auto eager2 = runner.run(v2, pool.pool, true, failures, "graph eager v2");
    DeviceBuffer dx(static_cast<std::size_t>(T) * kH * 2), di(static_cast<std::size_t>(T) * kK * 4),
        dw(static_cast<std::size_t>(T) * kK * 4), dt(kE * 4),
        dout(static_cast<std::size_t>(T) * kH * 4);
    auto upload = [&](const Call& c) {
        std::vector<std::uint16_t> xb(c.x.size());
        for (std::size_t i = 0; i < xb.size(); ++i) xb[i] = f32_to_bf16(c.x[i]);
        dx.copy_from_host(xb.data(), xb.size() * 2);
        di.copy_from_host(c.ids.data(), c.ids.size() * 4);
        dw.copy_from_host(c.w.data(), c.w.size() * 4);
        dt.copy_from_host(c.table.data(), kE * 4);
        dout.fill(0xFF);
        cuda_synchronize();
    };
    auto download = [&] {
        std::vector<float> v(static_cast<std::size_t>(T) * kH);
        dout.copy_to_host(v.data(), v.size() * 4);
        return v;
    };
    cudaStream_t s       = runner.stream();
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    upload(v1);
    cuda_check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
    {
        Tensor tx(dx.p, DType::BF16, {kH, T}), ti(di.p, DType::I32, {kK, T}),
            tw(dw.p, DType::FP32, {kK, T}), to(dout.p, DType::FP32, {kH, T});
        ops::moe_grouped_nvfp4(tx, ti, tw, runner.layer(v1, dt.p), pool.pool, to,
                               runner.workspace(), s);
    }
    cuda_check(cudaStreamEndCapture(s, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (int round = 0; round < 2; ++round) {
        upload(v1);
        cuda_check(cudaGraphLaunch(exec, s), "launch");
        cuda_synchronize(s);
        failures += verify_exact("graph v1", download(), eager1);
        upload(v2);
        cuda_check(cudaGraphLaunch(exec, s), "launch");
        cuda_synchronize(s);
        failures += verify_exact("graph v2", download(), eager2);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    if (eager1 == eager2) {
        std::cerr << "graph: v1 and v2 outputs are identical (the replay check is vacuous)\n";
        ++failures;
    }
    return failures;
}

int validation_cases(const Model& m, Runner& runner, const DevicePool& pool,
                     const std::vector<std::vector<int>>& place) {
    int failures = 0;
    Call c       = make_call(m, 20, 1000U);
    c.table      = table_from(m, place);
    reference(m, c, true);
    DeviceBuffer dx(20 * kH * 2), di(20 * kK * 4), dw(20 * kK * 4), dt(kE * 4), dout(20 * kH * 4);
    dt.copy_from_host(c.table.data(), kE * 4);
    auto expect_throw = [&](const char* label, auto&& fn) {
        try {
            fn();
        } catch (const std::invalid_argument&) { return; }
        std::cerr << label << ": expected std::invalid_argument\n";
        ++failures;
    };
    const auto layer = runner.layer(c, dt.p);
    Tensor tx(dx.p, DType::BF16, {kH, 20}), ti(di.p, DType::I32, {kK, 20}),
        tw(dw.p, DType::FP32, {kK, 20}), to(dout.p, DType::FP32, {kH, 20});
    expect_throw("contiguous-slot pool", [&] {
        auto p                 = pool.pool;
        p.gate_up_codes.stride = kSlotBytes;
        ops::moe_grouped_nvfp4(tx, ti, tw, layer, p, to, runner.workspace(), runner.stream());
    });
    expect_throw("BF16 out", [&] {
        Tensor o(dout.p, DType::BF16, {kH, 20});
        ops::moe_grouped_nvfp4(tx, ti, tw, layer, pool.pool, o, runner.workspace(),
                               runner.stream());
    });
    expect_throw("workspace too small", [&] {
        WorkspaceArena small(ops::moe_grouped_nvfp4_workspace_bytes(20) - 1024);
        ops::moe_grouped_nvfp4(tx, ti, tw, layer, pool.pool, to, small, runner.stream());
    });
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool benchmark = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
        if (argc > 1 && !benchmark) { throw std::invalid_argument("usage: test [--bench]"); }
        if (cuda_unavailable()) {
            std::cout << "SKIP: no CUDA device\n";
            return 77;
        }
        const Model m         = make_model(64, 21U);
        const int slots       = 3 * static_cast<int>(m.ids.size()) + 7;
        const auto place      = random_placement(m, slots, 22U);
        const DevicePool pool = make_pool(m, place, slots, Layout::Planar, 23U);
        Runner runner(m, benchmark ? 16384 : 2 * ops::kMoeGroupedChunkColumns + 77);
        int failures = 0;
        if (benchmark) {
            std::puts("case,size,columns,sample,us");
            for (int tokens : {1, 9, 64, 200, 256, 777, 2048, 2049, 4096, 4173, 16384}) {
                for (bool skewed : {false, true}) {
                    auto c  = make_call(m, tokens, 100U + tokens);
                    c.table = table_from(m, place);
                    c.a1    = 2688.0F / 4.5F * kDivisorJitter;
                    c.a2    = 2688.0F / 60.0F;
                    if (skewed) {
                        for (std::size_t i = 0; i < c.ids.size(); ++i) { c.ids[i] = m.ids[i % kK]; }
                    }
                    runner.run(c, pool.pool, true, failures,
                               skewed ? "moe-grouped-skewed" : "moe-grouped", true);
                }
            }
            return failures ? 1 : 0;
        }
        failures += oracle_cases(m, runner, pool, place);
        failures += permutation_case(m, runner, pool, place);
        failures += missing_case(m, runner, pool, place);
        failures += piece_case(m, runner, pool, place, 77, 600U);
        failures += piece_case(m, runner, pool, place, 0, 610U);
        failures += graph_case(m, runner, pool, place);
        failures += validation_cases(m, runner, pool, place);
        if (failures != 0) {
            std::cerr << "moe_grouped_nvfp4: " << failures << " failure(s)\n";
            return 1;
        }
        std::cout << "moe_grouped_nvfp4: all checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "moe_grouped_nvfp4 test: " << e.what() << '\n';
        return 1;
    }
}
