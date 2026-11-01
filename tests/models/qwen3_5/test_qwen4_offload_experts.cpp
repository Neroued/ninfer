// GPU test of the Qwen4Exp offload expert backend (execution/qwen4_experts.h, Qwen4OffloadExperts):
// the window protocol, the slot cache it drives (fills, hits, misses) and the host executor it
// hands the misses to, over synthetic expert banks of the real geometry (512 experts of 2560/640
// NVFP4 per layer, pinned and mapped host memory). Two cache layers: layer 0 stands for the Text
// layers, layer 1 for the MTP layer (text_layers = 1: called several times per window, observed at
// end_window).
//
// Oracle: Qwen4MappedExperts over the same banks (the M1 reference route: moe_slots_nvfp4 reading
// every expert in place, slot = expert id). An expert computed from a device slot runs the same
// kernel on a byte-identical copy, so a column whose experts are all resident must match bit for
// bit; an expert computed on the CPU quantizes its activations to int8 per 16 values (W4A8) instead
// of NVFP4 (W4A4), so a column computed on the CPU is compared with a bound (see
// kCpuRelativeBound).
//
// Checks:
//   - lifecycle: begin twice / end or observe without a window are refused, abandon closes;
//   - prompt window: every route missing, the CPU computes it, decide_prompt inserts the routed
//     experts, the fills land (freeze_after_prompt waits for them) and publish at the next window;
//   - decode window: resident experts are device hits (bit-exact against the oracle), a column of
//     non-resident experts is computed on the CPU;
//   - an executor failure (an expert id the host bank does not hold) fails the window and every
//     later one instead of silently dropping the CPU experts;
//   - the routing log of the MTP layer: every window observes exactly its own calls (a window of
//     MTP calls alone, as the prefix-reuse bridge, and a full prompt chunk followed by its drafts);
//   - the expert trace keeps every publish of a window whose MTP layer runs several times;
//   - columns that route nothing (ids -1, the verify columns past a lane's extent) are not counted
//     as tokens by the cache policy.
// Exit 77 without a CUDA device or AVX-512 VNNI.

#include "core/arena.h"
#include "core/device.h"
#include "models/qwen3_5/execution/qwen4_experts.h"
#include "ninfer/ops/moe_slots.h"
#include "ops/sparse_moe/host/host_moe_executor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models::qwen3_5::execution;

constexpr std::int32_t kExperts = 512;
constexpr std::int32_t kTopK    = ops::kMoeSlotsTopK;
constexpr std::int32_t kHidden  = ops::kMoeSlotsHidden;
constexpr std::int32_t kLayers  = 2; // layer 0: a Text layer, layer 1: the MTP layer
constexpr std::int32_t kSlots   = 16;
constexpr std::int32_t kPrompt  = 12; // largest prompt call (columns)
// Experts holding data (the others are never routed except to provoke a failure).
constexpr std::int32_t kResidentSet = 16;  // experts 0..15: the prompt's experts, made resident
constexpr std::int32_t kMissFirst   = 100; // experts 100..131: never resident
constexpr std::int32_t kMissCount   = 32;
// A column computed on the CPU (W4A8) against the W4A4 oracle: relative L2 distance of the column.
// The two paths differ only in the activation quantization of the gate|up and down inputs (NVFP4
// E2M1 per 16 values with an E4M3 scale, relative RMS error ~0.1 on Gaussian data, vs int8 per 16,
// ~0.005), which on these banks moves a column by ~0.1 (measured, see the printed values); a
// partial missing the CPU experts entirely is at distance 1.
constexpr double kCpuRelativeBound = 0.25;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::uint16_t to_bf16(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

template <class T>
T* host_mapped(std::size_t count) {
    void* p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, count * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));
    return static_cast<T*>(p);
}

// One layer's stacked bank in pinned mapped memory. Only the experts the test routes hold data:
// random E2M1 codes, E4M3 scales in [0.25, 1] and divisor 64 (finite, deterministic weights).
struct Bank {
    std::array<std::byte*, 4> plane{};
    float* gate_up_divisors = nullptr;
    float* down_divisors    = nullptr;
    Qwen4ExpertBank view;

    explicit Bank(std::uint64_t seed) {
        const std::int64_t bytes[4] = {ops::kMoeSlotsGateUpCodeBytes,
                                       ops::kMoeSlotsGateUpScaleBytes, ops::kMoeSlotsDownCodeBytes,
                                       ops::kMoeSlotsDownScaleBytes};
        for (int q = 0; q < 4; ++q) {
            plane[q] = host_mapped<std::byte>(static_cast<std::size_t>(bytes[q]) * kExperts);
        }
        gate_up_divisors = host_mapped<float>(2 * kExperts);
        down_divisors    = host_mapped<float>(kExperts);
        std::fill_n(gate_up_divisors, 2 * kExperts, 64.0F);
        std::fill_n(down_divisors, kExperts, 64.0F);
        std::uint64_t s = seed;
        const auto fill = [&](std::int32_t e) {
            for (int q = 0; q < 4; ++q) {
                auto* p = reinterpret_cast<std::uint8_t*>(plane[q] + bytes[q] * e);
                for (std::int64_t i = 0; i < bytes[q]; i += 8) {
                    std::uint64_t w = splitmix(s);
                    if (q % 2 == 1) { // scale planes: E4M3 0x30..0x38 (0.25 .. 1.0)
                        std::uint64_t v = 0;
                        for (int b = 0; b < 8; ++b) {
                            v |= (0x30ULL + ((w >> (8 * b)) % 9)) << (8 * b);
                        }
                        w = v;
                    }
                    std::memcpy(p + i, &w, 8);
                }
            }
        };
        for (std::int32_t e = 0; e < kResidentSet; ++e) { fill(e); }
        for (std::int32_t e = kMissFirst; e < kMissFirst + kMissCount; ++e) { fill(e); }
        view.bank = ops::MoeSlotPool{
            .gate_up_codes  = {plane[0], ops::kMoeSlotsGateUpCodeBytes},
            .gate_up_scales = {plane[1], ops::kMoeSlotsGateUpScaleBytes},
            .down_codes     = {plane[2], ops::kMoeSlotsDownCodeBytes},
            .down_scales    = {plane[3], ops::kMoeSlotsDownScaleBytes},
            .slots          = kExperts,
        };
        view.gate_up_divisors      = gate_up_divisors;
        view.down_divisors         = down_divisors;
        view.gate_up_input_divisor = 1.0F;
        view.down_input_divisor    = 1.0F;
    }

    ~Bank() {
        for (std::byte* p : plane) { (void)cudaFreeHost(p); }
        (void)cudaFreeHost(gate_up_divisors);
        (void)cudaFreeHost(down_divisors);
    }

    Bank(const Bank&)            = delete;
    Bank& operator=(const Bank&) = delete;
};

// Program-owned device memory of the backend (sized for kLayers x kSlots).
struct Memory {
    std::vector<void*> owned;
    Qwen4OffloadExperts::DeviceMemory view;
    Tensor identity;

    Memory() {
        const auto planes = Qwen4OffloadExperts::pool_plane_bytes(kLayers, kSlots);
        for (std::size_t q = 0; q < 4; ++q) {
            view.pool[q] = static_cast<std::byte*>(alloc(static_cast<std::size_t>(planes[q])));
        }
        view.table = static_cast<std::int32_t*>(alloc(sizeof(std::int32_t) * kLayers * kExperts));
        view.divisors = static_cast<float*>(
            alloc(sizeof(float) * kLayers * Qwen4OffloadExperts::kDivisorsPerLayer));
        view.identity = static_cast<std::int32_t*>(alloc(sizeof(std::int32_t) * kExperts));
        view.reads    = static_cast<std::int32_t*>(alloc(sizeof(std::int32_t) * kTopK * 8));
        std::vector<std::int32_t> ids(kExperts);
        for (std::int32_t e = 0; e < kExperts; ++e) { ids[e] = e; }
        CUDA_CHECK(cudaMemcpy(view.identity, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
        identity = Tensor(view.identity, DType::I32, {kExperts});
    }

    void* alloc(std::size_t bytes) {
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes));
        owned.push_back(p);
        return p;
    }

    ~Memory() {
        for (void* p : owned) { (void)cudaFree(p); }
    }
};

// One call's operands: x [2560,T] (Gaussian), ids [10,T], weights [10,T] (normalized).
struct Call {
    std::int32_t layer = 0;
    std::vector<std::vector<std::int32_t>> columns; // per column its 10 ids
};

struct Operands {
    Tensor x, ids, weights;
};

Operands upload(WorkspaceArena& work, const Call& call, std::uint64_t seed, cudaStream_t s) {
    const auto T = static_cast<std::int32_t>(call.columns.size());
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kHidden) * T);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) { v = to_bf16(normal(rng)); }
    std::vector<std::int32_t> ids;
    std::vector<float> weights;
    std::uniform_real_distribution<float> uniform(0.5F, 1.5F);
    for (const auto& column : call.columns) {
        float total = 0.0F;
        std::vector<float> w(kTopK);
        for (auto& v : w) { total += (v = uniform(rng)); }
        for (std::int32_t k = 0; k < kTopK; ++k) {
            ids.push_back(column[static_cast<std::size_t>(k)]);
            weights.push_back(w[static_cast<std::size_t>(k)] / total);
        }
    }
    Operands o;
    o.x       = work.alloc(DType::BF16, {kHidden, T});
    o.ids     = work.alloc(DType::I32, {kTopK, T});
    o.weights = work.alloc(DType::FP32, {kTopK, T});
    CUDA_CHECK(cudaMemcpyAsync(o.x.data, x.data(), x.size() * 2, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(o.ids.data, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(o.weights.data, weights.data(), weights.size() * 4,
                               cudaMemcpyHostToDevice, s));
    return o;
}

std::vector<std::int32_t> experts_from(std::int32_t first, std::int32_t count, std::uint64_t& s) {
    std::vector<std::int32_t> pool(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) { pool[static_cast<std::size_t>(i)] = first + i; }
    for (std::size_t i = pool.size() - 1; i > 0; --i) {
        std::swap(pool[i], pool[splitmix(s) % (i + 1)]);
    }
    pool.resize(kTopK);
    return pool;
}

Call make_call(std::int32_t layer, std::int32_t columns, std::int32_t first, std::int32_t count,
               std::uint64_t seed) {
    Call call{layer, {}};
    std::uint64_t s = seed;
    for (std::int32_t t = 0; t < columns; ++t) {
        call.columns.push_back(experts_from(first, count, s));
    }
    return call;
}

struct Fixture {
    cudaStream_t stream = nullptr;
    std::vector<std::unique_ptr<Bank>> banks;
    std::vector<Qwen4ExpertBank> views;
    Memory memory;
    WorkspaceArena work{std::size_t(256) << 20};

    Fixture() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (std::int32_t l = 0; l < kLayers; ++l) {
            banks.push_back(std::make_unique<Bank>(0x5eed0000ULL + static_cast<std::uint64_t>(l)));
            views.push_back(banks.back()->view);
        }
    }

    ~Fixture() { (void)cudaStreamDestroy(stream); }

    Qwen4OffloadOptions options() const {
        Qwen4OffloadOptions o;
        o.slots_per_layer     = kSlots;
        o.cpu_threads         = 2;
        o.insert_cap          = 2;
        o.prompt_tokens       = kPrompt;
        o.text_layers         = 1;
        o.freeze_after_prompt = true; // prompt fills land before the window closes
        return o;
    }

    std::unique_ptr<Qwen4OffloadExperts> make(const Qwen4OffloadOptions& o) {
        return std::make_unique<Qwen4OffloadExperts>(views, o, memory.view, 0, stream);
    }

    // One forward call of a layer (routed() then run(), as Qwen4Forward::moe); returns the FP32
    // partial on the host.
    std::vector<float> call(Qwen4RoutedExperts& backend, const Call& c, std::uint64_t seed) {
        auto scope       = work.scope();
        const Operands o = upload(work, c, seed, stream);
        const auto T     = static_cast<std::int32_t>(c.columns.size());
        Tensor partial   = work.alloc(DType::FP32, {kHidden, T});
        const auto& bank = views[static_cast<std::size_t>(c.layer)];
        backend.routed(c.layer, bank, o.x, o.ids, o.weights, work, stream);
        backend.run(c.layer, bank, o.x, o.ids, o.weights, partial, work, stream);
        std::vector<float> out(static_cast<std::size_t>(kHidden) * T);
        CUDA_CHECK(cudaMemcpyAsync(out.data(), partial.data, out.size() * 4, cudaMemcpyDeviceToHost,
                                   stream));
        return out;
    }

    // The oracle partial of the same call.
    std::vector<float> mapped(const Call& c, std::uint64_t seed) {
        Qwen4MappedExperts reference(memory.identity);
        auto out = call(reference, c, seed);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return out;
    }

    // One window of `calls` (in order); returns each call's partial.
    std::vector<std::vector<float>> window(Qwen4OffloadExperts& backend, Qwen4Window kind,
                                           const std::vector<Call>& calls, std::uint64_t seed) {
        backend.begin_window(stream, kind);
        std::vector<std::vector<float>> out;
        try {
            for (std::size_t i = 0; i < calls.size(); ++i) {
                out.push_back(call(backend, calls[i], seed + i));
            }
            backend.observe_window(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (...) {
            (void)cudaStreamSynchronize(stream);
            backend.abandon_window(stream);
            throw;
        }
        backend.end_window(stream);
        return out;
    }
};

double column_distance(const std::vector<float>& a, const std::vector<float>& b, std::int32_t t) {
    double d = 0.0, n = 0.0;
    for (std::int32_t h = 0; h < kHidden; ++h) {
        const std::size_t i = static_cast<std::size_t>(t) * kHidden + h;
        d += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        n += static_cast<double>(b[i]) * b[i];
    }
    return n > 0.0 ? std::sqrt(d / n) : (d > 0.0 ? INFINITY : 0.0);
}

bool column_equal(const std::vector<float>& a, const std::vector<float>& b, std::int32_t t) {
    return std::memcmp(a.data() + static_cast<std::size_t>(t) * kHidden,
                       b.data() + static_cast<std::size_t>(t) * kHidden,
                       kHidden * sizeof(float)) == 0;
}

template <class E, class Body>
bool throws(Body&& body) {
    try {
        body();
    } catch (const E&) { return true; } catch (...) {
        return false;
    }
    return false;
}

void lifecycle(Fixture& f) {
    auto backend           = f.make(f.options());
    Qwen4OffloadExperts& b = *backend;
    check(throws<std::logic_error>([&] { b.end_window(f.stream); }),
          "lifecycle: end without begin");
    check(throws<std::logic_error>([&] { b.observe_window(f.stream); }),
          "lifecycle: observe without begin");
    b.begin_window(f.stream, Qwen4Window::Decode);
    check(throws<std::logic_error>([&] { b.begin_window(f.stream, Qwen4Window::Decode); }),
          "lifecycle: begin twice");
    b.abandon_window(f.stream);
    b.abandon_window(f.stream); // closed: no-op
    b.begin_window(f.stream, Qwen4Window::Prompt);
    b.end_window(f.stream);
    check(b.counters().windows == 2, "lifecycle: two windows fenced (abandon closes its window)");
}

void fills_hits_misses(Fixture& f) {
    Qwen4OffloadOptions o   = f.options();
    o.freeze_after_prompt   = false; // the engine's policy: decode windows are observed too
    o.keep_decode_residency = false; // T69's rule: the prompt's routed experts become resident
    auto backend            = f.make(o);
    Qwen4OffloadExperts& b  = *backend;
    // Prompt window: 12 columns per layer over experts 0..15 (every one routed), nothing resident.
    const std::vector<Call> prompt  = {make_call(0, kPrompt, 0, kResidentSet, 11),
                                       make_call(1, kPrompt, 0, kResidentSet, 12)};
    const auto p                    = f.window(b, Qwen4Window::Prompt, prompt, 100);
    const Qwen4OffloadWindowStats w = b.last_window();
    std::printf("  prompt window: layers %d accesses %lld device_hits %lld missing %lld inserts "
                "%lld fills %lld pending %lld\n",
                w.layers, static_cast<long long>(w.accesses), static_cast<long long>(w.device_hits),
                static_cast<long long>(w.missing), static_cast<long long>(w.inserts),
                static_cast<long long>(w.fills_issued), static_cast<long long>(w.fills_pending));
    check(w.kind == Qwen4Window::Prompt && w.layers == 2 && w.accesses == 2 * kPrompt * kTopK &&
              w.device_hits == 0 && w.missing == 2 * kResidentSet,
          "prompt window: every route counted, none resident, 16 distinct missing per layer");
    check(w.inserts == 2 * kResidentSet && w.fills_issued == 2 * kResidentSet,
          "prompt window: the routed experts are inserted and their fills issued");
    // A fill publishes at the first window that begins after it completed.
    for (int i = 0; i < 2000 && b.counters().published < 2 * kResidentSet; ++i) {
        b.begin_window(f.stream, Qwen4Window::Decode);
        b.end_window(f.stream);
        CUDA_CHECK(cudaStreamSynchronize(f.stream));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(b.counters().published == 2 * kResidentSet, "the prompt's fills were published");
    double worst = 0.0;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        const auto ref = f.mapped(prompt[i], 100 + i);
        for (std::int32_t t = 0; t < kPrompt; ++t) {
            worst = std::max(worst, column_distance(p[i], ref, t));
        }
    }
    std::printf(
        "  prompt window (all experts on the CPU): worst column distance to the oracle %.4f\n",
        worst);
    check(worst < kCpuRelativeBound, "prompt window: CPU partials within the W4A8 bound");

    // Decode window: column 0 resident experts only, column 1 experts 100..131 (never resident).
    Call decode{0, {}};
    std::uint64_t s = 21;
    decode.columns.push_back(experts_from(0, kResidentSet, s));
    decode.columns.push_back(experts_from(kMissFirst, kMissCount, s));
    const auto d                     = f.window(b, Qwen4Window::Decode, {decode}, 200);
    const auto ref                   = f.mapped(decode, 200);
    const Qwen4OffloadWindowStats dw = b.last_window();
    const double miss                = column_distance(d[0], ref, 1);
    std::printf("  decode window: accesses %lld device_hits %lld missing %lld; resident column "
                "bit-exact %d, CPU column distance %.4f\n",
                static_cast<long long>(dw.accesses), static_cast<long long>(dw.device_hits),
                static_cast<long long>(dw.missing), column_equal(d[0], ref, 0) ? 1 : 0, miss);
    check(dw.accesses == 2 * kTopK && dw.device_hits == kTopK && dw.missing == kTopK,
          "decode window: 10 device hits, 10 distinct CPU experts");
    check(column_equal(d[0], ref, 0),
          "decode window: resident column equals the oracle bit for bit");
    check(miss < kCpuRelativeBound, "decode window: CPU column within the W4A8 bound");
}

// T211 (the default): a prompt window inserts nothing; the decode after it inserts up to the
// post-prompt boost cap (T69, on by default since T211).
void prompt_keeps_residency(Fixture& f) {
    Qwen4OffloadOptions o           = f.options();
    o.freeze_after_prompt           = false;
    auto backend                    = f.make(o);
    Qwen4OffloadExperts& b          = *backend;
    const std::vector<Call> prompt  = {make_call(0, kPrompt, 0, kResidentSet, 61),
                                       make_call(1, kPrompt, 0, kResidentSet, 62)};
    const auto p                    = f.window(b, Qwen4Window::Prompt, prompt, 600);
    const Qwen4OffloadWindowStats w = b.last_window();
    std::printf("  keep: prompt window accesses %lld missing %lld inserts %lld fills %lld\n",
                static_cast<long long>(w.accesses), static_cast<long long>(w.missing),
                static_cast<long long>(w.inserts), static_cast<long long>(w.fills_issued));
    check(w.accesses == 2 * kPrompt * kTopK && w.missing == 2 * kResidentSet,
          "keep: prompt window counted, 16 distinct missing per layer");
    check(w.inserts == 0 && w.fills_issued == 0 && w.fills_pending == 0,
          "keep: the prompt window inserts nothing");
    double worst = 0.0;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        const auto ref = f.mapped(prompt[i], 600 + i);
        for (std::int32_t t = 0; t < kPrompt; ++t) {
            worst = std::max(worst, column_distance(p[i], ref, t));
        }
    }
    check(worst < kCpuRelativeBound, "keep: prompt CPU partials within the W4A8 bound");
    Call decode{0, {}};
    std::uint64_t s = 63;
    decode.columns.push_back(experts_from(0, kResidentSet, s));
    (void)f.window(b, Qwen4Window::Decode, {decode}, 700);
    const Qwen4OffloadWindowStats dw = b.last_window();
    check(o.boost_windows > 0 && dw.missing == kTopK && dw.inserts == o.boost_cap,
          "keep: the decode window after it inserts up to the boost cap");
}

// T211b: the request's first text chunk decides. Neither kind of prompt inserts during its windows;
// when the first decode window begins, a new conversation (less than half of the prompt restored
// from the cache) applies its routing as T69 did, with at most new_prompt_cap targets per layer,
// and a continuation applies nothing.
void prompt_new_conversation(Fixture& f) {
    for (const bool continuation : {false, true}) {
        Qwen4OffloadOptions o          = f.options();
        o.freeze_after_prompt          = false;
        o.new_prompt_cap               = 4;
        auto backend                   = f.make(o);
        Qwen4OffloadExperts& b         = *backend;
        const std::vector<Call> prompt = {make_call(0, kPrompt, 0, kResidentSet, 71),
                                          make_call(1, kPrompt, 0, kResidentSet, 72)};
        // The chunk starts at 0 (nothing cached) or at 3 * kPrompt of 4 * kPrompt (75 % cached);
        // a second chunk of the same request keeps the request's decision.
        const std::int64_t begin = continuation ? 3 * kPrompt : 0;
        b.prepare_prompt(kPrompt, begin, begin + 2 * kPrompt);
        (void)f.window(b, Qwen4Window::Prompt, prompt, 800);
        const auto w1 = b.last_window();
        b.prepare_prompt(kPrompt, begin + kPrompt, begin + 2 * kPrompt);
        (void)f.window(b, Qwen4Window::Prompt, prompt, 810);
        const auto w2           = b.last_window();
        const std::int64_t from = b.counters().inserts;
        (void)f.window(b, Qwen4Window::Decode, {}, 820);
        const std::int64_t applied = b.counters().inserts - from;
        std::printf("  %s: prompt inserts %lld %lld, applied at the decode %lld\n",
                    continuation ? "continuation" : "new", static_cast<long long>(w1.inserts),
                    static_cast<long long>(w2.inserts), static_cast<long long>(applied));
        check(w1.inserts == 0 && w2.inserts == 0, "prompt windows insert nothing");
        check(continuation ? applied == 0 : applied == 2 * o.new_prompt_cap,
              continuation ? "continuation: nothing applied at the decode"
                           : "new conversation: new_prompt_cap targets per layer at the decode");
    }
    // The default (new_prompt_cap -1): the new conversation's counts reach the frequencies only.
    Qwen4OffloadOptions o = f.options();
    o.freeze_after_prompt = false;
    check(o.new_prompt_cap == -1, "default: frequencies only");
    auto backend = f.make(o);
    backend->prepare_prompt(kPrompt, 0, kPrompt);
    (void)f.window(
        *backend, Qwen4Window::Prompt,
        {make_call(0, kPrompt, 0, kResidentSet, 91), make_call(1, kPrompt, 0, kResidentSet, 92)},
        900);
    const std::int64_t from = backend->counters().inserts;
    (void)f.window(*backend, Qwen4Window::Decode, {}, 910);
    check(backend->counters().inserts == from, "default: no insertion applied at the decode");
}

void executor_failure(Fixture& f) {
    auto backend           = f.make(f.options());
    Qwen4OffloadExperts& b = *backend;
    // Expert 512 does not exist: the device skips it (out of its table), the host executor throws.
    Call bad{0, {{512, 100, 101, 102, 103, 104, 105, 106, 107, 108}}};
    bool surfaced = false;
    try {
        (void)f.window(b, Qwen4Window::Decode, {bad}, 300);
    } catch (const std::runtime_error& e) {
        surfaced = std::string(e.what()).find("executor failed") != std::string::npos;
        std::printf("  executor failure surfaced: %s\n", e.what());
    }
    check(surfaced, "executor failure: the window that saw it fails");
    const Call good = make_call(0, 1, kMissFirst, kMissCount, 31);
    check(throws<std::runtime_error>([&] { (void)f.window(b, Qwen4Window::Decode, {good}, 301); }),
          "executor failure: every later window fails too");
}

// Records of the routing dump: {window, kind, layer, tokens} then tokens x 10 ids.
struct RouteRecord {
    std::int32_t window = 0, kind = 0, layer = 0, tokens = 0;
    std::vector<std::int32_t> ids;
};

std::vector<RouteRecord> read_routing(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<RouteRecord> out;
    RouteRecord r;
    while (in.read(reinterpret_cast<char*>(&r.window), 16)) {
        r.ids.resize(static_cast<std::size_t>(r.tokens) * kTopK);
        in.read(reinterpret_cast<char*>(r.ids.data()),
                static_cast<std::streamsize>(r.ids.size() * 4));
        out.push_back(r);
    }
    return out;
}

std::vector<std::int32_t> flat_ids(const std::vector<Call>& calls) {
    std::vector<std::int32_t> out;
    for (const Call& c : calls) {
        for (const auto& column : c.columns) {
            out.insert(out.end(), column.begin(), column.end());
        }
    }
    return out;
}

void mtp_routing_log(Fixture& f) {
    const std::string path =
        "/tmp/ninfer_qwen4_offload_routing_" + std::to_string(::getpid()) + ".bin";
    Qwen4OffloadOptions o = f.options();
    o.routing_path        = path;
    // Window 0 and 1: MTP calls alone (the prefix-reuse bridge); window 2: a full prompt chunk of
    // both layers, then three single-column draft calls of the MTP layer.
    const Call bridge_a           = make_call(1, 3, kMissFirst, kMissCount, 41);
    const Call bridge_b           = make_call(1, 2, kMissFirst, kMissCount, 42);
    const std::vector<Call> chunk = {make_call(0, kPrompt, kMissFirst, kMissCount, 43),
                                     make_call(1, kPrompt, kMissFirst, kMissCount, 44),
                                     make_call(1, 1, kMissFirst, kMissCount, 45),
                                     make_call(1, 1, kMissFirst, kMissCount, 46),
                                     make_call(1, 1, kMissFirst, kMissCount, 47)};
    {
        auto backend = f.make(o);
        (void)f.window(*backend, Qwen4Window::Prompt, {bridge_a}, 400);
        (void)f.window(*backend, Qwen4Window::Prompt, {bridge_b}, 401);
        (void)f.window(*backend, Qwen4Window::Prompt, chunk, 402);
    }
    const auto records = read_routing(path);
    std::remove(path.c_str());
    const auto find = [&](std::int32_t window, std::int32_t layer) -> const RouteRecord* {
        for (const auto& r : records) {
            if (r.window == window && r.layer == layer) { return &r; }
        }
        return nullptr;
    };
    const RouteRecord* a  = find(0, 1);
    const RouteRecord* b  = find(1, 1);
    const RouteRecord* c0 = find(2, 0);
    const RouteRecord* c1 = find(2, 1);
    std::printf("  MTP routing log tokens: bridge %d, bridge %d, chunk text %d, chunk MTP %d\n",
                a ? a->tokens : -1, b ? b->tokens : -1, c0 ? c0->tokens : -1, c1 ? c1->tokens : -1);
    check(a != nullptr && a->ids == flat_ids({bridge_a}),
          "routing log: bridge window 0 observes its call");
    check(b != nullptr && b->ids == flat_ids({bridge_b}),
          "routing log: bridge window 1 observes its own call only (no stale ids)");
    check(c0 != nullptr && c0->ids == flat_ids({chunk[0]}), "routing log: chunk Text layer");
    check(c1 != nullptr && c1->ids == flat_ids({chunk[1], chunk[2], chunk[3], chunk[4]}),
          "routing log: a full chunk's MTP call and its drafts are all observed");
}

double json_number(const std::string& line, const std::string& key) {
    const auto at = line.find("\"" + key + "\":");
    if (at == std::string::npos) { return NAN; }
    return std::stod(line.substr(at + key.size() + 3));
}

void mtp_trace(Fixture& f) {
    const std::string path =
        "/tmp/ninfer_qwen4_offload_trace_" + std::to_string(::getpid()) + ".jsonl";
    Qwen4OffloadOptions o = f.options();
    o.trace_path          = path;
    // A decode window: the Text layer once, the MTP layer three times (alignment and two drafts),
    // every call with 10 distinct non-resident experts: 4 publishes of 10 CPU experts each.
    const std::vector<Call> calls = {
        make_call(0, 1, kMissFirst, kMissCount, 51), make_call(1, 2, kMissFirst, kMissCount, 52),
        make_call(1, 1, kMissFirst, kMissCount, 53), make_call(1, 1, kMissFirst, kMissCount, 54)};
    std::int64_t expected = 0;
    for (const Call& c : calls) {
        std::vector<std::int32_t> ids = flat_ids({c});
        std::sort(ids.begin(), ids.end());
        expected += std::unique(ids.begin(), ids.end()) - ids.begin();
    }
    {
        auto backend = f.make(o);
        (void)f.window(*backend, Qwen4Window::Decode, calls, 500);
    }
    std::ifstream in(path);
    std::string line;
    double cpu_experts = NAN;
    while (std::getline(in, line)) {
        if (line.find("\"window\":0,") != std::string::npos) {
            cpu_experts = json_number(line, "cpu_experts");
        }
    }
    std::remove(path.c_str());
    std::printf("  trace of an MTP decode window: cpu_experts %.0f (expected %lld)\n", cpu_experts,
                static_cast<long long>(expected));
    check(cpu_experts == static_cast<double>(expected),
          "trace: every publish of the window is recorded (MTP layer called three times)");
}

void unrouted_columns(Fixture& f) {
    Qwen4OffloadOptions o  = f.options();
    o.freeze_after_prompt  = false; // decode windows reach the policy
    auto backend           = f.make(o);
    Qwen4OffloadExperts& b = *backend;
    // A verify-like call of 4 columns of which 2 route nothing (moe_route's invalid columns).
    Call c = make_call(0, 4, kMissFirst, kMissCount, 61);
    c.columns[1].assign(kTopK, -1);
    c.columns[3].assign(kTopK, -1);
    const auto out                  = f.window(b, Qwen4Window::Decode, {c}, 600);
    const Qwen4OffloadWindowStats w = b.last_window();
    bool zero                       = true;
    for (const std::int32_t t : {1, 3}) {
        for (std::int32_t h = 0; h < kHidden; ++h) {
            zero = zero && out[0][static_cast<std::size_t>(t) * kHidden + h] == 0.0F;
        }
    }
    std::printf("  unrouted columns: tokens %d accesses %lld\n", w.tokens,
                static_cast<long long>(w.accesses));
    check(zero, "unrouted columns: partial is zero");
    check(w.accesses == 2 * kTopK, "unrouted columns: only routed columns are accessed");
    check(w.tokens == 2, "unrouted columns: the policy counts routed columns as tokens");
}

} // namespace

// The prompt ring the startup planner counts (startup.cpp) is the one the runtime lends: half a
// layer of experts per GPU at tp 2 (o-prefill: the 1-GPU count halved FP4-Local's prompt chunk).
void test_ring_layers() {
    ::unsetenv("NINFER_QWEN4_TP2_HALF_RING");
    const auto uniform = [](std::int32_t K, std::size_t layers = 48) {
        return std::vector<std::int32_t>(layers, K);
    };
    check(Qwen4OffloadExperts::ring_slots_for(576, 2) == 288, "tp 2 ring is half");
    check(Qwen4OffloadExperts::ring_layers(576, uniform(59), 48, 2) == 5,
          "tp 2 ring layers at 59 slots");
    check(Qwen4OffloadExperts::ring_layers(576, uniform(61), 48, 1) == 10,
          "tp 1 ring layers at 61 slots");
    check(Qwen4OffloadExperts::ring_layers(512, uniform(64), 48, 1) == 8, "exact layers");
    ::setenv("NINFER_QWEN4_TP2_HALF_RING", "0", 1);
    check(Qwen4OffloadExperts::ring_layers(576, uniform(59), 48, 2) == 10,
          "HALF_RING=0 keeps the 1-GPU ring");
    ::unsetenv("NINFER_QWEN4_TP2_HALF_RING");
    check(Qwen4OffloadExperts::layers_below(uniform(10), 5, 51) == 6,
          "layers_below: end + 1 when the layers below do not hold the slots");
    // A ring wider than every Text layer (ring_layers = text_layers + 1) leaves the workspace an
    // `end` of -1: nothing below, no read outside the profile.
    check(Qwen4OffloadExperts::ring_layers(512, uniform(24, 4), 4, 1) == 5,
          "ring_layers: text_layers + 1 when the Text layers cannot hold the ring");
    check(Qwen4OffloadExperts::layers_below(uniform(24, 4), -1, 1) == 1 &&
              Qwen4OffloadExperts::layers_below(uniform(24, 4), 0, 1) == 1,
          "layers_below: an end <= 0 holds nothing");
    check(Qwen4OffloadExperts::may_lend(uniform(66), 48, 36, 2) &&
              !Qwen4OffloadExperts::may_lend(uniform(66), 48, 37, 2),
          "uniform layers, tp 2: at most 36 of 48 lent");
    check(Qwen4OffloadExperts::may_lend(uniform(66), 48, 43, 1) &&
              !Qwen4OffloadExperts::may_lend(uniform(66), 48, 44, 1),
          "uniform layers, one GPU: at most 43 of 48 lent (0.9)");
}

// prod/cache: the per-layer slot profile keeps the pool (layers * slots_per_layer base slots),
// stays within a partition's experts next to the elastic columns, applies to Flash-Next at tp 2
// only (NINFER_QWEN4_SLOT_PROFILE=0: uniform), and still lends the production prompt workspace.
void test_slot_profile() {
    ::unsetenv("NINFER_QWEN4_SLOT_PROFILE");
    for (const std::int32_t layers : {48, 49}) {
        for (const std::int32_t K : {10, 59, 61, 66, 120, 200}) {
            const auto b     = Qwen4OffloadExperts::layer_slot_profile(layers, 48, K, 21, 2);
            std::int64_t sum = 0;
            bool bounded     = b.size() == static_cast<std::size_t>(layers);
            for (const std::int32_t x : b) {
                sum += x;
                bounded = bounded && x >= 1 && x + 21 <= 256;
            }
            check(sum == static_cast<std::int64_t>(layers) * K && bounded,
                  "profile: layers * K base slots, each within the partition");
        }
    }
    const auto b = Qwen4OffloadExperts::layer_slot_profile(49, 48, 66, 21, 2);
    check(b[0] == 149 && b[1] == 119 && b[15] == 29 && b[47] == 59 && b[48] == 109,
          "profile at the production 66 slots: the fitted table");
    const std::vector<std::int32_t> uniform(49, 66);
    check(Qwen4OffloadExperts::layer_slot_profile(49, 48, 66, 21, 1) == uniform,
          "profile: tp 1 stays uniform");
    check(Qwen4OffloadExperts::layer_slot_profile(41, 40, 66, 21, 2) ==
              std::vector<std::int32_t>(41, 66),
          "profile: other layer counts stay uniform");
    ::setenv("NINFER_QWEN4_SLOT_PROFILE", "0", 1);
    check(Qwen4OffloadExperts::layer_slot_profile(49, 48, 66, 21, 2) == uniform,
          "NINFER_QWEN4_SLOT_PROFILE=0 keeps every layer at slots_per_layer");
    ::unsetenv("NINFER_QWEN4_SLOT_PROFILE");
    // The 16384-column prompt workspace of production (24 uniform layers of 66 slots) and the
    // tp 2 ring still fit the lendable layers.
    const std::int32_t ring = Qwen4OffloadExperts::ring_layers(576, b, 48, 2);
    const std::int32_t ws   = Qwen4OffloadExperts::layers_below(b, 48 - ring, 24 * 66);
    check(ring == 6 && ws == 28 && Qwen4OffloadExperts::may_lend(b, 48, ring + ws, 2),
          "profile: ring + workspace lendable at 66 slots");
}

int main() {
    test_ring_layers();
    test_slot_profile();
    int count             = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (!ops::host::host_experts_supported()) {
        std::cout << "SKIP: the host expert executor needs AVX2 + FMA + F16C\n";
        return 77;
    }
    try {
        CUDA_CHECK(cudaSetDevice(0));
        Fixture f;
        const std::vector<std::pair<const char*, std::function<void(Fixture&)>>> cases = {
            {"lifecycle", lifecycle},
            {"fills, hits, misses", fills_hits_misses},
            {"prompt keeps the residency", prompt_keeps_residency},
            {"prompt of a new conversation", prompt_new_conversation},
            {"executor failure", executor_failure},
            {"MTP routing log", mtp_routing_log},
            {"MTP trace", mtp_trace},
            {"unrouted columns", unrouted_columns},
        };
        for (const auto& [name, body] : cases) {
            std::printf("%s:\n", name);
            try {
                body(f);
            } catch (const std::exception& e) {
                std::cerr << "FAIL: " << name << ": " << e.what() << '\n';
                ++failures;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::printf("qwen4 offload experts: all checks passed\n");
    return 0;
}
