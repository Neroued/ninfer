#include "ninfer/ops/sliding_window_attention.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int D = 128, Hq = 32, Hkv = 8, Lanes = 8;
constexpr float Scale = 0.08838834764831844055f;
constexpr ops::AttentionHeadGeometry Geometry{D, Hq, Hkv};
// BF16 output alone can round by 1/256 relative. Allow that final rounding plus
// FP16 P/V arithmetic, while bounding aggregate error more tightly than the old BF16 partial route.
constexpr ReductionCriterion Criterion{2.3e-3, 3e-4, 4.5e-3};
constexpr std::array<int, 8> LaneOrder{7, 0, 4, 2, 6, 1, 5, 3};

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) bits[i] = f32_to_bf16(v[i]);
    return bits;
}

std::vector<std::uint16_t> fp16_bits(std::vector<float>& v) {
    std::vector<std::uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        const __half h = __float2half_rn(v[i]);
        std::memcpy(&bits[i], &h, sizeof(h));
        // Persistent V is represented FP16, independently of its producer's earlier BF16 value.
        v[i] = __half2float(h);
    }
    return bits;
}

struct Cache {
    int window, padded;
    bool boundary;
    std::vector<float> k, v;
    std::vector<std::uint16_t> k_bits, v_bits;
    GuardedDeviceBuffer device_k, device_v;

    std::size_t index(int lane, int head, int position, int d) const {
        return (((static_cast<std::size_t>(lane) * Hkv + head) * padded +
                 (position & (window - 1))) *
                D) +
               d;
    }

    Cache(int w, bool edge = false)
        : window(w), padded(w + 8), boundary(edge),
          k(static_cast<std::size_t>(D) * padded * Hkv * Lanes), v(k.size()),
          device_k(k.size() * 2), device_v(v.size() * 2) {
        if (!edge) {
            fill_uniform(k, 401U + w, -0.4f, 0.4f);
            fill_uniform(v, 503U + w, -0.8f, 0.8f);
        } else {
            for (int lane = 0; lane < Lanes; ++lane)
                for (int head = 0; head < Hkv; ++head)
                    for (int d = 0; d < D; ++d) {
                        v[index(lane, head, 0, d)] = 512.0f;
                        v[index(lane, head, 1, d)] = 256.0f;
                    }
        }
        round_to_bf16(k);
        round_to_bf16(v);
        k_bits = bf16_bits(k);
        v_bits = fp16_bits(v);
        device_k.copy_from_host(k_bits.data(), k_bits.size() * 2);
        device_v.copy_from_host(v_bits.data(), v_bits.size() * 2);
    }

    CyclicKVCacheLayerView view() {
        return {.k               = Tensor(device_k.data(), DType::BF16, {D, padded, Hkv, Lanes}),
                .v               = Tensor(device_v.data(), DType::FP16, {D, padded, Hkv, Lanes}),
                .capacity        = static_cast<std::uint32_t>(window),
                .padded_capacity = static_cast<std::uint32_t>(padded),
                .num_kv_heads    = Hkv,
                .head_dim        = D,
                .lane_capacity   = Lanes};
    }

    int verify() const {
        int failures =
            device_k.verify_guards("context K guards") + device_v.verify_guards("context V guards");
        failures +=
            verify_exact("context K unchanged",
                         from_device<std::uint16_t>(device_k.data(), k_bits.size()), k_bits);
        failures +=
            verify_exact("context V unchanged",
                         from_device<std::uint16_t>(device_v.data(), v_bits.size()), v_bits);
        return failures;
    }
};

// One naive FP64 oracle for both cache windows, every route, and every execution mode. The
// cached interval and |key_position-query_position| predicate come from the public formula.
std::vector<double> oracle(const Cache& cache, int width, int batch, const std::vector<float>& q,
                           const std::vector<float>& qk, const std::vector<float>& qv,
                           const std::vector<int>& positions, const std::vector<int>& valid,
                           const std::vector<int>& lanes) {
    std::vector<double> out(static_cast<std::size_t>(D) * Hq * width * batch, 0.0);
    std::vector<std::thread> workers;
    for (int b = 0; b < batch; ++b)
        workers.emplace_back([&, b] {
            if (valid[b] == 0) return;
            const int length        = positions[b * width];
            const int start         = std::max(0, length - cache.window);
            const int context_count = length - start;
            for (int token = 0; token < valid[b]; ++token) {
                const int p = positions[b * width + token];
                naive_dense_softmax_attention(
                    Geometry, 1, context_count + valid[b], static_cast<double>(Scale),
                    [&](int d, int h, int) {
                        return static_cast<double>(
                            q[((static_cast<std::size_t>(b) * width + token) * Hq + h) * D + d]);
                    },
                    [&](int d, int h, int key) {
                        return key < context_count
                                   ? static_cast<double>(
                                         cache.k[cache.index(lanes[b], h, start + key, d)])
                                   : static_cast<double>(qk[((static_cast<std::size_t>(b) * width +
                                                              key - context_count) *
                                                                 Hkv +
                                                             h) *
                                                                D +
                                                            d]);
                    },
                    [&](int d, int h, int key) {
                        return key < context_count
                                   ? static_cast<double>(
                                         cache.v[cache.index(lanes[b], h, start + key, d)])
                                   : static_cast<double>(qv[((static_cast<std::size_t>(b) * width +
                                                              key - context_count) *
                                                                 Hkv +
                                                             h) *
                                                                D +
                                                            d]);
                    },
                    [&](int, int key) {
                        const int kp = key < context_count
                                           ? start + key
                                           : positions[b * width + key - context_count];
                        return std::abs(static_cast<std::int64_t>(kp) - p) < cache.window;
                    },
                    [&](int d, int h, int, double value) {
                        out[((static_cast<std::size_t>(b) * width + token) * Hq + h) * D + d] =
                            value;
                    });
            }
        });
    for (auto& worker : workers) worker.join();
    return out;
}

enum class Prefix { Full, Mixed, Zero };

int run_case(Cache& cache, int width, int batch, int base_context, int envelope_max, Prefix prefix,
             bool replay, std::size_t* observed = nullptr) {
    const std::size_t q_count  = static_cast<std::size_t>(D) * Hq * width * batch;
    const std::size_t kv_count = static_cast<std::size_t>(D) * Hkv * width * batch;
    std::vector<float> q(q_count), qk(kv_count), qv(kv_count);
    if (!cache.boundary) {
        fill_uniform(q, 101U + width * 131 + batch, -0.35f, 0.35f);
        fill_uniform(qk, 211U + width * 137 + batch, -0.4f, 0.4f);
        fill_uniform(qv, 307U + width * 139 + batch, -0.8f, 0.8f);
    } else {
        for (int b = 0; b < batch; ++b)
            for (int h = 0; h < Hkv; ++h)
                for (int d = 0; d < D; ++d)
                    qv[((static_cast<std::size_t>(b) * width + width - 1) * Hkv + h) * D + d] =
                        64.0f;
    }
    round_to_bf16(q);
    round_to_bf16(qk);
    round_to_bf16(qv);
    const auto q_base = bf16_bits(q), k_base = bf16_bits(qk), v_base = bf16_bits(qv);
    std::vector<int> lengths(batch), positions(width * batch), valid(batch), lanes(batch);
    constexpr std::array<int, 8> small_lengths{0, 1, 31, 32, 63, 64, 95, 96};
    for (int b = 0; b < batch; ++b) {
        lengths[b] = base_context < 0 ? small_lengths[(b + width) % 8]
                                      : (cache.boundary || batch == 1 || b % 3 == 0 ? base_context
                                         : b % 3 == 1 ? std::max(0, base_context - 63)
                                                      : std::min(128, base_context));
        valid[b]   = prefix == Prefix::Zero   ? 0
                     : prefix == Prefix::Full ? width
                     : b % 4 == 0             ? width
                     : b % 4 == 1             ? std::max(1, width - 1)
                     : b % 4 == 2             ? 1
                                              : 0;
        lanes[b]   = LaneOrder[b];
    }
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{
        0, static_cast<std::uint32_t>(envelope_max)};
    const auto capacity = ops::sliding_window_attention_workspace_capacity_bytes(
        Geometry, cache.window, envelope, width, width, batch);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    DeviceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 1)});
    GuardedDeviceBuffer dq(q_count * 2), dk(kv_count * 2), dv(kv_count * 2),
        dp(positions.size() * 4), dvalid(batch * 4), dlane(batch * 4), output(q_count * 2);
    Tensor tq(dq.data(), DType::BF16, {D, Hq, width, batch}),
        tk(dk.data(), DType::BF16, {D, Hkv, width, batch}),
        tv(dv.data(), DType::BF16, {D, Hkv, width, batch});
    Tensor tp(dp.data(), DType::I32, {width, batch}), tvalid(dvalid.data(), DType::I32, {batch}),
        tlane(dlane.data(), DType::I32, {batch}),
        to(output.data(), DType::BF16, {D, Hq, width, batch});
    auto context = cache.view();
    DeviceContext device;
    const auto launch = [&] {
        ops::sliding_window_attention(tq, tk, tv, tp, tvalid, tlane, Geometry, cache.window, Scale,
                                      context, envelope, workspace, to, device.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase)
            for (int b = 0; b < batch; ++b) {
                lengths[b] = (b % 2 == 0 ? 17 : lengths[b] + 17) % (envelope_max + 1);
                lanes[b]   = LaneOrder[(b + 3) % 8];
                if (prefix != Prefix::Zero) valid[b] = valid[b] == 0 ? width : valid[b] - 1;
            }
        for (int b = 0; b < batch; ++b)
            for (int t = 0; t < width; ++t)
                positions[b * width + t] = t < valid[b] ? lengths[b] + t : -1234567;
        auto q_bits = q_base, k_bits = k_base, v_bits = v_base;
        auto value = qv;
        if (phase) {
            for (auto& bits : v_bits) bits ^= 0x8000;
            for (auto& x : value) x = -x;
        }
        // Inert physical tails may contain arbitrary bits, including NaNs.
        for (int b = 0; b < batch; ++b)
            for (int t = valid[b]; t < width; ++t) {
                std::fill_n(q_bits.begin() + (static_cast<std::size_t>(b) * width + t) * Hq * D,
                            Hq * D, 0x7fc1);
                std::fill_n(k_bits.begin() + (static_cast<std::size_t>(b) * width + t) * Hkv * D,
                            Hkv * D, 0x7fc1);
                std::fill_n(v_bits.begin() + (static_cast<std::size_t>(b) * width + t) * Hkv * D,
                            Hkv * D, 0x7fc1);
            }
        dq.copy_from_host(q_bits.data(), q_bits.size() * 2);
        dk.copy_from_host(k_bits.data(), k_bits.size() * 2);
        dv.copy_from_host(v_bits.data(), v_bits.size() * 2);
        dp.copy_from_host(positions.data(), positions.size() * 4);
        dvalid.copy_from_host(valid.data(), batch * 4);
        dlane.copy_from_host(lanes.data(), batch * 4);
        output.fill(0xff);
        scratch.fill(0xa7);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(device.stream);
        else
            launch();
        cuda_synchronize(device.stream);
        const auto expected = oracle(cache, width, batch, q, qk, value, positions, valid, lanes);
        const std::string label = "SWA window=" + std::to_string(cache.window) +
                                  " W=" + std::to_string(width) + " B=" + std::to_string(batch) +
                                  " L=" + std::to_string(base_context) +
                                  " envelope=" + std::to_string(envelope_max) +
                                  (replay ? " graph " : " eager ") + std::to_string(phase);
        failures +=
            verify_reduction(label, from_device_bf16(output.data(), q_count), expected, Criterion);
        const auto out_bits = from_device<std::uint16_t>(output.data(), q_count);
        bool zero           = true;
        for (int b = 0; b < batch; ++b)
            for (int t = valid[b]; t < width; ++t)
                for (int i = 0; i < Hq * D; ++i)
                    zero = zero &&
                           out_bits[(static_cast<std::size_t>(b) * width + t) * Hq * D + i] == 0;
        if (!zero) {
            std::cerr << label << ": inert tail is not exact zero\n";
            ++failures;
        }
        failures += output.verify_guards(label) + scratch.verify_guards(label + " scratch");
        if (workspace.used() != 0 || workspace.peak_used() != capacity) {
            std::cerr << label << ": workspace query/scope mismatch\n";
            ++failures;
        }
        if (observed) *observed = std::max(*observed, workspace.peak_used());
        failures += verify_exact((label + " q unchanged").c_str(),
                                 from_device<std::uint16_t>(dq.data(), q_count), q_bits);
        failures += verify_exact((label + " query K unchanged").c_str(),
                                 from_device<std::uint16_t>(dk.data(), kv_count), k_bits);
        failures += verify_exact((label + " query V unchanged").c_str(),
                                 from_device<std::uint16_t>(dv.data(), kv_count), v_bits);
        failures += verify_exact((label + " positions unchanged").c_str(),
                                 from_device<int>(dp.data(), positions.size()), positions);
        failures += verify_exact((label + " valid unchanged").c_str(),
                                 from_device<int>(dvalid.data(), batch), valid);
        failures += verify_exact((label + " lanes unchanged").c_str(),
                                 from_device<int>(dlane.data(), batch), lanes);
    }
    failures += dq.verify_guards("Q guards") + dk.verify_guards("query K guards") +
                dv.verify_guards("query V guards") + dp.verify_guards("positions guards") +
                dvalid.verify_guards("valid guards") + dlane.verify_guards("lanes guards");
    return failures;
}

int qualify() {
    int failures = 0;
    Cache cache(2048);
    for (int batch = 1; batch <= 8; ++batch) {
        const int maximum = batch % 2 ? 96 : 2048;
        std::array<std::size_t, 17> peaks{};
        for (int width = 1; width <= 16; ++width) {
            failures += run_case(cache, width, batch, -1, maximum, Prefix::Mixed,
                                 batch == 1 || batch == 8, &peaks[width]);
        }
        for (const auto [lo, hi] : {std::pair{1, 16}, std::pair{4, 5}, std::pair{8, 9}}) {
            const auto peak = *std::max_element(peaks.begin() + lo, peaks.begin() + hi + 1);
            if (peak !=
                ops::sliding_window_attention_workspace_capacity_bytes(
                    Geometry, 2048, {0, static_cast<std::uint32_t>(maximum)}, lo, hi, batch)) {
                std::cerr << "SWA interval capacity differs from observed peak\n";
                ++failures;
            }
        }
    }

    for (int width : {2, 3, 8, 16}) {
        failures += run_case(cache, width, 8, -1, 96, Prefix::Mixed, true);
        failures += run_case(cache, width, 1, -1, 2048, Prefix::Mixed, true);
    }
    for (int length : {64, 96, 97, 128})
        for (int batch : {1, 8})
            failures += run_case(cache, 3, batch, length, length, Prefix::Mixed, false);
    for (int width : {2, 8, 16})
        for (int batch : {1, 8}) {
            const int length = width == 16 ? 262144 : width == 8 ? 4103 : 2048;
            failures += run_case(cache, width, batch, length, 262144, Prefix::Mixed, true);
        }
    for (int width : {1, 3, 16})
        for (int batch : {1, 8})
            failures += run_case(cache, width, batch, 2048, 262144, Prefix::Zero, true);
    // Every width/CTA and batch dispatch boundary with full active split work.
    for (int width : {4, 5, 8, 9, 16})
        for (int batch = 1; batch <= 8; ++batch)
            failures += run_case(cache, width, batch, 2048, 2048, Prefix::Mixed, false);
    for (int width : {8, 9, 16})
        for (int batch : {1, 5, 6, 8})
            for (int length : {96, 97, 128, 129})
                failures += run_case(cache, width, batch, length, length, Prefix::Mixed, false);
    for (int width : {8, 9, 16})
        for (int batch : {2, 4, 6})
            for (int length : {511, 512, 513, 1024})
                failures += run_case(cache, width, batch, length, length, Prefix::Mixed, false);
    failures += cache.verify();
    Cache boundary(2048, true);
    for (int width : {2, 3, 16})
        failures += run_case(boundary, width, width == 3 ? 8 : 1, 2048, 2048, Prefix::Full, true);
    failures += boundary.verify();
    {
        Cache legacy(4096);
        failures += run_case(legacy, 1, 1, 0, 0, Prefix::Full, false);
        failures += run_case(legacy, 16, 1, 1, 1, Prefix::Full, false);
        failures += run_case(legacy, 8, 1, 96, 4096, Prefix::Full, true);
        failures += run_case(legacy, 16, 1, 4096, 4096, Prefix::Full, true);
        failures += run_case(legacy, 2, 2, 8194, 8194, Prefix::Mixed, true);
        failures += run_case(legacy, 8, 1, 65, 96, Prefix::Zero, true);
        failures += run_case(legacy, 8, 1, 4096, 4096, Prefix::Zero, true);
        failures += legacy.verify();
        Cache edge(4096, true);
        failures += run_case(edge, 2, 1, 4096, 4096, Prefix::Full, true);
        failures += edge.verify();
    }
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    try {
        const int failures = qualify();
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " sliding_window_attention\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "sliding_window_attention: " << error.what() << '\n';
        return 1;
    }
}
