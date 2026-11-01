// Two-device stream-split hyper-connection mix (ninfer/ops/tp2/hyper_connection.h, T30) at the
// Qwen3.8-Flash-Next geometry (4 streams of 2560, low rank 320, Q8_G32_FP16 Down/Up).
//
// Rank r holds streams [2r, 2r+2): its half of the residual, the norm rows, the Down|inject
// columns and the Up rows of those streams. Each shard is packed from the column/row block of the
// same logical FP32 matrix as the whole weight; Q8 row-split groups are 32 columns inside each
// row, and the column cut at 5120 is a whole number of groups, so the shard codes and scales equal
// the parent's slice (checked through the packer's dequantized values).
//
// Oracles and criteria (fixed before the first run):
//  * FP64 formula of hyper_connection_mix (test_hyper_connection.cpp's oracle over the whole
//    residual and the whole decoded weights), with that suite's criteria: mixed normwise relative
//    L2 4e-3, gross 1e-4 + 1.6e-2 * max; inject absolute 3e-3. The split adds one BF16 rounding of
//    each rank's p partial (the projection is a BF16 Linear output on each rank, then an FP32 sum)
//    on top of the general route's single BF16 p: the inject bound's derivation (|dgate| <= |dp|/8,
//    BF16 p) covers two such roundings within its 2x margin.
//  * against the single-device hyper_connection_mix of the whole weights on device 0: within twice
//    the oracle criterion (both qualified against the same oracle), as the 27B parity suites.
//  * both ranks' mixed outputs are bit-identical; rank r's inject equals gates [2r, 2r+2) of the
//    oracle; the residual is read only.
//  * combine over the split residual (hyper_connection_combine on each rank with its gates) equals
//    the whole combine's rows bit for bit.
//  * captured into a two-device graph on the mailbox: replays bit-identical to the eager call.
// --bench: per-mixer cost of the split mix (two exchanges) inside one two-device graph of 24
// mixers against the replicated single-device mix (both GPUs, concurrently), T = 1, 4, 8.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ninfer/ops/tp2/hyper_connection.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/split_test_support.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStreams = 4;
constexpr std::int32_t kHidden  = 2560;
constexpr std::int32_t kWide    = kStreams * kHidden;
constexpr std::int32_t kLocal   = kWide / 2;
constexpr std::int32_t kRank    = 320;
constexpr float kEps            = 1.0e-6F;

constexpr ReductionCriterion kMixed{4.0e-3, 1.0e-4, 1.6e-2};
constexpr ReductionCriterion kMixedParity{8.0e-3, 2.0e-4, 3.2e-2};
constexpr PointwiseCriterion kInject{3.0e-3, 0.0};
constexpr PointwiseCriterion kInjectParity{6.0e-3, 0.0};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
    return out;
}

std::vector<float> random_bf16(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

std::vector<float> columns_of(const std::vector<float>& m, int rows, int cols, int first,
                              int count) {
    std::vector<float> out(static_cast<std::size_t>(rows) * count);
    for (int r = 0; r < rows; ++r) {
        std::copy_n(m.begin() + static_cast<std::ptrdiff_t>(r) * cols + first, count,
                    out.begin() + static_cast<std::ptrdiff_t>(r) * count);
    }
    return out;
}

struct DeviceWeights {
    DeviceBuffer norm, down, up;
    ops::HyperConnectionWeights weights;
};

struct Fixture {
    bool inject;
    std::int32_t rows;
    std::vector<float> norm, down_source, up_source;
    quantized_weight::PackedWeight down, up; // whole
    std::array<quantized_weight::PackedWeight, 2> down_r, up_r;
    std::array<std::vector<float>, 2> norm_r;
    DeviceWeights whole;                // device 0
    std::array<DeviceWeights, 2> shard; // device r

    Fixture(const ExecutionContext& ec, bool with_inject, std::uint32_t seed)
        : inject(with_inject), rows(kRank + (with_inject ? kStreams : 0)) {
        norm = random_bf16(kWide, seed, -0.125F, 0.125F);
        down_source.resize(static_cast<std::size_t>(rows) * kWide);
        fill_uniform(down_source, seed + 1U, -0.04F, 0.04F);
        for (std::size_t i = static_cast<std::size_t>(kRank) * kWide; i < down_source.size(); ++i)
            down_source[i] *= 0.5F;
        up_source.resize(static_cast<std::size_t>(kWide) * kRank);
        fill_uniform(up_source, seed + 2U, -0.25F, 0.25F);
        down = quantized_weight::pack_q8_g32_row_split(down_source, rows, kWide);
        up   = quantized_weight::pack_q8_g32_row_split(up_source, kWide, kRank);
        for (int r = 0; r < 2; ++r) {
            down_r[r] = quantized_weight::pack_q8_g32_row_split(
                columns_of(down_source, rows, kWide, r * kLocal, kLocal), rows, kLocal);
            up_r[r] = quantized_weight::pack_q8_g32_row_split(
                std::vector<float>(
                    up_source.begin() + static_cast<std::ptrdiff_t>(r) * kLocal * kRank,
                    up_source.begin() + static_cast<std::ptrdiff_t>(r + 1) * kLocal * kRank),
                kLocal, kRank);
            norm_r[r] =
                std::vector<float>(norm.begin() + r * kLocal, norm.begin() + (r + 1) * kLocal);
        }
        set_device(ec, 0);
        whole = upload(norm, down, up, kWide);
        for (int r = 0; r < 2; ++r) {
            set_device(ec, r);
            shard[r] = upload(norm_r[r], down_r[r], up_r[r], kLocal);
        }
    }

    static DeviceWeights upload(const std::vector<float>& n,
                                const quantized_weight::PackedWeight& d,
                                const quantized_weight::PackedWeight& u, std::int32_t width) {
        DeviceWeights w;
        w.norm         = to_device(bits(n));
        w.down         = to_device(d.payload);
        w.up           = to_device(u.payload);
        w.weights.norm = Tensor(w.norm.p, DType::BF16, {width});
        w.weights.down = d.device_weight(w.down.p);
        w.weights.up   = u.device_weight(w.up.p);
        return w;
    }

    // The shard's decoded values equal the parent's block (the packer quantizes per 32-column
    // group of a row, and 5120 is a group boundary).
    int check_shards() const {
        int failures = 0;
        for (int r = 0; r < 2; ++r) {
            for (int row = 0; row < rows; ++row) {
                for (int k = 0; k < kLocal; k += 97) {
                    if (down_r[r].dequant[static_cast<std::size_t>(row) * kLocal + k] !=
                        down.dequant[static_cast<std::size_t>(row) * kWide + r * kLocal + k]) {
                        ++failures;
                    }
                }
            }
            for (int row = 0; row < kLocal; row += 13) {
                for (int k = 0; k < kRank; ++k) {
                    if (up_r[r].dequant[static_cast<std::size_t>(row) * kRank + k] !=
                        up.dequant[static_cast<std::size_t>(r * kLocal + row) * kRank + k]) {
                        ++failures;
                    }
                }
            }
        }
        if (failures != 0) { std::cerr << "shards differ from the parent blocks\n"; }
        return failures;
    }

    // FP64 oracle (test_hyper_connection.cpp) over the whole residual.
    void reference(const std::vector<float>& residual, std::int32_t tokens,
                   std::vector<double>& mixed, std::vector<double>& gates) const {
        mixed.assign(static_cast<std::size_t>(kHidden) * tokens, 0.0);
        gates.assign(static_cast<std::size_t>(kStreams) * tokens, 0.0);
        std::vector<double> n(kWide), low(kRank), p(rows);
        for (std::int32_t t = 0; t < tokens; ++t) {
            const float* x = residual.data() + static_cast<std::size_t>(t) * kWide;
            for (std::int32_t s = 0; s < kStreams; ++s) {
                double sum = 0.0;
                for (std::int32_t h = 0; h < kHidden; ++h)
                    sum += double(x[s * kHidden + h]) * x[s * kHidden + h];
                const double inv = 1.0 / std::sqrt(sum / kHidden + double(kEps));
                for (std::int32_t h = 0; h < kHidden; ++h)
                    n[s * kHidden + h] = x[s * kHidden + h] * inv * (1.0 + norm[s * kHidden + h]);
            }
            for (std::int32_t r = 0; r < rows; ++r) {
                const float* w = down.dequant.data() + static_cast<std::size_t>(r) * kWide;
                double sum     = 0.0;
                for (std::int32_t k = 0; k < kWide; ++k) sum += double(w[k]) * n[k];
                p[r] = sum / kStreams;
            }
            for (std::int32_t r = 0; r < kRank; ++r) low[r] = p[r] * sigmoid(p[r]);
            for (std::int32_t s = 0; inject && s < kStreams; ++s)
                gates[static_cast<std::size_t>(t) * kStreams + s] = 2.0 * sigmoid(p[kRank + s]);
            for (std::int32_t h = 0; h < kHidden; ++h) {
                double value = 0.0;
                for (std::int32_t s = 0; s < kStreams; ++s) {
                    const std::int32_t row = s * kHidden + h;
                    const float* w = up.dequant.data() + static_cast<std::size_t>(row) * kRank;
                    double logit   = 0.0;
                    for (std::int32_t r = 0; r < kRank; ++r) logit += double(w[r]) * low[r];
                    value += sigmoid(logit) * n[row];
                }
                mixed[static_cast<std::size_t>(t) * kHidden + h] = value / kStreams;
            }
        }
    }
};

// Per-call device buffers of both ranks and of the single-device run.
struct Call {
    std::int32_t tokens;
    std::vector<float> residual; // whole [kWide, T]
    DeviceBuffer whole_residual;
    std::unique_ptr<GuardedDeviceBuffer> whole_mixed, whole_inject;
    std::array<DeviceBuffer, 2> residual_r;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> mixed_r, inject_r;
    std::array<std::unique_ptr<WorkspaceArena>, 2> workspace;
    std::unique_ptr<WorkspaceArena> whole_workspace;
    std::array<Tensor, 2> residual_t, mixed_t;
    std::array<Tensor, 2> inject_tensors;
    std::array<Tensor*, 2> inject_t{nullptr, nullptr};

    Call(const ExecutionContext& ec, const Fixture& f, std::int32_t t, std::uint32_t seed)
        : tokens(t) {
        residual = random_bf16(static_cast<std::size_t>(kWide) * t, seed, -0.75F, 0.75F);
        set_device(ec, 0);
        whole_residual = to_device(bits(residual));
        whole_mixed =
            std::make_unique<GuardedDeviceBuffer>(static_cast<std::size_t>(kHidden) * t * 2);
        whole_inject =
            std::make_unique<GuardedDeviceBuffer>(static_cast<std::size_t>(kStreams) * t * 4);
        whole_workspace = std::make_unique<WorkspaceArena>(
            ops::hyper_connection_mix_workspace_bytes(t, t, f.inject, kHidden, kRank));
        for (int r = 0; r < 2; ++r) {
            set_device(ec, r);
            std::vector<float> half(static_cast<std::size_t>(kLocal) * t);
            for (int c = 0; c < t; ++c) {
                std::copy_n(residual.begin() + static_cast<std::ptrdiff_t>(c) * kWide + r * kLocal,
                            kLocal, half.begin() + static_cast<std::ptrdiff_t>(c) * kLocal);
            }
            residual_r[r] = to_device(bits(half));
            mixed_r[r] =
                std::make_unique<GuardedDeviceBuffer>(static_cast<std::size_t>(kHidden) * t * 2);
            inject_r[r] =
                std::make_unique<GuardedDeviceBuffer>(static_cast<std::size_t>(2) * t * 4);
            mixed_r[r]->fill(0xff);
            inject_r[r]->fill(0xff);
            workspace[r] = std::make_unique<WorkspaceArena>(
                ops::hyper_connection_mix_split_workspace_bytes(t, t, f.inject));
            residual_t[r]     = Tensor(residual_r[r].p, DType::BF16, {kLocal, t});
            mixed_t[r]        = Tensor(mixed_r[r]->data(), DType::BF16, {kHidden, t});
            inject_tensors[r] = Tensor(inject_r[r]->data(), DType::FP32, {2, t});
            inject_t[r]       = f.inject ? &inject_tensors[r] : nullptr;
        }
        retire_staging(ec);
    }

    void run_split(const ExecutionContext& ec, const Fixture& f, const ops::PeerEvents& events) {
        ops::hyper_connection_mix_split(residual_t, {f.shard[0].weights, f.shard[1].weights}, kEps,
                                        mixed_t, inject_t, {workspace[0].get(), workspace[1].get()},
                                        ec, events);
    }

    void run_whole(const ExecutionContext& ec, const Fixture& f) {
        set_device(ec, 0);
        Tensor r(whole_residual.p, DType::BF16, {kWide, tokens});
        Tensor m(whole_mixed->data(), DType::BF16, {kHidden, tokens});
        Tensor i(whole_inject->data(), DType::FP32, {kStreams, tokens});
        ops::hyper_connection_mix(r, f.whole.weights, kEps, m, f.inject ? &i : nullptr,
                                  *whole_workspace, ec.dev[0]->stream);
        cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "whole mix");
    }
};

std::vector<double> read_f32(const void* device, std::size_t n) {
    const auto v = from_device<float>(device, n);
    return {v.begin(), v.end()};
}

struct Observed {
    std::array<std::vector<std::uint16_t>, 2> mixed;
    std::array<std::vector<float>, 2> inject;
};

Observed observe(const ExecutionContext& ec, Call& call, int& failures, const std::string& label) {
    Observed o;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * call.tokens;
    for (int r = 0; r < 2; ++r) {
        set_device(ec, r);
        o.mixed[r] = from_device<std::uint16_t>(call.mixed_r[r]->data(), narrow);
        o.inject[r] =
            from_device<float>(call.inject_r[r]->data(), static_cast<std::size_t>(2) * call.tokens);
        failures += call.mixed_r[r]->verify_guards(label + " mixed rank " + std::to_string(r));
        failures += call.inject_r[r]->verify_guards(label + " inject rank " + std::to_string(r));
    }
    return o;
}

int check_call(const ExecutionContext& ec, const Fixture& f, Call& call, const Observed& o,
               const std::string& label) {
    int failures             = 0;
    const std::int32_t T     = call.tokens;
    const std::size_t narrow = static_cast<std::size_t>(kHidden) * T;
    std::vector<double> mixed_ref, gates_ref;
    f.reference(call.residual, T, mixed_ref, gates_ref);
    failures += verify_exact((label + " ranks agree").c_str(), o.mixed[1], o.mixed[0]);
    std::vector<double> mixed(narrow);
    for (std::size_t i = 0; i < narrow; ++i) { mixed[i] = bf16_to_f32(o.mixed[0][i]); }
    failures += verify_reduction(label + " mixed vs FP64", mixed, mixed_ref, kMixed);
    call.run_whole(ec, f);
    set_device(ec, 0);
    const auto whole_mixed = from_device_bf16(call.whole_mixed->data(), narrow);
    failures += verify_reduction(label + " mixed vs 1 GPU", mixed, whole_mixed, kMixedParity);
    if (f.inject) {
        const auto whole_gates =
            read_f32(call.whole_inject->data(), static_cast<std::size_t>(kStreams) * T);
        for (int r = 0; r < 2; ++r) {
            std::vector<double> got, ref, whole;
            for (int t = 0; t < T; ++t) {
                for (int s = 0; s < 2; ++s) {
                    got.push_back(o.inject[r][static_cast<std::size_t>(t) * 2 + s]);
                    ref.push_back(gates_ref[static_cast<std::size_t>(t) * kStreams + 2 * r + s]);
                    whole.push_back(
                        whole_gates[static_cast<std::size_t>(t) * kStreams + 2 * r + s]);
                }
            }
            failures += verify_pointwise(label + " inject rank " + std::to_string(r) + " vs FP64",
                                         got, ref, kInject);
            failures += verify_pointwise(label + " inject rank " + std::to_string(r) + " vs 1 GPU",
                                         got, whole, kInjectParity);
        }
    }
    // The residual is read only.
    for (int r = 0; r < 2; ++r) {
        set_device(ec, r);
        std::vector<float> half(static_cast<std::size_t>(kLocal) * T);
        for (int c = 0; c < T; ++c) {
            std::copy_n(call.residual.begin() + static_cast<std::ptrdiff_t>(c) * kWide + r * kLocal,
                        kLocal, half.begin() + static_cast<std::ptrdiff_t>(c) * kLocal);
        }
        failures +=
            verify_exact((label + " residual preserved").c_str(),
                         from_device<std::uint16_t>(call.residual_r[r], half.size()), bits(half));
    }
    return failures;
}

// Captures one split mix into a two-device graph; `composed` forces the composed route
// (NINFER_HC_SPLIT_ROUTE=composed while capturing). Returns the graph's node count.
std::size_t capture_split(const ExecutionContext& ec, const Fixture& f, Call& call,
                          const ops::PeerEvents& events, bool composed,
                          DecodeGraphExecutable& executable) {
    if (composed) { ::setenv("NINFER_HC_SPLIT_ROUTE", "composed", 1); }
    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    set_device(ec, 0);
    definition.capture(
        ec.dev[0]->stream,
        [&] {
            call.run_split(ec, f, events);
            set_device(ec, 0);
        },
        DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
    if (composed) { ::unsetenv("NINFER_HC_SPLIT_ROUTE"); }
    executable.instantiate(definition);
    return definition.node_count();
}

Observed replay(const ExecutionContext& ec, Call& call, DecodeGraphExecutable& executable,
                int& failures, const std::string& label) {
    for (int r = 0; r < 2; ++r) {
        set_device(ec, r);
        call.mixed_r[r]->fill(0xff);
        call.inject_r[r]->fill(0xff);
    }
    retire_staging(ec);
    set_device(ec, 0);
    executable.launch(ec.dev[0]->stream);
    synchronize_both(ec);
    return observe(ec, call, failures, label);
}

// The head form has no inject output (the buffer keeps its NaN fill): compare mixed only.
int same(const Observed& a, const Observed& b, const std::string& label, bool inject) {
    int failures = 0;
    for (int r = 0; r < 2; ++r) {
        failures += verify_exact((label + " mixed").c_str(), a.mixed[r], b.mixed[r]);
        if (inject) {
            failures += verify_exact((label + " inject").c_str(), a.inject[r], b.inject[r]);
        }
    }
    return failures;
}

// Eager (composed route, staged exchanges); captured on the mailbox with the default route (the
// fused kernel for T <= 8: two nodes, one per device) replayed twice (replays identical, each
// judged like the eager call); captured with the composed route forced (equal to eager bit for
// bit: same kernels, the mailbox and the staged path add identically).
int run_case(const ExecutionContext& ec, const Fixture& f, std::int32_t T, std::uint32_t seed,
             const ops::PeerEvents& staged, const ops::PeerEvents& mailbox_events) {
    const std::string label =
        std::string("hc split ") + (f.inject ? "inject" : "head") + " T=" + std::to_string(T);
    int failures = 0;
    Call call(ec, f, T, seed);
    call.run_split(ec, f, staged);
    synchronize_both(ec);
    const Observed eager = observe(ec, call, failures, label + " eager");
    failures += check_call(ec, f, call, eager, label + " eager composed");

    DecodeGraphExecutable fused;
    const std::size_t nodes = capture_split(ec, f, call, mailbox_events, false, fused);
    const bool expect_fused = T <= 8;
    if (expect_fused && nodes != 2) {
        std::cerr << label << ": " << nodes << " graph nodes, expected the fused route's 2\n";
        ++failures;
    }
    std::cout << label << ": default captured route " << (nodes == 2 ? "fused" : "composed") << " ("
              << nodes << " nodes)\n";
    const Observed first = replay(ec, call, fused, failures, label + " graph");
    failures += check_call(ec, f, call, first, label + (expect_fused ? " graph fused" : " graph"));
    const Observed second = replay(ec, call, fused, failures, label + " graph replay 2");
    failures += same(second, first, label + " graph replays agree", f.inject);
    if (!expect_fused) { failures += same(first, eager, label + " graph == eager", f.inject); }

    DecodeGraphExecutable composed;
    (void)capture_split(ec, f, call, mailbox_events, true, composed);
    const Observed forced = replay(ec, call, composed, failures, label + " graph composed");
    failures += same(forced, eager, label + " graph composed == eager", f.inject);
    return failures;
}

// combine_mix_split over the whole oracle: the committed residual within one BF16 rounding of the
// whole combine's, mixed against the FP64 oracle of the committed residual and against the
// single-device combine_mix; eager (composed) and captured (fused for T <= 8).
int combine_mix_case(const ExecutionContext& ec, const Fixture& f, std::int32_t T,
                     std::uint32_t seed, const ops::PeerEvents& staged,
                     const ops::PeerEvents& mailbox_events) {
    const std::string label = "hc split combine_mix T=" + std::to_string(T);
    int failures            = 0;
    const auto y = random_bf16(static_cast<std::size_t>(kHidden) * T, seed + 1U, -0.5F, 0.5F);
    std::vector<float> gates(static_cast<std::size_t>(kStreams) * T);
    fill_uniform(gates, seed + 2U, 0.0F, 2.0F);
    for (const bool graph : {false, true}) {
        Call call(ec, f, T, seed);
        // Single device: combine_mix of the whole residual.
        set_device(ec, 0);
        DeviceBuffer dy0 = to_device(bits(y)), dg0 = to_device(gates);
        DeviceBuffer committed_whole = to_device(bits(call.residual));
        {
            Tensor r(committed_whole.p, DType::BF16, {kWide, T});
            Tensor m(call.whole_mixed->data(), DType::BF16, {kHidden, T});
            Tensor i(call.whole_inject->data(), DType::FP32, {kStreams, T});
            ops::hyper_connection_combine_mix(r, Tensor(dy0.p, DType::BF16, {kHidden, T}),
                                              Tensor(dg0.p, DType::FP32, {kStreams, T}),
                                              f.whole.weights, kEps, m, f.inject ? &i : nullptr,
                                              *call.whole_workspace, ec.dev[0]->stream);
            cuda_check(cudaStreamSynchronize(ec.dev[0]->stream), "whole combine_mix");
        }
        const auto whole_mixed =
            from_device_bf16(call.whole_mixed->data(), static_cast<std::size_t>(kHidden) * T);
        // The exact combine (FP64) of the residual, rounded once: the oracle's input.
        std::vector<float> committed(call.residual.size());
        for (int c = 0; c < T; ++c)
            for (int s = 0; s < kStreams; ++s)
                for (int h = 0; h < kHidden; ++h) {
                    const std::size_t i = static_cast<std::size_t>(c) * kWide + s * kHidden + h;
                    committed[i]        = bf16_to_f32(f32_to_bf16(static_cast<float>(
                        double(call.residual[i]) +
                        double(gates[static_cast<std::size_t>(c) * kStreams + s]) *
                            double(y[static_cast<std::size_t>(c) * kHidden + h]))));
                }
        std::array<DeviceBuffer, 2> dy, dg;
        std::array<Tensor, 2> ty, tg;
        for (int r = 0; r < 2; ++r) {
            set_device(ec, r);
            std::vector<float> rank_gates(static_cast<std::size_t>(2) * T);
            for (int c = 0; c < T; ++c)
                for (int s = 0; s < 2; ++s)
                    rank_gates[static_cast<std::size_t>(c) * 2 + s] =
                        gates[static_cast<std::size_t>(c) * kStreams + 2 * r + s];
            dy[r] = to_device(bits(y));
            dg[r] = to_device(rank_gates);
            ty[r] = Tensor(dy[r].p, DType::BF16, {kHidden, T});
            tg[r] = Tensor(dg[r].p, DType::FP32, {2, T});
        }
        retire_staging(ec);
        const auto run = [&](const ops::PeerEvents& events) {
            ops::hyper_connection_combine_mix_split(
                call.residual_t, ty, tg, {f.shard[0].weights, f.shard[1].weights}, kEps,
                call.mixed_t, call.inject_t, {call.workspace[0].get(), call.workspace[1].get()}, ec,
                events);
        };
        if (graph) {
            // The devices were prepared by the eager calls of run_case: the fused route for T <= 8.
            const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
            DecodeGraphDefinition definition;
            set_device(ec, 0);
            definition.capture(
                ec.dev[0]->stream,
                [&] {
                    run(mailbox_events);
                    set_device(ec, 0);
                },
                DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
            DecodeGraphExecutable executable;
            executable.instantiate(definition);
            set_device(ec, 0);
            executable.launch(ec.dev[0]->stream);
        } else {
            run(staged);
        }
        synchronize_both(ec);
        const std::string mode = label + (graph ? " graph" : " eager");
        std::vector<double> mixed_ref, gates_ref;
        f.reference(committed, T, mixed_ref, gates_ref);
        set_device(ec, 0);
        const auto mixed0 =
            from_device_bf16(call.mixed_r[0]->data(), static_cast<std::size_t>(kHidden) * T);
        failures += verify_reduction(mode + " mixed vs FP64", mixed0, mixed_ref, kMixed);
        failures += verify_reduction(mode + " mixed vs 1 GPU", mixed0, whole_mixed, kMixedParity);
        for (int r = 0; r < 2; ++r) {
            set_device(ec, r);
            std::vector<double> got =
                from_device_bf16(call.residual_r[r], static_cast<std::size_t>(kLocal) * T);
            std::vector<double> want(got.size());
            for (int c = 0; c < T; ++c)
                for (int i = 0; i < kLocal; ++i)
                    want[static_cast<std::size_t>(c) * kLocal + i] =
                        double(
                            call.residual[static_cast<std::size_t>(c) * kWide + r * kLocal + i]) +
                        double(
                            gates[static_cast<std::size_t>(c) * kStreams + 2 * r + i / kHidden]) *
                            double(y[static_cast<std::size_t>(c) * kHidden + i % kHidden]);
            failures += verify_pointwise(mode + " committed residual rank " + std::to_string(r),
                                         got, want, PointwiseCriterion{1.0e-6, 1.0 / 128.0});
            const auto m = from_device<std::uint16_t>(call.mixed_r[r]->data(),
                                                      static_cast<std::size_t>(kHidden) * T);
            set_device(ec, 0);
            failures += verify_exact((mode + " ranks agree").c_str(), m,
                                     from_device<std::uint16_t>(call.mixed_r[0]->data(), m.size()));
        }
    }
    return failures;
}

// Combine over the split residual equals the whole combine's rows bit for bit.
int combine_case(const ExecutionContext& ec, std::int32_t T, std::uint32_t seed) {
    const auto residual = random_bf16(static_cast<std::size_t>(kWide) * T, seed, -2.0F, 2.0F);
    const auto y = random_bf16(static_cast<std::size_t>(kHidden) * T, seed + 1U, -1.0F, 1.0F);
    std::vector<float> inject(static_cast<std::size_t>(kStreams) * T);
    fill_uniform(inject, seed + 2U, 0.0F, 2.0F);
    set_device(ec, 0);
    DeviceBuffer whole = to_device(bits(residual)), dy0 = to_device(bits(y)),
                 di = to_device(inject);
    Tensor tw(whole.p, DType::BF16, {kWide, T});
    ops::hyper_connection_combine(Tensor(dy0.p, DType::BF16, {kHidden, T}),
                                  Tensor(di.p, DType::FP32, {kStreams, T}), tw, nullptr);
    cuda_check(cudaDeviceSynchronize(), "whole combine");
    const auto whole_bits = from_device<std::uint16_t>(whole, residual.size());
    int failures          = 0;
    for (int r = 0; r < 2; ++r) {
        set_device(ec, r);
        std::vector<float> half(static_cast<std::size_t>(kLocal) * T),
            gates(static_cast<std::size_t>(2) * T);
        std::vector<std::uint16_t> want(half.size());
        for (int c = 0; c < T; ++c) {
            for (int i = 0; i < kLocal; ++i) {
                half[static_cast<std::size_t>(c) * kLocal + i] =
                    residual[static_cast<std::size_t>(c) * kWide + r * kLocal + i];
                want[static_cast<std::size_t>(c) * kLocal + i] =
                    whole_bits[static_cast<std::size_t>(c) * kWide + r * kLocal + i];
            }
            gates[static_cast<std::size_t>(c) * 2] =
                inject[static_cast<std::size_t>(c) * kStreams + 2 * r];
            gates[static_cast<std::size_t>(c) * 2 + 1] =
                inject[static_cast<std::size_t>(c) * kStreams + 2 * r + 1];
        }
        DeviceBuffer dr = to_device(bits(half)), dy = to_device(bits(y)), dg = to_device(gates);
        Tensor tr(dr.p, DType::BF16, {kLocal, T});
        ops::hyper_connection_combine(Tensor(dy.p, DType::BF16, {kHidden, T}),
                                      Tensor(dg.p, DType::FP32, {2, T}), tr, nullptr);
        cuda_check(cudaDeviceSynchronize(), "split combine");
        failures += verify_exact(
            ("hc split combine T=" + std::to_string(T) + " rank " + std::to_string(r)).c_str(),
            from_device<std::uint16_t>(dr, half.size()), want);
    }
    return failures;
}

// Per-mixer cost: one graph of 24 split mixers (48 exchanges) vs one of 24 replicated mixes on
// each device (both devices, as at tp 2 with HC replicated). Median of 7 batches of 20 replays.
void bench(const ExecutionContext& ec, const Fixture& f) {
    constexpr int kMixers = 24, kBatches = 7, kReplays = 20;
    std::printf("bench: us per mixer (graph of %d mixers), median of %d x %d replays\n", kMixers,
                kBatches, kReplays);
    std::printf("  T  replicated(fused/general, both GPUs)  split(2 exchanges)\n");
    for (const std::int32_t T : {1, 4, 8}) {
        // The Program's slot: the widest exchange of the round, the collapse partial [2560, T].
        ops::PeerMailbox mailbox(ec, static_cast<std::size_t>(kHidden) * T * 4);
        ops::PeerEvents mailbox_events(ec);
        mailbox_events.attach_mailbox(&mailbox);
        Call call(ec, f, T, 0xBE00U + static_cast<std::uint32_t>(T));
        // Replicated: each device mixes the whole residual with its own copy of the weights.
        std::array<DeviceBuffer, 2> res, mix, inj;
        std::array<std::unique_ptr<WorkspaceArena>, 2> ws;
        std::array<DeviceBuffer, 2> norm, down, up;
        std::array<ops::HyperConnectionWeights, 2> weights;
        for (int r = 0; r < 2; ++r) {
            set_device(ec, r);
            res[r] = to_device(bits(call.residual));
            mix[r] = DeviceBuffer(static_cast<std::size_t>(kHidden) * T * 2);
            inj[r] = DeviceBuffer(static_cast<std::size_t>(kStreams) * T * 4);
            ws[r]  = std::make_unique<WorkspaceArena>(
                ops::hyper_connection_mix_workspace_bytes(T, T, true, kHidden, kRank));
            norm[r]         = to_device(bits(f.norm));
            down[r]         = to_device(f.down.payload);
            up[r]           = to_device(f.up.payload);
            weights[r].norm = Tensor(norm[r].p, DType::BF16, {kWide});
            weights[r].down = f.down.device_weight(down[r].p);
            weights[r].up   = f.up.device_weight(up[r].p);
        }
        retire_staging(ec);
        const auto time_graph = [&](const std::function<void()>& body) {
            const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
            DecodeGraphDefinition definition;
            set_device(ec, 0);
            definition.capture(
                ec.dev[0]->stream,
                [&] {
                    body();
                    set_device(ec, 0);
                },
                DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
            DecodeGraphExecutable executable;
            executable.instantiate(definition);
            for (int i = 0; i < 5; ++i) {
                set_device(ec, 0);
                executable.launch(ec.dev[0]->stream);
                synchronize_both(ec);
            }
            std::vector<double> per_mixer;
            for (int b = 0; b < kBatches; ++b) {
                const auto start = std::chrono::steady_clock::now();
                set_device(ec, 0);
                for (int i = 0; i < kReplays; ++i) { executable.launch(ec.dev[0]->stream); }
                synchronize_both(ec);
                const std::chrono::duration<double, std::micro> elapsed =
                    std::chrono::steady_clock::now() - start;
                per_mixer.push_back(elapsed.count() / (kReplays * kMixers));
            }
            std::sort(per_mixer.begin(), per_mixer.end());
            return per_mixer[per_mixer.size() / 2];
        };
        const double replicated = time_graph([&] {
            for (int m = 0; m < kMixers; ++m) {
                for (int r = 0; r < 2; ++r) {
                    set_device(ec, r);
                    Tensor tr(res[r].p, DType::BF16, {kWide, T});
                    Tensor tm(mix[r].p, DType::BF16, {kHidden, T});
                    Tensor ti(inj[r].p, DType::FP32, {kStreams, T});
                    ops::hyper_connection_mix(tr, weights[r], kEps, tm, &ti, *ws[r],
                                              ec.dev[r]->stream);
                }
            }
        });
        const double split      = time_graph([&] {
            for (int m = 0; m < kMixers; ++m) { call.run_split(ec, f, mailbox_events); }
        });
        std::printf("  %d  %8.2f  %8.2f\n", T, replicated, split);
    }
}

} // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        int devices = 0;
        cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        if (devices < 2) {
            std::cout << "SKIP: the split hyper-connection needs two CUDA devices\n";
            return 77;
        }
        const ExecutionContext ec({0, 1});
        (void)ops::enable_peer_access(ec);
        const ops::PeerEvents staged(ec);
        // Slot for the widest exchange: the collapse partial [2560, 17] FP32.
        ops::PeerMailbox mailbox(ec, static_cast<std::size_t>(kHidden) * 17 * 4);
        ops::PeerEvents mailbox_events(ec);
        mailbox_events.attach_mailbox(&mailbox);
        int failures = 0;
        const Fixture with_inject(ec, true, 0x9000U);
        const Fixture head(ec, false, 0xA000U);
        failures += with_inject.check_shards() + head.check_shards();
        for (const std::int32_t T : {1, 2, 4, 8, 9, 17}) {
            failures += run_case(ec, with_inject, T, 0x5000U + T, staged, mailbox_events);
        }
        for (const std::int32_t T : {1, 4}) {
            failures += run_case(ec, head, T, 0x7000U + T, staged, mailbox_events);
        }
        for (const std::int32_t T : {1, 4, 9}) { failures += combine_case(ec, T, 0x6000U + T); }
        for (const std::int32_t T : {1, 4, 9}) {
            failures += combine_mix_case(ec, with_inject, T, 0x6100U + T, staged, mailbox_events);
        }
        if (mailbox.hang_reported()) {
            std::cerr << "mailbox hang reported\n";
            ++failures;
        }
        if (do_bench) { bench(ec, with_inject); }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " hyper_connection_split\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "hyper_connection_split: " << error.what() << '\n';
        return 1;
    }
}
