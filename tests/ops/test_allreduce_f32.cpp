// Two-device qualification of allreduce_sum_f32 (include/ninfer/ops/allreduce.h): FP32 partials
// summed across the ranks and rounded to BF16 once, the exchange of the Qwen4Exp MoE partials.
//
// Oracle: exact. The Op's arithmetic is defined bit for bit: sum = fl32(a + b), one IEEE
// round-to-nearest-even FP32 addition, and bf16 = RNE(sum). The host evaluates both with its own
// IEEE float addition and the test's RNE encoder, so every FP32 and BF16 output bit is compared.
// Both ranks must hold the same bits (the addition is commutative).
//
// Transports: the staged path eagerly and captured, the pipelined mailbox captured (replayed with
// fresh inputs, back-to-back launches without host steps), the Legacy mailbox (which has no FP32
// form and must stay staged inside the graph), FP32 payloads (with and without the BF16 output)
// interleaved with BF16 exchanges so that one mailbox slot carries both in one capture, and node
// counts proving which transport each capture selected.
//
// --bench: per-exchange cost, inside a graph of 96 exchanges, of the BF16 payload [2560,T] (the
// attention/GDN output all-reduce) and the FP32 payload [2560,T] (the MoE partials) at T = 1, 2,
// 4, 8 on a mailbox whose slot fits the FP32 payload (the Program's sizing). Informative.
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/peer_mailbox.h"
#include "ops/op_tester.h"
#include "ops/split_test_support.h"

#include "core/decode_graph.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden = 2560;

std::vector<std::uint32_t> f32_bits(const std::vector<float>& values) {
    std::vector<std::uint32_t> bits(values.size());
    std::memcpy(bits.data(), values.data(), values.size() * sizeof(float));
    return bits;
}

// Partials of a realistic spread: most elements O(1), some tiny and some large, and exact
// cancellations, so the FP32 sum and the BF16 rounding see every regime.
std::vector<float> make_partial(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -4.0F, 4.0F);
    for (std::size_t i = 0; i < count; i += 7) { values[i] *= 1.0e-5F; }
    for (std::size_t i = 3; i < count; i += 11) { values[i] *= 3.0e3F; }
    return values;
}

struct Expected {
    std::vector<std::uint32_t> sum;
    std::vector<std::uint16_t> bf16;
};

Expected oracle(const std::vector<float>& a, const std::vector<float>& b, double scale) {
    Expected out;
    std::vector<float> sum(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        // One IEEE float addition (x86-64 SSE, round to nearest even), then an exact power of two.
        const volatile float s = a[i] + b[i];
        sum[i]                 = static_cast<float>(static_cast<double>(s) * scale);
    }
    out.sum = f32_bits(sum);
    out.bf16.resize(sum.size());
    for (std::size_t i = 0; i < sum.size(); ++i) { out.bf16[i] = f32_to_bf16(sum[i]); }
    return out;
}

void capture_two_devices(const ExecutionContext& ec, const DecodeGraphPeerBridge& bridge,
                         DecodeGraphDefinition& definition, const std::function<void()>& body) {
    set_device(ec, 0);
    definition.capture(
        ec.dev[0]->stream,
        [&] {
            body();
            set_device(ec, 0);
        },
        DecodeGraphPeerCapture{.bridge = &bridge, .stream = ec.dev[1]->stream});
}

void launch_two_devices(const ExecutionContext& ec, DecodeGraphExecutable& executable) {
    set_device(ec, 0);
    executable.launch(ec.dev[0]->stream);
    synchronize_both(ec);
}

// Per-rank buffers of one case.
struct RankBuffers {
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> partial, staging, out;
    std::array<Tensor, 2> partial_t, staging_t, out_t;

    RankBuffers(const ExecutionContext& ec, std::int32_t ne0, std::int32_t ne1) {
        const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            partial[rank] = std::make_unique<GuardedDeviceBuffer>(count * 4);
            staging[rank] = std::make_unique<GuardedDeviceBuffer>(count * 4);
            out[rank]     = std::make_unique<GuardedDeviceBuffer>(count * 2);
            staging[rank]->fill(0);
            out[rank]->fill(0xff);
            partial_t[rank] = Tensor(partial[rank]->data(), DType::FP32, {ne0, ne1});
            staging_t[rank] = Tensor(staging[rank]->data(), DType::FP32, {ne0, ne1});
            out_t[rank]     = Tensor(out[rank]->data(), DType::BF16, {ne0, ne1});
        }
    }

    void upload(const ExecutionContext& ec, const std::vector<float>& a,
                const std::vector<float>& b) {
        set_device(ec, 0);
        partial[0]->copy_from_host(a.data(), a.size() * 4);
        set_device(ec, 1);
        partial[1]->copy_from_host(b.data(), b.size() * 4);
        retire_staging(ec);
    }

    int verify(const ExecutionContext& ec, const std::string& label, const Expected& expected,
               bool with_out) {
        int failures            = 0;
        const std::size_t count = expected.sum.size();
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            const std::string rank_label = label + " rank " + std::to_string(rank);
            failures += verify_exact((rank_label + " fp32 sum").c_str(),
                                     from_device<std::uint32_t>(partial[rank]->data(), count),
                                     expected.sum);
            if (with_out) {
                failures += verify_exact((rank_label + " bf16 out").c_str(),
                                         from_device<std::uint16_t>(out[rank]->data(), count),
                                         expected.bf16);
            }
            failures += partial[rank]->verify_guards(rank_label + " partial");
            failures += staging[rank]->verify_guards(rank_label + " staging");
            failures += out[rank]->verify_guards(rank_label + " out");
        }
        return failures;
    }
};

int run_eager_case(const char* label, std::int32_t ne0, std::int32_t ne1, std::uint32_t seed,
                   bool with_out, const ExecutionContext& ec, const ops::PeerEvents& events) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    const auto a            = make_partial(count, seed);
    const auto b            = make_partial(count, seed + 1);
    RankBuffers buffers(ec, ne0, ne1);
    buffers.upload(ec, a, b);
    ops::allreduce_sum_f32(buffers.partial_t, buffers.staging_t,
                           with_out ? &buffers.out_t : nullptr, ec, events);
    synchronize_both(ec);
    return buffers.verify(ec, label, oracle(a, b, 1.0), with_out);
}

// `sites` exchanges of the same buffers in one two-device graph, replayed with fresh inputs: the
// first site leaves s = fl(a + b), every later one doubles it exactly. A captured memset on rank 1
// skews the ranks so a stale publication would be combined (see test_allreduce.cpp).
int run_captured_case(const char* label, std::int32_t ne0, std::int32_t ne1, int sites,
                      std::size_t expect_nodes_per_site, const ExecutionContext& ec,
                      const ops::PeerEvents& events, const ops::PeerMailbox* mailbox) {
    constexpr int kReplays           = 4;
    constexpr int kBackToBack        = 3;
    constexpr std::size_t kSkewBytes = 64u << 20;
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    RankBuffers buffers(ec, ne0, ne1);
    set_device(ec, 1);
    DeviceBuffer skew(kSkewBytes);
    retire_staging(ec);

    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        set_device(ec, 1);
        cuda_check(cudaMemsetAsync(skew.p, 0, skew.bytes, ec.dev[1]->stream), "skew memset");
        for (int site = 0; site < sites; ++site) {
            ops::allreduce_sum_f32(buffers.partial_t, buffers.staging_t, &buffers.out_t, ec,
                                   events);
        }
    });
    int failures                = 0;
    const std::size_t nodes     = definition.node_count();
    const bool mailbox_expected = expect_nodes_per_site == 2;
    if (mailbox_expected ? nodes != 1 + 2 * static_cast<std::size_t>(sites)
                         : nodes <= 1 + 2 * static_cast<std::size_t>(sites)) {
        std::cerr << label << ": " << nodes << " graph nodes do not match the "
                  << (mailbox_expected ? "mailbox" : "staged") << " transport\n";
        ++failures;
    }
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    const double scale = std::ldexp(1.0, sites - 1);
    std::vector<float> a, b;
    for (int replay = 0; replay < kReplays; ++replay) {
        a = make_partial(count, 701u + 2u * static_cast<std::uint32_t>(replay));
        b = make_partial(count, 702u + 2u * static_cast<std::uint32_t>(replay));
        buffers.upload(ec, a, b);
        launch_two_devices(ec, executable);
        failures += buffers.verify(ec, std::string(label) + " replay " + std::to_string(replay),
                                   oracle(a, b, scale), true);
    }
    set_device(ec, 0);
    for (int launch = 0; launch < kBackToBack; ++launch) { executable.launch(ec.dev[0]->stream); }
    synchronize_both(ec);
    failures += buffers.verify(ec, std::string(label) + " back-to-back",
                               oracle(a, b, scale * std::ldexp(1.0, kBackToBack * sites)), true);
    if (mailbox != nullptr && mailbox->hang_reported()) {
        std::cerr << label << ": a mailbox exchange reported a hang\n";
        ++failures;
    }
    return failures;
}

// The Qwen4Exp block's exchanges on one mailbox, in one graph: per site a BF16 all-reduce
// (attention or GDN output), an FP32 one with its BF16 output (MoE partials) and an FP32 one
// without (bf16_out == nullptr). Three exchanges per site on the two-slot round robin, so each slot
// carries BF16 and FP32 payloads (different lane counts) within the capture whatever slot the
// round robin starts at. Replayed with fresh inputs: both ranks' FP32 sums and BF16 outputs against
// the exact oracle (every FP32 exchange after the first doubles the sum exactly), the BF16 buffers
// of both ranks against the same graph on the staged path; node counts prove each transport.
int run_interleaved_case(const ExecutionContext& ec, const ops::PeerEvents& staged,
                         const ops::PeerEvents& mailbox_events, const ops::PeerMailbox& mailbox) {
    constexpr std::int32_t kTokens = 4;
    constexpr int kSites           = 2;
    constexpr int kReplays         = 3;
    const std::size_t count        = static_cast<std::size_t>(kHidden) * kTokens;
    // The last FP32 exchange leaves (a + b) * 2^(2 kSites - 1); the last one with a BF16 output
    // wrote the rounding of (a + b) * 2^(2 kSites - 2).
    const double sum_scale = std::ldexp(1.0, 2 * kSites - 1);
    const double out_scale = std::ldexp(1.0, 2 * kSites - 2);
    const auto run = [&](const ops::PeerEvents& events, const char* transport, std::size_t& nodes,
                         int& failures) {
        RankBuffers moe(ec, kHidden, kTokens);
        std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> attention, attention_staging;
        std::array<Tensor, 2> attention_t, attention_staging_t;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            attention[rank]         = std::make_unique<GuardedDeviceBuffer>(count * 2);
            attention_staging[rank] = std::make_unique<GuardedDeviceBuffer>(count * 2);
            attention_staging[rank]->fill(0);
            attention_t[rank] = Tensor(attention[rank]->data(), DType::BF16, {kHidden, kTokens});
            attention_staging_t[rank] =
                Tensor(attention_staging[rank]->data(), DType::BF16, {kHidden, kTokens});
        }
        const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
        DecodeGraphDefinition definition;
        capture_two_devices(ec, bridge, definition, [&] {
            for (int site = 0; site < kSites; ++site) {
                ops::allreduce_sum(attention_t, attention_staging_t, ec, events);
                ops::allreduce_sum_f32(moe.partial_t, moe.staging_t, &moe.out_t, ec, events);
                ops::allreduce_sum_f32(moe.partial_t, moe.staging_t, nullptr, ec, events);
            }
        });
        nodes = definition.node_count();
        DecodeGraphExecutable executable;
        executable.instantiate(definition);
        std::vector<std::vector<std::uint16_t>> attention_out;
        for (int replay = 0; replay < kReplays; ++replay) {
            const auto seed = 821u + 4u * static_cast<std::uint32_t>(replay);
            for (int rank = 0; rank < 2; ++rank) {
                set_device(ec, rank);
                std::vector<float> values(count);
                fill_uniform(values, seed + 2u + static_cast<std::uint32_t>(rank), -4.0F, 4.0F);
                std::vector<std::uint16_t> bits(count);
                for (std::size_t i = 0; i < count; ++i) { bits[i] = f32_to_bf16(values[i]); }
                attention[rank]->copy_from_host(bits.data(), count * 2);
            }
            const auto a = make_partial(count, seed);
            const auto b = make_partial(count, seed + 1u);
            moe.upload(ec, a, b);
            launch_two_devices(ec, executable);
            Expected expected = oracle(a, b, sum_scale);
            expected.bf16     = oracle(a, b, out_scale).bf16;
            failures += moe.verify(
                ec, std::string("interleaved ") + transport + " replay " + std::to_string(replay),
                expected, true);
            for (int rank = 0; rank < 2; ++rank) {
                set_device(ec, rank);
                attention_out.push_back(from_device<std::uint16_t>(attention[rank]->data(), count));
                failures += attention[rank]->verify_guards(std::string("interleaved attention ") +
                                                           transport);
            }
        }
        return attention_out;
    };
    int failures              = 0;
    std::size_t mailbox_nodes = 0, staged_nodes = 0;
    const auto via_mailbox = run(mailbox_events, "mailbox", mailbox_nodes, failures);
    const auto via_staged  = run(staged, "staged", staged_nodes, failures);
    if (mailbox_nodes != 2 * 3 * kSites || staged_nodes <= 2 * 3 * kSites) {
        std::cerr << "interleaved: " << mailbox_nodes << " / " << staged_nodes
                  << " graph nodes do not match the mailbox / staged transports\n";
        ++failures;
    }
    for (std::size_t i = 0; i < via_mailbox.size(); ++i) {
        const std::string label = "interleaved bf16 all-reduce replay " + std::to_string(i / 2) +
                                  " rank " + std::to_string(i % 2) + " equals staged";
        failures += verify_exact(label.c_str(), via_mailbox[i], via_staged[i]);
    }
    if (mailbox.hang_reported()) {
        std::cerr << "interleaved: a mailbox exchange reported a hang\n";
        ++failures;
    }
    return failures;
}

int run_rejections(const ExecutionContext& ec, const ops::PeerEvents& events) {
    int failures = 0;
    RankBuffers buffers(ec, 16, 1);
    const auto expect_throw = [&](const char* what, auto&& body) {
        try {
            body();
        } catch (const std::invalid_argument&) { return; } catch (const std::exception& e) {
            std::cerr << "rejection " << what << ": wrong exception " << e.what() << '\n';
            ++failures;
            return;
        }
        std::cerr << "rejection " << what << ": accepted\n";
        ++failures;
    };
    expect_throw("bf16 partial", [&] {
        auto partial     = buffers.partial_t;
        partial[0].dtype = DType::BF16;
        ops::allreduce_sum_f32(partial, buffers.staging_t, nullptr, ec, events);
    });
    expect_throw("aliased staging", [&] {
        ops::allreduce_sum_f32(buffers.partial_t, buffers.partial_t, nullptr, ec, events);
    });
    expect_throw("fp32 output", [&] {
        auto out = buffers.out_t;
        out[1]   = buffers.staging_t[1];
        ops::allreduce_sum_f32(buffers.partial_t, buffers.staging_t, &out, ec, events);
    });
    expect_throw("shape mismatch", [&] {
        auto staging = buffers.staging_t;
        staging[1]   = Tensor(staging[1].data, DType::FP32, {8, 2});
        ops::allreduce_sum_f32(buffers.partial_t, staging, nullptr, ec, events);
    });
    return failures;
}

// Mean per-exchange time of `sites` captured all-reduces of [2560,T], median over batches.
double captured_exchange_us(const ExecutionContext& ec, const ops::PeerEvents& events,
                            std::int32_t tokens, bool fp32) {
    constexpr int kSites   = 96;
    constexpr int kWarmup  = 10;
    constexpr int kBatches = 7;
    constexpr int kReplays = 40;
    RankBuffers buffers(ec, kHidden, tokens);
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> bf16, bf16_staging;
    std::array<Tensor, 2> bf16_t, bf16_staging_t;
    const std::size_t count = static_cast<std::size_t>(kHidden) * tokens;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        bf16[rank]         = std::make_unique<GuardedDeviceBuffer>(count * 2);
        bf16_staging[rank] = std::make_unique<GuardedDeviceBuffer>(count * 2);
        bf16[rank]->fill(0);
        bf16_staging[rank]->fill(0);
        bf16_t[rank]         = Tensor(bf16[rank]->data(), DType::BF16, {kHidden, tokens});
        bf16_staging_t[rank] = Tensor(bf16_staging[rank]->data(), DType::BF16, {kHidden, tokens});
    }
    buffers.upload(ec, std::vector<float>(count, 0.0F), std::vector<float>(count, 0.0F));
    const DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
    DecodeGraphDefinition definition;
    capture_two_devices(ec, bridge, definition, [&] {
        for (int site = 0; site < kSites; ++site) {
            if (fp32) {
                ops::allreduce_sum_f32(buffers.partial_t, buffers.staging_t, &buffers.out_t, ec,
                                       events);
            } else {
                ops::allreduce_sum(bf16_t, bf16_staging_t, ec, events);
            }
        }
    });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    for (int i = 0; i < kWarmup; ++i) { launch_two_devices(ec, executable); }
    std::vector<double> per_exchange;
    for (int batch = 0; batch < kBatches; ++batch) {
        const auto started = std::chrono::steady_clock::now();
        set_device(ec, 0);
        for (int i = 0; i < kReplays; ++i) { executable.launch(ec.dev[0]->stream); }
        synchronize_both(ec);
        const std::chrono::duration<double, std::micro> elapsed =
            std::chrono::steady_clock::now() - started;
        per_exchange.push_back(elapsed.count() / (static_cast<double>(kReplays) * kSites));
    }
    std::sort(per_exchange.begin(), per_exchange.end());
    return per_exchange[per_exchange.size() / 2];
}

void run_bench(const ExecutionContext& ec) {
    std::cout << "bench: captured all-reduce of [2560,T], graph of 96 exchanges, median of 7 "
                 "batches of 40 replays (us per exchange)\n";
    std::cout << "  T  bf16_bytes  bf16_us  fp32_bytes  fp32_us  fp32-bf16_us\n";
    for (const std::int32_t tokens : {1, 2, 4, 8}) {
        // The Program sizes the slot for the widest exchange of the round: the FP32 partials.
        ops::PeerMailbox mailbox(ec, static_cast<std::size_t>(kHidden) * tokens * 4);
        ops::PeerEvents events(ec);
        events.attach_mailbox(&mailbox);
        const double bf16 = captured_exchange_us(ec, events, tokens, false);
        const double fp32 = captured_exchange_us(ec, events, tokens, true);
        std::printf("  %d  %10d  %7.2f  %10d  %7.2f  %+.2f\n", tokens, kHidden * tokens * 2, bf16,
                    kHidden * tokens * 4, fp32, fp32 - bf16);
        if (mailbox.hang_reported()) { std::cout << "  (mailbox hang reported)\n"; }
    }
}

// T56: the eager HostExchange transport. `calls` back-to-back exchanges of the same buffers with no
// host step between them and a 64 MB memset skewing rank 1 before each: the first leaves
// s = fl(a + b), each later one doubles it exactly, so a host buffer overwritten before the peer
// read it (or an operand overwritten before it was copied out) breaks the bits.
int run_host_exchange_case(const char* label, std::int32_t ne0, std::int32_t ne1, int calls,
                           std::uint32_t seed, const ExecutionContext& ec,
                           const ops::PeerEvents& events) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    const auto a            = make_partial(count, seed);
    const auto b            = make_partial(count, seed + 1);
    RankBuffers buffers(ec, ne0, ne1);
    buffers.upload(ec, a, b);
    set_device(ec, 1);
    DeviceBuffer skew(64u << 20);
    synchronize_both(ec);
    for (int call = 0; call < calls; ++call) {
        set_device(ec, 1);
        cuda_check(cudaMemsetAsync(skew.p, call, skew.bytes, ec.dev[1]->stream), "skew memset");
        ops::allreduce_sum_f32(buffers.partial_t, buffers.staging_t, &buffers.out_t, ec, events);
    }
    synchronize_both(ec);
    return buffers.verify(ec, label, oracle(a, b, std::ldexp(1.0, calls - 1)), true);
}

// The BF16 allreduce_sum through the HostExchange, back to back, against the exact host oracle:
// s = RNE_bf16(x0 + x1) after the first call, doubled (exactly) by every later call. (T65: this
// case used to compare with the driver-staged pull of the same calls, but without peer access that
// pull is not ordered after the previous call's in-place combine on the source device and reads
// stale operands back to back, ~25 % of runs. The staged result is still computed and reported,
// not counted: it is the reference no longer, and production eager exchanges take the
// HostExchange.)
int run_host_exchange_bf16_case(const char* label, std::int32_t ne0, std::int32_t ne1, int calls,
                                std::uint32_t seed, const ExecutionContext& ec,
                                const ops::PeerEvents& staged, const ops::PeerEvents& host) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    std::array<std::vector<std::uint16_t>, 2> input;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<float> v(count);
        fill_uniform(v, seed + static_cast<std::uint32_t>(rank), -2.0F, 2.0F);
        input[rank].resize(count);
        for (std::size_t i = 0; i < count; ++i) { input[rank][i] = f32_to_bf16(v[i]); }
    }
    std::array<std::array<std::vector<std::uint16_t>, 2>, 2> result;
    for (int variant = 0; variant < 2; ++variant) {
        std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> buffer, staging;
        std::array<Tensor, 2> buffer_t, staging_t;
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            buffer[rank]  = std::make_unique<GuardedDeviceBuffer>(count * 2);
            staging[rank] = std::make_unique<GuardedDeviceBuffer>(count * 2);
            buffer[rank]->copy_from_host(input[rank].data(), count * 2);
            buffer_t[rank]  = Tensor(buffer[rank]->data(), DType::BF16, {ne0, ne1});
            staging_t[rank] = Tensor(staging[rank]->data(), DType::BF16, {ne0, ne1});
        }
        synchronize_both(ec);
        for (int call = 0; call < calls; ++call) {
            ops::allreduce_sum(buffer_t, staging_t, ec, variant == 0 ? staged : host);
        }
        synchronize_both(ec);
        for (int rank = 0; rank < 2; ++rank) {
            set_device(ec, rank);
            result[variant][rank] = from_device<std::uint16_t>(buffer[rank]->data(), count);
        }
    }
    std::vector<std::uint16_t> oracle(count);
    const auto widen = [](std::uint16_t b) {
        const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
        float v               = 0.0F;
        std::memcpy(&v, &u, sizeof(v));
        return v;
    };
    for (std::size_t i = 0; i < count; ++i) {
        const volatile float s = widen(input[0][i]) + widen(input[1][i]);
        std::uint16_t b        = f32_to_bf16(s);
        for (int call = 1; call < calls; ++call) {
            const volatile float d = widen(b) + widen(b);
            b                      = f32_to_bf16(d);
        }
        oracle[i] = b;
    }
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        failures += verify_exact((std::string(label) + " rank " + std::to_string(rank)).c_str(),
                                 result[1][rank], oracle);
        std::size_t stale = 0;
        for (std::size_t i = 0; i < count; ++i) { stale += result[0][rank][i] != oracle[i]; }
        if (stale != 0) {
            std::cout << label << " rank " << rank << ": note: driver-staged pull " << stale
                      << " of " << count << " values stale back to back (not counted)\n";
        }
    }
    return failures;
}

// Prefill overlap: the split-phase BF16 sum (allreduce_sum_begin / allreduce_sum_finish) on two
// lanes over the two column halves [0,split) and [split,ne1) of one buffer, issued as the
// overlapped prefill issues them (begin A, begin B, then per round finish A, begin A, finish B,
// begin B), a 64 MB memset skewing rank 1 before every begin, then one allreduce_sum() of the
// whole buffer through the same exchange. Exact oracle: RNE_bf16(x0 + x1) after a half's first
// call, doubled exactly by every later call, so a lane finished early, a host buffer or staging
// reused before the peer read it, or an operand overwritten before it was copied out breaks the
// bits.
int run_split_lane_case(const char* label, std::int32_t ne0, std::int32_t ne1, std::int32_t split,
                        int rounds, std::uint32_t seed, const ExecutionContext& ec,
                        const ops::PeerEvents& host) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    std::array<std::vector<std::uint16_t>, 2> input;
    for (int rank = 0; rank < 2; ++rank) {
        std::vector<float> v(count);
        fill_uniform(v, seed + static_cast<std::uint32_t>(rank), -2.0F, 2.0F);
        input[rank].resize(count);
        for (std::size_t i = 0; i < count; ++i) { input[rank][i] = f32_to_bf16(v[i]); }
    }
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> buffer, staging;
    std::array<Tensor, 2> buffer_t, staging_t;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        buffer[rank]  = std::make_unique<GuardedDeviceBuffer>(count * 2);
        staging[rank] = std::make_unique<GuardedDeviceBuffer>(count * 2);
        buffer[rank]->copy_from_host(input[rank].data(), count * 2);
        buffer_t[rank]  = Tensor(buffer[rank]->data(), DType::BF16, {ne0, ne1});
        staging_t[rank] = Tensor(staging[rank]->data(), DType::BF16, {ne0, ne1});
    }
    std::array<std::array<Tensor, 2>, 2> half_buffer, half_staging;
    for (int h = 0; h < 2; ++h) {
        const std::int32_t first = h == 0 ? 0 : split;
        const std::int32_t n     = h == 0 ? split : ne1 - split;
        for (int rank = 0; rank < 2; ++rank) {
            half_buffer[h][rank]  = buffer_t[rank].slice(1, first, n);
            half_staging[h][rank] = staging_t[rank].slice(1, first, n);
        }
    }
    set_device(ec, 1);
    DeviceBuffer skew(64u << 20);
    synchronize_both(ec);
    int failures = 0;
    if (!ops::allreduce_sum_split_available(ec, host)) {
        std::cerr << label << ": split all-reduce unavailable with a host exchange\n";
        return 1;
    }
    for (int round = 0; round < rounds; ++round) {
        for (int h = 0; h < 2; ++h) {
            if (round > 0) { ops::allreduce_sum_finish(ec, host, h); }
            set_device(ec, 1);
            cuda_check(cudaMemsetAsync(skew.p, round * 2 + h, skew.bytes, ec.dev[1]->stream),
                       "skew memset");
            ops::allreduce_sum_begin(half_buffer[h], half_staging[h], ec, host, h);
        }
    }
    ops::allreduce_sum_finish(ec, host, 0);
    ops::allreduce_sum_finish(ec, host, 1);
    ops::allreduce_sum(buffer_t, staging_t, ec, host);
    synchronize_both(ec);
    std::vector<std::uint16_t> oracle(count);
    const auto widen = [](std::uint16_t b) {
        const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
        float v               = 0.0F;
        std::memcpy(&v, &u, sizeof(v));
        return v;
    };
    for (std::size_t i = 0; i < count; ++i) {
        const volatile float s = widen(input[0][i]) + widen(input[1][i]);
        std::uint16_t b        = f32_to_bf16(s);
        for (int call = 0; call < rounds; ++call) {
            const volatile float d = widen(b) + widen(b);
            b                      = f32_to_bf16(d);
        }
        oracle[i] = b;
    }
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        const auto result = from_device<std::uint16_t>(buffer[rank]->data(), count);
        failures += verify_exact((std::string(label) + " rank " + std::to_string(rank)).c_str(),
                                 result, oracle);
    }
    return failures;
}

// v07/p2p-overlap: with PeerLanes attached next to a HostExchange, a split-phase sum opens on the
// lanes, never on the HostExchange, and allreduce_sum_finish closes it there.
int run_peer_lane_selection(const ExecutionContext& ec, const ops::PeerEvents& events,
                            const ops::PeerLanes& lanes, const ops::HostExchange& exchange) {
    constexpr std::int32_t ne0 = 16, ne1 = 8;
    const std::size_t bytes = static_cast<std::size_t>(ne0) * ne1 * 2;
    std::array<std::unique_ptr<GuardedDeviceBuffer>, 2> buffer, staging;
    std::array<Tensor, 2> buffer_t, staging_t;
    const std::vector<std::uint16_t> zeros(bytes / 2, 0);
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        buffer[rank]  = std::make_unique<GuardedDeviceBuffer>(bytes);
        staging[rank] = std::make_unique<GuardedDeviceBuffer>(bytes);
        buffer[rank]->copy_from_host(zeros.data(), bytes);
        buffer_t[rank]  = Tensor(buffer[rank]->data(), DType::BF16, {ne0, ne1});
        staging_t[rank] = Tensor(staging[rank]->data(), DType::BF16, {ne0, ne1});
    }
    synchronize_both(ec);
    int failures = 0;
    if (!ops::allreduce_sum_split_available(ec, events)) {
        std::cerr << "peer lane selection: split all-reduce unavailable\n";
        return 1;
    }
    ops::allreduce_sum_begin(buffer_t, staging_t, ec, events, 1);
    if (!lanes.sum_open(1) || exchange.sum_open(1)) {
        std::cerr << "peer lane selection: lane 1 did not open on the PeerLanes\n";
        ++failures;
    }
    ops::allreduce_sum_finish(ec, events, 1);
    if (lanes.sum_open(1) || exchange.sum_open(1)) {
        std::cerr << "peer lane selection: lane 1 still open after finish\n";
        ++failures;
    }
    synchronize_both(ec);
    return failures;
}

// T62: the pipelined prefill exchange (HostExchange::begin_sum_f32 / columns_ready /
// finish_sum_f32). Oracle, exact: S = fl32(n(a) + n(b)) with n = identity (FP32), RNE to binary16
// (FP16, the host's _Float16 conversion) or RNE to bfloat16 (BF16); both ranks hold S and
// RNE_bf16(S). `calls` back-to-back calls on the same buffers (call k sums call k-1's S with
// itself), each rank reporting its columns in its own groups; with `mixed`, an
// allreduce_sum_f32 through the same HostExchange (pull path, shared pinned buffers) follows.
float narrow_host(float v, ops::HostExchange::Payload payload) {
    switch (payload) {
    case ops::HostExchange::Payload::FP16: {
        // T128b: the device payload is f16(sat(v * 2^-k)) * 2^k, k = NINFER_TP_AR_SHIFT (default
        // 2).
        const char* e = std::getenv("NINFER_TP_AR_SHIFT");
        int k         = e != nullptr ? std::atoi(e) : 2;
        k             = k < 0 ? 0 : (k > 14 ? 14 : k);
        float x       = std::ldexp(v, -k);
        if (std::fabs(x) > 65504.0F) { x = std::copysign(65504.0F, x); }
        return std::ldexp(static_cast<float>(static_cast<_Float16>(x)), k);
    }
    case ops::HostExchange::Payload::BF16: {
        const std::uint32_t bits = static_cast<std::uint32_t>(f32_to_bf16(v)) << 16;
        float out                = 0.0F;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }
    default:
        return v;
    }
}

// INT8 payload (opt-in): per column, 32-row blocks; s = fp16 rounded up of min(max(amax / 127,
// 2^-14), 65504), q = rint(x / s) clamped to [-127, 127], value q * s; an all-zero block is 0; a
// block with NaN or Inf is NaN. Every float operation is rounded on its own (volatile), as the
// device kernel does (no contraction). Columns whose rows are not a multiple of 32 use the FP16
// payload, as begin_sum_f32 does.
std::vector<float> narrow_columns_host(const std::vector<float>& v, std::int32_t rows,
                                       ops::HostExchange::Payload payload) {
    using Payload = ops::HostExchange::Payload;
    std::vector<float> out(v.size());
    if (payload != Payload::INT8 || rows % 32 != 0) {
        const Payload p = payload == Payload::INT8 ? Payload::FP16 : payload;
        for (std::size_t i = 0; i < v.size(); ++i) { out[i] = narrow_host(v[i], p); }
        return out;
    }
    for (std::size_t base = 0; base < v.size(); base += 32) {
        bool bad   = false;
        float amax = 0.0F;
        for (std::size_t j = 0; j < 32; ++j) {
            const float x = v[base + j];
            if (!std::isfinite(x)) { bad = true; }
            amax = std::fmax(amax, std::fabs(x));
        }
        if (bad) {
            for (std::size_t j = 0; j < 32; ++j) {
                out[base + j] = std::numeric_limits<float>::quiet_NaN();
            }
            continue;
        }
        if (amax == 0.0F) {
            for (std::size_t j = 0; j < 32; ++j) { out[base + j] = 0.0F; }
            continue;
        }
        volatile float s = amax / 127.0F;
        const float c    = std::fmin(std::fmax(s, 6.103515625e-05F), 65504.0F);
        _Float16 h       = static_cast<_Float16>(c);
        if (static_cast<float>(h) < c) { // round up: the next binary16 above
            std::uint16_t bits = 0;
            std::memcpy(&bits, &h, 2);
            ++bits;
            std::memcpy(&h, &bits, 2);
        }
        const float sf     = static_cast<float>(h);
        volatile float inv = 1.0F / sf;
        for (std::size_t j = 0; j < 32; ++j) {
            volatile float prod = v[base + j] * inv;
            // The code is an integer: a code of zero is +0 whatever the sign of x.
            const float q = static_cast<float>(
                static_cast<int>(std::fmin(std::fmax(std::nearbyint(prod), -127.0F), 127.0F)));
            volatile float val = q * sf;
            out[base + j]      = val;
        }
    }
    return out;
}

int run_pipelined_case(const char* label, std::int32_t ne0, std::int32_t ne1,
                       ops::HostExchange::Payload payload,
                       const std::array<std::vector<std::int32_t>, 2>& groups, int calls,
                       bool mixed, std::uint32_t seed, const ExecutionContext& ec,
                       ops::HostExchange& exchange, const ops::PeerEvents& events) {
    const std::size_t count = static_cast<std::size_t>(ne0) * static_cast<std::size_t>(ne1);
    auto a                  = make_partial(count, seed);
    auto b                  = make_partial(count, seed + 1);
    RankBuffers buffers(ec, ne0, ne1);
    buffers.upload(ec, a, b);
    set_device(ec, 1);
    DeviceBuffer skew(64u << 20);
    synchronize_both(ec);
    std::vector<float> sum(count);
    for (int call = 0; call < calls; ++call) {
        const auto na = narrow_columns_host(a, ne0, payload);
        const auto nb = narrow_columns_host(b, ne0, payload);
        for (std::size_t i = 0; i < count; ++i) {
            const volatile float s = na[i] + nb[i];
            sum[i]                 = s;
        }
        a = sum;
        b = sum;
        if (!exchange.begin_sum_f32(buffers.partial_t, buffers.staging_t, &buffers.out_t, payload,
                                    ec)) {
            std::cerr << label << ": begin_sum_f32 refused an eager call\n";
            return 1;
        }
        // Rank 1 lags (a long memset ahead of its groups) so that the groups really interleave.
        set_device(ec, 1);
        cuda_check(cudaMemsetAsync(skew.p, call, skew.bytes, ec.dev[1]->stream), "skew memset");
        for (int rank = 0; rank < 2; ++rank) {
            std::int64_t first = 0;
            for (const std::int32_t n : groups[rank]) {
                exchange.columns_ready(rank, first, n);
                first += n;
            }
        }
        exchange.finish_sum_f32();
    }
    if (mixed) {
        ops::allreduce_sum_f32(buffers.partial_t, buffers.staging_t, &buffers.out_t, ec, events);
        for (std::size_t i = 0; i < count; ++i) {
            const volatile float s = a[i] + b[i];
            sum[i]                 = s;
        }
    }
    synchronize_both(ec);
    Expected expected;
    expected.sum = f32_bits(sum);
    expected.bf16.resize(count);
    for (std::size_t i = 0; i < count; ++i) { expected.bf16[i] = f32_to_bf16(sum[i]); }
    return buffers.verify(ec, label, expected, true);
}

// INT8 payload blocks at the edges, one call: an all-zero column, subnormal-only and tiny blocks, a
// value beyond the scale range (saturates to 127 * 65504), a NaN and an Inf (each turns its 32-row
// block into NaN on both ranks).
int run_int8_special_case(const ExecutionContext& ec, ops::HostExchange& exchange) {
    using Payload              = ops::HostExchange::Payload;
    constexpr std::int32_t ne0 = 64;
    constexpr std::int32_t ne1 = 6;
    const std::size_t count    = static_cast<std::size_t>(ne0) * ne1;
    auto a                     = make_partial(count, 401u);
    auto b                     = make_partial(count, 402u);
    for (std::int32_t i = 0; i < ne0; ++i) {
        a[static_cast<std::size_t>(i)]           = 0.0F;
        b[static_cast<std::size_t>(i)]           = 0.0F;
        a[static_cast<std::size_t>(ne0 + i)]     = 1.0e-40F;
        b[static_cast<std::size_t>(ne0 + i)]     = 1.0e-30F;
        a[static_cast<std::size_t>(2 * ne0 + i)] = 1.0F;
    }
    a[static_cast<std::size_t>(2 * ne0 + 7)]  = 1.0e8F;
    a[static_cast<std::size_t>(3 * ne0 + 5)]  = std::numeric_limits<float>::quiet_NaN();
    a[static_cast<std::size_t>(4 * ne0 + 40)] = std::numeric_limits<float>::infinity();
    RankBuffers buffers(ec, ne0, ne1);
    buffers.upload(ec, a, b);
    const auto na = narrow_columns_host(a, ne0, Payload::INT8);
    const auto nb = narrow_columns_host(b, ne0, Payload::INT8);
    std::vector<float> sum(count);
    for (std::size_t i = 0; i < count; ++i) {
        const volatile float v = na[i] + nb[i];
        sum[i]                 = v;
    }
    if (!exchange.begin_sum_f32(buffers.partial_t, buffers.staging_t, &buffers.out_t, Payload::INT8,
                                ec)) {
        std::cerr << "int8 edge blocks: begin_sum_f32 refused an eager call\n";
        return 1;
    }
    exchange.columns_ready(0, 0, ne1);
    exchange.columns_ready(1, 0, 3);
    exchange.columns_ready(1, 3, 3);
    exchange.finish_sum_f32();
    synchronize_both(ec);
    int failures = 0;
    for (int rank = 0; rank < 2; ++rank) {
        set_device(ec, rank);
        const auto got = from_device<std::uint32_t>(buffers.partial[rank]->data(), count);
        const auto out = from_device<std::uint16_t>(buffers.out[rank]->data(), count);
        int bad        = 0;
        for (std::size_t i = 0; i < count; ++i) {
            float g = 0.0F;
            std::memcpy(&g, &got[i], 4);
            std::uint32_t want = 0;
            std::memcpy(&want, &sum[i], 4);
            const bool ok = std::isnan(sum[i]) ? std::isnan(g) && (out[i] & 0x7FFFu) > 0x7F80u
                                               : got[i] == want && out[i] == f32_to_bf16(sum[i]);
            if (!ok && bad++ < 4) {
                std::cerr << "int8 edge blocks rank " << rank << " [" << i % ne0 << "," << i / ne0
                          << "]: got " << g << " want " << sum[i] << '\n';
            }
        }
        failures += bad != 0 ? 1 : 0;
        failures += buffers.staging[rank]->verify_guards("int8 edge blocks staging");
    }
    std::cout << "pipelined int8 edge blocks (zero, subnormal, saturated, NaN, Inf): "
              << (failures ? "FAIL" : "ok") << '\n';
    return failures;
}

} // namespace

// T68: HostExchange::copy, one way in each direction, sizes across many 4 KiB chunks, against the
// uploaded bytes; eager_exchange selects the attached exchange.
int run_host_copy_case(const ExecutionContext& ec, ops::HostExchange& exchange,
                       const ops::PeerEvents& events) {
    int failures = 0;
    if (ops::eager_exchange(ec, events) != &exchange) {
        std::cerr << "host copy: eager_exchange did not select the attached exchange\n";
        ++failures;
    }
    for (const std::size_t bytes :
         {std::size_t{1}, std::size_t{4096}, std::size_t{10001}, std::size_t{3} << 20}) {
        for (int from = 0; from < 2; ++from) {
            const int to = 1 - from;
            std::vector<std::uint8_t> host(bytes), back(bytes, 0);
            for (std::size_t i = 0; i < bytes; ++i) {
                host[i] = static_cast<std::uint8_t>(
                    (i * 131u + bytes + static_cast<std::size_t>(from) * 7u) & 0xFFu);
            }
            void* src = nullptr;
            void* dst = nullptr;
            cuda_check(cudaSetDevice(ec.dev[from]->device), "cudaSetDevice");
            cuda_check(cudaMalloc(&src, bytes), "cudaMalloc source");
            cuda_check(cudaMemcpy(src, host.data(), bytes, cudaMemcpyHostToDevice), "upload");
            cuda_check(cudaSetDevice(ec.dev[to]->device), "cudaSetDevice");
            cuda_check(cudaMalloc(&dst, bytes), "cudaMalloc destination");
            cuda_check(cudaMemset(dst, 0, bytes), "cudaMemset");
            // Both devices: the pageable upload of the source may still be in flight on its
            // device's default stream, which the exchange's non-blocking streams do not wait for.
            retire_staging(ec);
            cuda_check(cudaSetDevice(ec.dev[from]->device), "cudaSetDevice");
            cuda_check(cudaEventRecord(events.inputs_ready(from), ec.dev[from]->stream), "record");
            exchange.copy(from, src, dst, bytes, ec, events);
            cuda_check(cudaSetDevice(ec.dev[to]->device), "cudaSetDevice");
            cuda_check(cudaStreamSynchronize(ec.dev[to]->stream), "sync destination stream");
            cuda_check(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost), "download");
            if (back != host) {
                std::cerr << "host copy " << from << "->" << to << " " << bytes << " B: mismatch\n";
                ++failures;
            }
            cuda_check(cudaFree(dst), "cudaFree");
            cuda_check(cudaSetDevice(ec.dev[from]->device), "cudaSetDevice");
            cuda_check(cudaFree(src), "cudaFree");
        }
    }
    std::cout << "host copy one-way (1 B .. 3 MiB, both directions): " << (failures ? "FAIL" : "ok")
              << '\n';
    return failures;
}

// A pipelined call abandoned while open (its caller failed between begin_sum_f32 and
// finish_sum_f32, after rank 0 had reported part of its columns) leaves the exchange free: the next
// call on the same exchange opens and sums exactly. Abandoning without an open call does nothing.
int run_abandoned_case(const ExecutionContext& ec, ops::HostExchange& exchange,
                       const ops::PeerEvents& events) {
    using Payload = ops::HostExchange::Payload;
    {
        constexpr std::int32_t kColumns = 64;
        const std::size_t count         = static_cast<std::size_t>(kHidden) * kColumns;
        RankBuffers buffers(ec, kHidden, kColumns);
        buffers.upload(ec, make_partial(count, 313u), make_partial(count, 314u));
        synchronize_both(ec);
        exchange.abandon_sum_f32();
        if (!exchange.begin_sum_f32(buffers.partial_t, buffers.staging_t, &buffers.out_t,
                                    Payload::FP16, ec)) {
            std::cerr << "abandoned pipelined call: begin_sum_f32 refused an eager call\n";
            return 1;
        }
        exchange.columns_ready(0, 0, kColumns / 2);
        exchange.abandon_sum_f32();
        if (exchange.columns_done(0) != 0) {
            std::cerr << "abandoned pipelined call: the call is still open\n";
            return 1;
        }
        synchronize_both(ec);
    }
    return run_pipelined_case("pipelined fp16 [2560,64] after an abandoned call", kHidden, 64,
                              Payload::FP16, {{{32, 32}, {64}}}, 1, false, 315u, ec, exchange,
                              events);
}

int main(int argc, char** argv) {
    const bool bench      = argc > 1 && std::string(argv[1]) == "--bench";
    const bool bench_only = argc > 1 && std::string(argv[1]) == "--bench-only";
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cout << "SKIP: cross-device collectives require two CUDA devices, found "
                  << device_count << '\n';
        return 77;
    }
    const ExecutionContext ec({0, 1});
    std::cout << "peer access: " << (ops::enable_peer_access(ec) ? "enabled" : "unavailable")
              << '\n';
    if (bench_only) {
        run_bench(ec);
        return 0;
    }
    const ops::PeerEvents events(ec);
    int failures = 0;
    failures += run_eager_case("eager [2560]", kHidden, 1, 101u, true, ec, events);
    failures += run_eager_case("eager [2560,4]", kHidden, 4, 103u, true, ec, events);
    failures += run_eager_case("eager [2560,8] fp32 only", kHidden, 8, 105u, false, ec, events);
    failures += run_eager_case("eager [2560,48]", kHidden, 48, 107u, true, ec, events);
    failures += run_eager_case("eager [17,3]", 17, 3, 109u, true, ec, events);
    failures += run_eager_case("eager [1]", 1, 1, 111u, true, ec, events);
    failures += run_rejections(ec, events);

    // The Program's slot: [2560, K+1] FP32 at K = 3.
    ops::PeerMailbox mailbox(ec, static_cast<std::size_t>(kHidden) * 4 * 4);
    ops::PeerEvents mailbox_events(ec);
    mailbox_events.attach_mailbox(&mailbox);
    ops::PeerMailbox legacy(ec, static_cast<std::size_t>(kHidden) * 4 * 4, 2,
                            ops::PeerExchangeKernel::Legacy);
    ops::PeerEvents legacy_events(ec);
    legacy_events.attach_mailbox(&legacy);

    failures += run_captured_case("captured staged [2560]", kHidden, 1, 5, 0, ec, events, nullptr);
    failures += run_captured_case("captured mailbox [2560]", kHidden, 1, 5, 2, ec, mailbox_events,
                                  &mailbox);
    failures += run_captured_case("captured mailbox [2560,4]", kHidden, 4, 5, 2, ec, mailbox_events,
                                  &mailbox);
    failures += run_captured_case("captured mailbox [2560,3]", kHidden, 3, 3, 2, ec, mailbox_events,
                                  &mailbox);
    failures += run_captured_case("captured mailbox [4,1] minimal", 4, 1, 3, 2, ec, mailbox_events,
                                  &mailbox);
    failures += run_captured_case("captured oversized [2560,5] stays staged", kHidden, 5, 3, 0, ec,
                                  mailbox_events, &mailbox);
    failures += run_captured_case("captured [5,1] unaligned stays staged", 5, 1, 3, 0, ec,
                                  mailbox_events, &mailbox);
    failures += run_captured_case("captured legacy mailbox stays staged", kHidden, 4, 3, 0, ec,
                                  legacy_events, &legacy);
    failures += run_interleaved_case(ec, events, mailbox_events, mailbox);

    {
        // T56: eager exchanges through pinned host buffers. 4 KiB chunks: many chunks per call;
        // the first [2560,48] call grows the buffers (both devices drained first).
        ops::HostExchange exchange(ec, 4096);
        ops::PeerEvents host_events(ec);
        host_events.attach_host_exchange(&exchange);
        failures += run_eager_case("host exchange [2560]", kHidden, 1, 201u, true, ec, host_events);
        failures +=
            run_eager_case("host exchange [2560,48]", kHidden, 48, 203u, true, ec, host_events);
        failures +=
            run_eager_case("host exchange [17,3] fp32 only", 17, 3, 205u, false, ec, host_events);
        failures += run_host_exchange_case("host exchange back to back [2560,64] x4", kHidden, 64,
                                           4, 207u, ec, host_events);
        failures += run_host_exchange_bf16_case("host exchange bf16 [2560,33] x3", kHidden, 33, 3,
                                                209u, ec, events, host_events);
        failures += run_split_lane_case("split lanes bf16 [2560,96] 32+64 x3 + sum", kHidden, 96,
                                        32, 3, 213u, ec, host_events);
        if (ops::allreduce_sum_split_available(ec, events)) {
            std::cerr << "split all-reduce available without a host exchange\n";
            ++failures;
        }
        if (exchange.capacity() < static_cast<std::size_t>(kHidden) * 64 * 4) {
            std::cerr << "host exchange: buffers did not grow (" << exchange.capacity() << ")\n";
            ++failures;
        }
        // A capture ignores the HostExchange: the staged nodes, as without it.
        failures += run_captured_case("captured with host exchange stays staged", kHidden, 1, 5, 0,
                                      ec, host_events, nullptr);
        // Default chunk (8 MiB), a prefill-sized partial: [2560,1024] FP32 = 10 MiB, two chunks.
        ops::HostExchange wide(ec);
        ops::PeerEvents wide_events(ec);
        wide_events.attach_host_exchange(&wide);
        failures += run_host_exchange_case("host exchange 8 MiB chunks [2560,1024] x3", kHidden,
                                           1024, 3, 211u, ec, wide_events);
        // Prefill-sized halves: 5 MiB per lane, three 2 MiB chunks each.
        failures += run_split_lane_case("split lanes bf16 [2560,2048] 1024+1024 x2 + sum", kHidden,
                                        2048, 1024, 2, 215u, ec, wide_events);
    }

    {
        // v07/p2p-overlap: the split-phase sums on PeerLanes, issued as the overlapped prefill
        // issues them (run_split_lane_case). Next to a HostExchange (the Program's layout under
        // NINFER_TP_PEER_OVERLAP=force; the closing one-call sum takes the HostExchange) on every
        // pair: without P2P the lane copies are driver-staged and host-ordered (direct = false).
        // Alone (the Program's layout with P2P; the closing sum is the direct pull) only with
        // peer access. [5120,2048] 1024+1024 is the 27B's chunk-2048 prefill half.
        const bool p2p = ops::enable_peer_access(ec);
        ops::PeerLanes lanes(ec, p2p);
        ops::HostExchange exchange(ec);
        ops::PeerEvents mixed(ec);
        mixed.attach_host_exchange(&exchange);
        mixed.attach_peer_lanes(&lanes);
        // One line per case (the other cases print only failures): the line users send us.
        const auto report = [](const std::string& label, int case_failures) {
            std::cout << label << ": " << (case_failures != 0 ? "FAIL" : "ok") << '\n';
            return case_failures;
        };
        const std::string staged = p2p ? "direct" : "driver-staged";
        failures += report("peer lanes selection (" + staged + ")",
                           run_peer_lane_selection(ec, mixed, lanes, exchange));
        for (const auto& [ne0, ne1, split, rounds, seed] :
             {std::array<int, 5>{kHidden, 96, 32, 3, 217},
              std::array<int, 5>{5120, 2048, 1024, 3, 219}}) {
            const std::string label = "peer lanes (" + staged + ") + host exchange bf16 [" +
                                      std::to_string(ne0) + "," + std::to_string(ne1) + "] " +
                                      std::to_string(split) + "+" + std::to_string(ne1 - split) +
                                      " x" + std::to_string(rounds) + " + sum";
            failures +=
                report(label, run_split_lane_case(label.c_str(), ne0, ne1, split, rounds,
                                                  static_cast<std::uint32_t>(seed), ec, mixed));
        }
        // PR #15 (tzolkincz) shapes: odd column counts (odd lane bytes, the pull and the residual
        // add unaligned) and tiny lanes; then a graph capture with the lanes attached stays on
        // the staged nodes (the lanes are eager only).
        for (const auto& [ne0, ne1, split, rounds, seed] :
             {std::array<int, 5>{kHidden, 101, 37, 2, 231}, std::array<int, 5>{17, 5, 2, 2, 233}}) {
            const std::string label = "peer lanes (" + staged + ") + host exchange bf16 [" +
                                      std::to_string(ne0) + "," + std::to_string(ne1) + "] " +
                                      std::to_string(split) + "+" + std::to_string(ne1 - split) +
                                      " x" + std::to_string(rounds) + " + sum";
            failures +=
                report(label, run_split_lane_case(label.c_str(), ne0, ne1, split, rounds,
                                                  static_cast<std::uint32_t>(seed), ec, mixed));
        }
        failures +=
            report("captured with peer lanes + host exchange stays staged",
                   run_captured_case("captured with peer lanes + host exchange stays staged",
                                     kHidden, 1, 5, 0, ec, mixed, nullptr));
        if (p2p) {
            ops::PeerEvents direct(ec);
            direct.attach_peer_lanes(&lanes);
            // PR #15 (tzolkincz) direct-lane cases: odd and tiny lanes, one-call sums with only
            // the lanes attached (the plain peer pull), and a capture that stays staged.
            for (const auto& [ne0, ne1, split, rounds, seed] :
                 {std::array<int, 5>{kHidden, 101, 37, 2, 235},
                  std::array<int, 5>{17, 5, 2, 2, 237}}) {
                const std::string label = "peer lanes direct bf16 [" + std::to_string(ne0) + "," +
                                          std::to_string(ne1) + "] " + std::to_string(split) + "+" +
                                          std::to_string(ne1 - split) + " x" +
                                          std::to_string(rounds) + " + sum";
                failures += report(
                    label, run_split_lane_case(label.c_str(), ne0, ne1, split, rounds,
                                               static_cast<std::uint32_t>(seed), ec, direct));
            }
            failures += report("peer lanes direct: one-call [2560,48]",
                               run_eager_case("peer lanes direct: one-call [2560,48]", kHidden, 48,
                                              239u, true, ec, direct));
            failures += report("peer lanes direct: one-call [2560] fp32 only",
                               run_eager_case("peer lanes direct: one-call [2560] fp32 only",
                                              kHidden, 1, 241u, false, ec, direct));
            if (!ops::allreduce_sum_split_available(ec, direct)) {
                std::cerr << "split all-reduce unavailable with only peer lanes attached\n";
                ++failures;
            }
            failures += report("captured with peer lanes direct stays staged",
                               run_captured_case("captured with peer lanes direct stays staged",
                                                 kHidden, 1, 5, 0, ec, direct, nullptr));
            for (const auto& [ne0, ne1, split, rounds, seed] :
                 {std::array<int, 5>{kHidden, 96, 32, 3, 221},
                  std::array<int, 5>{5120, 2048, 1024, 4, 223}}) {
                const std::string label = "peer lanes direct bf16 [" + std::to_string(ne0) + "," +
                                          std::to_string(ne1) + "] " + std::to_string(split) + "+" +
                                          std::to_string(ne1 - split) + " x" +
                                          std::to_string(rounds) + " + sum";
                failures += report(
                    label, run_split_lane_case(label.c_str(), ne0, ne1, split, rounds,
                                               static_cast<std::uint32_t>(seed), ec, direct));
            }
        } else {
            std::cout << "peer lanes direct: skipped (no peer access)\n";
        }
    }

    {
        // T62: the pipelined prefill exchange, every payload, uneven per-rank groups, many
        // 4 KiB chunks per group, back to back, then a pull() through the same pinned buffers.
        using Payload = ops::HostExchange::Payload;
        ops::HostExchange exchange(ec, 4096);
        ops::PeerEvents host_events(ec);
        host_events.attach_host_exchange(&exchange);
        failures += run_pipelined_case("pipelined fp32 [2560,96] groups 32+64 / 96 x2 + pull",
                                       kHidden, 96, Payload::FP32, {{{32, 64}, {96}}}, 2, true,
                                       301u, ec, exchange, host_events);
        failures += run_pipelined_case("pipelined fp16 [2560,100] groups 10+40+50 / 64+36 x3",
                                       kHidden, 100, Payload::FP16, {{{10, 40, 50}, {64, 36}}}, 3,
                                       false, 303u, ec, exchange, host_events);
        failures += run_pipelined_case("pipelined bf16 [2560,48] one group x2 + pull", kHidden, 48,
                                       Payload::BF16, {{{48}, {48}}}, 2, true, 305u, ec, exchange,
                                       host_events);
        failures +=
            run_pipelined_case("pipelined fp16 [17,5] empty group", 17, 5, Payload::FP16,
                               {{{0, 5}, {2, 3}}}, 1, false, 307u, ec, exchange, host_events);
        failures += run_pipelined_case("pipelined int8 [2560,100] groups 10+40+50 / 64+36 x3",
                                       kHidden, 100, Payload::INT8, {{{10, 40, 50}, {64, 36}}}, 3,
                                       false, 313u, ec, exchange, host_events);
        failures += run_pipelined_case("pipelined int8 [192,40] groups 7+33 / 40 x2 + pull", 192,
                                       40, Payload::INT8, {{{7, 33}, {40}}}, 2, true, 315u, ec,
                                       exchange, host_events);
        failures +=
            run_pipelined_case("pipelined int8 [17,5] falls back to fp16", 17, 5, Payload::INT8,
                               {{{0, 5}, {2, 3}}}, 1, false, 317u, ec, exchange, host_events);
        failures += run_int8_special_case(ec, exchange);
        failures += run_abandoned_case(ec, exchange, host_events);
        ops::HostExchange wide(ec);
        ops::PeerEvents wide_events(ec);
        wide_events.attach_host_exchange(&wide);
        failures += run_pipelined_case("pipelined fp32 8 MiB chunks [2560,1024] 512+512 / 1024",
                                       kHidden, 1024, Payload::FP32, {{{512, 512}, {1024}}}, 2,
                                       true, 309u, ec, wide, wide_events);
        failures += run_pipelined_case(
            "pipelined fp16 8 MiB chunks [2560,2048] 4x512 / 2048", kHidden, 2048, Payload::FP16,
            {{{512, 512, 512, 512}, {2048}}}, 2, false, 311u, ec, wide, wide_events);
        failures += run_pipelined_case(
            "pipelined int8 8 MiB chunks [2560,2048] 4x512 / 2048", kHidden, 2048, Payload::INT8,
            {{{512, 512, 512, 512}, {2048}}}, 2, false, 319u, ec, wide, wide_events);
    }

    {
        ops::HostExchange exchange(ec, 4096);
        ops::PeerEvents host_events(ec);
        host_events.attach_host_exchange(&exchange);
        failures += run_host_copy_case(ec, exchange, host_events);
    }

    if (bench) { run_bench(ec); }
    std::cout << (failures ? "FAIL" : "OK") << " allreduce_f32\n";
    return failures ? 1 : 0;
}
