// Correctness + coverage for causal_conv1d, against the frozen op-test
// standard (docs/l1-op-test-standard.md): fp64 golden from bf16-rounded
// inputs, honest [-8,8] ranges, composite tolerance bf16_reduction.
#include "qus/kernels/causal_conv1d.h"
#include "kernels/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

static double silu(double x) { return x / (1.0 + std::exp(-x)); }

static std::size_t idx2(std::int32_t c, std::int32_t t, std::int32_t C) {
    return static_cast<std::size_t>(t) * static_cast<std::size_t>(C) + static_cast<std::size_t>(c);
}

static std::vector<std::uint16_t> from_device_u16(const void* ptr, std::size_t n) {
    std::vector<std::uint16_t> out(n);
    cudaMemcpy(out.data(), ptr, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return out;
}

static int verify_u16_equal(const char* label, const std::vector<std::uint16_t>& got,
                            const std::vector<std::uint16_t>& ref) {
    if (got.size() != ref.size()) {
        std::cerr << label << ": size mismatch got=" << got.size() << " ref=" << ref.size()
                  << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != ref[i]) {
            std::cerr << label << ": bit mismatch at " << i << " got=0x" << std::hex << got[i]
                      << " ref=0x" << ref[i] << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

static void cpu_prefill_ref(const std::vector<float>& x, const std::vector<float>& weight,
                            const std::vector<float>& initial_state, std::int32_t C, std::int32_t T,
                            std::vector<double>& out, std::vector<double>& final_state) {
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t c = 0; c < C; ++c) {
            double acc = 0.0;
            for (std::int32_t j = 0; j < 4; ++j) {
                const std::int32_t src_t = t - 3 + j;
                const double xv          = (src_t < 0)
                                               ? static_cast<double>(initial_state[idx2(c, 3 + src_t, C)])
                                               : static_cast<double>(x[idx2(c, src_t, C)]);
                acc += static_cast<double>(weight[idx2(c, j, C)]) * xv;
            }
            out[idx2(c, t, C)] = silu(acc);
        }
    }

    for (std::int32_t s = 0; s < 3; ++s) {
        const std::int32_t seq_pos = T + s;
        for (std::int32_t c = 0; c < C; ++c) {
            final_state[idx2(c, s, C)] =
                (seq_pos < 3) ? static_cast<double>(initial_state[idx2(c, seq_pos, C)])
                              : static_cast<double>(x[idx2(c, seq_pos - 3, C)]);
        }
    }
}

static void cpu_decode_ref(const std::vector<float>& x, const std::vector<float>& weight,
                           const std::vector<float>& initial_state, std::int32_t C,
                           std::vector<double>& out, std::vector<double>& final_state) {
    for (std::int32_t c = 0; c < C; ++c) {
        double acc = 0.0;
        for (std::int32_t j = 0; j < 3; ++j) {
            acc += static_cast<double>(weight[idx2(c, j, C)]) *
                   static_cast<double>(initial_state[idx2(c, j, C)]);
        }
        acc += static_cast<double>(weight[idx2(c, 3, C)]) * static_cast<double>(x[c]);
        out[c] = silu(acc);

        final_state[idx2(c, 0, C)] = static_cast<double>(initial_state[idx2(c, 1, C)]);
        final_state[idx2(c, 1, C)] = static_cast<double>(initial_state[idx2(c, 2, C)]);
        final_state[idx2(c, 2, C)] = static_cast<double>(x[c]);
    }
}

static int one_prefill_shape(const char* tag, std::int32_t C, std::int32_t T, std::uint32_t seed,
                             float lo = -8.f, float hi = 8.f) {
    const std::size_t n        = static_cast<std::size_t>(C) * static_cast<std::size_t>(T);
    const std::size_t state_n  = static_cast<std::size_t>(C) * 3u;
    const std::size_t weight_n = static_cast<std::size_t>(C) * 4u;

    std::vector<float> x(n), weight(weight_n), state(state_n);
    fill_uniform(x, seed, lo, hi);
    fill_uniform(weight, seed + 1000u, lo, hi);
    fill_uniform(state, seed + 2000u, lo, hi);
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(state);

    std::vector<double> ref_out(n), ref_state(state_n);
    cpu_prefill_ref(x, weight, state, C, T, ref_out, ref_state);

    DBuf dx = to_device_bf16(x), dw = to_device_bf16(weight), dstate = to_device_bf16(state),
         dout(n * 2);
    Tensor tx(dx.p, DType::BF16, {C, T});
    Tensor tw(dw.p, DType::BF16, {C, 4});
    Tensor ts(dstate.p, DType::BF16, {C, 3});
    Tensor tout(dout.p, DType::BF16, {C, T});
    kernels::causal_conv1d_prefill(tx, tw, ts, tout, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f += verify((std::string(tag) + " out").c_str(), from_device_bf16(dout, n), ref_out,
                Tolerance::bf16_reduction());
    f += verify((std::string(tag) + " state").c_str(), from_device_bf16(dstate, state_n), ref_state,
                Tolerance::bf16_reduction());
    return f;
}

static int one_decode_shape(const char* tag, std::int32_t C, std::uint32_t seed, float lo = -8.f,
                            float hi = 8.f) {
    const std::size_t state_n  = static_cast<std::size_t>(C) * 3u;
    const std::size_t weight_n = static_cast<std::size_t>(C) * 4u;

    std::vector<float> x(C), weight(weight_n), state(state_n);
    fill_uniform(x, seed, lo, hi);
    fill_uniform(weight, seed + 1000u, lo, hi);
    fill_uniform(state, seed + 2000u, lo, hi);
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(state);

    std::vector<double> ref_out(C), ref_state(state_n);
    cpu_decode_ref(x, weight, state, C, ref_out, ref_state);

    DBuf dx = to_device_bf16(x), dw = to_device_bf16(weight), dstate = to_device_bf16(state),
         dout(static_cast<std::size_t>(C) * 2u);
    Tensor tx(dx.p, DType::BF16, {C, 1});
    Tensor tw(dw.p, DType::BF16, {C, 4});
    Tensor ts(dstate.p, DType::BF16, {C, 3});
    Tensor tout(dout.p, DType::BF16, {C, 1});
    kernels::causal_conv1d_decode(tx, tw, ts, tout, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f += verify((std::string(tag) + " out").c_str(), from_device_bf16(dout, C), ref_out,
                Tolerance::bf16_reduction());
    f += verify((std::string(tag) + " state").c_str(), from_device_bf16(dstate, state_n), ref_state,
                Tolerance::bf16_reduction());
    return f;
}

static int decode_chain_equivalence(std::uint32_t seed) {
    constexpr std::int32_t C   = 10240;
    constexpr std::int32_t T   = 5;
    const std::size_t n        = static_cast<std::size_t>(C) * T;
    const std::size_t state_n  = static_cast<std::size_t>(C) * 3u;
    const std::size_t weight_n = static_cast<std::size_t>(C) * 4u;

    std::vector<float> x(n), weight(weight_n), state(state_n);
    fill_uniform(x, seed, -8.f, 8.f);
    fill_uniform(weight, seed + 1000u, -8.f, 8.f);
    fill_uniform(state, seed + 2000u, -8.f, 8.f);
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(state);

    DBuf dx_prefill = to_device_bf16(x), dw_prefill = to_device_bf16(weight),
         dstate_prefill = to_device_bf16(state), dout_prefill(n * 2);
    Tensor tx_prefill(dx_prefill.p, DType::BF16, {C, T});
    Tensor tw_prefill(dw_prefill.p, DType::BF16, {C, 4});
    Tensor ts_prefill(dstate_prefill.p, DType::BF16, {C, 3});
    Tensor tout_prefill(dout_prefill.p, DType::BF16, {C, T});
    kernels::causal_conv1d_prefill(tx_prefill, tw_prefill, ts_prefill, tout_prefill, nullptr);
    cudaDeviceSynchronize();

    DBuf dx_decode = to_device_bf16(x), dw_decode = to_device_bf16(weight),
         dstate_decode = to_device_bf16(state), dout_decode(n * 2);
    Tensor tw_decode(dw_decode.p, DType::BF16, {C, 4});
    Tensor ts_decode(dstate_decode.p, DType::BF16, {C, 3});
    for (std::int32_t t = 0; t < T; ++t) {
        auto* x_step = static_cast<unsigned char*>(dx_decode.p) +
                       static_cast<std::size_t>(t) * static_cast<std::size_t>(C) * 2u;
        Tensor tx_step(x_step, DType::BF16, {C, 1});
        auto* out_step = static_cast<unsigned char*>(dout_decode.p) +
                         static_cast<std::size_t>(t) * static_cast<std::size_t>(C) * 2u;
        Tensor tout_step(out_step, DType::BF16, {C, 1});
        kernels::causal_conv1d_decode(tx_step, tw_decode, ts_decode, tout_step, nullptr);
    }
    cudaDeviceSynchronize();

    int f = 0;
    f += verify("causal_conv1d decode-chain out", from_device_bf16(dout_decode, n),
                from_device_bf16(dout_prefill, n), Tolerance::bf16_reduction());
    f += verify("causal_conv1d decode-chain state", from_device_bf16(dstate_decode, state_n),
                from_device_bf16(dstate_prefill, state_n), Tolerance::bf16_reduction());
    return f;
}

static int snapshot_chain_equivalence(std::uint32_t seed, std::int32_t T, float lo = -8.f,
                                      float hi = 8.f) {
    constexpr std::int32_t C   = 10240;
    const std::size_t n        = static_cast<std::size_t>(C) * static_cast<std::size_t>(T);
    const std::size_t state_n  = static_cast<std::size_t>(C) * 3u;
    const std::size_t weight_n = static_cast<std::size_t>(C) * 4u;

    std::vector<float> x(n), weight(weight_n), state(state_n);
    fill_uniform(x, seed, lo, hi);
    fill_uniform(weight, seed + 1000u, lo, hi);
    fill_uniform(state, seed + 2000u, lo, hi);
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(state);

    std::vector<float> snapshot_state(state_n * static_cast<std::size_t>(T), 17.0f);
    std::copy(state.begin(), state.end(), snapshot_state.begin());

    DBuf dx_snapshot = to_device_bf16(x), dw_snapshot = to_device_bf16(weight),
         dstates_snapshot = to_device_bf16(snapshot_state), dout_snapshot(n * 2),
         dinitial_slot = to_device_i32({0});
    Tensor tx_snapshot(dx_snapshot.p, DType::BF16, {C, T});
    Tensor tw_snapshot(dw_snapshot.p, DType::BF16, {C, 4});
    Tensor ts_snapshot(dstates_snapshot.p, DType::BF16, {C, 3, T});
    Tensor tinitial_slot(dinitial_slot.p, DType::I32, {1});
    Tensor tout_snapshot(dout_snapshot.p, DType::BF16, {C, T});
    kernels::causal_conv1d_sequence_snapshot(tx_snapshot, tw_snapshot, ts_snapshot, tinitial_slot,
                                             tout_snapshot, nullptr);
    cudaDeviceSynchronize();

    DBuf dx_decode = to_device_bf16(x), dw_decode = to_device_bf16(weight),
         dstate_decode = to_device_bf16(state), dout_decode(n * 2);
    Tensor tw_decode(dw_decode.p, DType::BF16, {C, 4});
    Tensor ts_decode(dstate_decode.p, DType::BF16, {C, 3});

    std::vector<std::uint16_t> expected_slots(state_n * static_cast<std::size_t>(T));
    for (std::int32_t t = 0; t < T; ++t) {
        auto* x_step = static_cast<unsigned char*>(dx_decode.p) +
                       static_cast<std::size_t>(t) * static_cast<std::size_t>(C) * 2u;
        Tensor tx_step(x_step, DType::BF16, {C, 1});
        auto* out_step = static_cast<unsigned char*>(dout_decode.p) +
                         static_cast<std::size_t>(t) * static_cast<std::size_t>(C) * 2u;
        Tensor tout_step(out_step, DType::BF16, {C, 1});
        kernels::causal_conv1d_decode(tx_step, tw_decode, ts_decode, tout_step, nullptr);
        cudaDeviceSynchronize();
        const auto state_bits = from_device_u16(dstate_decode.p, state_n);
        std::memcpy(expected_slots.data() + static_cast<std::size_t>(t) * state_n,
                    state_bits.data(), state_n * sizeof(std::uint16_t));
    }
    cudaDeviceSynchronize();

    const std::string tag = "causal_conv1d snapshot T=" + std::to_string(T);
    int f                 = 0;
    f += verify_u16_equal((tag + " out bits").c_str(), from_device_u16(dout_snapshot.p, n),
                          from_device_u16(dout_decode.p, n));
    f += verify_u16_equal((tag + " state slot bits").c_str(),
                          from_device_u16(dstates_snapshot.p, state_n * static_cast<std::size_t>(T)),
                          expected_slots);
    return f;
}

static int selected_slot_snapshot_equivalence(std::uint32_t seed, std::int32_t T,
                                              std::int32_t initial_slot) {
    constexpr std::int32_t C     = 10240;
    constexpr std::int32_t Slots = 6;
    const std::size_t n          = static_cast<std::size_t>(C) * static_cast<std::size_t>(T);
    const std::size_t state_n    = static_cast<std::size_t>(C) * 3u;
    const std::size_t weight_n   = static_cast<std::size_t>(C) * 4u;

    std::vector<float> x(n), weight(weight_n), state(state_n);
    fill_uniform(x, seed, -8.f, 8.f);
    fill_uniform(weight, seed + 1000u, -8.f, 8.f);
    fill_uniform(state, seed + 2000u, -8.f, 8.f);
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(state);

    std::vector<float> snapshot_state(state_n * Slots, 17.0f);
    for (std::size_t i = 0; i < snapshot_state.size(); ++i) {
        snapshot_state[i] = bf16_to_f32(f32_to_bf16(snapshot_state[i]));
    }
    std::copy(state.begin(), state.end(),
              snapshot_state.begin() + static_cast<std::size_t>(initial_slot) * state_n);

    DBuf dx_snapshot = to_device_bf16(x), dw_snapshot = to_device_bf16(weight),
         dstates_snapshot = to_device_bf16(snapshot_state), dout_snapshot(n * 2),
         dinitial_slot = to_device_i32({initial_slot});
    Tensor tx_snapshot(dx_snapshot.p, DType::BF16, {C, T});
    Tensor tw_snapshot(dw_snapshot.p, DType::BF16, {C, 4});
    Tensor ts_snapshot(dstates_snapshot.p, DType::BF16, {C, 3, Slots});
    Tensor tinitial_slot(dinitial_slot.p, DType::I32, {1});
    Tensor tout_snapshot(dout_snapshot.p, DType::BF16, {C, T});
    kernels::causal_conv1d_sequence_snapshot(tx_snapshot, tw_snapshot, ts_snapshot, tinitial_slot,
                                             tout_snapshot, nullptr);
    cudaDeviceSynchronize();

    DBuf dx_decode = to_device_bf16(x), dw_decode = to_device_bf16(weight),
         dstate_decode = to_device_bf16(state), dout_decode(n * 2);
    Tensor tw_decode(dw_decode.p, DType::BF16, {C, 4});
    Tensor ts_decode(dstate_decode.p, DType::BF16, {C, 3});

    std::vector<std::uint16_t> expected_slots(snapshot_state.size());
    for (std::size_t i = 0; i < snapshot_state.size(); ++i) {
        expected_slots[i] = f32_to_bf16(snapshot_state[i]);
    }
    for (std::int32_t t = 0; t < T; ++t) {
        auto* x_step = static_cast<unsigned char*>(dx_decode.p) +
                       static_cast<std::size_t>(t) * static_cast<std::size_t>(C) * 2u;
        Tensor tx_step(x_step, DType::BF16, {C, 1});
        auto* out_step = static_cast<unsigned char*>(dout_decode.p) +
                         static_cast<std::size_t>(t) * static_cast<std::size_t>(C) * 2u;
        Tensor tout_step(out_step, DType::BF16, {C, 1});
        kernels::causal_conv1d_decode(tx_step, tw_decode, ts_decode, tout_step, nullptr);
        cudaDeviceSynchronize();
        const auto state_bits = from_device_u16(dstate_decode.p, state_n);
        std::memcpy(expected_slots.data() + static_cast<std::size_t>(t) * state_n,
                    state_bits.data(), state_n * sizeof(std::uint16_t));
    }
    cudaDeviceSynchronize();

    const std::string tag = "causal_conv1d selected snapshot slot=" +
                            std::to_string(initial_slot) + " T=" + std::to_string(T);
    int f = 0;
    f += verify_u16_equal((tag + " out bits").c_str(), from_device_u16(dout_snapshot.p, n),
                          from_device_u16(dout_decode.p, n));
    f += verify_u16_equal((tag + " slots").c_str(),
                          from_device_u16(dstates_snapshot.p, expected_slots.size()),
                          expected_slots);
    return f;
}

static int prefill_state_carry_equivalence(std::uint32_t seed, std::int32_t C, std::int32_t T,
                                           std::int32_t split) {
    const std::size_t n        = static_cast<std::size_t>(C) * T;
    const std::size_t state_n  = static_cast<std::size_t>(C) * 3u;
    const std::size_t weight_n = static_cast<std::size_t>(C) * 4u;

    std::vector<float> x(n), weight(weight_n), state(state_n);
    fill_uniform(x, seed, -8.f, 8.f);
    fill_uniform(weight, seed + 1000u, -8.f, 8.f);
    fill_uniform(state, seed + 2000u, -8.f, 8.f);
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(state);

    DBuf dx_whole = to_device_bf16(x), dw_whole = to_device_bf16(weight),
         dstate_whole = to_device_bf16(state), dout_whole(n * 2);
    Tensor tx_whole(dx_whole.p, DType::BF16, {C, T});
    Tensor tw_whole(dw_whole.p, DType::BF16, {C, 4});
    Tensor ts_whole(dstate_whole.p, DType::BF16, {C, 3});
    Tensor tout_whole(dout_whole.p, DType::BF16, {C, T});
    kernels::causal_conv1d_prefill(tx_whole, tw_whole, ts_whole, tout_whole, nullptr);
    cudaDeviceSynchronize();

    DBuf dx_split = to_device_bf16(x), dw_split = to_device_bf16(weight),
         dstate_split = to_device_bf16(state), dout_split(n * 2);
    Tensor tx_split(dx_split.p, DType::BF16, {C, T});
    Tensor tw_split(dw_split.p, DType::BF16, {C, 4});
    Tensor ts_split(dstate_split.p, DType::BF16, {C, 3});
    Tensor tout_split(dout_split.p, DType::BF16, {C, T});

    Tensor x0   = tx_split.slice(1, 0, split);
    Tensor out0 = tout_split.slice(1, 0, split);
    kernels::causal_conv1d_prefill(x0, tw_split, ts_split, out0, nullptr);
    const std::int32_t tail = T - split;
    Tensor x1               = tx_split.slice(1, split, tail);
    Tensor out1             = tout_split.slice(1, split, tail);
    kernels::causal_conv1d_prefill(x1, tw_split, ts_split, out1, nullptr);
    cudaDeviceSynchronize();

    const std::string tag = "causal_conv1d state-carry C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " split=" + std::to_string(split);
    int f = 0;
    f += verify((tag + " out").c_str(), from_device_bf16(dout_split, n),
                from_device_bf16(dout_whole, n), Tolerance::bf16_reduction());
    f += verify((tag + " state").c_str(), from_device_bf16(dstate_split, state_n),
                from_device_bf16(dstate_whole, state_n), Tolerance::bf16_reduction());
    return f;
}

// Prefix-append parity: reading the initial width-3 window from a selected slot and
// writing the running window to slot 0 must match the in-place run seeded with the
// same initial window. Same math -> bit-exact. read_slot == 0 exercises the in-place
// equivalence; T < 3 exercises reading the initial window from the selected slot.
static int prefill_from_slot_equivalence(std::uint32_t seed, std::int32_t C, std::int32_t T,
                                         std::int32_t slots, std::int32_t read_slot) {
    const std::size_t n       = static_cast<std::size_t>(C) * T;
    const std::size_t state_n = static_cast<std::size_t>(C) * 3u;
    const std::size_t weight_n = static_cast<std::size_t>(C) * 4u;

    std::vector<float> x(n), weight(weight_n), state(state_n);
    fill_uniform(x, seed, -8.f, 8.f);
    fill_uniform(weight, seed + 1000u, -8.f, 8.f);
    fill_uniform(state, seed + 2000u, -8.f, 8.f);
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(state);

    DBuf rx = to_device_bf16(x), rw = to_device_bf16(weight), rstate = to_device_bf16(state),
         rout(n * 2);
    Tensor tx(rx.p, DType::BF16, {C, T});
    Tensor tw(rw.p, DType::BF16, {C, 4});
    Tensor ts(rstate.p, DType::BF16, {C, 3});
    Tensor tout(rout.p, DType::BF16, {C, T});
    kernels::causal_conv1d_prefill(tx, tw, ts, tout, nullptr);
    cudaDeviceSynchronize();

    DBuf fx = to_device_bf16(x), fw = to_device_bf16(weight), fout(n * 2);
    DBuf fstates(state_n * static_cast<std::size_t>(slots) * 2u);
    cudaMemset(fstates.p, 0, fstates.bytes);
    std::vector<std::uint16_t> state_bits(state_n);
    for (std::size_t i = 0; i < state_n; ++i) { state_bits[i] = f32_to_bf16(state[i]); }
    cudaMemcpy(static_cast<unsigned char*>(fstates.p) +
                   static_cast<std::size_t>(read_slot) * state_n * 2u,
               state_bits.data(), state_n * 2u, cudaMemcpyHostToDevice);

    Tensor fx_t(fx.p, DType::BF16, {C, T});
    Tensor fw_t(fw.p, DType::BF16, {C, 4});
    Tensor fin(static_cast<unsigned char*>(fstates.p) +
                   static_cast<std::size_t>(read_slot) * state_n * 2u,
               DType::BF16, {C, 3});
    Tensor fout_state(fstates.p, DType::BF16, {C, 3});
    Tensor fout_t(fout.p, DType::BF16, {C, T});
    kernels::causal_conv1d_prefill(fx_t, fw_t, fin, fout_state, fout_t, nullptr);
    cudaDeviceSynchronize();

    const std::string tag = "causal_conv1d from-slot C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " slots=" + std::to_string(slots) +
                            " read_slot=" + std::to_string(read_slot);
    int f = 0;
    f += verify_u16_equal((tag + " out bits").c_str(), from_device_u16(fout.p, n),
                          from_device_u16(rout.p, n));
    f += verify_u16_equal((tag + " slot0 state bits").c_str(),
                          from_device_u16(fstates.p, state_n),
                          from_device_u16(rstate.p, state_n));
    return f;
}

template <typename Fn>
static int expect_invalid(const char* label, const Fn& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& e) {
        std::cerr << label << ": expected invalid_argument, got " << e.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

template <typename Fn>
static int expect_overflow(const char* label, const Fn& fn) {
    try {
        fn();
    } catch (const std::overflow_error&) { return 0; } catch (const std::exception& e) {
        std::cerr << label << ": expected overflow_error, got " << e.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected overflow_error\n";
    return 1;
}

static int validation_checks() {
    int f                      = 0;
    constexpr std::int32_t C   = 10240;
    constexpr std::int32_t T   = 7;
    const std::size_t n        = static_cast<std::size_t>(C) * T;
    const std::size_t state_n  = static_cast<std::size_t>(C) * 3u;
    const std::size_t weight_n = static_cast<std::size_t>(C) * 4u;

    std::vector<float> x_h(n, 0.25f), w_h(weight_n, 0.5f), s_h(state_n, -0.25f);
    round_to_bf16(x_h);
    round_to_bf16(w_h);
    round_to_bf16(s_h);
    DBuf dx = to_device_bf16(x_h), dw = to_device_bf16(w_h), dstate = to_device_bf16(s_h),
         dout(n * 2);
    Tensor x(dx.p, DType::BF16, {C, T});
    Tensor weight(dw.p, DType::BF16, {C, 4});
    Tensor state(dstate.p, DType::BF16, {C, 3});
    Tensor out(dout.p, DType::BF16, {C, T});

    f += expect_invalid("causal_conv1d validation x dtype", [&] {
        Tensor bad = x;
        bad.dtype  = DType::FP32;
        kernels::causal_conv1d_prefill(bad, weight, state, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation weight dtype", [&] {
        Tensor bad = weight;
        bad.dtype  = DType::FP32;
        kernels::causal_conv1d_prefill(x, bad, state, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation state dtype", [&] {
        Tensor bad = state;
        bad.dtype  = DType::FP32;
        kernels::causal_conv1d_prefill(x, weight, bad, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation out dtype", [&] {
        Tensor bad = out;
        bad.dtype  = DType::FP32;
        kernels::causal_conv1d_prefill(x, weight, state, bad, nullptr);
    });
    f += expect_invalid("causal_conv1d validation out shape", [&] {
        Tensor bad = out;
        bad.ne[1]  = T + 1;
        kernels::causal_conv1d_prefill(x, weight, state, bad, nullptr);
    });
    f += expect_invalid("causal_conv1d validation weight C", [&] {
        Tensor bad = weight;
        bad.ne[0]  = C + 1;
        kernels::causal_conv1d_prefill(x, bad, state, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation weight taps", [&] {
        Tensor bad = weight;
        bad.ne[1]  = 5;
        kernels::causal_conv1d_prefill(x, bad, state, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation state C", [&] {
        Tensor bad = state;
        bad.ne[0]  = C + 1;
        kernels::causal_conv1d_prefill(x, weight, bad, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation state width", [&] {
        Tensor bad = state;
        bad.ne[1]  = 2;
        kernels::causal_conv1d_prefill(x, weight, bad, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation decode T",
                        [&] { kernels::causal_conv1d_decode(x, weight, state, out, nullptr); });
    f += expect_invalid("causal_conv1d validation snapshot T exceeds slots", [&] {
        Tensor bad_state(dstate.p, DType::BF16, {C, 3, T - 1});
        DBuf d_initial_slot = to_device_i32({0});
        Tensor initial_slot(d_initial_slot.p, DType::I32, {1});
        kernels::causal_conv1d_sequence_snapshot(x, weight, bad_state, initial_slot, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation snapshot initial_slot dtype", [&] {
        Tensor snapshot_state(dstate.p, DType::BF16, {C, 3, T});
        DBuf d_initial_slot(sizeof(std::int32_t));
        Tensor initial_slot(d_initial_slot.p, DType::FP32, {1});
        kernels::causal_conv1d_sequence_snapshot(x, weight, snapshot_state, initial_slot, out,
                                                 nullptr);
    });
    f += expect_invalid("causal_conv1d validation contiguous", [&] {
        Tensor bad = out;
        bad.nb[0]  = 4;
        kernels::causal_conv1d_prefill(x, weight, state, bad, nullptr);
    });
    f += expect_invalid("causal_conv1d validation null data", [&] {
        Tensor bad(nullptr, DType::BF16, {C, T});
        kernels::causal_conv1d_prefill(bad, weight, state, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation negative dim", [&] {
        Tensor bad = x;
        bad.ne[1]  = -1;
        kernels::causal_conv1d_prefill(bad, weight, state, out, nullptr);
    });
    f += expect_invalid("causal_conv1d validation zero C", [&] {
        Tensor zx = x;
        Tensor zw = weight;
        Tensor zs = state;
        Tensor zo = out;
        zx.ne[0] = zw.ne[0] = zs.ne[0] = zo.ne[0] = 0;
        kernels::causal_conv1d_prefill(zx, zw, zs, zo, nullptr);
    });
    f += expect_overflow("causal_conv1d validation overflow", [&] {
        Tensor hx(nullptr, DType::BF16, {1});
        Tensor hw(nullptr, DType::BF16, {1});
        Tensor hs(nullptr, DType::BF16, {1});
        Tensor ho(nullptr, DType::BF16, {1});
        for (int d = 0; d < 4; ++d) {
            hx.ne[d] = std::numeric_limits<std::int32_t>::max();
            ho.ne[d] = std::numeric_limits<std::int32_t>::max();
        }
        hw.ne[0] = std::numeric_limits<std::int32_t>::max();
        hw.ne[1] = 4;
        hw.ne[2] = 1;
        hw.ne[3] = 1;
        hs.ne[0] = std::numeric_limits<std::int32_t>::max();
        hs.ne[1] = 3;
        hs.ne[2] = 1;
        hs.ne[3] = 1;
        kernels::causal_conv1d_prefill(hx, hw, hs, ho, nullptr);
    });

    try {
        Tensor zx = x;
        Tensor zo = out;
        zx.ne[1]  = 0;
        zo.ne[1]  = 0;
        kernels::causal_conv1d_prefill(zx, weight, state, zo, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "causal_conv1d validation zero T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    return f;
}

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int f = 0;
    f += validation_checks();

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_prefill_shape("causal_conv1d prefill [10240,1]", 10240, 1, seed);
        f += one_prefill_shape("causal_conv1d prefill [10240,7]", 10240, 7, seed);
        f += one_prefill_shape("causal_conv1d prefill [10240,64]", 10240, 64, seed);
        f += one_prefill_shape("causal_conv1d prefill [10240,4096]", 10240, 4096, seed);
        f += one_decode_shape("causal_conv1d decode [10240,1]", 10240, seed + 3000u);
        f += one_prefill_shape("causal_conv1d prefill [10241,64]", 10241, 64, seed + 4000u);
        f += decode_chain_equivalence(seed + 5000u);
        for (std::int32_t T : {1, 2, 3, 4, 5, 6}) {
            f += snapshot_chain_equivalence(seed + 6000u + static_cast<std::uint32_t>(T), T);
        }
        f += selected_slot_snapshot_equivalence(seed + 7000u, 4, 0);
        f += selected_slot_snapshot_equivalence(seed + 7100u, 4, 2);
        f += selected_slot_snapshot_equivalence(seed + 7200u, 5, 5);
    }
    f += prefill_state_carry_equivalence(6061u, 10240, 257, 129);
    f += prefill_state_carry_equivalence(6067u, 10241, 257, 2);
    f += prefill_from_slot_equivalence(6071u, 10240, 7, 8, 0);
    f += prefill_from_slot_equivalence(6073u, 10240, 7, 8, 3);
    f += prefill_from_slot_equivalence(6075u, 10240, 64, 8, 5);
    f += prefill_from_slot_equivalence(6077u, 10240, 2, 8, 4);
    f += prefill_from_slot_equivalence(6079u, 10241, 7, 8, 3);
    f += one_prefill_shape("causal_conv1d stress prefill [10241,64] [-32,32]", 10241, 64, 4242u,
                           -32.f, 32.f);
    f += one_decode_shape("causal_conv1d stress decode [10240,1] [-32,32]", 10240, 4343u, -32.f,
                          32.f);
    f += snapshot_chain_equivalence(4444u, 6, -32.f, 32.f);

    std::cout << (f ? "FAIL" : "OK") << " causal_conv1d correctness\n";
    return f ? 1 : 0;
}
