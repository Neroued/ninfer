#include "ninfer/ops/gated_delta_rule.h"
#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int S    = 128;
constexpr int H_qk = 16;
constexpr int H_v  = 48;
constexpr int B    = 1;
constexpr int BT   = 64;

std::vector<std::uint16_t> from_device_u16(const void* ptr, std::size_t n) {
    std::vector<std::uint16_t> out(n);
    cudaMemcpy(out.data(), ptr, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return out;
}

std::vector<std::uint32_t> from_device_u32(const void* ptr, std::size_t n) {
    std::vector<std::uint32_t> out(n);
    cudaMemcpy(out.data(), ptr, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    return out;
}

std::uint32_t f32_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename T>
int verify_bits_equal(const char* label, const std::vector<T>& got, const std::vector<T>& ref) {
    if (got.size() != ref.size()) {
        std::cerr << label << ": size mismatch got=" << got.size() << " ref=" << ref.size() << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != ref[i]) {
            std::cerr << label << ": bit mismatch at " << i << " got=0x" << std::hex
                      << static_cast<std::uint64_t>(got[i]) << " ref=0x"
                      << static_cast<std::uint64_t>(ref[i]) << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

void fill_uniform_shared(std::vector<float>& buf, std::mt19937& gen, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (float& x : buf) { x = d(gen); }
}

gdn_ref::Inputs make_inputs_for_geometry(int head_dim, int qk_heads, int value_heads, int T,
                                         std::uint32_t seed, bool stress_g) {
    gdn_ref::Inputs in{};
    in.S    = head_dim;
    in.H_qk = qk_heads;
    in.H_v  = value_heads;
    in.T    = T;
    in.B    = B;
    in.q.resize(static_cast<std::size_t>(B * T * qk_heads * head_dim));
    in.k.resize(static_cast<std::size_t>(B * T * qk_heads * head_dim));
    in.v.resize(static_cast<std::size_t>(B * T * value_heads * head_dim));
    in.g.resize(static_cast<std::size_t>(B * T * value_heads));
    in.beta.resize(static_cast<std::size_t>(B * T * value_heads));
    in.state.resize(static_cast<std::size_t>(B * value_heads * head_dim * head_dim));

    std::mt19937 gen(seed);
    fill_uniform_shared(in.q, gen, -1.0f, 1.0f);
    fill_uniform_shared(in.k, gen, -1.0f, 1.0f);
    fill_uniform_shared(in.v, gen, -1.0f, 1.0f);
    fill_uniform_shared(in.g, gen, stress_g ? -1.0f : -4.0f, stress_g ? -0.05f : 0.0f);
    fill_uniform_shared(in.beta, gen, 0.05f, 0.95f);
    fill_uniform_shared(in.state, gen, -0.1f, 0.1f);

    l2_normalize_rows(in.q, head_dim, static_cast<long long>(B * T * qk_heads));
    l2_normalize_rows(in.k, head_dim, static_cast<long long>(B * T * qk_heads));
    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    return in;
}

gdn_ref::Inputs make_inputs(int T, std::uint32_t seed, bool stress_g) {
    return make_inputs_for_geometry(S, H_qk, H_v, T, seed, stress_g);
}

std::size_t align_up_size(std::size_t n, std::size_t align) {
    return (n + align - 1) & ~(align - 1);
}

std::size_t chunked_workspace_bytes(int T_full) {
    if (T_full <= 0) { return 0; }
    const std::size_t T  = static_cast<std::size_t>(T_full);
    const std::size_t NT = static_cast<std::size_t>((T_full + BT - 1) / BT);
    std::size_t off      = 0;
    auto reserve         = [&](std::size_t bytes) {
        if (bytes == 0) { return; }
        off = align_up_size(off + bytes, 256);
    };
    reserve(T * H_v * sizeof(float));                  // g_cumsum
    reserve(T * H_v * S * sizeof(std::uint16_t));      // W
    reserve(T * H_v * S * sizeof(std::uint16_t));      // U
    reserve(T * H_v * S * sizeof(std::uint16_t));      // v_new
    reserve(NT * H_v * S * S * sizeof(std::uint16_t)); // h_chunk
    return off;
}

std::size_t chunked_arena_bytes(int T) {
    const int T_full         = (T / BT) * BT;
    const std::size_t stages = chunked_workspace_bytes(T_full);
    return stages + 4 * 1024 * 1024;
}

struct GpuResult {
    std::vector<double> out;
    std::vector<double> state;
};

GpuResult run_recurrent_gpu(const gdn_ref::Inputs& in) {
    DBuf dq     = to_device_bf16(in.q);
    DBuf dk     = to_device_bf16(in.k);
    DBuf dv     = to_device_bf16(in.v);
    DBuf dg     = to_device_f32(in.g);
    DBuf dbeta  = to_device_f32(in.beta);
    DBuf dstate = to_device_f32(in.state);
    DBuf dout(in.v.size() * 2);

    Tensor tq(dq.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
    Tensor tk(dk.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
    Tensor tv(dv.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
    Tensor tg(dg.p, DType::FP32, {H_v, static_cast<int>(in.T)});
    Tensor tbeta(dbeta.p, DType::FP32, {H_v, static_cast<int>(in.T)});
    Tensor tstate(dstate.p, DType::FP32, {S, S, H_v});
    Tensor tout(dout.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
    WorkspaceArena ws(chunked_arena_bytes(static_cast<int>(in.T)));

    ops::gated_delta_rule(tq, tk, tv, tg, tbeta, 1.0f / std::sqrt(float(S)), ws, tstate, tout,
                          nullptr);
    cudaDeviceSynchronize();
    return {from_device_bf16(dout, in.v.size()), from_device_f32(dstate, in.state.size())};
}

GpuResult run_recurrent_gpu_stepped(const gdn_ref::Inputs& in) {
    DBuf dq     = to_device_bf16(in.q);
    DBuf dk     = to_device_bf16(in.k);
    DBuf dv     = to_device_bf16(in.v);
    DBuf dg     = to_device_f32(in.g);
    DBuf dbeta  = to_device_f32(in.beta);
    DBuf dstate = to_device_f32(in.state);
    DBuf dout(in.v.size() * 2);

    Tensor tq(dq.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
    Tensor tk(dk.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
    Tensor tv(dv.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
    Tensor tg(dg.p, DType::FP32, {H_v, static_cast<int>(in.T)});
    Tensor tbeta(dbeta.p, DType::FP32, {H_v, static_cast<int>(in.T)});
    Tensor tstate(dstate.p, DType::FP32, {S, S, H_v});
    Tensor tout(dout.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
    WorkspaceArena ws(chunked_arena_bytes(static_cast<int>(in.T)));

    for (int t = 0; t < static_cast<int>(in.T); ++t) {
        Tensor q_t    = tq.slice(2, t, 1);
        Tensor k_t    = tk.slice(2, t, 1);
        Tensor v_t    = tv.slice(2, t, 1);
        Tensor g_t    = tg.slice(1, t, 1);
        Tensor beta_t = tbeta.slice(1, t, 1);
        Tensor out_t  = tout.slice(2, t, 1);
        ops::gated_delta_rule(q_t, k_t, v_t, g_t, beta_t, 1.0f / std::sqrt(float(S)), ws, tstate,
                              out_t, nullptr);
    }
    cudaDeviceSynchronize();
    return {from_device_bf16(dout, in.v.size()), from_device_f32(dstate, in.state.size())};
}

int snapshot_chain_equivalence_case(int T, std::uint32_t seed, bool stress_g) {
    const auto in = make_inputs(T, seed, stress_g);

    std::vector<float> snapshot_state(in.state.size() * static_cast<std::size_t>(T), 17.0f);
    std::copy(in.state.begin(), in.state.end(), snapshot_state.begin());

    DBuf dq_snapshot     = to_device_bf16(in.q);
    DBuf dk_snapshot     = to_device_bf16(in.k);
    DBuf dv_snapshot     = to_device_bf16(in.v);
    DBuf dg_snapshot     = to_device_f32(in.g);
    DBuf dbeta_snapshot  = to_device_f32(in.beta);
    DBuf dstate_snapshot = to_device_f32(snapshot_state);
    DBuf dout_snapshot(in.v.size() * 2);
    DBuf dinitial_slot = to_device_i32({0});
    WorkspaceArena ws_snapshot(chunked_arena_bytes(T));

    Tensor tq_snapshot(dq_snapshot.p, DType::BF16, {S, H_qk, T});
    Tensor tk_snapshot(dk_snapshot.p, DType::BF16, {S, H_qk, T});
    Tensor tv_snapshot(dv_snapshot.p, DType::BF16, {S, H_v, T});
    Tensor tg_snapshot(dg_snapshot.p, DType::FP32, {H_v, T});
    Tensor tbeta_snapshot(dbeta_snapshot.p, DType::FP32, {H_v, T});
    Tensor tstate_snapshot(dstate_snapshot.p, DType::FP32, {S, S, H_v, T});
    Tensor tinitial_slot(dinitial_slot.p, DType::I32, {1});
    Tensor tout_snapshot(dout_snapshot.p, DType::BF16, {S, H_v, T});

    ops::gated_delta_rule_snapshot(tq_snapshot, tk_snapshot, tv_snapshot, tg_snapshot,
                                   tbeta_snapshot, 1.0f / std::sqrt(float(S)), ws_snapshot,
                                   tstate_snapshot, tinitial_slot, tout_snapshot, nullptr);
    cudaDeviceSynchronize();

    DBuf dq_step     = to_device_bf16(in.q);
    DBuf dk_step     = to_device_bf16(in.k);
    DBuf dv_step     = to_device_bf16(in.v);
    DBuf dg_step     = to_device_f32(in.g);
    DBuf dbeta_step  = to_device_f32(in.beta);
    DBuf dstate_step = to_device_f32(in.state);
    DBuf dout_step(in.v.size() * 2);
    WorkspaceArena ws_step(chunked_arena_bytes(T));

    Tensor tq_step(dq_step.p, DType::BF16, {S, H_qk, T});
    Tensor tk_step(dk_step.p, DType::BF16, {S, H_qk, T});
    Tensor tv_step(dv_step.p, DType::BF16, {S, H_v, T});
    Tensor tg_step(dg_step.p, DType::FP32, {H_v, T});
    Tensor tbeta_step(dbeta_step.p, DType::FP32, {H_v, T});
    Tensor tstate_step(dstate_step.p, DType::FP32, {S, S, H_v});
    Tensor tout_step(dout_step.p, DType::BF16, {S, H_v, T});

    std::vector<std::uint32_t> expected_slots(in.state.size() * static_cast<std::size_t>(T));
    for (int t = 0; t < T; ++t) {
        Tensor q_t    = tq_step.slice(2, t, 1);
        Tensor k_t    = tk_step.slice(2, t, 1);
        Tensor v_t    = tv_step.slice(2, t, 1);
        Tensor g_t    = tg_step.slice(1, t, 1);
        Tensor beta_t = tbeta_step.slice(1, t, 1);
        Tensor out_t  = tout_step.slice(2, t, 1);
        ops::gated_delta_rule(q_t, k_t, v_t, g_t, beta_t, 1.0f / std::sqrt(float(S)), ws_step,
                              tstate_step, out_t, nullptr);
        cudaDeviceSynchronize();
        const auto state_bits = from_device_u32(dstate_step.p, in.state.size());
        std::memcpy(expected_slots.data() + static_cast<std::size_t>(t) * in.state.size(),
                    state_bits.data(), in.state.size() * sizeof(std::uint32_t));
    }
    cudaDeviceSynchronize();

    const std::string tag =
        "gdn recurrent snapshot T=" + std::to_string(T) + (stress_g ? " stress" : " default");
    int failures = 0;
    failures += verify_bits_equal((tag + " out bits").c_str(),
                                  from_device_u16(dout_snapshot.p, in.v.size()),
                                  from_device_u16(dout_step.p, in.v.size()));
    failures += verify_bits_equal(
        (tag + " state slot bits").c_str(),
        from_device_u32(dstate_snapshot.p, in.state.size() * static_cast<std::size_t>(T)),
        expected_slots);
    return failures;
}

int selected_slot_snapshot_equivalence_case(int head_dim, int qk_heads, int value_heads, int T,
                                            int initial_slot, std::uint32_t seed) {
    constexpr int Slots = 7;
    const auto in       = make_inputs_for_geometry(head_dim, qk_heads, value_heads, T, seed, false);
    const std::size_t state_n = in.state.size();

    std::vector<float> snapshot_state(state_n * Slots, 17.0f);
    std::copy(in.state.begin(), in.state.end(),
              snapshot_state.begin() + static_cast<std::size_t>(initial_slot) * state_n);

    DBuf dq_snapshot     = to_device_bf16(in.q);
    DBuf dk_snapshot     = to_device_bf16(in.k);
    DBuf dv_snapshot     = to_device_bf16(in.v);
    DBuf dg_snapshot     = to_device_f32(in.g);
    DBuf dbeta_snapshot  = to_device_f32(in.beta);
    DBuf dstate_snapshot = to_device_f32(snapshot_state);
    DBuf dout_snapshot(in.v.size() * 2);
    DBuf dinitial_slot = to_device_i32({initial_slot});
    WorkspaceArena ws_snapshot(chunked_arena_bytes(T));

    Tensor tq_snapshot(dq_snapshot.p, DType::BF16, {head_dim, qk_heads, T});
    Tensor tk_snapshot(dk_snapshot.p, DType::BF16, {head_dim, qk_heads, T});
    Tensor tv_snapshot(dv_snapshot.p, DType::BF16, {head_dim, value_heads, T});
    Tensor tg_snapshot(dg_snapshot.p, DType::FP32, {value_heads, T});
    Tensor tbeta_snapshot(dbeta_snapshot.p, DType::FP32, {value_heads, T});
    Tensor tstate_snapshot(dstate_snapshot.p, DType::FP32,
                           {head_dim, head_dim, value_heads, Slots});
    Tensor tinitial_slot(dinitial_slot.p, DType::I32, {1});
    Tensor tout_snapshot(dout_snapshot.p, DType::BF16, {head_dim, value_heads, T});

    ops::gated_delta_rule_snapshot(tq_snapshot, tk_snapshot, tv_snapshot, tg_snapshot,
                                   tbeta_snapshot, 1.0f / std::sqrt(float(head_dim)), ws_snapshot,
                                   tstate_snapshot, tinitial_slot, tout_snapshot, nullptr);
    cudaDeviceSynchronize();

    DBuf dq_step     = to_device_bf16(in.q);
    DBuf dk_step     = to_device_bf16(in.k);
    DBuf dv_step     = to_device_bf16(in.v);
    DBuf dg_step     = to_device_f32(in.g);
    DBuf dbeta_step  = to_device_f32(in.beta);
    DBuf dstate_step = to_device_f32(in.state);
    DBuf dout_step(in.v.size() * 2);
    WorkspaceArena ws_step(chunked_arena_bytes(T));

    Tensor tq_step(dq_step.p, DType::BF16, {head_dim, qk_heads, T});
    Tensor tk_step(dk_step.p, DType::BF16, {head_dim, qk_heads, T});
    Tensor tv_step(dv_step.p, DType::BF16, {head_dim, value_heads, T});
    Tensor tg_step(dg_step.p, DType::FP32, {value_heads, T});
    Tensor tbeta_step(dbeta_step.p, DType::FP32, {value_heads, T});
    Tensor tstate_step(dstate_step.p, DType::FP32, {head_dim, head_dim, value_heads});
    Tensor tout_step(dout_step.p, DType::BF16, {head_dim, value_heads, T});

    std::vector<std::uint32_t> expected_slots(snapshot_state.size());
    for (std::size_t i = 0; i < snapshot_state.size(); ++i) {
        expected_slots[i] = f32_bits(snapshot_state[i]);
    }
    for (int t = 0; t < T; ++t) {
        Tensor q_t    = tq_step.slice(2, t, 1);
        Tensor k_t    = tk_step.slice(2, t, 1);
        Tensor v_t    = tv_step.slice(2, t, 1);
        Tensor g_t    = tg_step.slice(1, t, 1);
        Tensor beta_t = tbeta_step.slice(1, t, 1);
        Tensor out_t  = tout_step.slice(2, t, 1);
        ops::gated_delta_rule(q_t, k_t, v_t, g_t, beta_t, 1.0f / std::sqrt(float(head_dim)),
                              ws_step, tstate_step, out_t, nullptr);
        cudaDeviceSynchronize();
        const auto state_bits = from_device_u32(dstate_step.p, state_n);
        std::memcpy(expected_slots.data() + static_cast<std::size_t>(t) * state_n,
                    state_bits.data(), state_n * sizeof(std::uint32_t));
    }
    cudaDeviceSynchronize();

    const std::string tag = "gdn selected snapshot S=" + std::to_string(head_dim) +
                            " Hqk=" + std::to_string(qk_heads) +
                            " Hv=" + std::to_string(value_heads) +
                            " slot=" + std::to_string(initial_slot) + " T=" + std::to_string(T);
    int failures = 0;
    failures += verify_bits_equal((tag + " out bits").c_str(),
                                  from_device_u16(dout_snapshot.p, in.v.size()),
                                  from_device_u16(dout_step.p, in.v.size()));
    failures += verify_bits_equal((tag + " slots").c_str(),
                                  from_device_u32(dstate_snapshot.p, expected_slots.size()),
                                  expected_slots);
    return failures;
}

GpuResult run_chunked_gpu(const gdn_ref::Inputs& in) {
    DBuf dq     = to_device_bf16(in.q);
    DBuf dk     = to_device_bf16(in.k);
    DBuf dv     = to_device_bf16(in.v);
    DBuf dg     = to_device_f32(in.g);
    DBuf dbeta  = to_device_f32(in.beta);
    DBuf dstate = to_device_f32(in.state);
    DBuf dout(in.v.size() * 2);
    WorkspaceArena ws(chunked_arena_bytes(static_cast<int>(in.T)));

    Tensor tq(dq.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
    Tensor tk(dk.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
    Tensor tv(dv.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
    Tensor tg(dg.p, DType::FP32, {H_v, static_cast<int>(in.T)});
    Tensor tbeta(dbeta.p, DType::FP32, {H_v, static_cast<int>(in.T)});
    Tensor tstate(dstate.p, DType::FP32, {S, S, H_v});
    Tensor tout(dout.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});

    ops::gated_delta_rule(tq, tk, tv, tg, tbeta, 1.0f / std::sqrt(float(S)), ws, tstate, tout,
                          nullptr);
    cudaDeviceSynchronize();
    return {from_device_bf16(dout, in.v.size()), from_device_f32(dstate, in.state.size())};
}

GpuResult run_chunked_gpu_split(const gdn_ref::Inputs& in, int split) {
    DBuf dq     = to_device_bf16(in.q);
    DBuf dk     = to_device_bf16(in.k);
    DBuf dv     = to_device_bf16(in.v);
    DBuf dg     = to_device_f32(in.g);
    DBuf dbeta  = to_device_f32(in.beta);
    DBuf dstate = to_device_f32(in.state);
    DBuf dout(in.v.size() * 2);
    WorkspaceArena ws(chunked_arena_bytes(static_cast<int>(in.T)));

    Tensor tq(dq.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
    Tensor tk(dk.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
    Tensor tv(dv.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
    Tensor tg(dg.p, DType::FP32, {H_v, static_cast<int>(in.T)});
    Tensor tbeta(dbeta.p, DType::FP32, {H_v, static_cast<int>(in.T)});
    Tensor tstate(dstate.p, DType::FP32, {S, S, H_v});
    Tensor tout(dout.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});

    Tensor q0    = tq.slice(2, 0, split);
    Tensor k0    = tk.slice(2, 0, split);
    Tensor v0    = tv.slice(2, 0, split);
    Tensor g0    = tg.slice(1, 0, split);
    Tensor beta0 = tbeta.slice(1, 0, split);
    Tensor out0  = tout.slice(2, 0, split);
    ops::gated_delta_rule(q0, k0, v0, g0, beta0, 1.0f / std::sqrt(float(S)), ws, tstate, out0,
                          nullptr);

    const int tail = static_cast<int>(in.T) - split;
    Tensor q1      = tq.slice(2, split, tail);
    Tensor k1      = tk.slice(2, split, tail);
    Tensor v1      = tv.slice(2, split, tail);
    Tensor g1      = tg.slice(1, split, tail);
    Tensor beta1   = tbeta.slice(1, split, tail);
    Tensor out1    = tout.slice(2, split, tail);
    ops::gated_delta_rule(q1, k1, v1, g1, beta1, 1.0f / std::sqrt(float(S)), ws, tstate, out1,
                          nullptr);

    cudaDeviceSynchronize();
    return {from_device_bf16(dout, in.v.size()), from_device_f32(dstate, in.state.size())};
}

int general_geometry_case(int head_dim, int qk_heads, int value_heads, int T, std::uint32_t seed) {
    const auto in      = make_inputs_for_geometry(head_dim, qk_heads, value_heads, T, seed, false);
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<double> ref_out(in.v.size());
    std::vector<double> ref_state(in.state.size());
    gdn_ref::forward_chunked(in.q.data(), in.k.data(), in.v.data(), in.g.data(), in.beta.data(),
                             in.state.data(), ref_out.data(), ref_state.data(), head_dim, qk_heads,
                             value_heads, T, B, scale, BT);

    DBuf dq        = to_device_bf16(in.q);
    DBuf dk        = to_device_bf16(in.k);
    DBuf dv        = to_device_bf16(in.v);
    DBuf dg        = to_device_f32(in.g);
    DBuf dbeta     = to_device_f32(in.beta);
    DBuf dstate_in = to_device_f32(in.state);
    DBuf dstate_out(in.state.size() * sizeof(float));
    DBuf dout(in.v.size() * sizeof(std::uint16_t));

    Tensor tq(dq.p, DType::BF16, {head_dim, qk_heads, T});
    Tensor tk(dk.p, DType::BF16, {head_dim, qk_heads, T});
    Tensor tv(dv.p, DType::BF16, {head_dim, value_heads, T});
    Tensor tg(dg.p, DType::FP32, {value_heads, T});
    Tensor tbeta(dbeta.p, DType::FP32, {value_heads, T});
    Tensor tstate_in(dstate_in.p, DType::FP32, {head_dim, head_dim, value_heads});
    Tensor tstate_out(dstate_out.p, DType::FP32, {head_dim, head_dim, value_heads});
    Tensor tout(dout.p, DType::BF16, {head_dim, value_heads, T});
    const std::size_t workspace_bytes =
        ops::gated_delta_rule_workspace_bytes(head_dim, qk_heads, value_heads, T);
    WorkspaceArena ws(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_rule(tq, tk, tv, tg, tbeta, static_cast<float>(scale), ws, tstate_in,
                          tstate_out, tout, nullptr);
    cudaDeviceSynchronize();

    const std::string tag = "gdn geometry S=" + std::to_string(head_dim) +
                            " Hqk=" + std::to_string(qk_heads) +
                            " Hv=" + std::to_string(value_heads) + " T=" + std::to_string(T);
    int failures = 0;
    failures += verify((tag + " out").c_str(), from_device_bf16(dout, ref_out.size()), ref_out,
                       Tolerance::gdn_output_bf16());
    failures += verify((tag + " state").c_str(), from_device_f32(dstate_out, ref_state.size()),
                       ref_state, Tolerance::gdn_state_fp32());
    return failures;
}

int recurrent_case(int T, std::uint32_t seed, bool stress_g) {
    const auto in      = make_inputs(T, seed, stress_g);
    const double scale = 1.0 / std::sqrt(static_cast<double>(S));
    std::vector<double> ref_out(static_cast<std::size_t>(B * T * H_v * S));
    std::vector<double> ref_state(static_cast<std::size_t>(B * H_v * S * S));
    gdn_ref::forward_recurrent(in.q.data(), in.k.data(), in.v.data(), in.g.data(), in.beta.data(),
                               in.state.data(), ref_out.data(), ref_state.data(), S, H_qk, H_v, T,
                               B, scale);

    DBuf dq     = to_device_bf16(in.q);
    DBuf dk     = to_device_bf16(in.k);
    DBuf dv     = to_device_bf16(in.v);
    DBuf dg     = to_device_f32(in.g);
    DBuf dbeta  = to_device_f32(in.beta);
    DBuf dstate = to_device_f32(in.state);
    DBuf dout(ref_out.size() * 2);

    Tensor tq(dq.p, DType::BF16, {S, H_qk, T});
    Tensor tk(dk.p, DType::BF16, {S, H_qk, T});
    Tensor tv(dv.p, DType::BF16, {S, H_v, T});
    Tensor tg(dg.p, DType::FP32, {H_v, T});
    Tensor tbeta(dbeta.p, DType::FP32, {H_v, T});
    Tensor tout(dout.p, DType::BF16, {S, H_v, T});
    WorkspaceArena ws(chunked_arena_bytes(T));

    Tensor tstate(dstate.p, DType::FP32, {S, S, H_v});

    ops::gated_delta_rule(tq, tk, tv, tg, tbeta, static_cast<float>(scale), ws, tstate, tout,
                          nullptr);
    cudaDeviceSynchronize();

    const std::string tag =
        std::string("gdn recurrent T=") + std::to_string(T) + (stress_g ? " stress" : " default");
    int failures = 0;
    failures += verify((tag + " out").c_str(), from_device_bf16(dout, ref_out.size()), ref_out,
                       Tolerance::gdn_output_bf16());
    const std::vector<double> got_state = from_device_f32(dstate, ref_state.size());
    failures += verify((tag + " state").c_str(), got_state, ref_state, Tolerance::gdn_state_fp32());
    return failures;
}

int chunked_case(int T, std::uint32_t seed, bool compare_recurrent, bool compare_ar_golden,
                 bool stress_g = false) {
    const auto in      = make_inputs(T, seed, stress_g);
    const double scale = 1.0 / std::sqrt(static_cast<double>(S));
    std::vector<double> chunked_ref_out(static_cast<std::size_t>(B * T * H_v * S));
    std::vector<double> chunked_ref_state(static_cast<std::size_t>(B * H_v * S * S));
    gdn_ref::forward_chunked(in.q.data(), in.k.data(), in.v.data(), in.g.data(), in.beta.data(),
                             in.state.data(), chunked_ref_out.data(), chunked_ref_state.data(), S,
                             H_qk, H_v, T, B, scale, BT);

    std::vector<double> ar_ref_out;
    std::vector<double> ar_ref_state;
    if (compare_ar_golden) {
        ar_ref_out.resize(static_cast<std::size_t>(B * T * H_v * S));
        ar_ref_state.resize(static_cast<std::size_t>(B * H_v * S * S));
        gdn_ref::forward_recurrent(in.q.data(), in.k.data(), in.v.data(), in.g.data(),
                                   in.beta.data(), in.state.data(), ar_ref_out.data(),
                                   ar_ref_state.data(), S, H_qk, H_v, T, B, scale);
    }

    int failures = 0;
    try {
        const GpuResult got = run_chunked_gpu(in);
        const std::string tag =
            std::string("gdn chunked T=") + std::to_string(T) + (stress_g ? " slow-decay" : "");
        failures += verify((tag + " vs chunked-ref out").c_str(), got.out, chunked_ref_out,
                           Tolerance::gdn_output_bf16());
        failures += verify((tag + " vs chunked-ref state").c_str(), got.state, chunked_ref_state,
                           Tolerance::gdn_state_fp32());
        if (compare_ar_golden) {
            failures += verify((tag + " vs AR-golden out").c_str(), got.out, ar_ref_out,
                               Tolerance::gdn_output_bf16());
            failures += verify((tag + " vs AR-golden state").c_str(), got.state, ar_ref_state,
                               Tolerance::gdn_state_fp32());
        }

        if (compare_recurrent) {
            const GpuResult recurrent = run_recurrent_gpu(in);
            failures += verify((tag + " vs recurrent out").c_str(), got.out, recurrent.out,
                               Tolerance::gdn_output_bf16());
            failures += verify((tag + " vs recurrent state").c_str(), got.state, recurrent.state,
                               Tolerance::gdn_state_fp32());
        }
    } catch (const std::exception& e) {
        std::cerr << "gdn chunked T=" << T << (stress_g ? " slow-decay" : "")
                  << ": unexpected exception: " << e.what() << '\n';
        return 1;
    }

    return failures;
}

int chunked_chain_equivalence_case(int T, std::uint32_t seed) {
    const auto in = make_inputs(T, seed, false);

    int failures = 0;
    try {
        const GpuResult chunked = run_chunked_gpu(in);
        const GpuResult ar_step = run_recurrent_gpu_stepped(in);
        const std::string tag = std::string("gdn chunked chain-equivalence T=") + std::to_string(T);
        failures +=
            verify((tag + " out").c_str(), chunked.out, ar_step.out, Tolerance::gdn_output_bf16());
        failures += verify((tag + " state").c_str(), chunked.state, ar_step.state,
                           Tolerance::gdn_state_fp32());
    } catch (const std::exception& e) {
        std::cerr << "gdn chunked chain-equivalence T=" << T
                  << ": unexpected exception: " << e.what() << '\n';
        return 1;
    }

    return failures;
}

int chunked_state_carry_equivalence_case(int T, int split, std::uint32_t seed) {
    const auto in = make_inputs(T, seed, false);

    int failures = 0;
    try {
        const GpuResult whole     = run_chunked_gpu(in);
        const GpuResult split_run = run_chunked_gpu_split(in, split);
        const std::string tag     = std::string("gdn chunked state-carry T=") + std::to_string(T) +
                                " split=" + std::to_string(split);
        failures +=
            verify((tag + " out").c_str(), split_run.out, whole.out, Tolerance::gdn_output_bf16());
        failures += verify((tag + " state").c_str(), split_run.state, whole.state,
                           Tolerance::gdn_state_fp32());
    } catch (const std::exception& e) {
        std::cerr << "gdn chunked state-carry T=" << T << " split=" << split
                  << ": unexpected exception: " << e.what() << '\n';
        return 1;
    }

    return failures;
}

// Prefix-append parity: reading the initial recurrent state from a selected slot
// and writing the running state to slot 0 must match the in-place run seeded with
// the same initial state. Same math/chunk boundaries -> bit-exact. read_slot == 0
// exercises the in-place equivalence; read_slot > 0 with T < BT exercises the
// tail-only recurrent-inout path.
int chunked_from_slot_equivalence_case(int T, int slots, int read_slot, std::uint32_t seed) {
    const auto in                 = make_inputs(T, seed, false);
    const float scale             = 1.0f / std::sqrt(float(S));
    const std::size_t slot_floats = in.state.size();

    DBuf rq = to_device_bf16(in.q), rk = to_device_bf16(in.k), rv = to_device_bf16(in.v);
    DBuf rg = to_device_f32(in.g), rbeta = to_device_f32(in.beta), rstate = to_device_f32(in.state);
    DBuf rout(in.v.size() * 2);
    WorkspaceArena ws(chunked_arena_bytes(static_cast<int>(in.T)));

    int failures = 0;
    try {
        {
            Tensor tq(rq.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
            Tensor tk(rk.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
            Tensor tv(rv.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
            Tensor tg(rg.p, DType::FP32, {H_v, static_cast<int>(in.T)});
            Tensor tbeta(rbeta.p, DType::FP32, {H_v, static_cast<int>(in.T)});
            Tensor tstate(rstate.p, DType::FP32, {S, S, H_v});
            Tensor tout(rout.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
            ops::gated_delta_rule(tq, tk, tv, tg, tbeta, scale, ws, tstate, tout, nullptr);
        }
        cudaDeviceSynchronize();

        DBuf fq = to_device_bf16(in.q), fk = to_device_bf16(in.k), fv = to_device_bf16(in.v);
        DBuf fg = to_device_f32(in.g), fbeta = to_device_f32(in.beta);
        DBuf fstates(slot_floats * static_cast<std::size_t>(slots) * 4);
        cudaMemset(fstates.p, 0, fstates.bytes);
        cudaMemcpy(static_cast<float*>(fstates.p) +
                       static_cast<std::size_t>(read_slot) * slot_floats,
                   in.state.data(), slot_floats * 4, cudaMemcpyHostToDevice);
        DBuf fout(in.v.size() * 2);
        {
            Tensor tq(fq.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
            Tensor tk(fk.p, DType::BF16, {S, H_qk, static_cast<int>(in.T)});
            Tensor tv(fv.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
            Tensor tg(fg.p, DType::FP32, {H_v, static_cast<int>(in.T)});
            Tensor tbeta(fbeta.p, DType::FP32, {H_v, static_cast<int>(in.T)});
            Tensor tin(static_cast<float*>(fstates.p) +
                           static_cast<std::size_t>(read_slot) * slot_floats,
                       DType::FP32, {S, S, H_v});
            Tensor tout_state(fstates.p, DType::FP32, {S, S, H_v});
            Tensor tout(fout.p, DType::BF16, {S, H_v, static_cast<int>(in.T)});
            ops::gated_delta_rule(tq, tk, tv, tg, tbeta, scale, ws, tin, tout_state, tout, nullptr);
        }
        cudaDeviceSynchronize();

        const std::string tag = std::string("gdn chunked from-slot T=") + std::to_string(T) +
                                " slots=" + std::to_string(slots) +
                                " read_slot=" + std::to_string(read_slot);
        failures +=
            verify_bits_equal((tag + " out bits").c_str(), from_device_u16(fout.p, in.v.size()),
                              from_device_u16(rout.p, in.v.size()));
        failures += verify_bits_equal((tag + " slot0 state bits").c_str(),
                                      from_device_u32(fstates.p, slot_floats),
                                      from_device_u32(rstate.p, slot_floats));
    } catch (const std::exception& e) {
        std::cerr << "gdn chunked from-slot T=" << T << " read_slot=" << read_slot
                  << ": unexpected exception: " << e.what() << '\n';
        return 1;
    }
    return failures;
}

int validation_case() {
    try {
        Tensor q(nullptr, DType::BF16, {S, H_qk, 1});
        Tensor k(nullptr, DType::BF16, {S, H_qk, 1});
        Tensor v(nullptr, DType::BF16, {S, H_v, 1});
        Tensor g(nullptr, DType::FP32, {H_v, 1});
        Tensor beta(nullptr, DType::FP32, {H_v, 1});
        Tensor state(nullptr, DType::FP32, {S, S, H_v});
        Tensor out(nullptr, DType::BF16, {S, H_v, 1});
        WorkspaceArena ws(1024 * 1024);
        ops::gated_delta_rule(q, k, v, g, beta, 1.0f / std::sqrt(float(S)), ws, state, out,
                              nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "gdn recurrent null validation: expected invalid_argument\n";
    return 1;
}

int chunked_validation_case() {
    const auto in = make_inputs(BT, 5028u, false);
    DBuf dq       = to_device_bf16(in.q);
    DBuf dk       = to_device_bf16(in.k);
    DBuf dv       = to_device_bf16(in.v);
    DBuf dg       = to_device_f32(in.g);
    DBuf dbeta    = to_device_f32(in.beta);
    DBuf dstate   = to_device_f32(in.state);
    DBuf dout(in.v.size() * 2);
    WorkspaceArena ws(chunked_arena_bytes(BT));

    Tensor tq(dq.p, DType::BF16, {S, H_qk, BT});
    Tensor tk(dk.p, DType::BF16, {S, H_qk, BT});
    Tensor tv(dv.p, DType::BF16, {S, H_v, BT});
    Tensor tg(dg.p, DType::FP32, {H_v, BT});
    Tensor tbeta(dbeta.p, DType::FP32, {H_v, BT});
    Tensor tstate(dstate.p, DType::FP32, {S, S, H_v});
    Tensor tout(dout.p, DType::BF16, {S, H_v, BT});

    int failures = 0;
    try {
        Tensor tq_bad(dq.p, DType::BF16, {S - 1, H_qk, BT});
        ops::gated_delta_rule(tq_bad, tk, tv, tg, tbeta, 1.0f / std::sqrt(float(S)), ws, tstate,
                              tout, nullptr);
        std::cerr << "gdn chunked bad shape validation: expected invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& e) {
        std::cerr << "gdn chunked bad shape validation: wrong exception: " << e.what() << '\n';
        ++failures;
    }

    try {
        Tensor too_few_slots(dstate.p, DType::FP32, {S, S, H_v, BT - 1});
        DBuf d_initial_slot = to_device_i32({0});
        Tensor initial_slot(d_initial_slot.p, DType::I32, {1});
        ops::gated_delta_rule_snapshot(tq, tk, tv, tg, tbeta, 1.0f / std::sqrt(float(S)), ws,
                                       too_few_slots, initial_slot, tout, nullptr);
        std::cerr << "gdn snapshot T exceeds slots validation: expected invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& e) {
        std::cerr << "gdn snapshot T exceeds slots validation: wrong exception: " << e.what()
                  << '\n';
        ++failures;
    }

    try {
        Tensor states(dstate.p, DType::FP32, {S, S, H_v, BT});
        DBuf d_initial_slot(sizeof(std::int32_t));
        Tensor initial_slot(d_initial_slot.p, DType::FP32, {1});
        ops::gated_delta_rule_snapshot(tq, tk, tv, tg, tbeta, 1.0f / std::sqrt(float(S)), ws,
                                       states, initial_slot, tout, nullptr);
        std::cerr << "gdn snapshot bad initial_slot validation: expected invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    } catch (const std::exception& e) {
        std::cerr << "gdn snapshot bad initial_slot validation: wrong exception: " << e.what()
                  << '\n';
        ++failures;
    }

    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    for (int T : {1, 2, 7, 64}) {
        failures += recurrent_case(T, 2026u + static_cast<std::uint32_t>(T), false);
    }
    failures += recurrent_case(1, 3027u, true);
    failures += recurrent_case(7, 3033u, true);
    failures += recurrent_case(2, 4028u, false);
    for (std::uint32_t seed : {7028u, 8128u}) {
        for (int T : {1, 2, 3, 4, 5, 6}) {
            failures +=
                snapshot_chain_equivalence_case(T, seed + static_cast<std::uint32_t>(T), false);
        }
    }
    failures += snapshot_chain_equivalence_case(6, 9028u, true);
    failures += selected_slot_snapshot_equivalence_case(S, H_qk, H_v, 4, 0, 10028u);
    failures += selected_slot_snapshot_equivalence_case(S, H_qk, H_v, 4, 2, 10038u);
    failures += selected_slot_snapshot_equivalence_case(S, H_qk, H_v, 5, 5, 10048u);
    failures += selected_slot_snapshot_equivalence_case(128, 16, 32, 6, 6, 10058u);
    failures += validation_case();
    failures += general_geometry_case(16, 4, 4, 1, 12016u);
    failures += general_geometry_case(32, 4, 8, 3, 12032u);
    failures += general_geometry_case(64, 4, 16, 70, 12064u);
    failures += general_geometry_case(128, 16, 32, 128, 12128u);
    failures += chunked_case(32, 4528u, true, true);
    failures += chunked_case(64, 4090u, true, true);
    failures += chunked_case(128, 4154u, true, true);
    failures += chunked_case(200, 4226u, true, true);
    failures += chunked_case(256, 4282u, true, true);
    failures += chunked_case(512, 8534u, true, false);
    failures += chunked_chain_equivalence_case(128, 6122u);
    failures += chunked_case(4096, 8122u, false, false);
    failures += chunked_case(4096, 9122u, false, false, true);
    failures += chunked_state_carry_equivalence_case(200, 96, 9822u);
    failures += chunked_state_carry_equivalence_case(512, 128, 9922u);
    failures += chunked_from_slot_equivalence_case(7, 8, 0, 11028u);
    failures += chunked_from_slot_equivalence_case(7, 8, 3, 11038u);
    failures += chunked_from_slot_equivalence_case(64, 8, 5, 11048u);
    failures += chunked_from_slot_equivalence_case(128, 8, 3, 11058u);
    failures += chunked_from_slot_equivalence_case(200, 8, 7, 11068u);
    failures += chunked_validation_case();

    std::cout << (failures ? "FAIL" : "OK") << " gated_delta_rule correctness\n";
    return failures ? 1 : 0;
}
