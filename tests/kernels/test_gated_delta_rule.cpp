#include "qus/kernels/gated_delta_rule.h"
#include "qus/core/state_store.h"
#include "kernels/gdn_ref.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

constexpr int S    = 128;
constexpr int H_qk = 16;
constexpr int H_v  = 48;
constexpr int B    = 1;

void fill_uniform_shared(std::vector<float>& buf, std::mt19937& gen, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (float& x : buf) { x = d(gen); }
}

gdn_ref::Inputs make_inputs(int T, std::uint32_t seed, bool stress_g) {
    gdn_ref::Inputs in{};
    in.S    = S;
    in.H_qk = H_qk;
    in.H_v  = H_v;
    in.T    = T;
    in.B    = B;
    in.q.resize(static_cast<std::size_t>(B * T * H_qk * S));
    in.k.resize(static_cast<std::size_t>(B * T * H_qk * S));
    in.v.resize(static_cast<std::size_t>(B * T * H_v * S));
    in.g.resize(static_cast<std::size_t>(B * T * H_v));
    in.beta.resize(static_cast<std::size_t>(B * T * H_v));
    in.state.resize(static_cast<std::size_t>(B * H_v * S * S));

    std::mt19937 gen(seed);
    fill_uniform_shared(in.q, gen, -1.0f, 1.0f);
    fill_uniform_shared(in.k, gen, -1.0f, 1.0f);
    fill_uniform_shared(in.v, gen, -1.0f, 1.0f);
    fill_uniform_shared(in.g, gen, stress_g ? -1.0f : -4.0f, stress_g ? -0.05f : 0.0f);
    fill_uniform_shared(in.beta, gen, 0.05f, 0.95f);
    fill_uniform_shared(in.state, gen, -0.1f, 0.1f);

    l2_normalize_rows(in.q, S, static_cast<long long>(B * T * H_qk));
    l2_normalize_rows(in.k, S, static_cast<long long>(B * T * H_qk));
    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    return in;
}

int recurrent_case(int T, std::uint32_t seed, bool stress_g, bool use_gdn_state = false) {
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

    Tensor tstate(dstate.p, DType::FP32, {S, S, H_v});
    std::optional<DeviceArena> state_arena;
    std::optional<GdnState> gdn_state;
    if (use_gdn_state) {
        state_arena.emplace(4 * 1024 * 1024);
        gdn_state.emplace(*state_arena, 1, 1, 1, H_v, S, S);
        cudaMemcpy(gdn_state->ssm[0].data, in.state.data(), in.state.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        tstate = gdn_state->ssm[0];
    }

    kernels::gated_delta_rule_recurrent(tq, tk, tv, tg, tbeta, static_cast<float>(scale), tstate,
                                        tout, nullptr);
    cudaDeviceSynchronize();

    const std::string tag =
        std::string("gdn recurrent T=") + std::to_string(T) + (stress_g ? " stress" : " default");
    int failures = 0;
    failures += verify((tag + " out").c_str(), from_device_bf16(dout, ref_out.size()), ref_out,
                       Tolerance::gdn_output_bf16());
    std::vector<double> got_state;
    if (use_gdn_state) {
        std::vector<float> state_f32(ref_state.size());
        cudaMemcpy(state_f32.data(), gdn_state->ssm[0].data, state_f32.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        got_state.resize(state_f32.size());
        for (std::size_t i = 0; i < state_f32.size(); ++i) { got_state[i] = state_f32[i]; }
    } else {
        got_state = from_device_f32(dstate, ref_state.size());
    }
    failures += verify((tag + " state").c_str(), got_state, ref_state, Tolerance::gdn_state_fp32());
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
        kernels::gated_delta_rule_recurrent(q, k, v, g, beta, 1.0f / std::sqrt(float(S)), state,
                                            out, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "gdn recurrent null validation: expected invalid_argument\n";
    return 1;
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
    failures += recurrent_case(2, 4028u, false, true);
    failures += validation_case();

    std::cout << (failures ? "FAIL" : "OK") << " gated_delta_rule_recurrent correctness\n";
    return failures ? 1 : 0;
}
