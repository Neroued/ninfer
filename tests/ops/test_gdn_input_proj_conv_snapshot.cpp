#include "ninfer/ops/gdn_input_proj.h"

#include "ops/op_tester.h"
#include "ops/row_split_pack.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kQueryRows = 2048;
constexpr int kKeyRows   = 2048;
constexpr int kSlots     = 7;
constexpr int kInitial   = 6;

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    std::vector<std::int32_t> result;
    for (const int row : {0, 1, rows / 3, rows / 2, rows - 2, rows - 1}) {
        if (std::find(result.begin(), result.end(), row) == result.end()) result.push_back(row);
    }
    return result;
}

DBuf to_device_bits(const std::vector<std::uint16_t>& bits) {
    DBuf device(bits.size() * sizeof(std::uint16_t));
    cudaMemcpy(device.p, bits.data(), device.bytes, cudaMemcpyHostToDevice);
    return device;
}

std::vector<std::uint16_t> from_device_bits(const DBuf& device, std::size_t count) {
    std::vector<std::uint16_t> bits(count);
    cudaMemcpy(bits.data(), device.p, bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return bits;
}

std::vector<double> from_device_values(const DBuf& device, std::size_t count) {
    const auto bits = from_device_bits(device, count);
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i) values[i] = bf16_to_f32(bits[i]);
    return values;
}

double silu(double x) {
    if (x >= 0.0) return x / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return x * e / (1.0 + e);
}

struct OracleViews {
    std::vector<double> query;
    std::vector<double> key;
    std::vector<double> value;
    std::vector<double> state;
};

template <class Projection>
OracleViews make_oracle(std::int32_t tokens, std::int32_t value_rows,
                        const std::vector<float>& conv_weight,
                        const std::vector<std::uint16_t>& initial_state, Projection&& projection) {
    const int channels = kQueryRows + kKeyRows + value_rows;
    OracleViews result;
    const auto q_rows = sampled_rows(kQueryRows);
    const auto k_rows = sampled_rows(kKeyRows);
    const auto v_rows = sampled_rows(value_rows);
    result.query.reserve(q_rows.size() * tokens);
    result.key.reserve(k_rows.size() * tokens);
    result.value.reserve(v_rows.size() * tokens);
    result.state.reserve((q_rows.size() + k_rows.size() + v_rows.size()) * 3 * tokens);

    const auto evaluate = [&](int global_row, std::vector<double>& output) {
        double s0       = bf16_to_f32(initial_state[global_row]);
        double s1       = bf16_to_f32(initial_state[channels + global_row]);
        double s2       = bf16_to_f32(initial_state[2 * channels + global_row]);
        const double w0 = conv_weight[global_row];
        const double w1 = conv_weight[channels + global_row];
        const double w2 = conv_weight[2 * channels + global_row];
        const double w3 = conv_weight[3 * channels + global_row];
        for (int token = 0; token < tokens; ++token) {
            const double p = projection(global_row, token);
            const double y = silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * p);
            output.push_back(y);
            result.state.push_back(s1);
            result.state.push_back(s2);
            result.state.push_back(p);
            s0 = s1;
            s1 = s2;
            s2 = p;
        }
    };

    for (const int row : q_rows) evaluate(row, result.query);
    for (const int row : k_rows) evaluate(kQueryRows + row, result.key);
    for (const int row : v_rows) evaluate(kQueryRows + kKeyRows + row, result.value);
    return result;
}

std::vector<double> gather_outputs(const std::vector<double>& full, std::int32_t rows,
                                   std::int32_t tokens) {
    std::vector<double> gathered;
    for (const int row : sampled_rows(rows)) {
        for (int token = 0; token < tokens; ++token) {
            gathered.push_back(full[static_cast<std::size_t>(token) * rows + row]);
        }
    }
    return gathered;
}

std::vector<double> gather_states(const std::vector<std::uint16_t>& full, std::int32_t channels,
                                  std::int32_t value_rows, std::int32_t tokens) {
    std::vector<double> gathered;
    const auto append = [&](int global_row) {
        for (int token = 0; token < tokens; ++token) {
            const std::size_t base = static_cast<std::size_t>(token) * 3 * channels;
            for (int history = 0; history < 3; ++history) {
                gathered.push_back(bf16_to_f32(full[base + history * channels + global_row]));
            }
        }
    };
    for (const int row : sampled_rows(kQueryRows)) append(row);
    for (const int row : sampled_rows(kKeyRows)) append(kQueryRows + row);
    for (const int row : sampled_rows(value_rows)) append(kQueryRows + kKeyRows + row);
    return gathered;
}

int verify_untouched_slots(const char* label, const std::vector<std::uint16_t>& before,
                           const std::vector<std::uint16_t>& after, std::int32_t channels,
                           std::int32_t tokens) {
    const std::size_t stride = static_cast<std::size_t>(channels) * 3;
    for (int slot = tokens; slot < kSlots; ++slot) {
        const auto begin = static_cast<std::size_t>(slot) * stride;
        if (!std::equal(before.begin() + begin, before.begin() + begin + stride,
                        after.begin() + begin)) {
            std::cerr << label << ": state slot " << slot << " was modified\n";
            return 1;
        }
    }
    return 0;
}

std::vector<std::uint16_t> make_states(std::int32_t channels) {
    const std::size_t stride = static_cast<std::size_t>(channels) * 3;
    std::vector<std::uint16_t> bits(stride * kSlots, f32_to_bf16(17.0F));
    std::vector<float> initial(stride);
    fill_uniform(initial, 9001u + channels, -0.05F, 0.05F);
    round_to_bf16(initial);
    for (std::size_t i = 0; i < stride; ++i) {
        bits[static_cast<std::size_t>(kInitial) * stride + i] = f32_to_bf16(initial[i]);
    }
    return bits;
}

int one_w8(std::int32_t tokens) {
    constexpr int kHidden    = 2048;
    constexpr int kValueRows = 4096;
    constexpr int kChannels  = 8192;
    constexpr int kParent    = 12288;
    std::vector<float> x(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(x, 101u + tokens, -0.01F, 0.01F);
    round_to_bf16(x);
    auto packed = row_split::make_patterned_weight(QType::W8G32_F16S, kParent, kHidden, 103u);
    std::vector<float> conv_weight(static_cast<std::size_t>(kChannels) * 4);
    fill_uniform(conv_weight, 107u, -0.02F, 0.02F);
    round_to_bf16(conv_weight);
    const auto states = make_states(kChannels);

    DBuf dx = to_device_bf16(x), dp(packed.payload.size()), dw = to_device_bf16(conv_weight),
         ds = to_device_bits(states), di = to_device_i32({kInitial});
    cudaMemcpy(dp.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    DBuf dq(static_cast<std::size_t>(kQueryRows) * tokens * 2),
        dk(static_cast<std::size_t>(kKeyRows) * tokens * 2),
        dv(static_cast<std::size_t>(kValueRows) * tokens * 2),
        dz(static_cast<std::size_t>(kValueRows) * tokens * 2);
    Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    Tensor tw(dw.p, DType::BF16, {kChannels, 4});
    Tensor ts(ds.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor ti(di.p, DType::I32, {1});
    Tensor tq(dq.p, DType::BF16, {kQueryRows, tokens});
    Tensor tk(dk.p, DType::BF16, {kKeyRows, tokens});
    Tensor tv(dv.p, DType::BF16, {kValueRows, tokens});
    Tensor tz(dz.p, DType::BF16, {kValueRows, tokens});
    WorkspaceArena workspace(
        std::max<std::size_t>(1, ops::gdn_input_proj_conv_snapshot_workspace_bytes(
                                     kQueryRows, kKeyRows, kValueRows, tokens)));
    ops::gdn_input_proj_conv_snapshot(tx, packed.device_weight(dp.p), tw, ts, ti, tq, tk, tv, tz,
                                      workspace, nullptr);
    cudaDeviceSynchronize();

    const std::size_t initial_base = static_cast<std::size_t>(kInitial) * 3 * kChannels;
    std::vector<std::uint16_t> initial_state(states.begin() + initial_base,
                                             states.begin() + initial_base + 3 * kChannels);
    const auto oracle =
        make_oracle(tokens, kValueRows, conv_weight, initial_state, [&](int row, int token) {
            return row_split::dot_row_split_lowbit_fp64(
                packed, row, x.data() + static_cast<std::size_t>(token) * kHidden, kHidden);
        });
    const auto got_states    = from_device_bits(ds, states.size());
    const std::string suffix = " W8 T=" + std::to_string(tokens);
    int failures             = 0;
    failures +=
        verify(("fused q" + suffix).c_str(),
               gather_outputs(from_device_values(dq, static_cast<std::size_t>(kQueryRows) * tokens),
                              kQueryRows, tokens),
               oracle.query, Tolerance::bf16_reduction());
    failures +=
        verify(("fused k" + suffix).c_str(),
               gather_outputs(from_device_values(dk, static_cast<std::size_t>(kKeyRows) * tokens),
                              kKeyRows, tokens),
               oracle.key, Tolerance::bf16_reduction());
    failures +=
        verify(("fused v" + suffix).c_str(),
               gather_outputs(from_device_values(dv, static_cast<std::size_t>(kValueRows) * tokens),
                              kValueRows, tokens),
               oracle.value, Tolerance::bf16_reduction());
    failures += verify(("fused state" + suffix).c_str(),
                       gather_states(got_states, kChannels, kValueRows, tokens), oracle.state,
                       Tolerance::linear_bf16());
    failures +=
        verify_untouched_slots(("fused" + suffix).c_str(), states, got_states, kChannels, tokens);

    std::vector<double> z_ref;
    std::vector<double> z_got =
        from_device_values(dz, static_cast<std::size_t>(kValueRows) * tokens);
    for (const int row : sampled_rows(kValueRows)) {
        for (int token = 0; token < tokens; ++token) {
            z_ref.push_back(row_split::dot_row_split_lowbit_fp64(
                packed, kChannels + row, x.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden));
        }
    }
    failures += verify(("fused z" + suffix).c_str(), gather_outputs(z_got, kValueRows, tokens),
                       z_ref, Tolerance::linear_bf16());
    return failures;
}

int one_q4_q5(std::int32_t tokens) {
    constexpr int kHidden    = 5120;
    constexpr int kValueRows = 6144;
    constexpr int kChannels  = 10240;
    std::vector<float> x(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(x, 201u + tokens, -0.01F, 0.01F);
    round_to_bf16(x);
    auto qk = row_split::make_patterned_weight(QType::Q4G64_F16S, 4096, kHidden, 203u);
    auto vv = row_split::make_patterned_weight(QType::Q5G64_F16S, kValueRows, kHidden, 211u);
    std::vector<float> conv_weight(static_cast<std::size_t>(kChannels) * 4);
    fill_uniform(conv_weight, 223u, -0.02F, 0.02F);
    round_to_bf16(conv_weight);
    const auto states = make_states(kChannels);

    DBuf dx = to_device_bf16(x), dqk(qk.payload.size()), dvv(vv.payload.size()),
         dw = to_device_bf16(conv_weight), ds = to_device_bits(states),
         di = to_device_i32({kInitial});
    cudaMemcpy(dqk.p, qk.payload.data(), qk.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvv.p, vv.payload.data(), vv.payload.size(), cudaMemcpyHostToDevice);
    DBuf dq(static_cast<std::size_t>(kQueryRows) * tokens * 2),
        dk(static_cast<std::size_t>(kKeyRows) * tokens * 2),
        dv(static_cast<std::size_t>(kValueRows) * tokens * 2);
    Tensor tx(dx.p, DType::BF16, {kHidden, tokens});
    Tensor tw(dw.p, DType::BF16, {kChannels, 4});
    Tensor ts(ds.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor ti(di.p, DType::I32, {1});
    Tensor tq(dq.p, DType::BF16, {kQueryRows, tokens});
    Tensor tk(dk.p, DType::BF16, {kKeyRows, tokens});
    Tensor tv(dv.p, DType::BF16, {kValueRows, tokens});
    WorkspaceArena workspace(
        std::max<std::size_t>(1, ops::gdn_input_proj_conv_snapshot_workspace_bytes(
                                     kQueryRows, kKeyRows, kValueRows, tokens)));
    ops::gdn_input_proj_conv_snapshot(tx, qk.device_weight(dqk.p), vv.device_weight(dvv.p), tw, ts,
                                      ti, tq, tk, tv, workspace, nullptr);
    cudaDeviceSynchronize();

    const std::size_t initial_base = static_cast<std::size_t>(kInitial) * 3 * kChannels;
    std::vector<std::uint16_t> initial_state(states.begin() + initial_base,
                                             states.begin() + initial_base + 3 * kChannels);
    const auto oracle =
        make_oracle(tokens, kValueRows, conv_weight, initial_state, [&](int row, int token) {
            const float* input = x.data() + static_cast<std::size_t>(token) * kHidden;
            return row < 4096
                       ? row_split::dot_row_split_lowbit_fp64(qk, row, input, kHidden)
                       : row_split::dot_row_split_lowbit_fp64(vv, row - 4096, input, kHidden);
        });
    const auto got_states    = from_device_bits(ds, states.size());
    const std::string suffix = " Q4/Q5 T=" + std::to_string(tokens);
    int failures             = 0;
    failures +=
        verify(("fused q" + suffix).c_str(),
               gather_outputs(from_device_values(dq, static_cast<std::size_t>(kQueryRows) * tokens),
                              kQueryRows, tokens),
               oracle.query, Tolerance::bf16_reduction());
    failures +=
        verify(("fused k" + suffix).c_str(),
               gather_outputs(from_device_values(dk, static_cast<std::size_t>(kKeyRows) * tokens),
                              kKeyRows, tokens),
               oracle.key, Tolerance::bf16_reduction());
    failures +=
        verify(("fused v" + suffix).c_str(),
               gather_outputs(from_device_values(dv, static_cast<std::size_t>(kValueRows) * tokens),
                              kValueRows, tokens),
               oracle.value, Tolerance::bf16_reduction());
    failures += verify(("fused state" + suffix).c_str(),
                       gather_states(got_states, kChannels, kValueRows, tokens), oracle.state,
                       Tolerance::linear_bf16());
    failures +=
        verify_untouched_slots(("fused" + suffix).c_str(), states, got_states, kChannels, tokens);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int failures = 0;
    for (int tokens = 1; tokens <= 6; ++tokens) {
        failures += one_w8(tokens);
        failures += one_q4_q5(tokens);
    }
    std::cout << (failures ? "FAIL" : "OK") << " fused GDN input projection/snapshot\n";
    return failures == 0 ? 0 : 1;
}
