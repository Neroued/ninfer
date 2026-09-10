#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The parity checks below compare the fused Op against the Ops it replaces, which cannot see a
// defect the two share. These are the independent half: the same naive FP64 oracle and the same
// criterion tests/ops/test_rmsnorm.cpp judges ops::rmsnorm by, evaluated here from the represented
// BF16 inputs and compared against the fused output directly.
constexpr ReductionCriterion rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 3.95e-3};
}

std::vector<double> rmsnorm_oracle(const std::vector<std::uint16_t>& input,
                                   const std::vector<std::uint16_t>& weight, std::int32_t hidden,
                                   std::int32_t tokens, float eps) {
    std::vector<double> output(input.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * hidden;
        double sum_squares     = 0.0;
        for (std::int32_t row = 0; row < hidden; ++row) {
            const double value = bf16_to_f32(input[base + row]);
            sum_squares += value * value;
        }
        const double inverse =
            1.0 / std::sqrt(sum_squares / static_cast<double>(hidden) + static_cast<double>(eps));
        for (std::int32_t row = 0; row < hidden; ++row) {
            output[base + row] =
                bf16_to_f32(input[base + row]) * inverse * (1.0 + bf16_to_f32(weight[row]));
        }
    }
    return output;
}

std::vector<std::uint16_t> bit_pattern(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> values(count);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state     = state * 1664525u + 1013904223u;
        values[i] = static_cast<std::uint16_t>((state >> 16) ^ static_cast<std::uint32_t>(i));
    }
    return values;
}

int pack_case(std::int32_t hidden, std::int32_t tokens) {
    const std::int32_t output_rows = 2 * hidden;
    const std::size_t input_count  = static_cast<std::size_t>(hidden) * tokens;
    const auto embedding           = bit_pattern(input_count, 0x1234'5678u);
    const auto hidden_values       = bit_pattern(input_count, 0x8765'4321u);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(output_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < hidden; ++row) {
            expected[static_cast<std::size_t>(token) * output_rows + row] =
                embedding[static_cast<std::size_t>(token) * hidden + row];
            expected[static_cast<std::size_t>(token) * output_rows + hidden + row] =
                hidden_values[static_cast<std::size_t>(token) * hidden + row];
        }
    }

    GuardedDeviceBuffer device_embedding(embedding.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_values.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(expected.size() * sizeof(std::uint16_t));
    device_embedding.copy_from_host(embedding.data(), embedding.size() * sizeof(std::uint16_t));
    device_hidden.copy_from_host(hidden_values.data(),
                                 hidden_values.size() * sizeof(std::uint16_t));
    device_output.fill(0xcd);

    Tensor embedding_tensor(device_embedding.data(), DType::BF16, {hidden, tokens});
    Tensor hidden_tensor(device_hidden.data(), DType::BF16, {hidden, tokens});
    Tensor output_tensor(device_output.data(), DType::BF16, {output_rows, tokens});
    ops::mtp_pack_fc_input(embedding_tensor, hidden_tensor, output_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        "mtp_pack_fc_input D=" + std::to_string(hidden) + " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_exact(
        label.c_str(), from_device<std::uint16_t>(device_output.data(), expected.size()), expected);
    failures += verify_exact((label + " preserves embedding").c_str(),
                             from_device<std::uint16_t>(device_embedding.data(), embedding.size()),
                             embedding);
    failures += verify_exact((label + " preserves hidden").c_str(),
                             from_device<std::uint16_t>(device_hidden.data(), hidden_values.size()),
                             hidden_values);
    failures += device_embedding.verify_guards((label + " embedding").c_str());
    failures += device_hidden.verify_guards((label + " hidden").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

int split_case(std::int32_t tokens) {
    constexpr std::int32_t kInputRows = 14336;
    constexpr std::int32_t kQueryRows = 6144;
    constexpr std::int32_t kKvRows    = 1024;
    const auto input = bit_pattern(static_cast<std::size_t>(kInputRows) * tokens, 0x2468'ace0u);
    std::vector<std::uint16_t> expected_query(static_cast<std::size_t>(kQueryRows) * tokens);
    std::vector<std::uint16_t> expected_key(static_cast<std::size_t>(kKvRows) * tokens);
    std::vector<std::uint16_t> expected_gate(static_cast<std::size_t>(kQueryRows) * tokens);
    std::vector<std::uint16_t> expected_value(static_cast<std::size_t>(kKvRows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t input_base = static_cast<std::size_t>(token) * kInputRows;
        for (std::int32_t row = 0; row < kQueryRows; ++row) {
            expected_query[static_cast<std::size_t>(token) * kQueryRows + row] =
                input[input_base + row];
            expected_gate[static_cast<std::size_t>(token) * kQueryRows + row] =
                input[input_base + kQueryRows + kKvRows + row];
        }
        for (std::int32_t row = 0; row < kKvRows; ++row) {
            expected_key[static_cast<std::size_t>(token) * kKvRows + row] =
                input[input_base + kQueryRows + row];
            expected_value[static_cast<std::size_t>(token) * kKvRows + row] =
                input[input_base + kQueryRows + kKvRows + kQueryRows + row];
        }
    }

    GuardedDeviceBuffer device_input(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_query(expected_query.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_key(expected_key.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_gate(expected_gate.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_value(expected_value.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
    device_query.fill(0xcd);
    device_key.fill(0xcd);
    device_gate.fill(0xcd);
    device_value.fill(0xcd);

    Tensor input_tensor(device_input.data(), DType::BF16, {kInputRows, tokens});
    Tensor query_tensor(device_query.data(), DType::BF16, {256, 24, tokens});
    Tensor key_tensor(device_key.data(), DType::BF16, {256, 4, tokens});
    Tensor gate_tensor(device_gate.data(), DType::BF16, {256, 24, tokens});
    Tensor value_tensor(device_value.data(), DType::BF16, {256, 4, tokens});
    ops::mtp_split_attn_in(input_tensor, query_tensor, key_tensor, gate_tensor, value_tensor,
                           nullptr);
    cuda_synchronize();

    const std::string label = "mtp_split_attn_in T=" + std::to_string(tokens);
    int failures            = 0;
    failures += verify_exact((label + " query").c_str(),
                             from_device<std::uint16_t>(device_query.data(), expected_query.size()),
                             expected_query);
    failures += verify_exact((label + " key").c_str(),
                             from_device<std::uint16_t>(device_key.data(), expected_key.size()),
                             expected_key);
    failures += verify_exact((label + " gate").c_str(),
                             from_device<std::uint16_t>(device_gate.data(), expected_gate.size()),
                             expected_gate);
    failures += verify_exact((label + " value").c_str(),
                             from_device<std::uint16_t>(device_value.data(), expected_value.size()),
                             expected_value);
    failures += verify_exact((label + " preserves input").c_str(),
                             from_device<std::uint16_t>(device_input.data(), input.size()), input);
    failures += device_input.verify_guards((label + " input").c_str());
    failures += device_query.verify_guards((label + " query").c_str());
    failures += device_key.verify_guards((label + " key").c_str());
    failures += device_gate.verify_guards((label + " gate").c_str());
    failures += device_value.verify_guards((label + " value").c_str());
    return failures;
}

// The fused stem Op has to produce the bytes the three-Op path produced, not merely bytes
// close to them: it exists to remove two graph nodes, and anything that shifts a bit would show up
// as a different draft token. The reference here is the product path itself -- two rmsnorm calls
// and the pack -- run on the same inputs in the same process.
int norm_pack_case(std::int32_t hidden, std::int32_t tokens) {
    const std::int32_t output_rows = 2 * hidden;
    const std::size_t count        = static_cast<std::size_t>(hidden) * tokens;
    std::vector<float> embedding(count), hidden_state(count), weight_e(hidden), weight_h(hidden);
    fill_uniform(embedding, 0x51ed'0001u, -4.0F, 4.0F);
    fill_uniform(hidden_state, 0x51ed'0002u, -4.0F, 4.0F);
    fill_uniform(weight_e, 0x51ed'0003u, -0.5F, 0.5F);
    fill_uniform(weight_h, 0x51ed'0004u, -0.5F, 0.5F);

    auto pack16 = [](const std::vector<float>& v) {
        std::vector<std::uint16_t> out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) { out[i] = f32_to_bf16(v[i]); }
        return out;
    };
    const auto embedding16     = pack16(embedding);
    const auto hidden16        = pack16(hidden_state);
    const auto weight_e16      = pack16(weight_e);
    const auto weight_h16      = pack16(weight_h);
    const std::size_t in_bytes = count * sizeof(std::uint16_t);
    const std::size_t w_bytes  = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    const std::size_t out_bytes =
        static_cast<std::size_t>(output_rows) * tokens * sizeof(std::uint16_t);

    GuardedDeviceBuffer d_embedding(in_bytes), d_hidden(in_bytes);
    GuardedDeviceBuffer d_weight_e(w_bytes), d_weight_h(w_bytes);
    GuardedDeviceBuffer d_norm_e(in_bytes), d_norm_h(in_bytes);
    GuardedDeviceBuffer d_reference(out_bytes), d_fused(out_bytes);
    d_embedding.copy_from_host(embedding16.data(), in_bytes);
    d_hidden.copy_from_host(hidden16.data(), in_bytes);
    d_weight_e.copy_from_host(weight_e16.data(), w_bytes);
    d_weight_h.copy_from_host(weight_h16.data(), w_bytes);
    d_reference.fill(0xcd);
    d_fused.fill(0xcd);

    Tensor t_embedding(d_embedding.data(), DType::BF16, {hidden, tokens});
    Tensor t_hidden(d_hidden.data(), DType::BF16, {hidden, tokens});
    Tensor t_weight_e(d_weight_e.data(), DType::BF16, {hidden});
    Tensor t_weight_h(d_weight_h.data(), DType::BF16, {hidden});
    Tensor t_norm_e(d_norm_e.data(), DType::BF16, {hidden, tokens});
    Tensor t_norm_h(d_norm_h.data(), DType::BF16, {hidden, tokens});
    Tensor t_reference(d_reference.data(), DType::BF16, {output_rows, tokens});
    Tensor t_fused(d_fused.data(), DType::BF16, {output_rows, tokens});

    constexpr float kStemEps = 1.0e-6F;
    ops::rmsnorm(t_embedding, t_weight_e, kStemEps, true, t_norm_e, nullptr);
    ops::rmsnorm(t_hidden, t_weight_h, kStemEps, true, t_norm_h, nullptr);
    ops::mtp_pack_fc_input(t_norm_e, t_norm_h, t_reference, nullptr);
    cuda_synchronize();

    const std::string label =
        "mtp_norm_pack_fc_input D=" + std::to_string(hidden) + " T=" + std::to_string(tokens);
    int failures = 0;
    if (!ops::mtp_norm_pack_fc_input_supported(t_embedding, t_weight_e, t_hidden, t_weight_h,
                                               t_fused)) {
        std::cout << label << ": route declined, three-Op path stands\n";
        return 0;
    }
    ops::mtp_norm_pack_fc_input(t_embedding, t_weight_e, t_hidden, t_weight_h, t_fused, kStemEps,
                                nullptr);
    cuda_synchronize();

    const std::size_t elements = static_cast<std::size_t>(output_rows) * tokens;
    const auto reference       = from_device<std::uint16_t>(d_reference.data(), elements);
    const auto fused           = from_device<std::uint16_t>(d_fused.data(), elements);
    failures += verify_exact(label.c_str(), fused, reference);

    const auto norm_e_oracle = rmsnorm_oracle(embedding16, weight_e16, hidden, tokens, kStemEps);
    const auto norm_h_oracle = rmsnorm_oracle(hidden16, weight_h16, hidden, tokens, kStemEps);
    std::vector<double> oracle(elements), produced(elements);
    for (std::int32_t token = 0; token < tokens; ++token)
        for (std::int32_t row = 0; row < hidden; ++row) {
            const std::size_t in   = static_cast<std::size_t>(token) * hidden + row;
            const std::size_t out  = static_cast<std::size_t>(token) * output_rows + row;
            oracle[out]            = norm_e_oracle[in];
            oracle[out + hidden]   = norm_h_oracle[in];
            produced[out]          = bf16_to_f32(fused[out]);
            produced[out + hidden] = bf16_to_f32(fused[out + hidden]);
        }
    failures += verify_reduction((label + " against the oracle").c_str(), produced, oracle,
                                 rmsnorm_bf16_criterion());

    failures += d_embedding.verify_guards((label + " embedding").c_str());
    failures += d_hidden.verify_guards((label + " hidden").c_str());
    failures += d_fused.verify_guards((label + " fused").c_str());
    return failures;
}

// An eps that is not positive and finite has to be refused, not normalised with: the composition
// these Ops replace refuses it in ops::rmsnorm, and a kernel handed it returns NaN instead.
int eps_contract(std::int32_t hidden) {
    const std::size_t bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    GuardedDeviceBuffer d_embedding(bytes), d_hidden(bytes), d_weight_e(bytes), d_weight_h(bytes);
    GuardedDeviceBuffer d_residual(bytes), d_out(bytes), d_pack(2 * bytes);
    Tensor t_embedding(d_embedding.data(), DType::BF16, {hidden, 1});
    Tensor t_hidden(d_hidden.data(), DType::BF16, {hidden, 1});
    Tensor t_weight_e(d_weight_e.data(), DType::BF16, {hidden});
    Tensor t_weight_h(d_weight_h.data(), DType::BF16, {hidden});
    Tensor t_residual(d_residual.data(), DType::BF16, {hidden, 1});
    Tensor t_out(d_out.data(), DType::BF16, {hidden, 1});
    Tensor t_pack(d_pack.data(), DType::BF16, {2 * hidden, 1});

    int failures       = 0;
    const auto refuses = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; }
        std::cout << "mtp eps contract D=" << hidden << " " << what
                  << " FAILED: accepted an invalid eps\n";
        ++failures;
    };
    for (const float eps : {0.0F, -1.0e-6F, std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity()}) {
        refuses("norm_pack", [&] {
            ops::mtp_norm_pack_fc_input(t_embedding, t_weight_e, t_hidden, t_weight_h, t_pack, eps,
                                        nullptr);
        });
        refuses("residual_norm", [&] {
            ops::mtp_residual_norm(t_embedding, t_residual, t_weight_e, t_out, eps, nullptr);
        });
    }
    return failures;
}

// The weight contract is [D]. A view with the right element count but the wrong shape has to be
// refused here exactly as ops::rmsnorm refuses it, or the fused route would run on an operand the
// composed path would not accept.
int weight_shape_contract(std::int32_t hidden) {
    const std::size_t bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    GuardedDeviceBuffer d_embedding(bytes), d_hidden(bytes), d_weight(bytes), d_residual(bytes);
    GuardedDeviceBuffer d_out(bytes), d_pack(2 * bytes);
    Tensor t_embedding(d_embedding.data(), DType::BF16, {hidden, 1});
    Tensor t_hidden(d_hidden.data(), DType::BF16, {hidden, 1});
    Tensor t_good(d_weight.data(), DType::BF16, {hidden});
    Tensor t_residual(d_residual.data(), DType::BF16, {hidden, 1});
    Tensor t_out(d_out.data(), DType::BF16, {hidden, 1});
    Tensor t_pack(d_pack.data(), DType::BF16, {2 * hidden, 1});

    int failures       = 0;
    const auto refuses = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; }
        std::cout << "mtp weight shape D=" << hidden << " " << what
                  << " FAILED: accepted a weight that is not [D]\n";
        ++failures;
    };
    const Tensor bad_rowed(d_weight.data(), DType::BF16, {1, hidden});
    const Tensor bad_split(d_weight.data(), DType::BF16, {hidden / 2, 2});
    for (const Tensor* bad : {&bad_rowed, &bad_split}) {
        refuses("norm_pack embedding_weight", [&] {
            ops::mtp_norm_pack_fc_input(t_embedding, *bad, t_hidden, t_good, t_pack, 1.0e-6F,
                                        nullptr);
        });
        refuses("norm_pack hidden_weight", [&] {
            ops::mtp_norm_pack_fc_input(t_embedding, t_good, t_hidden, *bad, t_pack, 1.0e-6F,
                                        nullptr);
        });
        refuses("residual_norm weight", [&] {
            ops::mtp_residual_norm(t_embedding, t_residual, *bad, t_out, 1.0e-6F, nullptr);
        });
    }
    return failures;
}

// The route predicate is part of the contract in both directions. If a stem width stopped being
// admitted, the exactness case above would print "route declined" and pass while covering nothing;
// if a width outside the mirrored ops::rmsnorm ladder started being admitted, the kernel would run
// a reduction ops::rmsnorm does not, and the output would stop being bit-exact.
int norm_pack_route(std::int32_t hidden, bool expected) {
    const std::size_t bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    GuardedDeviceBuffer d_embedding(bytes), d_hidden(bytes), d_weight_e(bytes), d_weight_h(bytes);
    GuardedDeviceBuffer d_out(2 * bytes);
    Tensor t_embedding(d_embedding.data(), DType::BF16, {hidden, 1});
    Tensor t_hidden(d_hidden.data(), DType::BF16, {hidden, 1});
    Tensor t_weight_e(d_weight_e.data(), DType::BF16, {hidden});
    Tensor t_weight_h(d_weight_h.data(), DType::BF16, {hidden});
    Tensor t_out(d_out.data(), DType::BF16, {2 * hidden, 1});
    const bool admitted =
        ops::mtp_norm_pack_fc_input_supported(t_embedding, t_weight_e, t_hidden, t_weight_h, t_out);
    if (admitted != expected) {
        std::cout << "mtp_norm_pack_fc_input route D=" << hidden << " FAILED: expected "
                  << (expected ? "admitted" : "declined") << "\n";
        return 1;
    }
    return 0;
}

// Runs the same shape twice with one input element changed and requires the two fused outputs to
// differ. Without this, "identical" above could be reporting on a kernel that never wrote anything
// the comparison reads. The change is a whole unit and not a ulp on purpose: a ulp of an input can
// round back onto the same output BF16, which would make the control itself flaky.
int norm_pack_strength(std::int32_t hidden, std::int32_t tokens) {
    const std::int32_t output_rows = 2 * hidden;
    const std::size_t elements     = static_cast<std::size_t>(output_rows) * tokens;
    std::vector<std::vector<std::uint16_t>> results;
    for (bool perturb : {false, true}) {
        const std::size_t count = static_cast<std::size_t>(hidden) * tokens;
        std::vector<float> embedding(count), hidden_state(count), weight_e(hidden),
            weight_h(hidden);
        fill_uniform(embedding, 0x51ed'0001u, -4.0F, 4.0F);
        fill_uniform(hidden_state, 0x51ed'0002u, -4.0F, 4.0F);
        fill_uniform(weight_e, 0x51ed'0003u, -0.5F, 0.5F);
        fill_uniform(weight_h, 0x51ed'0004u, -0.5F, 0.5F);
        if (perturb) { embedding[count / 2] += 1.0F; }
        auto pack16 = [](const std::vector<float>& v) {
            std::vector<std::uint16_t> out(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) { out[i] = f32_to_bf16(v[i]); }
            return out;
        };
        const auto embedding16     = pack16(embedding);
        const auto hidden16        = pack16(hidden_state);
        const auto weight_e16      = pack16(weight_e);
        const auto weight_h16      = pack16(weight_h);
        const std::size_t in_bytes = count * sizeof(std::uint16_t);
        const std::size_t w_bytes  = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
        GuardedDeviceBuffer d_embedding(in_bytes), d_hidden(in_bytes);
        GuardedDeviceBuffer d_weight_e(w_bytes), d_weight_h(w_bytes);
        GuardedDeviceBuffer d_fused(elements * sizeof(std::uint16_t));
        d_embedding.copy_from_host(embedding16.data(), in_bytes);
        d_hidden.copy_from_host(hidden16.data(), in_bytes);
        d_weight_e.copy_from_host(weight_e16.data(), w_bytes);
        d_weight_h.copy_from_host(weight_h16.data(), w_bytes);
        d_fused.fill(0xcd);
        Tensor t_embedding(d_embedding.data(), DType::BF16, {hidden, tokens});
        Tensor t_hidden(d_hidden.data(), DType::BF16, {hidden, tokens});
        Tensor t_weight_e(d_weight_e.data(), DType::BF16, {hidden});
        Tensor t_weight_h(d_weight_h.data(), DType::BF16, {hidden});
        Tensor t_fused(d_fused.data(), DType::BF16, {output_rows, tokens});
        if (!ops::mtp_norm_pack_fc_input_supported(t_embedding, t_weight_e, t_hidden, t_weight_h,
                                                   t_fused)) {
            return 0;
        }
        ops::mtp_norm_pack_fc_input(t_embedding, t_weight_e, t_hidden, t_weight_h, t_fused, 1.0e-6F,
                                    nullptr);
        cuda_synchronize();
        results.push_back(from_device<std::uint16_t>(d_fused.data(), elements));
    }
    if (results[0] == results[1]) {
        std::cout << "mtp_norm_pack_fc_input strength control FAILED: an input change left "
                     "the output unchanged\n";
        return 1;
    }
    return 0;
}

// The fused residual-and-norm has to leave both the updated residual and the normalised
// output byte-for-byte where the two Ops left them, so the reference here is those two Ops run on
// the same inputs in the same process.
int residual_norm_case(std::int32_t hidden, std::int32_t tokens) {
    const std::size_t count = static_cast<std::size_t>(hidden) * tokens;
    std::vector<float> delta(count), residual(count), weight(hidden);
    fill_uniform(delta, 0x9e11'0001u, -4.0F, 4.0F);
    fill_uniform(residual, 0x9e11'0002u, -4.0F, 4.0F);
    fill_uniform(weight, 0x9e11'0003u, -0.5F, 0.5F);
    auto pack16 = [](const std::vector<float>& v) {
        std::vector<std::uint16_t> out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) { out[i] = f32_to_bf16(v[i]); }
        return out;
    };
    const auto delta16        = pack16(delta);
    const auto residual16     = pack16(residual);
    const auto weight16       = pack16(weight);
    const std::size_t bytes   = count * sizeof(std::uint16_t);
    const std::size_t w_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);

    GuardedDeviceBuffer d_delta(bytes), d_weight(w_bytes);
    GuardedDeviceBuffer d_ref_residual(bytes), d_ref_out(bytes);
    GuardedDeviceBuffer d_fused_residual(bytes), d_fused_out(bytes);
    d_delta.copy_from_host(delta16.data(), bytes);
    d_weight.copy_from_host(weight16.data(), w_bytes);
    d_ref_residual.copy_from_host(residual16.data(), bytes);
    d_fused_residual.copy_from_host(residual16.data(), bytes);
    d_ref_out.fill(0xcd);
    d_fused_out.fill(0xcd);

    Tensor t_delta(d_delta.data(), DType::BF16, {hidden, tokens});
    Tensor t_weight(d_weight.data(), DType::BF16, {hidden});
    Tensor t_ref_residual(d_ref_residual.data(), DType::BF16, {hidden, tokens});
    Tensor t_ref_out(d_ref_out.data(), DType::BF16, {hidden, tokens});
    Tensor t_fused_residual(d_fused_residual.data(), DType::BF16, {hidden, tokens});
    Tensor t_fused_out(d_fused_out.data(), DType::BF16, {hidden, tokens});

    constexpr float kTailEps = 1.0e-6F;
    ops::residual_add(t_delta, t_ref_residual, nullptr);
    ops::rmsnorm(t_ref_residual, t_weight, kTailEps, true, t_ref_out, nullptr);
    cuda_synchronize();

    const std::string label =
        "mtp_residual_norm D=" + std::to_string(hidden) + " T=" + std::to_string(tokens);
    if (!ops::mtp_residual_norm_supported(t_delta, t_fused_residual, t_weight, t_fused_out)) {
        std::cout << label << ": route declined, two-Op path stands\n";
        return 0;
    }
    ops::mtp_residual_norm(t_delta, t_fused_residual, t_weight, t_fused_out, kTailEps, nullptr);
    cuda_synchronize();

    int failures = 0;
    // The updated residual is a BF16 sum of two BF16 values, which is exact in double, so this
    // half of the fused Op has an independent oracle that is itself exact.
    std::vector<std::uint16_t> residual_oracle(count);
    for (std::size_t i = 0; i < count; ++i)
        residual_oracle[i] = f32_to_bf16(bf16_to_f32(delta16[i]) + bf16_to_f32(residual16[i]));
    failures +=
        verify_exact((label + " residual against the oracle").c_str(),
                     from_device<std::uint16_t>(d_fused_residual.data(), count), residual_oracle);
    {
        const auto out_oracle = rmsnorm_oracle(residual_oracle, weight16, hidden, tokens, kTailEps);
        const auto produced   = from_device_bf16(d_fused_out.data(), count);
        failures += verify_reduction((label + " output against the oracle").c_str(), produced,
                                     out_oracle, rmsnorm_bf16_criterion());
    }
    failures += verify_exact((label + " residual").c_str(),
                             from_device<std::uint16_t>(d_fused_residual.data(), count),
                             from_device<std::uint16_t>(d_ref_residual.data(), count));
    failures += verify_exact((label + " output").c_str(),
                             from_device<std::uint16_t>(d_fused_out.data(), count),
                             from_device<std::uint16_t>(d_ref_out.data(), count));
    failures += d_delta.verify_guards((label + " delta").c_str());
    failures += d_fused_residual.verify_guards((label + " residual").c_str());
    failures += d_fused_out.verify_guards((label + " out").c_str());
    return failures;
}

// Same contract in both directions as norm_pack_route: the tail width must be admitted, and a
// width outside the mirrored ops::rmsnorm ladder must not be.
int residual_norm_route(std::int32_t hidden, bool expected) {
    const std::size_t bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
    GuardedDeviceBuffer d_delta(bytes), d_residual(bytes), d_weight(bytes), d_out(bytes);
    Tensor t_delta(d_delta.data(), DType::BF16, {hidden, 1});
    Tensor t_residual(d_residual.data(), DType::BF16, {hidden, 1});
    Tensor t_weight(d_weight.data(), DType::BF16, {hidden});
    Tensor t_out(d_out.data(), DType::BF16, {hidden, 1});
    const bool admitted = ops::mtp_residual_norm_supported(t_delta, t_residual, t_weight, t_out);
    if (admitted != expected) {
        std::cout << "mtp_residual_norm route D=" << hidden << " FAILED: expected "
                  << (expected ? "admitted" : "declined") << "\n";
        return 1;
    }
    return 0;
}

// One changed input element has to reach the output, otherwise "identical" above would be
// reporting on a kernel that never wrote what the comparison reads. A whole unit and not a ulp: a
// ulp of an input can round back onto the same output BF16.
int residual_norm_strength(std::int32_t hidden, std::int32_t tokens) {
    const std::size_t count = static_cast<std::size_t>(hidden) * tokens;
    std::vector<std::vector<std::uint16_t>> results;
    for (bool perturb : {false, true}) {
        std::vector<float> delta(count), residual(count), weight(hidden);
        fill_uniform(delta, 0x9e11'0001u, -4.0F, 4.0F);
        fill_uniform(residual, 0x9e11'0002u, -4.0F, 4.0F);
        fill_uniform(weight, 0x9e11'0003u, -0.5F, 0.5F);
        if (perturb) { delta[count / 2] += 1.0F; }
        auto pack16 = [](const std::vector<float>& v) {
            std::vector<std::uint16_t> out(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) { out[i] = f32_to_bf16(v[i]); }
            return out;
        };
        const auto delta16        = pack16(delta);
        const auto residual16     = pack16(residual);
        const auto weight16       = pack16(weight);
        const std::size_t bytes   = count * sizeof(std::uint16_t);
        const std::size_t w_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
        GuardedDeviceBuffer d_delta(bytes), d_weight(w_bytes), d_residual(bytes), d_out(bytes);
        d_delta.copy_from_host(delta16.data(), bytes);
        d_weight.copy_from_host(weight16.data(), w_bytes);
        d_residual.copy_from_host(residual16.data(), bytes);
        d_out.fill(0xcd);
        Tensor t_delta(d_delta.data(), DType::BF16, {hidden, tokens});
        Tensor t_weight(d_weight.data(), DType::BF16, {hidden});
        Tensor t_residual(d_residual.data(), DType::BF16, {hidden, tokens});
        Tensor t_out(d_out.data(), DType::BF16, {hidden, tokens});
        if (!ops::mtp_residual_norm_supported(t_delta, t_residual, t_weight, t_out)) { return 0; }
        ops::mtp_residual_norm(t_delta, t_residual, t_weight, t_out, 1.0e-6F, nullptr);
        cuda_synchronize();
        results.push_back(from_device<std::uint16_t>(d_out.data(), count));
    }
    if (results[0] == results[1]) {
        std::cout << "mtp_residual_norm strength control FAILED: an input change left the "
                     "output unchanged\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += pack_case(5120, 1);
    failures += pack_case(5120, 6);
    failures += pack_case(5120, 48);
    failures += pack_case(2048, 1);
    failures += pack_case(2048, 6);
    failures += pack_case(2048, 48);
    failures += norm_pack_route(2048, true);
    failures += norm_pack_route(5120, true);
    failures += norm_pack_route(1536, false);
    failures += norm_pack_route(4096, false);
    failures += norm_pack_route(384, false);
    failures += norm_pack_case(2048, 1);
    failures += norm_pack_case(2048, 4);
    failures += norm_pack_case(2048, 48);
    failures += norm_pack_case(5120, 1);
    failures += norm_pack_case(5120, 6);
    failures += norm_pack_case(5120, 64);
    failures += residual_norm_route(2048, true);
    failures += residual_norm_route(5120, true);
    failures += residual_norm_route(1536, false);
    failures += residual_norm_route(384, false);
    failures += residual_norm_case(2048, 1);
    failures += residual_norm_case(2048, 4);
    failures += residual_norm_case(2048, 48);
    failures += residual_norm_case(5120, 1);
    failures += residual_norm_case(5120, 6);
    failures += residual_norm_case(5120, 64);
    failures += residual_norm_strength(2048, 4);
    failures += residual_norm_strength(5120, 1);
    failures += norm_pack_strength(2048, 4);
    failures += norm_pack_strength(5120, 1);
    failures += weight_shape_contract(2048);
    failures += weight_shape_contract(5120);
    failures += eps_contract(2048);
    failures += eps_contract(5120);
    failures += split_case(1);
    failures += split_case(6);
    failures += split_case(48);
    std::cout << (failures ? "FAIL" : "OK") << " mtp_pack\n";
    return failures ? 1 : 0;
}
