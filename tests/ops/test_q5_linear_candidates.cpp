#include "ninfer/ops/linear.h"
#include "ops/linear/q5/q5_rowsplit_launch.h"
#include "ops/linear/q5/q5_rowsplit_plan.h"
#include "ops/op_tester.h"
#include "ops/row_split_pack.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<double> cpu_q5_linear(const std::vector<float>& x,
                                  const row_split::PackedWeight& weight, std::int32_t n,
                                  std::int32_t k, std::int32_t t) {
    std::vector<double> out(static_cast<std::size_t>(n) * t, 0.0);
    for (std::int32_t col = 0; col < t; ++col) {
        for (std::int32_t row = 0; row < n; ++row) {
            double acc        = 0.0;
            const float* wrow = weight.dequant.data() + static_cast<std::size_t>(row) * k;
            const float* xcol = x.data() + static_cast<std::size_t>(col) * k;
            for (std::int32_t kk = 0; kk < k; ++kk) {
                acc += static_cast<double>(wrow[kk]) * static_cast<double>(xcol[kk]);
            }
            out[static_cast<std::size_t>(col) * n + row] = acc;
        }
    }
    return out;
}

int one_candidate(ops::detail::Q5ScheduleId schedule, ops::detail::Q5KernelVariant variant,
                  std::int32_t n, std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(k) * t);
    fill_uniform(source_weight, seed + 1000u, -0.125f, 0.125f);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(source_weight);
    round_to_bf16(x);

    row_split::PackedWeight packed =
        row_split::pack_row_split_lowbit(source_weight, n, k, QType::Q5G64_F16S);
    std::vector<double> ref = cpu_q5_linear(x, packed, n, k, t);

    DBuf dx = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    DBuf dout(static_cast<std::size_t>(n) * t * 2u);
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);

    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor ty(dout.p, DType::BF16, {n, t});
    ops::detail::q5_rowsplit_launch_candidate(schedule, variant, tx, packed.device_weight(dw.p), ty,
                                              nullptr);
    cudaDeviceSynchronize();

    const std::string label = std::string(ops::detail::q5_schedule_name(schedule)) + "." +
                              ops::detail::q5_kernel_variant_name(variant) + " [" +
                              std::to_string(n) + "," + std::to_string(k) +
                              "] C=" + std::to_string(t);
    const Tolerance tolerance = ops::detail::q5_schedule_uses_mma(schedule)
                                    ? Tolerance::linear_tc()
                                    : Tolerance::linear_bf16();
    return verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t), ref,
                  tolerance);
}

int legality_contract_rejections() {
    using S = ops::detail::Q5ScheduleId;
    using V = ops::detail::Q5KernelVariant;

    int failures      = 0;
    const auto reject = [&](const char* label, S schedule, V variant,
                            const ops::detail::Q5Problem& problem) {
        if (ops::detail::q5_candidate_is_legal(schedule, variant, problem)) {
            std::cerr << "q5 candidate legality accepted " << label << '\n';
            ++failures;
        }
    };

    reject("GEMV column mismatch", S::GemvR16S2X, V::None, {6144, 5120, 5120, 2});
    reject("GEMV shape mismatch", S::GemvR16S2X, V::None, {5120, 5120, 5120, 1});
    reject("split2 shape mismatch", S::SimtSplit2Exact, V::None, {6144, 5120, 5120, 2});
    reject("split4 column mismatch", S::SimtSplit4Exact, V::None, {6144, 5120, 5120, 7});
    reject("SIMT Full lifecycle mismatch", S::SimtR8C4, V::Full, {128, 1152, 1152, 4});
    reject("MMA None lifecycle mismatch", S::MmaR64C64, V::None, {128, 1152, 1152, 64});
    reject("MMA Full column mismatch", S::MmaR64C64, V::Full, {128, 1152, 1152, 32});
    reject("MMA Full Kpad mismatch", S::MmaR64C128, V::Full, {128, 4304, 4352, 128});
    reject("invalid padded K", S::SimtR8C4, V::None, {128, 1152, 1088, 4});
    reject("unknown schedule", static_cast<S>(999), V::Predicated, {128, 1152, 1152, 64});
    return failures;
}

int public_routes_match_fixed(std::int32_t n, std::int32_t k,
                              const std::vector<std::int32_t>& columns, std::uint32_t seed) {
    const std::int32_t max_cols = *std::max_element(columns.begin(), columns.end());
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(k) * max_cols);
    fill_uniform(source_weight, seed + 1000u, -0.125f, 0.125f);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(source_weight);
    round_to_bf16(x);

    row_split::PackedWeight packed =
        row_split::pack_row_split_lowbit(source_weight, n, k, QType::Q5G64_F16S);
    DBuf dx = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = packed.device_weight(dw.p);
    WorkspaceArena workspace(256);

    int failures = 0;
    for (const std::int32_t cols : columns) {
        DBuf public_output(static_cast<std::size_t>(n) * cols * 2u);
        DBuf fixed_output(static_cast<std::size_t>(n) * cols * 2u);
        Tensor tx(dx.p, DType::BF16, {k, cols});
        Tensor public_out(public_output.p, DType::BF16, {n, cols});
        Tensor fixed_out(fixed_output.p, DType::BF16, {n, cols});
        const ops::detail::Q5Plan plan =
            ops::detail::q5_rowsplit_resolve_plan({n, k, weight.padded_shape[1], cols});

        ops::detail::q5_rowsplit_execute_plan(plan, tx, weight, fixed_out, nullptr);
        ops::linear(tx, weight, public_out, workspace, nullptr);
        cudaDeviceSynchronize();

        const std::size_t count          = static_cast<std::size_t>(n) * cols;
        const std::vector<double> fixed  = from_device_bf16(fixed_output, count);
        const std::vector<double> actual = from_device_bf16(public_output, count);
        if (actual != fixed) {
            std::cerr << "q5 public route mismatch [" << n << "," << k << "] C=" << cols << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    using S      = ops::detail::Q5ScheduleId;
    using V      = ops::detail::Q5KernelVariant;
    int failures = 0;

    failures += one_candidate(S::GemvR16S2X, V::None, 6144, 5120, 1, 11u);
    failures += one_candidate(S::SimtR8C4, V::None, 128, 1152, 4, 13u);
    failures += one_candidate(S::SimtR8C8, V::None, 128, 4304, 9, 17u);
    failures += one_candidate(S::SimtSplit4Exact, V::None, 6144, 5120, 2, 19u);
    failures += one_candidate(S::SimtSplit2Exact, V::None, 5120, 6144, 2, 23u);

    failures += one_candidate(S::MmaR64C64, V::Full, 128, 1152, 64, 29u);
    failures += one_candidate(S::MmaR64C64, V::Predicated, 144, 4304, 65, 31u);
    failures += one_candidate(S::MmaR64C128, V::Full, 128, 1152, 128, 37u);
    failures += one_candidate(S::MmaR64C128, V::Predicated, 144, 4304, 129, 41u);

    failures += legality_contract_rejections();
    failures += public_routes_match_fixed(6144, 5120, {25, 64, 65, 128}, 43u);
    failures += public_routes_match_fixed(1152, 4304, {88, 128}, 47u);

    std::cout << (failures ? "FAIL" : "OK") << " Q5 Linear fixed candidates\n";
    return failures ? 1 : 0;
}
