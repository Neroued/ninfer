// Correctness + coverage for linear dense BF16/FP32, against the frozen
// op-test standard (docs/op-development.md): fp64 golden W @ x from
// bf16-rounded inputs and weights, composite tolerance linear_bf16.
#include "ninfer/ops/linear.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/op_tester.h"
#include "ops/linear/q4/q4_rowsplit_launch.h"
#include "ops/linear/q4/q4_rowsplit_plan.h"
#include "ops/linear/q5/q5_rowsplit_launch.h"
#include "ops/linear/q5/q5_rowsplit_plan.h"
#include "ops/linear/q6/q6_rowsplit_plan.h"
#include "ops/row_split_pack.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

const char* qtype_name(QType qtype) {
    switch (qtype) {
    case QType::BF16_CTRL:
        return "bf16";
    case QType::FP32_CTRL:
        return "fp32";
    case QType::Q4G64_F16S:
        return "q4";
    case QType::Q5G64_F16S:
        return "q5";
    case QType::Q6G64_F16S:
        return "q6";
    case QType::W8G32_F16S:
        return "w8g32";
    default:
        return "unsupported";
    }
}

Weight dense_weight(void* data, QType qtype, std::int32_t n, std::int32_t k) {
    Weight w{};
    w.qtype         = qtype;
    w.layout        = QuantLayout::Contiguous;
    w.payload       = data;
    w.payload_bytes = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k) *
                      ((qtype == QType::FP32_CTRL) ? 4u : 2u);
    w.qdata           = data;
    w.scales          = nullptr;
    w.group_size      = 0;
    w.group           = 0;
    w.ndim            = 2;
    w.shape[0]        = n;
    w.shape[1]        = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    w.n               = n;
    w.k               = k;
    return w;
}

void cpu_linear_dense(const std::vector<float>& x, const std::vector<float>& weight, std::int32_t n,
                      std::int32_t k, std::int32_t t, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(n) * t, 0.0);
    for (std::int32_t col = 0; col < t; ++col) {
        double* dst = out.data() + static_cast<std::size_t>(col) * n;
        for (std::int32_t row = 0; row < n; ++row) {
            const float* wrow = weight.data() + static_cast<std::size_t>(row) * k;
            for (std::int32_t kk = 0; kk < k; ++kk) {
                const double xv = static_cast<double>(x[static_cast<std::size_t>(col) * k + kk]);
                dst[row] += static_cast<double>(wrow[kk]) * xv;
            }
        }
    }
}

void cpu_linear_dequant(const std::vector<float>& x, const std::vector<float>& weight,
                        std::int32_t n, std::int32_t k, std::int32_t t, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(n) * t, 0.0);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const std::int32_t n_threads =
        std::min<std::int32_t>(n, static_cast<std::int32_t>(std::min<unsigned>(hw, 16u)));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n_threads));

    for (std::int32_t tid = 0; tid < n_threads; ++tid) {
        const std::int32_t row_begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * tid) / n_threads);
        const std::int32_t row_end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * (tid + 1)) / n_threads);
        threads.emplace_back([&, row_begin, row_end] {
            std::unique_ptr<double[]> acc(new double[static_cast<std::size_t>(t)]);
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                std::fill(acc.get(), acc.get() + t, 0.0);
                const float* wrow = weight.data() + static_cast<std::size_t>(row) * k;
                for (std::int32_t kk = 0; kk < k; ++kk) {
                    const double wv = static_cast<double>(wrow[kk]);
                    for (std::int32_t col = 0; col < t; ++col) {
                        acc[col] +=
                            wv * static_cast<double>(x[static_cast<std::size_t>(col) * k + kk]);
                    }
                }
                for (std::int32_t col = 0; col < t; ++col) {
                    out[static_cast<std::size_t>(col) * n + row] = acc[col];
                }
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
}

int one_dense_shape(std::int32_t n, std::int32_t k, std::int32_t t, QType qtype, std::uint32_t seed,
                    float x_abs = 8.0f, float weight_abs = 0.125f,
                    const char* case_name = nullptr) {
    std::vector<float> x(static_cast<std::size_t>(k) * t);
    std::vector<float> weight(static_cast<std::size_t>(n) * k);
    fill_uniform(x, seed, -x_abs, x_abs);
    fill_uniform(weight, seed + 1000u, -weight_abs, weight_abs);
    round_to_bf16(x);
    round_to_bf16(weight);

    std::vector<double> ref;
    cpu_linear_dense(x, weight, n, k, t, ref);

    DBuf dx = to_device_bf16(x);
    DBuf dw = (qtype == QType::FP32_CTRL) ? to_device_f32(weight) : to_device_bf16(weight);
    DBuf dout(static_cast<std::size_t>(n) * t * 2u);

    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor tout(dout.p, DType::BF16, {n, t});
    WorkspaceArena ws(64ULL << 20);
    ops::linear(tx, dense_weight(dw.p, qtype, n, k), tout, ws, nullptr);
    cudaDeviceSynchronize();

    const std::string suffix = case_name ? (" " + std::string(case_name)) : std::string();
    const std::string label  = "linear " + std::string(qtype_name(qtype)) + suffix + " [" +
                              std::to_string(n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(t) + " seed=" + std::to_string(seed);
    return verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t), ref,
                  Tolerance::linear_bf16());
}

int fp32_ctrl_first_column_consistency(std::int32_t n, std::int32_t k, std::int32_t t) {
    std::vector<float> x_one(static_cast<std::size_t>(k), 0.0f);
    std::vector<float> x_many(static_cast<std::size_t>(k) * t, 0.0f);
    std::vector<float> weight(static_cast<std::size_t>(n) * k, 0.0f);
    for (std::int32_t kk = 0; kk < std::min<std::int32_t>(k, 16); ++kk) {
        x_one[kk] = 3.0f;
        for (std::int32_t col = 0; col < t; ++col) {
            x_many[static_cast<std::size_t>(col) * k + kk] = 3.0f;
        }
    }
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t kk = 0; kk < std::min<std::int32_t>(k, 16); ++kk) {
            weight[static_cast<std::size_t>(row) * k + kk] = 1.003f;
        }
    }

    DBuf dx_one  = to_device_bf16(x_one);
    DBuf dx_many = to_device_bf16(x_many);
    DBuf dw      = to_device_f32(weight);
    DBuf out_one(static_cast<std::size_t>(n) * 2u);
    DBuf out_many(static_cast<std::size_t>(n) * t * 2u);

    Tensor tx_one(dx_one.p, DType::BF16, {k, 1});
    Tensor tx_many(dx_many.p, DType::BF16, {k, t});
    Tensor tout_one(out_one.p, DType::BF16, {n, 1});
    Tensor tout_many(out_many.p, DType::BF16, {n, t});
    Weight w = dense_weight(dw.p, QType::FP32_CTRL, n, k);
    WorkspaceArena ws(64ULL << 20);

    ops::linear(tx_one, w, tout_one, ws, nullptr);
    ops::linear(tx_many, w, tout_many, ws, nullptr);
    cudaDeviceSynchronize();

    const std::vector<double> got_one = from_device_bf16(out_one, n);
    const std::vector<double> got_many =
        from_device_bf16(out_many, static_cast<std::size_t>(n) * t);
    for (std::int32_t row = 0; row < n; ++row) {
        if (got_one[row] != got_many[row]) {
            std::cerr << "linear fp32 first-column consistency [" << n << "," << k << "] T=" << t
                      << ": row " << row << " T=1=" << got_one[row] << " T>1=" << got_many[row]
                      << '\n';
            return 1;
        }
    }
    return 0;
}

void launch_q4_test_reference(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    using S = ops::detail::Q4ScheduleId;
    using V = ops::detail::Q4KernelVariant;

    const ops::detail::Q4Problem problem{out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    S schedule;
    if (problem.cols == 1 && problem.k == 5120) {
        schedule = S::GemvR1W8Direct;
    } else if (problem.cols <= 4) {
        schedule = S::SimtR8C4;
    } else if (problem.cols <= 8) {
        schedule = S::SimtR8C8;
    } else if (problem.cols <= 64) {
        schedule = S::MmaR64C64;
    } else {
        schedule = S::MmaR64C128;
    }

    V variant = V::None;
    if (!ops::detail::q4_candidate_is_legal(schedule, variant, problem)) {
        variant = ops::detail::q4_candidate_is_legal(schedule, V::Full, problem) ? V::Full
                                                                                 : V::Predicated;
    }
    ops::detail::q4_rowsplit_launch_candidate(schedule, variant, x, w, out, stream);
}

void launch_q5_test_reference(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    using S = ops::detail::Q5ScheduleId;
    using V = ops::detail::Q5KernelVariant;

    const ops::detail::Q5Problem problem{out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    if (ops::detail::q5_rowsplit_admits(problem)) {
        const ops::detail::Q5Plan plan = ops::detail::q5_rowsplit_resolve_plan(problem);
        ops::detail::q5_rowsplit_execute_plan(plan, x, w, out, stream);
        return;
    }

    const S schedule = problem.cols <= 4    ? S::SimtR8C4
                       : problem.cols <= 16 ? S::SimtR8C8
                                            : S::MmaR64C128;
    V variant        = V::None;
    if (!ops::detail::q5_candidate_is_legal(schedule, variant, problem)) {
        variant = ops::detail::q5_candidate_is_legal(schedule, V::Full, problem) ? V::Full
                                                                                 : V::Predicated;
    }
    ops::detail::q5_rowsplit_launch_candidate(schedule, variant, x, w, out, stream);
}

int one_quant_shape(QType qtype, std::int32_t n, std::int32_t k,
                    const std::vector<std::int32_t>& ts, std::uint32_t seed, float x_abs = 8.0f,
                    float weight_abs = 0.125f, const char* case_name = nullptr) {
    const std::int32_t max_t = *std::max_element(ts.begin(), ts.end());
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k);
    fill_uniform(source_weight, seed + 2000u, -weight_abs, weight_abs);
    round_to_bf16(source_weight);
    row_split::PackedWeight packed = row_split::pack_row_split_lowbit(source_weight, n, k, qtype);
    std::vector<float>().swap(source_weight);

    std::vector<float> x(static_cast<std::size_t>(k) * max_t);
    fill_uniform(x, seed, -x_abs, x_abs);
    round_to_bf16(x);

    std::vector<double> ref_max_t;
    cpu_linear_dequant(x, packed.dequant, n, k, max_t, ref_max_t);

    DBuf dx = to_device_bf16(x);
    DBuf dweight(packed.payload.size());
    cudaMemcpy(dweight.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    WorkspaceArena ws(64ULL << 20);

    int failures = 0;
    for (std::int32_t t : ts) {
        DBuf dout(static_cast<std::size_t>(n) * t * 2u);
        Tensor tx(dx.p, DType::BF16, {k, t});
        Tensor tout(dout.p, DType::BF16, {n, t});
        try {
            ops::linear(tx, packed.device_weight(dweight.p), tout, ws, nullptr);
            cudaDeviceSynchronize();
        } catch (const std::exception& e) {
            std::cerr << "linear " << qtype_name(qtype) << " [" << n << "," << k << "] T=" << t
                      << " seed=" << seed << ": unexpected exception: " << e.what() << '\n';
            ++failures;
            continue;
        }

        const std::vector<double> ref(ref_max_t.begin(),
                                      ref_max_t.begin() + static_cast<std::size_t>(n) * t);
        const std::string suffix = case_name ? (" " + std::string(case_name)) : std::string();
        const std::string label  = "linear " + std::string(qtype_name(qtype)) + suffix + " [" +
                                  std::to_string(n) + "," + std::to_string(k) +
                                  "] T=" + std::to_string(t) + " seed=" + std::to_string(seed);
        Tolerance tol = t > 16 ? Tolerance::linear_tc() : Tolerance::linear_bf16();
        if (qtype == QType::Q4G64_F16S) {
            const ops::detail::Q4Plan plan =
                ops::detail::q4_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
            tol = ops::detail::q4_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                   : Tolerance::linear_bf16();
        } else if (qtype == QType::Q5G64_F16S) {
            const ops::detail::Q5Plan plan =
                ops::detail::q5_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
            tol = ops::detail::q5_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                   : Tolerance::linear_bf16();
        } else if (qtype == QType::Q6G64_F16S) {
            const ops::detail::Q6Plan plan =
                ops::detail::q6_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
            tol = ops::detail::q6_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                   : Tolerance::linear_bf16();
        }
        failures += verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t),
                           ref, tol);
    }
    return failures;
}

int paired_w8g32_shape(std::int32_t n, std::int32_t k, const std::vector<std::int32_t>& ts,
                       std::uint32_t seed) {
    const std::int32_t max_t = *std::max_element(ts.begin(), ts.end());
    std::vector<float> kw(static_cast<std::size_t>(n) * k);
    std::vector<float> vw(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(k) * max_t);
    fill_uniform(kw, seed + 2000u, -0.125f, 0.125f);
    fill_uniform(vw, seed + 3000u, -0.125f, 0.125f);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(kw);
    round_to_bf16(vw);
    round_to_bf16(x);

    row_split::PackedWeight kpacked = row_split::pack_row_split_lowbit(kw, n, k, QType::W8G32_F16S);
    row_split::PackedWeight vpacked = row_split::pack_row_split_lowbit(vw, n, k, QType::W8G32_F16S);
    std::vector<double> kref;
    std::vector<double> vref;
    cpu_linear_dequant(x, kpacked.dequant, n, k, max_t, kref);
    cpu_linear_dequant(x, vpacked.dequant, n, k, max_t, vref);

    DBuf dx = to_device_bf16(x);
    DBuf dkw(kpacked.payload.size());
    DBuf dvw(vpacked.payload.size());
    cudaMemcpy(dkw.p, kpacked.payload.data(), kpacked.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvw.p, vpacked.payload.data(), vpacked.payload.size(), cudaMemcpyHostToDevice);
    WorkspaceArena ws(64ULL << 20);

    int failures = 0;
    for (std::int32_t t : ts) {
        DBuf dkout(static_cast<std::size_t>(n) * t * 2u);
        DBuf dvout(static_cast<std::size_t>(n) * t * 2u);
        Tensor tx(dx.p, DType::BF16, {k, t});
        Tensor tk(dkout.p, DType::BF16, {n, t});
        Tensor tv(dvout.p, DType::BF16, {n, t});
        ops::linear_pair(tx, kpacked.device_weight(dkw.p), vpacked.device_weight(dvw.p), tk, tv, ws,
                         nullptr);
        cudaDeviceSynchronize();
        const std::size_t count = static_cast<std::size_t>(n) * t;
        const std::vector<double> kr(kref.begin(), kref.begin() + count);
        const std::vector<double> vr(vref.begin(), vref.begin() + count);
        const std::string base = "linear w8g32 paired [" + std::to_string(n) + "," +
                                 std::to_string(k) + "] T=" + std::to_string(t);
        failures += verify((base + " K").c_str(), from_device_bf16(dkout, count), kr,
                           Tolerance::linear_tc());
        failures += verify((base + " V").c_str(), from_device_bf16(dvout, count), vr,
                           Tolerance::linear_tc());
    }
    return failures;
}

row_split::PackedWeight patterned_weight(QType qtype, std::int32_t n, std::int32_t k,
                                         std::uint32_t seed) {
    const auto spec                  = row_split::detail::quant_spec(qtype);
    const std::int32_t padded_k      = row_split::detail::align_up(k, 128);
    const std::int32_t kg            = padded_k / spec.group_size;
    const std::uint64_t groups       = static_cast<std::uint64_t>(n) * kg;
    const std::uint64_t nibble_bytes = groups * row_split::detail::nibble_bytes_per_group(spec);
    const std::uint64_t high_bytes   = groups * row_split::detail::high_bytes_per_group(spec);

    row_split::PackedWeight packed;
    packed.nibble_plane_bytes = nibble_bytes;
    packed.high_plane_offset  = row_split::detail::align_up_size(nibble_bytes, 256);
    packed.high_plane_bytes   = high_bytes;
    packed.scale_plane_offset =
        packed.high_plane_offset + row_split::detail::align_up_size(high_bytes, 256);
    packed.scale_plane_bytes = groups * 2;
    packed.payload.resize(
        static_cast<std::size_t>(packed.scale_plane_offset + packed.scale_plane_bytes));
    for (std::uint64_t i = 0; i < nibble_bytes; ++i) {
        packed.payload[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((i * 17u + seed) & 0xffu);
    }
    for (std::uint64_t i = 0; i < high_bytes; ++i) {
        packed.payload[static_cast<std::size_t>(packed.high_plane_offset + i)] =
            static_cast<std::uint8_t>((i * 13u + seed * 3u) & 0xffu);
    }
    for (std::uint64_t i = 0; i < groups; ++i) {
        row_split::detail::store_u16_le(packed.payload,
                                        static_cast<std::size_t>(packed.scale_plane_offset + i * 2),
                                        0x3c00u); // fp16 1.0
    }
    packed.weight.qtype            = qtype;
    packed.weight.layout           = QuantLayout::RowSplit;
    packed.weight.scale_dtype      = DType::FP16;
    packed.weight.payload_bytes    = packed.payload.size();
    packed.weight.high_plane_bytes = high_bytes;
    packed.weight.group_size       = spec.group_size;
    packed.weight.group            = spec.group_size;
    packed.weight.ndim             = 2;
    packed.weight.shape[0]         = n;
    packed.weight.shape[1]         = k;
    packed.weight.padded_shape[0]  = n;
    packed.weight.padded_shape[1]  = padded_k;
    packed.weight.n                = n;
    packed.weight.k                = k;
    return packed;
}

int grouped_attention_correctness(std::int32_t t) {
    constexpr int kHidden = 5120;
    constexpr int kQRows  = 6144;
    constexpr int kKvRows = 1024;
    std::vector<float> x(static_cast<std::size_t>(kHidden) * t);
    fill_uniform(x, 301u, -0.01f, 0.01f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);

    auto qw = patterned_weight(QType::Q4G64_F16S, kQRows, kHidden, 11u);
    auto gw = patterned_weight(QType::Q5G64_F16S, kQRows, kHidden, 13u);
    auto kw = patterned_weight(QType::Q4G64_F16S, kKvRows, kHidden, 17u);
    auto vw = patterned_weight(QType::Q5G64_F16S, kKvRows, kHidden, 19u);
    DBuf dqw(qw.payload.size()), dgw(gw.payload.size()), dkw(kw.payload.size()),
        dvw(vw.payload.size());
    cudaMemcpy(dqw.p, qw.payload.data(), qw.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dgw.p, gw.payload.data(), gw.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dkw.p, kw.payload.data(), kw.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvw.p, vw.payload.data(), vw.payload.size(), cudaMemcpyHostToDevice);

    const std::size_t q_count  = static_cast<std::size_t>(kQRows) * t;
    const std::size_t kv_count = static_cast<std::size_t>(kKvRows) * t;
    DBuf q_ref(q_count * 2), g_ref(q_count * 2), k_ref(kv_count * 2), v_ref(kv_count * 2);
    DBuf q_got(q_count * 2), g_got(q_count * 2), k_got(kv_count * 2), v_got(kv_count * 2);
    Tensor tx(dx.p, DType::BF16, {kHidden, t});
    Tensor tq_ref(q_ref.p, DType::BF16, {kQRows, t});
    Tensor tg_ref(g_ref.p, DType::BF16, {kQRows, t});
    Tensor tk_ref(k_ref.p, DType::BF16, {kKvRows, t});
    Tensor tv_ref(v_ref.p, DType::BF16, {kKvRows, t});
    Tensor tq_got(q_got.p, DType::BF16, {kQRows, t});
    Tensor tg_got(g_got.p, DType::BF16, {kQRows, t});
    Tensor tk_got(k_got.p, DType::BF16, {kKvRows, t});
    Tensor tv_got(v_got.p, DType::BF16, {kKvRows, t});
    WorkspaceArena ws(256ULL << 20);
    const Weight qweight = qw.device_weight(dqw.p);
    const Weight gweight = gw.device_weight(dgw.p);
    const Weight kweight = kw.device_weight(dkw.p);
    const Weight vweight = vw.device_weight(dvw.p);
    launch_q4_test_reference(tx, qweight, tq_ref, nullptr);
    launch_q5_test_reference(tx, gweight, tg_ref, nullptr);
    launch_q4_test_reference(tx, kweight, tk_ref, nullptr);
    launch_q5_test_reference(tx, vweight, tv_ref, nullptr);
    ops::attn_input_proj(tx, qweight, gweight, kweight, vweight, tq_got, tg_got, tk_got, tv_got, ws,
                         nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f += verify("linear grouped attention q", from_device_bf16(q_got, q_count),
                from_device_bf16(q_ref, q_count), Tolerance::linear_tc());
    f += verify("linear grouped attention gate", from_device_bf16(g_got, q_count),
                from_device_bf16(g_ref, q_count), Tolerance::linear_tc());
    f += verify("linear grouped attention k", from_device_bf16(k_got, kv_count),
                from_device_bf16(k_ref, kv_count), Tolerance::linear_tc());
    f += verify("linear grouped attention v", from_device_bf16(v_got, kv_count),
                from_device_bf16(v_ref, kv_count), Tolerance::linear_tc());
    return f;
}

int grouped_gdn_correctness(std::int32_t t) {
    constexpr int kHidden = 5120;
    constexpr int kQkRows = 4096;
    constexpr int kVRows  = 6144;
    constexpr int kRows   = kQkRows + kVRows;
    std::vector<float> x(static_cast<std::size_t>(kHidden) * t);
    fill_uniform(x, 311u, -0.01f, 0.01f);
    round_to_bf16(x);
    DBuf dx  = to_device_bf16(x);
    auto qkw = patterned_weight(QType::Q4G64_F16S, kQkRows, kHidden, 23u);
    auto vw  = patterned_weight(QType::Q5G64_F16S, kVRows, kHidden, 29u);
    DBuf dqkw(qkw.payload.size()), dvw(vw.payload.size());
    cudaMemcpy(dqkw.p, qkw.payload.data(), qkw.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvw.p, vw.payload.data(), vw.payload.size(), cudaMemcpyHostToDevice);
    const std::size_t qk_count = static_cast<std::size_t>(kQkRows) * t;
    const std::size_t v_count  = static_cast<std::size_t>(kVRows) * t;
    DBuf qk_ref(qk_count * 2), v_ref(v_count * 2), got(static_cast<std::size_t>(kRows) * t * 2);
    Tensor tx(dx.p, DType::BF16, {kHidden, t});
    Tensor tqk(qk_ref.p, DType::BF16, {kQkRows, t});
    Tensor tv(v_ref.p, DType::BF16, {kVRows, t});
    Tensor tqkv(got.p, DType::BF16, {kRows, t});
    WorkspaceArena ws(256ULL << 20);
    const Weight qkweight = qkw.device_weight(dqkw.p);
    const Weight vweight  = vw.device_weight(dvw.p);
    launch_q4_test_reference(tx, qkweight, tqk, nullptr);
    ops::linear(tx, vweight, tv, ws, nullptr);
    ops::gdn_input_proj(tx, qkweight, vweight, tqkv, ws, nullptr);
    cudaDeviceSynchronize();
    std::vector<double> ref(static_cast<std::size_t>(kRows) * t);
    const auto qkh = from_device_bf16(qk_ref, qk_count);
    const auto vh  = from_device_bf16(v_ref, v_count);
    for (int col = 0; col < t; ++col) {
        std::copy_n(qkh.data() + static_cast<std::size_t>(col) * kQkRows, kQkRows,
                    ref.data() + static_cast<std::size_t>(col) * kRows);
        std::copy_n(vh.data() + static_cast<std::size_t>(col) * kVRows, kVRows,
                    ref.data() + static_cast<std::size_t>(col) * kRows + kQkRows);
    }
    return verify("linear grouped GDN qkv",
                  from_device_bf16(got, static_cast<std::size_t>(kRows) * t), ref,
                  Tolerance::linear_tc());
}

int gate_up_silu_correctness(std::int32_t t) {
    constexpr int kHidden = 5120;
    constexpr int kInter  = 17408;
    auto packed           = patterned_weight(QType::Q4G64_F16S, 2 * kInter, kHidden, 31u);
    DBuf dw(packed.payload.size());
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    std::vector<float> x(static_cast<std::size_t>(kHidden) * t);
    fill_uniform(x, 313u, -0.01f, 0.01f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);
    DBuf gate_up(static_cast<std::size_t>(2 * kInter) * t * 2);
    DBuf ref(static_cast<std::size_t>(kInter) * t * 2);
    DBuf got(static_cast<std::size_t>(kInter) * t * 2);
    Tensor tx(dx.p, DType::BF16, {kHidden, t});
    Tensor tgate_up(gate_up.p, DType::BF16, {2 * kInter, t});
    Tensor tref(ref.p, DType::BF16, {kInter, t});
    Tensor tgot(got.p, DType::BF16, {kInter, t});
    WorkspaceArena ws(256ULL << 20);
    const Weight weight = packed.device_weight(dw.p);
    launch_q4_test_reference(tx, weight, tgate_up, nullptr);
    ops::silu_mul(tgate_up.slice(0, 0, kInter), tgate_up.slice(0, kInter, kInter), tref, nullptr);
    ops::linear_swiglu(tx, weight, tgot, ws, nullptr);
    cudaDeviceSynchronize();
    return verify("linear Q4 gate/up SiLU paired",
                  from_device_bf16(got, static_cast<std::size_t>(kInter) * t),
                  from_device_bf16(ref, static_cast<std::size_t>(kInter) * t),
                  Tolerance::linear_tc());
}

int residual_epilogue_correctness(std::int32_t t) {
    constexpr int kN = 5120;
    constexpr int kK = 6144;
    auto packed      = patterned_weight(QType::Q5G64_F16S, kN, kK, 37u);
    DBuf dw(packed.payload.size());
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    std::vector<float> x(static_cast<std::size_t>(kK) * t);
    std::vector<float> residual(static_cast<std::size_t>(kN) * t);
    fill_uniform(x, 317u, -0.01f, 0.01f);
    fill_uniform(residual, 319u, -1.0f, 1.0f);
    round_to_bf16(x);
    round_to_bf16(residual);
    DBuf dx   = to_device_bf16(x);
    DBuf dref = to_device_bf16(residual);
    DBuf dgot = to_device_bf16(residual);
    DBuf y(static_cast<std::size_t>(kN) * t * 2);
    Tensor tx(dx.p, DType::BF16, {kK, t});
    Tensor ty(y.p, DType::BF16, {kN, t});
    Tensor tref(dref.p, DType::BF16, {kN, t});
    Tensor tgot(dgot.p, DType::BF16, {kN, t});
    WorkspaceArena ws(256ULL << 20);
    const Weight weight = packed.device_weight(dw.p);
    launch_q5_test_reference(tx, weight, ty, nullptr);
    ops::residual_add(ty, tref, nullptr);
    ops::linear_add(tx, weight, tgot, ws, nullptr);
    cudaDeviceSynchronize();
    return verify("linear Q5 residual epilogue",
                  from_device_bf16(dgot, static_cast<std::size_t>(kN) * t),
                  from_device_bf16(dref, static_cast<std::size_t>(kN) * t), Tolerance::linear_tc());
}

int prefill_fusion_correctness() {
    int f = 0;
    for (const std::int32_t t : {4, 17, 128, 129}) {
        f += grouped_attention_correctness(t);
        f += grouped_gdn_correctness(t);
        f += gate_up_silu_correctness(t);
        f += residual_epilogue_correctness(t);
    }
    return f;
}

int dense_metadata_validation() {
    int f = 0;
    std::vector<float> x(4, 1.0f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);
    DBuf dout(2 * 2u);
    Tensor tx(dx.p, DType::BF16, {4, 1});
    Tensor tout(dout.p, DType::BF16, {2, 1});
    WorkspaceArena ws(64ULL << 20);

    Weight w = dense_weight(dx.p, QType::BF16_CTRL, 2, 4);

    try {
        Weight bad = w;
        bad.scales = dx.p;
        ops::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear dense scale plane: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[0] = 64;
        ops::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear dense padded N: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[1] = 64;
        ops::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear dense padded K: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[2] = 2;
        ops::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear dense padded trailing dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor empty_tx(nullptr, DType::BF16, {4, 1});
        Tensor empty_out(nullptr, DType::BF16, {2, 1});
        empty_tx.ne[1]  = 0;
        empty_out.ne[1] = 0;
        ops::linear(empty_tx, w, empty_out, ws, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "linear dense empty T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    return f;
}

int dense_alignment_validation() {
    int f                    = 0;
    constexpr std::int32_t n = 2;
    constexpr std::int32_t k = 5;
    DBuf dx(static_cast<std::size_t>(k) * 2u + 16u);
    DBuf dw(static_cast<std::size_t>(n) * k * 2u + 16u);
    DBuf dout(static_cast<std::size_t>(n) * 2u + 16u);
    cudaMemset(dx.p, 0, dx.bytes);
    cudaMemset(dw.p, 0, dw.bytes);
    cudaMemset(dout.p, 0, dout.bytes);

    auto* x_bytes   = static_cast<unsigned char*>(dx.p);
    auto* w_bytes   = static_cast<unsigned char*>(dw.p);
    auto* out_bytes = static_cast<unsigned char*>(dout.p);
    Tensor tx(dx.p, DType::BF16, {k, 1});
    Tensor tout(dout.p, DType::BF16, {n, 1});
    Weight w = dense_weight(dw.p, QType::BF16_CTRL, n, k);
    WorkspaceArena ws(64ULL << 20);

    try {
        Tensor bad_x(x_bytes + 2, DType::BF16, {k, 1});
        ops::linear(bad_x, w, tout, ws, nullptr);
        cudaDeviceSynchronize();
        std::cerr << "linear dense unaligned x: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_w = dense_weight(w_bytes + 2, QType::BF16_CTRL, n, k);
        ops::linear(tx, bad_w, tout, ws, nullptr);
        cudaDeviceSynchronize();
        std::cerr << "linear dense unaligned weight: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(out_bytes + 2, DType::BF16, {n, 1});
        ops::linear(tx, w, bad_out, ws, nullptr);
        cudaDeviceSynchronize();
        std::cerr << "linear dense unaligned out: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    return f;
}

int lowbit_metadata_validation(QType qtype) {
    int f                    = 0;
    constexpr std::int32_t n = 64;
    constexpr std::int32_t k = 64;
    std::vector<float> source(static_cast<std::size_t>(n) * k);
    fill_uniform(source, 123u, -0.125f, 0.125f);
    round_to_bf16(source);
    row_split::PackedWeight packed = row_split::pack_row_split_lowbit(source, n, k, qtype);

    std::vector<float> x(k, 1.0f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);
    DBuf dout(static_cast<std::size_t>(n) * 2u);
    Tensor tx(dx.p, DType::BF16, {k, 1});
    Tensor tout(dout.p, DType::BF16, {n, 1});
    WorkspaceArena ws(64ULL << 20);
    const std::uint32_t expected_group = qtype == QType::W8G32_F16S ? 32u : 64u;

    try {
        Weight bad        = packed.device_weight(dx.p);
        bad.payload_bytes = 0;
        ops::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " payload size: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad     = packed.device_weight(dx.p);
        bad.group      = expected_group == 32u ? 64 : 32;
        bad.group_size = expected_group == 32u ? 64u : 32u;
        ops::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " group size: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad      = packed.device_weight(dx.p);
        bad.scale_dtype = DType::FP32;
        ops::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " scale dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = packed.device_weight(dx.p);
        bad.padded_shape[1] = bad.padded_shape[1] + 128;
        ops::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " padded K: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    if (qtype == QType::W8G32_F16S) {
        try {
            Weight bad           = packed.device_weight(dx.p);
            bad.qhigh            = dx.p;
            bad.high_plane_bytes = 1;
            ops::linear(tx, bad, tout, ws, nullptr);
            std::cerr << "linear w8g32 high plane: expected invalid_argument\n";
            ++f;
        } catch (const std::invalid_argument&) {}
    }

    try {
        Tensor empty_tx  = tx;
        Tensor empty_out = tout;
        empty_tx.ne[1]   = 0;
        empty_out.ne[1]  = 0;
        Weight bad       = packed.device_weight(nullptr);
        ops::linear(empty_tx, bad, empty_out, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype)
                  << " empty T null payload: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor empty_tx  = tx;
        Tensor empty_out = tout;
        empty_tx.data    = nullptr;
        empty_out.data   = nullptr;
        empty_tx.ne[1]   = 0;
        empty_out.ne[1]  = 0;
        ops::linear(empty_tx, packed.device_weight(dx.p), empty_out, ws, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "linear " << qtype_name(qtype) << " empty T: expected no throw, got "
                  << e.what() << '\n';
        ++f;
    }

    return f;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    if (std::getenv("NINFER_LINEAR_TEST_W8G32_ONLY") != nullptr) {
        int f = 0;
        f += one_quant_shape(QType::W8G32_F16S, 64, 256, {128}, 119u);
        f += one_quant_shape(QType::W8G32_F16S, 70, 130, {17}, 121u);
        f += paired_w8g32_shape(64, 256, {17, 128}, 127u);
        std::cout << (f ? "FAIL" : "OK") << " linear W8G32 focused correctness\n";
        return f ? 1 : 0;
    }
    if (std::getenv("NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY") != nullptr) {
        const int f = prefill_fusion_correctness();
        std::cout << (f ? "FAIL" : "OK") << " linear prefill fusion correctness\n";
        return f ? 1 : 0;
    }

    int f = 0;
    f += prefill_fusion_correctness();
    f += dense_metadata_validation();
    f += dense_alignment_validation();
    f += lowbit_metadata_validation(QType::Q4G64_F16S);
    f += lowbit_metadata_validation(QType::Q5G64_F16S);
    f += lowbit_metadata_validation(QType::Q6G64_F16S);
    f += lowbit_metadata_validation(QType::W8G32_F16S);

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        for (QType qtype : {QType::BF16_CTRL, QType::FP32_CTRL}) {
            f += one_dense_shape(48, 5120, 1, qtype, seed);
            f += one_dense_shape(48, 5120, 7, qtype, seed);
        }
    }
    for (QType qtype : {QType::BF16_CTRL, QType::FP32_CTRL}) {
        if (qtype == QType::BF16_CTRL) { f += one_dense_shape(64, 64, 32, qtype, 11u); }
        f += one_dense_shape(64, 96, 64, qtype, 13u);
        f += one_dense_shape(96, 513, 33, qtype, 15u);
        f += one_dense_shape(5120, 6144, 1, qtype, 17u);
        f += one_dense_shape(5120, 6144, 7, qtype, 17u);
        f += one_dense_shape(37, 513, 19, qtype, 19u);
        f += one_dense_shape(80, 130, 17, qtype, 101u, 8.0f, 1.5f, "stress");
    }
    f += one_dense_shape(96, 128, 512, QType::BF16_CTRL, 53u);
    f += fp32_ctrl_first_column_consistency(48, 64, 7);
    f += fp32_ctrl_first_column_consistency(64, 64, 32);
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_quant_shape(QType::W8G32_F16S, 70, 130, {1, 2, 5, 17}, seed);
        f += one_quant_shape(QType::W8G32_F16S, 70, 131, {1, 2, 5, 17}, seed);
        for (auto [n, k] : {std::pair<std::int32_t, std::int32_t>{5120, 10240},
                            std::pair<std::int32_t, std::int32_t>{14336, 5120},
                            std::pair<std::int32_t, std::int32_t>{5120, 6144},
                            std::pair<std::int32_t, std::int32_t>{34816, 5120},
                            std::pair<std::int32_t, std::int32_t>{5120, 17408}}) {
            f += one_quant_shape(QType::W8G32_F16S, n, k, {1, 2, 5, 17}, seed);
        }
    }
    f += one_quant_shape(QType::W8G32_F16S, 96, 130, {1, 2, 5, 17}, 113u, 8.0f, 1.25f, "stress");
    f += paired_w8g32_shape(64, 256, {17, 128}, 127u);
    // Q4 public correctness is exact-admission only. The dedicated Q4 plan/dispatch tests cover
    // every route seam and compare public auto against the fixed entry word-for-word; these
    // oracle points cover the split-K/SIMT numerical boundary at real product geometries.
    f += one_quant_shape(QType::Q4G64_F16S, 1024, 5120, {1, 2, 15, 16}, 23u);
    f += one_quant_shape(QType::Q4G64_F16S, 4096, 5120, {1}, 25u);
    f += one_quant_shape(QType::Q4G64_F16S, 3456, 1152, {4, 40}, 27u);
    // Q5 public correctness is exact-admission only. Fused input projections own their
    // large-column parent operations; linear_add owns T=1 and T>=25 residual epilogues.
    f += one_quant_shape(QType::Q5G64_F16S, 1024, 5120, {1, 4, 5, 16}, 29u);
    f += one_quant_shape(QType::Q5G64_F16S, 6144, 5120, {1, 2, 6, 7, 24, 25, 64, 65, 512}, 31u);
    f += one_quant_shape(QType::Q5G64_F16S, 5120, 6144, {2, 6, 7, 24}, 37u);
    f += one_quant_shape(QType::Q5G64_F16S, 5120, 17408, {2, 6, 7, 24}, 41u);
    f += one_quant_shape(QType::Q5G64_F16S, 1152, 1152, {4, 56, 60, 128}, 43u);
    f += one_quant_shape(QType::Q5G64_F16S, 1152, 4304, {4, 84, 88, 128}, 47u);
    // Q6 public correctness is exact-admission only. The dedicated Q6 plan/dispatch tests cover
    // every route seam and compare public auto against the fixed entry word-for-word. These
    // independent fp64-oracle points cover the real head and Vision geometries.
    f += one_quant_shape(QType::Q6G64_F16S, 248320, 5120, {1}, 211u);
    f += one_quant_shape(QType::Q6G64_F16S, 1152, 1536, {4, 40, 128}, 227u);

    std::cout << (f ? "FAIL" : "OK") << " linear correctness\n";
    return f ? 1 : 0;
}
