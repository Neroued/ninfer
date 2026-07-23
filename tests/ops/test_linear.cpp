// Correctness and exact-admission coverage for pure Linear format backends.
#include "ninfer/ops/linear.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/op_tester.h"
#include "ops/linear/bf16/bf16_contiguous_plan.h"
#include "ops/linear/q4/q4_rowsplit_plan.h"
#include "ops/linear/q5/q5_rowsplit_plan.h"
#include "ops/linear/q6/q6_rowsplit_plan.h"
#include "ops/linear/w8/w8_rowsplit_plan.h"
#include "ops/linear_add/q5/q5_linear_add_plan.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"
#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"
#include "ops/row_split_pack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

const char* qtype_name(QType qtype) {
    switch (qtype) {
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

std::vector<std::int32_t> sampled_indices(std::int32_t limit) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index : {0, 1, limit / 3, limit / 2, limit - 2, limit - 1}) {
        if (index >= 0 && index < limit &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    return result;
}

constexpr std::size_t kOutputGuardBytes = 256;
constexpr std::uint8_t kOutputGuardByte = 0xa5;

struct GuardedBf16Output {
    explicit GuardedBf16Output(std::size_t words)
        : storage(words * sizeof(std::uint16_t) + 2 * kOutputGuardBytes), words(words) {
        cudaMemset(storage.p, kOutputGuardByte, storage.bytes);
    }

    void* data() const { return static_cast<std::uint8_t*>(storage.p) + kOutputGuardBytes; }

    int verify_guards(const std::string& label) const {
        std::vector<std::uint8_t> prefix(kOutputGuardBytes);
        std::vector<std::uint8_t> suffix(kOutputGuardBytes);
        cudaMemcpy(prefix.data(), storage.p, prefix.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(suffix.data(),
                   static_cast<const std::uint8_t*>(data()) + words * sizeof(std::uint16_t),
                   suffix.size(), cudaMemcpyDeviceToHost);
        const auto intact = [](const std::vector<std::uint8_t>& bytes) {
            return std::all_of(bytes.begin(), bytes.end(),
                               [](std::uint8_t byte) { return byte == kOutputGuardByte; });
        };
        if (intact(prefix) && intact(suffix)) { return 0; }
        std::cerr << label << ": output canary was overwritten\n";
        return 1;
    }

    int verify_initialized(const std::string& label) const {
        std::vector<std::uint16_t> bits(words);
        cudaMemcpy(bits.data(), data(), bits.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost);
        if (std::find(bits.begin(), bits.end(), std::uint16_t{0xa5a5}) == bits.end()) { return 0; }
        std::cerr << label << ": output contains an unpublished poison word\n";
        return 1;
    }

    DBuf storage;
    std::size_t words;
};

std::vector<double> from_device_bf16_ptr(const void* device, std::size_t count) {
    std::vector<std::uint16_t> bits(count);
    cudaMemcpy(bits.data(), device, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    std::vector<double> result(count);
    for (std::size_t i = 0; i < count; ++i) { result[i] = bf16_to_f32(bits[i]); }
    return result;
}

std::vector<std::uint16_t> from_device_bf16_bits(const void* device, std::size_t count) {
    std::vector<std::uint16_t> bits(count);
    cudaMemcpy(bits.data(), device, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return bits;
}

template <class Launch>
void capture_and_replay(Launch&& launch) {
    cudaStream_t stream  = nullptr;
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launch(stream);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}

int verify_sampled_projection(const std::string& label, const std::vector<double>& full_output,
                              std::int32_t output_stride, std::int32_t output_row_offset,
                              const std::vector<float>& x, std::int32_t k, std::int32_t t,
                              const row_split::PackedWeight& packed, const Tolerance& tolerance,
                              std::int32_t weight_row_offset = 0, std::int32_t logical_rows = -1) {
    if (logical_rows < 0) { logical_rows = packed.weight.shape[0] - weight_row_offset; }
    const auto rows = sampled_indices(logical_rows);
    const auto cols = sampled_indices(t);
    std::vector<double> actual;
    std::vector<double> reference;
    actual.reserve(rows.size() * cols.size());
    reference.reserve(actual.capacity());
    for (const std::int32_t col : cols) {
        const float* xcol = x.data() + static_cast<std::size_t>(col) * k;
        for (const std::int32_t row : rows) {
            actual.push_back(full_output[static_cast<std::size_t>(col) * output_stride +
                                         output_row_offset + row]);
            reference.push_back(
                row_split::dot_row_split_lowbit_fp64(packed, weight_row_offset + row, xcol, k));
        }
    }
    return verify(label.c_str(), actual, reference, tolerance);
}

int verify_complete_projection(const std::string& label, const std::vector<double>& actual,
                               const std::vector<float>& x, std::int32_t k, std::int32_t t,
                               const row_split::PackedWeight& packed,
                               std::int32_t weight_row_offset, std::int32_t rows,
                               const Tolerance& tolerance) {
    std::vector<double> reference(static_cast<std::size_t>(rows) * t);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const std::int32_t n_threads =
        std::min<std::int32_t>(rows, static_cast<std::int32_t>(std::min<unsigned>(hw, 16u)));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n_threads));
    for (std::int32_t tid = 0; tid < n_threads; ++tid) {
        const std::int32_t row_begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * tid) / n_threads);
        const std::int32_t row_end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * (tid + 1)) / n_threads);
        threads.emplace_back([&, row_begin, row_end] {
            for (std::int32_t col = 0; col < t; ++col) {
                const float* xcol = x.data() + static_cast<std::size_t>(col) * k;
                for (std::int32_t row = row_begin; row < row_end; ++row) {
                    reference[static_cast<std::size_t>(col) * rows + row] =
                        row_split::dot_row_split_lowbit_fp64(packed, weight_row_offset + row, xcol,
                                                             k);
                }
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    return verify(label.c_str(), actual, reference, tolerance);
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
        } else if (qtype == QType::W8G32_F16S) {
            const ops::detail::W8Plan plan =
                ops::detail::w8_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
            tol = ops::detail::w8_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                   : Tolerance::linear_bf16();
        }
        failures += verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t),
                           ref, tol);
    }
    return failures;
}

int one_quant_shape_sampled(QType qtype, std::int32_t n, std::int32_t k, std::int32_t t,
                            std::uint32_t seed) {
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k);
    fill_uniform(source_weight, seed + 2000u, -0.125f, 0.125f);
    round_to_bf16(source_weight);
    row_split::PackedWeight packed = row_split::pack_row_split_lowbit(source_weight, n, k, qtype);
    std::vector<float>().swap(source_weight);

    std::vector<float> x(static_cast<std::size_t>(k) * t);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(x);

    DBuf dx = to_device_bf16(x);
    DBuf dweight(packed.payload.size());
    DBuf dout(static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
    cudaMemcpy(dweight.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    WorkspaceArena ws(64ULL << 20);
    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor tout(dout.p, DType::BF16, {n, t});
    try {
        ops::linear(tx, packed.device_weight(dweight.p), tout, ws, nullptr);
        cudaDeviceSynchronize();
    } catch (const std::exception& e) {
        std::cerr << "linear " << qtype_name(qtype) << " sampled [" << n << "," << k << "] T=" << t
                  << " seed=" << seed << ": unexpected exception: " << e.what() << '\n';
        return 1;
    }

    const std::vector<double> full_output = from_device_bf16(dout, static_cast<std::size_t>(n) * t);
    std::vector<std::int32_t> sampled_rows;
    std::vector<std::int32_t> sampled_cols;
    const auto add_unique = [](std::vector<std::int32_t>& values, std::int32_t value,
                               std::int32_t limit) {
        if (value >= 0 && value < limit &&
            std::find(values.begin(), values.end(), value) == values.end()) {
            values.push_back(value);
        }
    };
    for (const std::int32_t row : {0, 1, n / 8, n / 3, n / 2, n - 2, n - 1}) {
        add_unique(sampled_rows, row, n);
    }
    for (const std::int32_t col : {0, 1, t / 8, t / 3, t / 2, t - 2, t - 1}) {
        add_unique(sampled_cols, col, t);
    }

    std::vector<double> actual;
    std::vector<double> reference;
    actual.reserve(sampled_rows.size() * sampled_cols.size());
    reference.reserve(actual.capacity());
    for (const std::int32_t col : sampled_cols) {
        const float* xcol = x.data() + static_cast<std::size_t>(col) * k;
        for (const std::int32_t row : sampled_rows) {
            const float* wrow = packed.dequant.data() + static_cast<std::size_t>(row) * k;
            double acc        = 0.0;
            for (std::int32_t kk = 0; kk < k; ++kk) {
                acc += static_cast<double>(wrow[kk]) * static_cast<double>(xcol[kk]);
            }
            actual.push_back(full_output[static_cast<std::size_t>(col) * n + row]);
            reference.push_back(acc);
        }
    }

    Tolerance tolerance = Tolerance::linear_bf16();
    if (qtype == QType::Q6G64_F16S) {
        const ops::detail::Q6Plan plan =
            ops::detail::q6_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
        tolerance = ops::detail::q6_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                     : Tolerance::linear_bf16();
    } else if (qtype == QType::W8G32_F16S) {
        const ops::detail::W8Plan plan =
            ops::detail::w8_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
        tolerance = ops::detail::w8_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                     : Tolerance::linear_bf16();
    } else {
        throw std::invalid_argument("sampled Linear oracle supports only Q6 and W8");
    }
    const std::string label = "linear " + std::string(qtype_name(qtype)) + " sampled [" +
                              std::to_string(n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(t) + " seed=" + std::to_string(seed);
    return verify(label.c_str(), actual, reference, tolerance);
}

int one_patterned_quant_shape_sampled(QType qtype, std::int32_t n, std::int32_t k, std::int32_t t,
                                      std::uint32_t seed) {
    row_split::PackedWeight packed = row_split::make_patterned_weight(qtype, n, k, seed + 2000u);
    std::vector<float> x(static_cast<std::size_t>(k) * t);
    fill_uniform(x, seed, -0.01f, 0.01f);
    round_to_bf16(x);

    DBuf dx = to_device_bf16(x);
    DBuf dweight(packed.payload.size());
    DBuf dout(static_cast<std::size_t>(n) * t * sizeof(std::uint16_t));
    cudaMemcpy(dweight.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    WorkspaceArena ws(64ULL << 20);
    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor tout(dout.p, DType::BF16, {n, t});
    try {
        ops::linear(tx, packed.device_weight(dweight.p), tout, ws, nullptr);
        cudaDeviceSynchronize();
    } catch (const std::exception& e) {
        std::cerr << "linear " << qtype_name(qtype) << " patterned sampled [" << n << "," << k
                  << "] T=" << t << " seed=" << seed << ": unexpected exception: " << e.what()
                  << '\n';
        return 1;
    }

    Tolerance tolerance = Tolerance::linear_bf16();
    if (qtype == QType::Q4G64_F16S) {
        const auto plan =
            ops::detail::q4_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
        tolerance = ops::detail::q4_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                     : Tolerance::linear_bf16();
    } else if (qtype == QType::Q5G64_F16S) {
        const auto plan =
            ops::detail::q5_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
        tolerance = ops::detail::q5_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                     : Tolerance::linear_bf16();
    } else if (qtype == QType::Q6G64_F16S) {
        const auto plan =
            ops::detail::q6_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
        tolerance = ops::detail::q6_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                     : Tolerance::linear_bf16();
    } else if (qtype == QType::W8G32_F16S) {
        const auto plan =
            ops::detail::w8_rowsplit_resolve_plan({n, k, packed.weight.padded_shape[1], t});
        tolerance = ops::detail::w8_schedule_uses_mma(plan.schedule) ? Tolerance::linear_tc()
                                                                     : Tolerance::linear_bf16();
    } else {
        throw std::invalid_argument("patterned sampled Linear oracle received an unsupported type");
    }

    const std::string label = "linear " + std::string(qtype_name(qtype)) +
                              " expanded-T patterned FP64 oracle [" + std::to_string(n) + "," +
                              std::to_string(k) + "] T=" + std::to_string(t);
    return verify_sampled_projection(label, from_device_bf16(dout, static_cast<std::size_t>(n) * t),
                                     n, 0, x, k, t, packed, tolerance);
}

int q6_output_head_dflash_oracle() {
    constexpr std::int32_t n     = 248320;
    constexpr std::int32_t k     = 2048;
    constexpr std::int32_t max_t = 16;

    row_split::PackedWeight packed =
        row_split::make_patterned_weight(QType::Q6G64_F16S, n, k, 4226u);
    std::vector<float> x(static_cast<std::size_t>(k) * max_t);
    fill_uniform(x, 2226u, -0.01f, 0.01f);
    round_to_bf16(x);

    DBuf dx = to_device_bf16(x);
    DBuf dweight(packed.payload.size());
    cudaMemcpy(dweight.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = packed.device_weight(dweight.p);
    WorkspaceArena ws(64ULL << 20);

    std::vector<std::int32_t> rows = sampled_indices(n);
    const auto add_row             = [&](std::int32_t row) {
        if (std::find(rows.begin(), rows.end(), row) == rows.end()) { rows.push_back(row); }
    };
    for (const std::int32_t row : {7, 8, 31, 32, 63, 64, 127, 128}) { add_row(row); }

    int failures = 0;
    for (std::int32_t t = 1; t <= max_t; ++t) {
        GuardedBf16Output output(static_cast<std::size_t>(n) * t);
        Tensor tx(dx.p, DType::BF16, {k, t});
        Tensor tout(output.data(), DType::BF16, {n, t});
        const std::string label =
            "linear q6 dFlash output head sampled FP64 oracle T=" + std::to_string(t);
        try {
            ops::linear(tx, weight, tout, ws, nullptr);
            cudaDeviceSynchronize();
            if (t == max_t) {
                capture_and_replay(
                    [&](cudaStream_t stream) { ops::linear(tx, weight, tout, ws, stream); });
            }
        } catch (const std::exception& error) {
            std::cerr << label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }

        const std::vector<std::uint16_t> output_bits =
            from_device_bf16_bits(output.data(), static_cast<std::size_t>(n) * t);
        if (std::find(output_bits.begin(), output_bits.end(), std::uint16_t{0xa5a5u}) !=
            output_bits.end()) {
            std::cerr << label << ": poisoned output word remained\n";
            ++failures;
        }
        failures += output.verify_guards(label);

        std::vector<double> actual;
        std::vector<double> reference;
        actual.reserve(rows.size() * static_cast<std::size_t>(t));
        reference.reserve(actual.capacity());
        for (std::int32_t col = 0; col < t; ++col) {
            const float* xcol = x.data() + static_cast<std::size_t>(col) * k;
            for (const std::int32_t row : rows) {
                actual.push_back(bf16_to_f32(output_bits[static_cast<std::size_t>(col) * n + row]));
                reference.push_back(row_split::dot_row_split_lowbit_fp64(packed, row, xcol, k));
            }
        }
        const auto plan           = ops::detail::q6_rowsplit_resolve_plan({n, k, k, t});
        const Tolerance tolerance = ops::detail::q6_schedule_uses_mma(plan.schedule)
                                        ? Tolerance::linear_tc()
                                        : Tolerance::linear_bf16();
        failures += verify(label.c_str(), actual, reference, tolerance);
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

int grouped_attention_correctness(std::int32_t t, bool use_graph) {
    constexpr int kHidden = 5120;
    constexpr int kQRows  = 6144;
    constexpr int kKvRows = 1024;
    std::vector<float> x(static_cast<std::size_t>(kHidden) * t);
    fill_uniform(x, 301u, -0.01f, 0.01f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);

    auto qkw = row_split::make_patterned_weight(QType::Q4G64_F16S, kQRows + kKvRows, kHidden, 11u);
    auto gvw = row_split::make_patterned_weight(QType::Q5G64_F16S, kQRows + kKvRows, kHidden, 13u);
    DBuf dqkw(qkw.payload.size()), dgvw(gvw.payload.size());
    cudaMemcpy(dqkw.p, qkw.payload.data(), qkw.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dgvw.p, gvw.payload.data(), gvw.payload.size(), cudaMemcpyHostToDevice);

    const std::size_t q_count  = static_cast<std::size_t>(kQRows) * t;
    const std::size_t kv_count = static_cast<std::size_t>(kKvRows) * t;
    GuardedBf16Output q_got(q_count), g_got(q_count), k_got(kv_count), v_got(kv_count);
    Tensor tx(dx.p, DType::BF16, {kHidden, t});
    Tensor tq_got(q_got.data(), DType::BF16, {kQRows, t});
    Tensor tg_got(g_got.data(), DType::BF16, {kQRows, t});
    Tensor tk_got(k_got.data(), DType::BF16, {kKvRows, t});
    Tensor tv_got(v_got.data(), DType::BF16, {kKvRows, t});
    WorkspaceArena ws(1);
    const Weight qkweight = qkw.device_weight(dqkw.p);
    const Weight gvweight = gvw.device_weight(dgvw.p);
    const auto launch     = [&](cudaStream_t stream) {
        ops::attn_input_proj(tx, qkweight, gvweight, tq_got, tg_got, tk_got, tv_got, ws, stream);
    };
    if (use_graph) {
        capture_and_replay(launch);
    } else {
        launch(nullptr);
        cudaDeviceSynchronize();
    }

    int f = 0;
    f += verify_sampled_projection(
        "linear grouped attention q FP64 oracle", from_device_bf16_ptr(q_got.data(), q_count),
        kQRows, 0, x, kHidden, t, qkw, t <= 16 ? Tolerance::linear_bf16() : Tolerance::linear_tc(),
        0, kQRows);
    f += verify_sampled_projection(
        "linear grouped attention gate FP64 oracle", from_device_bf16_ptr(g_got.data(), q_count),
        kQRows, 0, x, kHidden, t, gvw, t <= 16 ? Tolerance::linear_bf16() : Tolerance::linear_tc(),
        0, kQRows);
    f += verify_sampled_projection(
        "linear grouped attention k FP64 oracle", from_device_bf16_ptr(k_got.data(), kv_count),
        kKvRows, 0, x, kHidden, t, qkw, t <= 16 ? Tolerance::linear_bf16() : Tolerance::linear_tc(),
        kQRows, kKvRows);
    f += verify_sampled_projection(
        "linear grouped attention v FP64 oracle", from_device_bf16_ptr(v_got.data(), kv_count),
        kKvRows, 0, x, kHidden, t, gvw, t <= 16 ? Tolerance::linear_bf16() : Tolerance::linear_tc(),
        kQRows, kKvRows);
    f += q_got.verify_guards("attention q T=" + std::to_string(t));
    f += g_got.verify_guards("attention gate T=" + std::to_string(t));
    f += k_got.verify_guards("attention k T=" + std::to_string(t));
    f += v_got.verify_guards("attention v T=" + std::to_string(t));
    return f;
}

int grouped_gdn_correctness(std::int32_t t, bool use_graph) {
    constexpr int kHidden = 5120;
    constexpr int kQkRows = 4096;
    constexpr int kVRows  = 6144;
    constexpr int kRows   = kQkRows + kVRows;
    std::vector<float> x(static_cast<std::size_t>(kHidden) * t);
    fill_uniform(x, 311u, -0.01f, 0.01f);
    round_to_bf16(x);
    DBuf dx  = to_device_bf16(x);
    auto qkw = row_split::make_patterned_weight(QType::Q4G64_F16S, kQkRows, kHidden, 23u);
    auto vw  = row_split::make_patterned_weight(QType::Q5G64_F16S, kVRows, kHidden, 29u);
    DBuf dqkw(qkw.payload.size()), dvw(vw.payload.size());
    cudaMemcpy(dqkw.p, qkw.payload.data(), qkw.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvw.p, vw.payload.data(), vw.payload.size(), cudaMemcpyHostToDevice);
    const std::size_t output_count = static_cast<std::size_t>(kRows) * t;
    GuardedBf16Output got(output_count);
    Tensor tx(dx.p, DType::BF16, {kHidden, t});
    Tensor tqkv(got.data(), DType::BF16, {kRows, t});
    WorkspaceArena ws(1);
    const Weight qkweight = qkw.device_weight(dqkw.p);
    const Weight vweight  = vw.device_weight(dvw.p);
    Tensor qk_view        = tqkv.slice(0, 0, kQkRows);
    Tensor value_view     = tqkv.slice(0, kQkRows, kVRows);
    int f                 = 0;
    if (qk_view.nb[1] != tqkv.nb[1] || value_view.nb[1] != tqkv.nb[1] ||
        qk_view.nb[1] != static_cast<std::size_t>(kRows) * sizeof(std::uint16_t) ||
        value_view.data != static_cast<std::uint8_t*>(tqkv.data) +
                               static_cast<std::size_t>(kQkRows) * sizeof(std::uint16_t)) {
        std::cerr << "GDN output row views have the wrong shared pitch/offset T=" << t << '\n';
        ++f;
    }
    const auto launch = [&](cudaStream_t stream) {
        ops::gdn_input_proj(tx, qkweight, vweight, tqkv, ws, stream);
    };
    if (use_graph) {
        capture_and_replay(launch);
    } else {
        launch(nullptr);
        cudaDeviceSynchronize();
    }
    const std::vector<double> full = from_device_bf16_ptr(got.data(), output_count);
    f += verify_sampled_projection("linear grouped GDN qk FP64 oracle", full, kRows, 0, x, kHidden,
                                   t, qkw,
                                   t <= 16 ? Tolerance::linear_bf16() : Tolerance::linear_tc());
    f += verify_sampled_projection("linear grouped GDN v FP64 oracle", full, kRows, kQkRows, x,
                                   kHidden, t, vw,
                                   t <= 16 ? Tolerance::linear_bf16() : Tolerance::linear_tc());
    if (t <= 16) {
        DBuf qk_contiguous(static_cast<std::size_t>(kQkRows) * t * sizeof(std::uint16_t));
        DBuf v_contiguous(static_cast<std::size_t>(kVRows) * t * sizeof(std::uint16_t));
        DBuf materialized(output_count * sizeof(std::uint16_t));
        Tensor tqk(qk_contiguous.p, DType::BF16, {kQkRows, t});
        Tensor tv(v_contiguous.p, DType::BF16, {kVRows, t});
        ops::linear(tx, qkweight, tqk, ws, nullptr);
        ops::linear(tx, vweight, tv, ws, nullptr);
        cudaMemcpy2DAsync(materialized.p, static_cast<std::size_t>(kRows) * sizeof(std::uint16_t),
                          qk_contiguous.p,
                          static_cast<std::size_t>(kQkRows) * sizeof(std::uint16_t),
                          static_cast<std::size_t>(kQkRows) * sizeof(std::uint16_t), t,
                          cudaMemcpyDeviceToDevice, nullptr);
        cudaMemcpy2DAsync(static_cast<std::uint8_t*>(materialized.p) +
                              static_cast<std::size_t>(kQkRows) * sizeof(std::uint16_t),
                          static_cast<std::size_t>(kRows) * sizeof(std::uint16_t), v_contiguous.p,
                          static_cast<std::size_t>(kVRows) * sizeof(std::uint16_t),
                          static_cast<std::size_t>(kVRows) * sizeof(std::uint16_t), t,
                          cudaMemcpyDeviceToDevice, nullptr);
        cudaDeviceSynchronize();
        if (from_device_bf16_bits(got.data(), output_count) !=
            from_device_bf16_bits(materialized.p, output_count)) {
            std::cerr << "GDN direct output differs from materialize/copy control T=" << t << '\n';
            ++f;
        }
    }
    f += got.verify_guards("GDN qkv T=" + std::to_string(t));
    return f;
}

int w8_attention_input_correctness() {
    constexpr int kHidden = 2048;
    constexpr int kQRows  = 4096;
    constexpr int kKvRows = 512;
    constexpr int kRows   = 9216;
    constexpr int kMaxT   = 1024;

    std::vector<float> x(static_cast<std::size_t>(kHidden) * kMaxT);
    fill_uniform(x, 1701u, -0.25f, 0.25f);
    round_to_bf16(x);
    auto packed = row_split::make_patterned_weight(QType::W8G32_F16S, kRows, kHidden, 1703u);
    DBuf dx     = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = packed.device_weight(dw.p);
    WorkspaceArena ws(1);

    int failures = 0;
    for (const std::int32_t t :
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 128, 129, 1024}) {
        const std::size_t q_count  = static_cast<std::size_t>(kQRows) * t;
        const std::size_t kv_count = static_cast<std::size_t>(kKvRows) * t;
        GuardedBf16Output q_got(q_count), gate_got(q_count), k_got(kv_count), v_got(kv_count);
        Tensor tx(dx.p, DType::BF16, {kHidden, t});
        Tensor tq(q_got.data(), DType::BF16, {kQRows, t});
        Tensor tgate(gate_got.data(), DType::BF16, {kQRows, t});
        Tensor tk(k_got.data(), DType::BF16, {kKvRows, t});
        Tensor tv(v_got.data(), DType::BF16, {kKvRows, t});
        const auto launch = [&](cudaStream_t stream) {
            ops::attn_input_proj(tx, weight, tq, tgate, tk, tv, ws, stream);
        };
        if (t == 1 || t == 16) {
            capture_and_replay(launch);
        } else {
            launch(nullptr);
            cudaDeviceSynchronize();
        }

        const auto plan =
            ops::detail::w8_attn_input_resolve_plan({kHidden, kQRows, kKvRows, kRows, kHidden, t});
        const bool mma = plan.schedule != ops::detail::W8AttnInputScheduleId::DecodeR8Direct &&
                         plan.schedule != ops::detail::W8AttnInputScheduleId::SimtR8C4;
        const Tolerance tolerance = mma ? Tolerance::linear_tc() : Tolerance::linear_bf16();
        const std::string suffix  = " T=" + std::to_string(t);
        failures += verify_sampled_projection("W8 attention q FP64 oracle" + suffix,
                                              from_device_bf16_ptr(q_got.data(), q_count), kQRows,
                                              0, x, kHidden, t, packed, tolerance, 0, kQRows);
        failures += verify_sampled_projection("W8 attention k FP64 oracle" + suffix,
                                              from_device_bf16_ptr(k_got.data(), kv_count), kKvRows,
                                              0, x, kHidden, t, packed, tolerance, kQRows, kKvRows);
        failures +=
            verify_sampled_projection("W8 attention gate FP64 oracle" + suffix,
                                      from_device_bf16_ptr(gate_got.data(), q_count), kQRows, 0, x,
                                      kHidden, t, packed, tolerance, kQRows + kKvRows, kQRows);
        failures += verify_sampled_projection(
            "W8 attention v FP64 oracle" + suffix, from_device_bf16_ptr(v_got.data(), kv_count),
            kKvRows, 0, x, kHidden, t, packed, tolerance, 2 * kQRows + kKvRows, kKvRows);
        failures += q_got.verify_guards("W8 attention q" + suffix);
        failures += gate_got.verify_guards("W8 attention gate" + suffix);
        failures += k_got.verify_guards("W8 attention k" + suffix);
        failures += v_got.verify_guards("W8 attention v" + suffix);
        failures += q_got.verify_initialized("W8 attention q" + suffix);
        failures += gate_got.verify_initialized("W8 attention gate" + suffix);
        failures += k_got.verify_initialized("W8 attention k" + suffix);
        failures += v_got.verify_initialized("W8 attention v" + suffix);
    }
    return failures;
}

int w8_companion_attention_input_correctness() {
    constexpr int kHidden = 2048;
    constexpr int kQRows  = 4096;
    constexpr int kKvRows = 1024;
    constexpr int kRows   = 6144;
    constexpr int kMaxT   = 1024;

    std::vector<float> x(static_cast<std::size_t>(kHidden) * kMaxT);
    fill_uniform(x, 1751u, -0.25f, 0.25f);
    round_to_bf16(x);
    auto packed = row_split::make_patterned_weight(QType::W8G32_F16S, kRows, kHidden, 1753u);
    DBuf dx     = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = packed.device_weight(dw.p);
    WorkspaceArena ws(1);
    std::vector<std::uint8_t> input_before(dx.bytes);
    std::vector<std::uint8_t> weight_before(dw.bytes);
    dx.copy_to_host(input_before.data(), input_before.size());
    dw.copy_to_host(weight_before.data(), weight_before.size());

    int failures = 0;
    for (const std::int32_t t :
         {1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
          17,  31,  32,  33,  47,  48,  49,  63,  64,  65,  95,  96,  97,  192, 193, 288,
          289, 320, 321, 384, 385, 448, 449, 480, 481, 512, 513, 559, 560, 561, 1024}) {
        const std::size_t q_count  = static_cast<std::size_t>(kQRows) * t;
        const std::size_t kv_count = static_cast<std::size_t>(kKvRows) * t;
        GuardedBf16Output q_got(q_count), k_got(kv_count), v_got(kv_count);
        Tensor tx(dx.p, DType::BF16, {kHidden, t});
        Tensor tq(q_got.data(), DType::BF16, {kQRows, t});
        Tensor tk(k_got.data(), DType::BF16, {kKvRows, t});
        Tensor tv(v_got.data(), DType::BF16, {kKvRows, t});
        const auto launch = [&](cudaStream_t stream) {
            ops::attn_input_proj(tx, weight, tq, tk, tv, ws, stream);
        };
        if (t == 2 || t == 16) {
            capture_and_replay(launch);
        } else {
            launch(nullptr);
            cudaDeviceSynchronize();
        }

        const auto plan =
            ops::detail::w8_attn_input_resolve_plan({kHidden, kQRows, kKvRows, kRows, kHidden, t});
        const bool mma = plan.schedule != ops::detail::W8AttnInputScheduleId::DecodeR8Direct &&
                         plan.schedule != ops::detail::W8AttnInputScheduleId::SimtR8C4;
        const Tolerance tolerance = mma ? Tolerance::linear_tc() : Tolerance::linear_bf16();
        const std::string suffix  = " T=" + std::to_string(t);
        const std::vector<double> q_output = from_device_bf16_ptr(q_got.data(), q_count);
        const std::vector<double> k_output = from_device_bf16_ptr(k_got.data(), kv_count);
        const std::vector<double> v_output = from_device_bf16_ptr(v_got.data(), kv_count);
        failures +=
            verify_sampled_projection("W8 companion attention q FP64 oracle" + suffix, q_output,
                                      kQRows, 0, x, kHidden, t, packed, tolerance, 0, kQRows);
        failures += verify_sampled_projection("W8 companion attention k FP64 oracle" + suffix,
                                              k_output, kKvRows, 0, x, kHidden, t, packed,
                                              tolerance, kQRows, kKvRows);
        failures += verify_sampled_projection("W8 companion attention v FP64 oracle" + suffix,
                                              v_output, kKvRows, 0, x, kHidden, t, packed,
                                              tolerance, kQRows + kKvRows, kKvRows);
        if (t == 2) {
            failures += verify_complete_projection(
                "W8 companion attention q complete FP64 exact-decode oracle", q_output, x, kHidden,
                t, packed, 0, kQRows, tolerance);
            failures += verify_complete_projection(
                "W8 companion attention k complete FP64 exact-decode oracle", k_output, x, kHidden,
                t, packed, kQRows, kKvRows, tolerance);
            failures += verify_complete_projection(
                "W8 companion attention v complete FP64 exact-decode oracle", v_output, x, kHidden,
                t, packed, kQRows + kKvRows, kKvRows, tolerance);
        }
        failures += q_got.verify_guards("W8 companion attention q" + suffix);
        failures += k_got.verify_guards("W8 companion attention k" + suffix);
        failures += v_got.verify_guards("W8 companion attention v" + suffix);
        failures += q_got.verify_initialized("W8 companion attention q" + suffix);
        failures += k_got.verify_initialized("W8 companion attention k" + suffix);
        failures += v_got.verify_initialized("W8 companion attention v" + suffix);
    }

    std::vector<std::uint8_t> input_after(dx.bytes);
    std::vector<std::uint8_t> weight_after(dw.bytes);
    dx.copy_to_host(input_after.data(), input_after.size());
    dw.copy_to_host(weight_after.data(), weight_after.size());
    if (input_after != input_before) {
        std::cerr << "W8 companion attention modified input\n";
        ++failures;
    }
    if (weight_after != weight_before) {
        std::cerr << "W8 companion attention modified packed weight\n";
        ++failures;
    }
    return failures;
}

int w8_gdn_input_correctness() {
    constexpr int kHidden  = 2048;
    constexpr int kQRows   = 2048;
    constexpr int kKRows   = 2048;
    constexpr int kVRows   = 4096;
    constexpr int kZRows   = 4096;
    constexpr int kQkvRows = kQRows + kKRows + kVRows;
    constexpr int kRows    = kQkvRows + kZRows;
    constexpr int kMaxT    = 1024;

    std::vector<float> x(static_cast<std::size_t>(kHidden) * kMaxT);
    fill_uniform(x, 1801u, -0.25f, 0.25f);
    round_to_bf16(x);
    auto packed = row_split::make_patterned_weight(QType::W8G32_F16S, kRows, kHidden, 1803u);
    DBuf dx     = to_device_bf16(x);
    DBuf dw(packed.payload.size());
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = packed.device_weight(dw.p);
    WorkspaceArena ws(1);

    int failures = 0;
    for (const std::int32_t t : {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,  14,  15,
                                 16, 17, 24, 25, 32, 33, 48, 49, 64, 65, 96, 97, 128, 129, 1024}) {
        const std::size_t qkv_count = static_cast<std::size_t>(kQkvRows) * t;
        const std::size_t z_count   = static_cast<std::size_t>(kZRows) * t;
        GuardedBf16Output qkv_got(qkv_count), z_got(z_count);
        Tensor tx(dx.p, DType::BF16, {kHidden, t});
        Tensor tqkv(qkv_got.data(), DType::BF16, {kQkvRows, t});
        Tensor tz(z_got.data(), DType::BF16, {kZRows, t});
        const auto launch = [&](cudaStream_t stream) {
            ops::gdn_input_proj(tx, weight, tqkv, tz, ws, stream);
        };
        if (t == 1 || t == 2 || t == 4 || t == 16) {
            capture_and_replay(launch);
        } else {
            launch(nullptr);
            cudaDeviceSynchronize();
        }

        const auto plan =
            ops::detail::w8_gdn_input_resolve_plan({kHidden, kQkvRows, kZRows, kRows, kHidden, t});
        const bool mma            = plan.schedule == ops::detail::W8GdnInputScheduleId::MmaR64C128;
        const Tolerance tolerance = mma ? Tolerance::linear_tc() : Tolerance::linear_bf16();
        const std::vector<double> qkv = from_device_bf16_ptr(qkv_got.data(), qkv_count);
        const std::string suffix      = " T=" + std::to_string(t);
        failures += verify_sampled_projection("W8 GDN q FP64 oracle" + suffix, qkv, kQkvRows, 0, x,
                                              kHidden, t, packed, tolerance, 0, kQRows);
        failures +=
            verify_sampled_projection("W8 GDN k FP64 oracle" + suffix, qkv, kQkvRows, kQRows, x,
                                      kHidden, t, packed, tolerance, kQRows, kKRows);
        failures += verify_sampled_projection("W8 GDN v FP64 oracle" + suffix, qkv, kQkvRows,
                                              kQRows + kKRows, x, kHidden, t, packed, tolerance,
                                              kQRows + kKRows, kVRows);
        failures += verify_sampled_projection(
            "W8 GDN z FP64 oracle" + suffix, from_device_bf16_ptr(z_got.data(), z_count), kZRows, 0,
            x, kHidden, t, packed, tolerance, kQkvRows, kZRows);
        failures += qkv_got.verify_guards("W8 GDN qkv" + suffix);
        failures += z_got.verify_guards("W8 GDN z" + suffix);
    }
    return failures;
}

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exp_value = std::exp(value);
    return value * exp_value / (1.0 + exp_value);
}

int gate_up_silu_route_correctness() {
    constexpr int kHidden = 5120;
    constexpr int kInter  = 17408;
    auto packed = row_split::make_patterned_weight(QType::Q4G64_F16S, 2 * kInter, kHidden, 31u);
    DBuf dw(packed.payload.size());
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    WorkspaceArena ws(256ULL << 20);
    const Weight weight = packed.device_weight(dw.p);

    int failures = 0;
    for (const std::int32_t t :
         {1, 2, 128, 129, 256, 257, 384, 385, 512, 513, 640, 641, 1024, 1025, 2048}) {
        std::vector<float> x(static_cast<std::size_t>(kHidden) * t);
        fill_uniform(x, 313u + static_cast<std::uint32_t>(t), -0.01f, 0.01f);
        round_to_bf16(x);
        DBuf dx = to_device_bf16(x);
        DBuf got(static_cast<std::size_t>(kInter) * t * 2);
        Tensor tx(dx.p, DType::BF16, {kHidden, t});
        Tensor tgot(got.p, DType::BF16, {kInter, t});
        ops::linear_swiglu(tx, weight, tgot, ws, nullptr);
        cudaDeviceSynchronize();
        const std::vector<double> full =
            from_device_bf16(got, static_cast<std::size_t>(kInter) * t);
        const auto rows = sampled_indices(kInter);
        const auto cols = sampled_indices(t);
        std::vector<double> actual;
        std::vector<double> reference;
        actual.reserve(rows.size() * cols.size());
        reference.reserve(actual.capacity());
        for (const std::int32_t col : cols) {
            const float* xcol = x.data() + static_cast<std::size_t>(col) * kHidden;
            for (const std::int32_t row : rows) {
                const double gate =
                    row_split::dot_row_split_lowbit_fp64(packed, row, xcol, kHidden);
                const double up =
                    row_split::dot_row_split_lowbit_fp64(packed, kInter + row, xcol, kHidden);
                actual.push_back(full[static_cast<std::size_t>(col) * kInter + row]);
                reference.push_back(silu_fp64(gate) * up);
            }
        }
        const std::string label = "linear_swiglu Q4 sampled FP64 oracle T=" + std::to_string(t);
        failures += verify(label.c_str(), actual, reference, Tolerance::linear_tc());
    }
    return failures;
}

int q5_residual_epilogue_correctness(std::int32_t k, std::uint32_t seed) {
    constexpr std::int32_t kN    = 5120;
    constexpr std::int32_t kMaxT = 129;
    auto packed                  = row_split::make_patterned_weight(QType::Q5G64_F16S, kN, k, seed);
    DBuf dw(packed.payload.size());
    cudaMemcpy(dw.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    std::vector<float> x(static_cast<std::size_t>(k) * kMaxT);
    std::vector<float> residual(static_cast<std::size_t>(kN) * kMaxT);
    fill_uniform(x, seed + 1000u, -0.01f, 0.01f);
    fill_uniform(residual, seed + 2000u, -1.0f, 1.0f);
    round_to_bf16(x);
    round_to_bf16(residual);
    DBuf dx = to_device_bf16(x);
    WorkspaceArena ws(256ULL << 20);
    const Weight weight = packed.device_weight(dw.p);

    int failures = 0;
    for (const std::int32_t t : {1, 2, 24, 25, 128, 129}) {
        std::vector<float> initial(residual.begin(),
                                   residual.begin() + static_cast<std::size_t>(kN) * t);
        DBuf dout = to_device_bf16(initial);
        Tensor tx(dx.p, DType::BF16, {k, t});
        Tensor tout(dout.p, DType::BF16, {kN, t});
        ops::linear_add(tx, weight, tout, ws, nullptr);
        cudaDeviceSynchronize();

        const std::vector<double> full = from_device_bf16(dout, static_cast<std::size_t>(kN) * t);
        const auto rows                = sampled_indices(kN);
        const auto cols                = sampled_indices(t);
        std::vector<double> actual;
        std::vector<double> reference;
        actual.reserve(rows.size() * cols.size());
        reference.reserve(actual.capacity());
        for (const std::int32_t col : cols) {
            const float* xcol = x.data() + static_cast<std::size_t>(col) * k;
            for (const std::int32_t row : rows) {
                const std::size_t index = static_cast<std::size_t>(col) * kN + row;
                actual.push_back(full[index]);
                reference.push_back(static_cast<double>(residual[index]) +
                                    row_split::dot_row_split_lowbit_fp64(packed, row, xcol, k));
            }
        }

        const auto plan = ops::detail::q5_linear_add_resolve_plan({kN, k, k, t});
        const bool uses_mma =
            plan.schedule == ops::detail::Q5LinearAddScheduleId::MmaResidualR64C64 ||
            plan.schedule == ops::detail::Q5LinearAddScheduleId::MmaResidualR64C128;
        const Tolerance tolerance = uses_mma ? Tolerance::linear_tc() : Tolerance::linear_bf16();
        const std::string label   = "linear_add Q5 [5120," + std::to_string(k) +
                                  "] sampled FP64 oracle T=" + std::to_string(t);
        failures += verify(label.c_str(), actual, reference, tolerance);
    }
    return failures;
}

int w8_residual_epilogue_correctness() {
    constexpr std::int32_t kN    = 2048;
    constexpr std::int32_t kK    = 4096;
    constexpr std::int32_t kMaxT = 1024;

    std::vector<float> x(static_cast<std::size_t>(kK) * kMaxT);
    std::vector<float> residual(static_cast<std::size_t>(kN) * kMaxT);
    fill_uniform(x, 3503u, -0.25f, 0.25f);
    fill_uniform(residual, 3505u, -2.0f, 2.0f);
    round_to_bf16(x);
    round_to_bf16(residual);
    row_split::PackedWeight packed =
        row_split::make_patterned_weight(QType::W8G32_F16S, kN, kK, 3501u);

    DBuf dx = to_device_bf16(x);
    DBuf dweight(packed.payload.size());
    cudaMemcpy(dweight.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    const Weight weight = packed.device_weight(dweight.p);
    WorkspaceArena ws(1);

    int failures = 0;
    for (const std::int32_t t :
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 52, 53, 640, 641, 1024}) {
        std::vector<float> initial(residual.begin(),
                                   residual.begin() + static_cast<std::size_t>(kN) * t);
        GuardedBf16Output got(static_cast<std::size_t>(kN) * t);
        std::vector<std::uint16_t> initial_bits(initial.size());
        std::transform(initial.begin(), initial.end(), initial_bits.begin(), f32_to_bf16);
        cudaMemcpy(got.data(), initial_bits.data(), initial_bits.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice);
        Tensor tx(dx.p, DType::BF16, {kK, t});
        Tensor tout(got.data(), DType::BF16, {kN, t});

        const auto launch = [&](cudaStream_t stream) {
            ops::linear_add(tx, weight, tout, ws, stream);
        };
        if (t == 16 || t == kMaxT) {
            capture_and_replay(launch);
        } else {
            launch(nullptr);
            cudaDeviceSynchronize();
        }

        const std::vector<double> full =
            from_device_bf16_ptr(got.data(), static_cast<std::size_t>(kN) * t);
        const auto sample_rows = sampled_indices(kN);
        const auto sample_cols = sampled_indices(t);

        std::vector<double> actual;
        std::vector<double> reference;
        actual.reserve(sample_rows.size() * sample_cols.size());
        reference.reserve(actual.capacity());
        for (const std::int32_t col : sample_cols) {
            const float* xcol = x.data() + static_cast<std::size_t>(col) * kK;
            for (const std::int32_t row : sample_rows) {
                const std::size_t index = static_cast<std::size_t>(col) * kN + row;
                actual.push_back(full[index]);
                reference.push_back(static_cast<double>(residual[index]) +
                                    row_split::dot_row_split_lowbit_fp64(packed, row, xcol, kK));
            }
        }

        const auto plan           = ops::detail::w8_linear_add_resolve_plan({kN, kK, kK, t});
        const Tolerance tolerance = ops::detail::w8_linear_add_schedule_uses_mma(plan.schedule)
                                        ? Tolerance::linear_tc()
                                        : Tolerance::linear_bf16();
        const std::string label =
            "linear_add W8 [2048,4096] sampled FP64 oracle T=" + std::to_string(t);
        failures += verify(label.c_str(), actual, reference, tolerance);
        failures += got.verify_guards(label);
    }
    return failures;
}

int prefill_fusion_correctness() {
    int f = 0;
    for (std::int32_t t = 1; t <= 16; ++t) {
        f += grouped_attention_correctness(t, t == 1 || t == 6);
    }
    for (const std::int32_t t : {17, 127, 128, 129}) {
        f += grouped_attention_correctness(t, false);
    }
    for (const std::int32_t t : {1, 2, 4, 5, 6, 7, 16, 17, 128, 129}) {
        f += grouped_gdn_correctness(t, t == 6);
    }
    f += gate_up_silu_route_correctness();
    f += q5_residual_epilogue_correctness(6144, 37u);
    f += q5_residual_epilogue_correctness(17408, 41u);
    f += w8_residual_epilogue_correctness();
    f += w8_attention_input_correctness();
    f += w8_companion_attention_input_correctness();
    f += w8_gdn_input_correctness();
    return f;
}

int bf16_placeholder_contract() {
    int f                    = 0;
    constexpr std::int32_t n = 48;
    constexpr std::int32_t k = 5120;
    DBuf dx(static_cast<std::size_t>(k) * 2u);
    DBuf dw_bf16(static_cast<std::size_t>(n) * k * 2u);
    DBuf dw_fp32(static_cast<std::size_t>(n) * k * 4u);
    DBuf dout(static_cast<std::size_t>(n) * 2u);
    Tensor tx(dx.p, DType::BF16, {k, 1});
    Tensor tout(dout.p, DType::BF16, {n, 1});
    WorkspaceArena ws(256);

    const auto weight = [&](void* data, QType qtype) {
        Weight w{};
        w.qtype         = qtype;
        w.layout        = QuantLayout::Contiguous;
        w.payload       = data;
        w.payload_bytes = static_cast<std::uint64_t>(n) * k * (qtype == QType::FP32_CTRL ? 4u : 2u);
        w.qdata         = data;
        w.ndim          = 2;
        w.shape[0]      = n;
        w.shape[1]      = k;
        w.padded_shape[0] = n;
        w.padded_shape[1] = k;
        w.n               = n;
        w.k               = k;
        return w;
    };

    if (ops::detail::bf16_contiguous_admits({n, k, 1})) {
        std::cerr << "linear BF16 placeholder unexpectedly admits a problem\n";
        ++f;
    }
    try {
        ops::linear(tx, weight(dw_bf16.p, QType::BF16_CTRL), tout, ws, nullptr);
        std::cerr << "linear BF16 placeholder unexpectedly executed\n";
        ++f;
    } catch (const std::invalid_argument&) {}
    try {
        ops::linear(tx, weight(dw_fp32.p, QType::FP32_CTRL), tout, ws, nullptr);
        std::cerr << "linear FP32 unexpectedly remains supported\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    Tensor empty_x   = tx;
    Tensor empty_out = tout;
    empty_x.ne[1]    = 0;
    empty_out.ne[1]  = 0;
    try {
        ops::linear(empty_x, weight(dw_bf16.p, QType::BF16_CTRL), empty_out, ws, nullptr);
        std::cerr << "linear BF16 placeholder accepted an empty problem\n";
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

    if (std::getenv("NINFER_LINEAR_TEST_W8_ATTN_ONLY") != nullptr) {
        int f = 0;
        f += w8_attention_input_correctness();
        f += w8_companion_attention_input_correctness();
        std::cout << (f ? "FAIL" : "OK") << " W8 attention-input dFlash correctness\n";
        return f ? 1 : 0;
    }
    if (std::getenv("NINFER_LINEAR_TEST_W8_LINEAR_ADD_ONLY") != nullptr) {
        const int f = w8_residual_epilogue_correctness();
        std::cout << (f ? "FAIL" : "OK") << " W8 LinearAdd dFlash correctness\n";
        return f ? 1 : 0;
    }
    if (std::getenv("NINFER_LINEAR_TEST_W8G32_ONLY") != nullptr) {
        int f = 0;
        f += one_quant_shape(QType::W8G32_F16S, 1024, 5120, {1, 4, 5, 16, 17}, 119u);
        f += one_quant_shape(QType::W8G32_F16S, 4608, 4608, {5, 6}, 121u);
        f +=
            one_quant_shape(QType::W8G32_F16S, 2048, 4608,
                            {14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31, 32, 33}, 123u);
        f += one_quant_shape_sampled(QType::W8G32_F16S, 2048, 4608, 16384, 125u);
        f += paired_w8g32_shape(1024, 5120, {4, 5, 57}, 127u);
        f += one_quant_shape(QType::W8G32_F16S, 2048, 4096, {56, 57}, 129u);
        f += one_quant_shape_sampled(QType::W8G32_F16S, 2048, 4096, 1024, 131u);
        f += one_quant_shape(QType::W8G32_F16S, 12288, 2048, {16, 17}, 133u);
        f += one_quant_shape_sampled(QType::W8G32_F16S, 12288, 2048, 1024, 135u);
        f += one_quant_shape(QType::W8G32_F16S, 9216, 2048, {13, 14, 128, 129}, 137u);
        f += one_quant_shape_sampled(QType::W8G32_F16S, 9216, 2048, 1024, 139u);
        f += w8_attention_input_correctness();
        f += w8_companion_attention_input_correctness();
        f += w8_gdn_input_correctness();
        std::cout << (f ? "FAIL" : "OK") << " linear W8G32 focused correctness\n";
        return f ? 1 : 0;
    }
    if (std::getenv("NINFER_LINEAR_TEST_PREFILL_FUSIONS_ONLY") != nullptr) {
        const int f = prefill_fusion_correctness();
        std::cout << (f ? "FAIL" : "OK") << " linear prefill fusion correctness\n";
        return f ? 1 : 0;
    }
    if (std::getenv("NINFER_LINEAR_TEST_Q6_HEAD_ONLY") != nullptr) {
        const int f = q6_output_head_dflash_oracle();
        std::cout << (f ? "FAIL" : "OK") << " Q6 output-head dFlash correctness\n";
        return f ? 1 : 0;
    }

    int f = 0;
    f += prefill_fusion_correctness();
    f += bf16_placeholder_contract();
    f += lowbit_metadata_validation(QType::Q4G64_F16S);
    f += lowbit_metadata_validation(QType::Q5G64_F16S);
    f += lowbit_metadata_validation(QType::Q6G64_F16S);
    f += lowbit_metadata_validation(QType::W8G32_F16S);

    // W8 public correctness is exact-admission only. The dedicated W8 dispatch test covers every
    // registered view and route family word-for-word; these independent fp64-oracle points cover
    // both SIMT schedules, both MMA row tiles, the Vision crossover, and the pair topology seam.
    f += one_quant_shape(QType::W8G32_F16S, 1024, 5120, {1, 4, 5, 16, 17}, 113u);
    f += one_quant_shape(QType::W8G32_F16S, 14336, 5120, {8, 9}, 117u);
    f += one_quant_shape(QType::W8G32_F16S, 4608, 4608, {5, 6}, 121u);
    f += one_quant_shape(QType::W8G32_F16S, 2048, 4608,
                         {14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31, 32, 33}, 123u);
    f += one_quant_shape_sampled(QType::W8G32_F16S, 2048, 4608, 16384, 125u);
    f += paired_w8g32_shape(1024, 5120, {4, 5, 57}, 127u);
    f += one_quant_shape(QType::W8G32_F16S, 2048, 4096, {56, 57}, 129u);
    f += one_quant_shape_sampled(QType::W8G32_F16S, 2048, 4096, 1024, 131u);
    f += one_quant_shape(QType::W8G32_F16S, 12288, 2048, {16, 17}, 133u);
    f += one_quant_shape_sampled(QType::W8G32_F16S, 12288, 2048, 1024, 135u);
    f += one_quant_shape(QType::W8G32_F16S, 9216, 2048, {13, 14, 128, 129}, 137u);
    f += one_quant_shape_sampled(QType::W8G32_F16S, 9216, 2048, 1024, 139u);
    f += one_patterned_quant_shape_sampled(QType::W8G32_F16S, 2048, 4096, 1025, 141u);
    // Q4 public correctness is exact-admission only. The dedicated Q4 plan/dispatch tests cover
    // every route seam and compare public auto against the fixed entry word-for-word; these
    // oracle points cover both registered head K values and the split-K/SIMT numerical boundary
    // at real product geometries.
    f += one_quant_shape(QType::Q4G64_F16S, 1024, 5120, {1, 2, 15, 16}, 23u);
    f += one_quant_shape(QType::Q4G64_F16S, 4096, 5120, {1}, 25u);
    f += one_quant_shape(QType::Q4G64_F16S, 131072, 2048, {1}, 26u);
    f += one_quant_shape(QType::Q4G64_F16S, 3456, 1152, {4, 40}, 27u);
    f += one_patterned_quant_shape_sampled(QType::Q4G64_F16S, 34816, 5120, 1, 28u);
    f += one_patterned_quant_shape_sampled(QType::Q4G64_F16S, 34816, 5120, 17, 28u);
    // Q5 public correctness is exact-admission only. Fused input projections own their
    // large-column parent operations; linear_add owns the residual epilogue for every T.
    f += one_quant_shape(QType::Q5G64_F16S, 1024, 5120, {1, 4, 5, 16}, 29u);
    f += one_quant_shape(QType::Q5G64_F16S, 6144, 5120, {1, 2, 6, 7, 24, 25, 64, 65, 512}, 31u);
    f += one_quant_shape(QType::Q5G64_F16S, 5120, 6144, {2, 6, 7, 24}, 37u);
    f += one_quant_shape(QType::Q5G64_F16S, 5120, 17408, {2, 6, 7, 24}, 41u);
    f += one_quant_shape(QType::Q5G64_F16S, 1152, 1152, {4, 56, 60, 128}, 43u);
    f += one_quant_shape(QType::Q5G64_F16S, 1152, 4304, {4, 84, 88, 128}, 47u);
    f += one_patterned_quant_shape_sampled(QType::Q5G64_F16S, 6144, 5120, 1025, 49u);
    // Q6 public correctness is exact-admission only. The dedicated Q6 plan/dispatch tests cover
    // every route seam and compare public auto against the fixed entry word-for-word. These
    // independent fp64-oracle points cover the real head and Vision geometries.
    f += one_quant_shape(QType::Q6G64_F16S, 248320, 5120, {1}, 211u);
    f += one_patterned_quant_shape_sampled(QType::Q6G64_F16S, 248320, 5120, 7, 212u);
    f += q6_output_head_dflash_oracle();
    f += one_quant_shape(QType::Q6G64_F16S, 1152, 1536, {4, 40, 128}, 227u);

    std::cout << (f ? "FAIL" : "OK") << " linear correctness\n";
    return f ? 1 : 0;
}
