#pragma once
#include "fixture.cuh"
#include "common/roofline.h"
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "ops/linear/projection/mxfp8/dispatch.h"
#include <numeric>
#include <sstream>

namespace ninfer::bench::kda {
struct Traffic {
    std::uint64_t read = 0, write = 0, state_read = 0, state_write = 0, record_read = 0,
                  record_write   = 0;
    std::uint64_t workspace_read = 0, workspace_write = 0;
    double projection_flops = 0;
    double matrix_tensor_us = 0;
    bool mxfp8_reference    = false;

    std::string csv(double us) const {
        const auto bytes = read + write;
        std::ostringstream s;
        s << read << ',' << write << ',' << state_read << ',' << state_write << ',' << record_read
          << ',' << record_write << ',' << bytes / (us * 1000) << ','
          << kRoofline.sustained_read_gbs << ',' << bytes / (kRoofline.sustained_read_gbs * 1000)
          << ',' << projection_flops << ',';
        if (projection_flops) s << projection_flops / (kRoofline.bf16_f32_tflops * 1e6);
        s << ',';
        if (mxfp8_reference && projection_flops)
            s << projection_flops / (kRoofline.mxfp8_f32_tflops * 1e6);
        s << ',' << workspace_read << ',' << workspace_write << ',' << projection_flops / us / 1e6
          << ',' << bytes / (us * 1000) / kRoofline.sustained_read_gbs * 100 << ','
          << bytes / (us * 1000) / kRoofline.memory_gbs * 100 << ',' << matrix_tensor_us;
        return s.str();
    }

    static constexpr const char* columns =
        "logical_read_bytes,logical_write_bytes,state_read_bytes,state_write_bytes,record_read_"
        "bytes,record_write_bytes,logical_GBps,"
        "memory_roofline_GBps,streaming_memory_reference_us,projection_flops,bf16_projection_"
        "reference_us,mxfp8_projection_reference_us,workspace_read_bytes,workspace_write_bytes,"
        "useful_matrix_TF_s,logical_read_reference_pct,logical_nominal_pct,matrix_tensor_"
        "reference_us";
};

inline Traffic traffic(Fixture& f, const std::string& op) {
    const auto n = std::accumulate(f.valid_counts.begin(), f.valid_counts.end(), std::uint64_t(0));
    const std::uint64_t state         = 4ULL * 128 * 128 * f.h * f.b;
    const std::uint64_t record_column = 2ULL * 128 * f.qh + 6ULL * 128 * f.h + 4ULL * f.h;
    Traffic out;
    const std::uint64_t m = std::uint64_t(f.t) * f.b;
    if (op.starts_with("layer-")) {
        const auto input    = traffic(f, f.recorded_layer() ? "input-record" : "input-update");
        const auto rec      = traffic(f, f.recorded_layer()     ? "record"
                                         : op == "layer-update" ? "update"
                                                                : "core");
        const auto fold     = op == "layer-chain" ? traffic(f, "fold") : Traffic{};
        out.read            = input.read + rec.read + fold.read;
        out.write           = input.write + rec.write + fold.write;
        out.state_read      = input.state_read + rec.state_read + fold.state_read;
        out.state_write     = input.state_write + rec.state_write + fold.state_write;
        out.record_read     = fold.record_read;
        out.record_write    = input.record_write + rec.record_write;
        out.workspace_read  = input.workspace_read + rec.workspace_read;
        out.workspace_write = input.workspace_write + rec.workspace_write;
        // Control split, two low-rank contractions, norm/gate and O projection.
        out.read += f.control_weight.model_weight_bytes() + f.forget_weight.model_weight_bytes() +
                    f.gate_weight.model_weight_bytes() + f.output_weight.model_weight_bytes() +
                    2ULL * (4096 + 256 + 4096 + 4096 + 4096) * m + f.gamma.size();
        out.write += 2ULL * (288 + 8192 + 4096 + 4096) * m;
        const double control_flops = 2.0 * (288 * 4096 + 2 * 4096 * 128) * m;
        const double output_flops  = 2.0 * 4096 * 4096 * m;
        out.projection_flops       = input.projection_flops + control_flops + output_flops;
        const auto& route =
            ops::detail::select_mxfp8_linear_route(4096, 4096, int(m), ops::LinearPolicy::AllowA8);
        const bool a8      = route.compute == ops::detail::Mxfp8LinearCompute::A8;
        const auto staging = a8 ? 4096ULL * route.code_tokens(int(m)) + 128ULL * m : 0;
        out.workspace_read += staging;
        out.workspace_write += staging;
        out.read += staging;
        out.write += staging;
        out.matrix_tensor_us =
            input.matrix_tensor_us + control_flops / (kRoofline.bf16_f32_tflops * 1e6) +
            output_flops / ((a8 ? kRoofline.mxfp8_f32_tflops : kRoofline.bf16_f32_tflops) * 1e6);
        return out;
    }
    if (op == "fold") {
        for (int m : f.commit_counts)
            if (m) {
                const auto bytes = 4ULL * 128 * 128 * f.h * f.layers;
                out.state_read += bytes + 2ULL * f.channels * f.layers * std::max(0, 3 - m);
                out.state_write += bytes + 6ULL * f.channels * f.layers;
                out.record_read +=
                    record_column * f.layers * m + 2ULL * f.channels * f.layers * std::min(m, 3);
                out.read += bytes + record_column * f.layers * m + 6ULL * f.channels * f.layers +
                            2 * sizeof(std::int64_t);
                out.write += bytes + 6ULL * f.channels * f.layers;
            }
        out.read += f.b * sizeof(std::int32_t);
        return out;
    }
    if (op == "input-update" || op == "input-record") {
        out.state_read   = 6ULL * f.channels * f.b;
        out.state_write  = op == "input-update" ? out.state_read : 0;
        out.record_write = op == "input-record" ? 2ULL * f.channels * n : 0;
        out.read =
            f.weight.model_weight_bytes() + 2ULL * 4096 * m + f.conv_weight.size() + out.state_read;
        out.write            = 2ULL * f.channels * f.t * f.b + out.state_write + out.record_write;
        out.projection_flops = 2.0 * 12288 * 4096 * n;
        out.mxfp8_reference =
            f.weight.weight.qtype == QType::MXFP8 && f.input_policy() == ops::LinearPolicy::AllowA8;
        if (f.weight.weight.qtype == QType::MXFP8) {
            const auto plan = ops::detail::kda_input::mxfp8_execution_workspace(
                f.t, f.b, op == "input-update", f.input_policy());
            out.workspace_read = out.workspace_write =
                plan.codes.bytes + plan.scales.bytes + plan.convolution.bytes;
            out.matrix_tensor_us =
                2.0 * 12288 * 4096 * m /
                ((plan.codes.bytes ? kRoofline.mxfp8_f32_tflops : kRoofline.bf16_f32_tflops) * 1e6);
            out.read += out.workspace_read;
            out.write += out.workspace_write;
        }
    } else if (op == "chain") {
        auto input = traffic(f, "input-record"), rec = traffic(f, "record"),
             fold = traffic(f, "fold");
        // Sum per-Op I/O; intermediate records/outputs intentionally count producer and consumer.
        out.read             = input.read + rec.read + fold.read;
        out.write            = input.write + rec.write + fold.write;
        out.state_read       = input.state_read + rec.state_read + fold.state_read;
        out.state_write      = fold.state_write;
        out.record_read      = fold.record_read;
        out.record_write     = input.record_write + rec.record_write;
        out.projection_flops = input.projection_flops;
        out.mxfp8_reference  = input.mxfp8_reference;
        out.workspace_read   = input.workspace_read + rec.workspace_read;
        out.workspace_write  = input.workspace_write + rec.workspace_write;
        out.matrix_tensor_us = input.matrix_tensor_us;
    } else {
        out.state_read    = state;
        out.state_write   = op == "record" ? 0 : state;
        const auto copied = f.aliases_kv_records() ? 0 : 2ULL * 128 * (f.qh + f.h);
        out.record_write  = op == "record" ? (copied + 4ULL * 129 * f.h) * n : 0;
        out.read = (4ULL * 128 * f.qh + 4ULL * 128 * f.h + 2ULL * f.h) * n + 4ULL * 129 * f.h +
                   out.state_read;
        out.write      = 2ULL * 128 * f.h * f.t * f.b + out.state_write + out.record_write;
        namespace core = ops::detail::kimi_delta_attention;
        if (op == "core" && core::select_recurrence_route(f.t) == core::RecurrenceRoute::Chunked) {
            const int window      = core::recurrence_window_tokens(f.h, f.t);
            const int windows     = (f.t + window - 1) / window;
            std::uint64_t packets = 0;
            for (int begin = 0; begin < f.t; begin += window)
                packets += core::chunked_workspace_bytes(f.h, std::min(window, f.t - begin));
            out.workspace_read = out.workspace_write = packets + std::uint64_t(windows - 1) * state;
            out.read += out.workspace_read;
            out.write += out.workspace_write;
        }
    }
    return out;
}
} // namespace ninfer::bench::kda
