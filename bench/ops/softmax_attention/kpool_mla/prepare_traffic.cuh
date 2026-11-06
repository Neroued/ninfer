#pragma once

#include "ops/softmax_attention/kpool_mla/prepare/plan.h"

namespace ninfer::bench {
struct KpoolPrepareTraffic {
    std::uint64_t workspace_read = 0, workspace_write = 0;
    std::uint64_t repeated_weights = 0, quantization_input = 0;
    double matrix_tensor_us = 0, executed_matrix_flops = 0;
    bool simt = false;
};

inline KpoolPrepareTraffic kpool_prepare_traffic(const tool::kpool::Fixture& f,
                                                 CudaExecutionView cuda) {
    namespace impl = ops::detail::kpool;
    const auto geometry =
        impl::prepare_geometry(f.w.profile(), f.cache.latent.storage, f.cache.index.format, f.B,
                               f.T, f.T, cuda.multiprocessor_count());
    const auto layout = impl::prepare_workspace(geometry, f.update, f.T, f.B);
    const int total   = f.T * f.B;
    KpoolPrepareTraffic out;
    const bool empty =
        std::none_of(f.valid.host.begin(), f.valid.host.end(), [](int n) { return n > 0; });
    const auto simt = [](impl::ProjectionRoute r) {
        return r == impl::ProjectionRoute::Gemv || r == impl::ProjectionRoute::Simt2 ||
               r == impl::ProjectionRoute::Simt4 || r == impl::ProjectionRoute::Fp8Simt1 ||
               r == impl::ProjectionRoute::Fp8Simt4;
    };
    for (int begin = 0; begin < total; begin += layout.columns) {
        const int columns = std::min(layout.columns, total - begin);
        const auto plan   = impl::prepare_projection_plan(geometry, columns);
        const bool paired_skipped =
            empty && plan.paired != impl::PairedProjectionRoute::None && plan.skip_empty;
        std::uint64_t live = 0;
        for (int m = begin; m < begin + columns; ++m) live += m % f.T < f.valid.at(m / f.T);
        if (plan.splits == 1) {
            const auto projected = live * geometry.norm_dim() * sizeof(float);
            out.workspace_write += projected;
            out.workspace_read += projected;
            if (f.update) {
                out.workspace_write += live * 128 * 2;
                out.workspace_read += live * 128 * 2;
            }
        } else {
            out.workspace_write += std::uint64_t(columns) * geometry.rows * plan.splits * 4;
            out.workspace_read += live * geometry.rows * plan.splits * 4;
        }
        if (plan.quantize) {
            if (!empty) out.workspace_read += std::uint64_t(columns) * (4096 + 128);
            out.workspace_write += std::uint64_t(columns) * (4096 + 128);
            out.quantization_input += std::uint64_t(columns) * 4096 * 2;
        }
        if (!paired_skipped) out.quantization_input += (columns - live) * 4096 * 2;
        if ((begin || empty) && !paired_skipped) {
            auto bytes = f.w.bytes();
            if (empty && plan.paired == impl::PairedProjectionRoute::None) {
                bytes = geometry.weights.first.qtype == QType::BF16
                            ? std::uint64_t(geometry.weights.first.n) * 4096 * 2
                            : 0;
                if (geometry.weights.second && geometry.weights.second->qtype == QType::BF16)
                    bytes += std::uint64_t(geometry.weights.second->n) * 4096 * 2;
            }
            out.repeated_weights += bytes;
        }
        const auto matrix = [&](const ops::LinearProfile& w, impl::ProjectionRoute r) {
            if (empty &&
                (plan.paired != impl::PairedProjectionRoute::None ? plan.skip_empty
                                                                  : w.qtype != QType::BF16))
                return;
            out.executed_matrix_flops += 2.0 * w.n * w.k * columns;
            const bool a8 =
                plan.paired == impl::PairedProjectionRoute::None && impl::projection_uses_a8(r);
            out.matrix_tensor_us +=
                2.0 * w.n * w.k * columns /
                ((a8 ? kRoofline.mxfp8_f32_tflops : kRoofline.bf16_f32_tflops) * 1e6);
            out.simt |= plan.paired == impl::PairedProjectionRoute::Simt1 ||
                        (plan.paired == impl::PairedProjectionRoute::None && simt(r));
        };
        matrix(geometry.weights.first, plan.first);
        if (geometry.weights.second) matrix(*geometry.weights.second, plan.second);
        const int end = begin + columns;
        if (f.update && end < total && end % f.T && end % f.T < f.valid.at(end / f.T)) {
            out.workspace_write += 1536;
            out.workspace_read += (f.bases[end / f.T] + end % f.T) % 4 * 512;
        }
    }
    return out;
}
} // namespace ninfer::bench
