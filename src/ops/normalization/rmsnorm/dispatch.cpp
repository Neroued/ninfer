#include "ops/normalization/rmsnorm/dispatch.h"
#include "ops/normalization/rmsnorm/launch.h"
#include "core/platform.h"

#include <algorithm>
#include <limits>

namespace ninfer::ops::detail::rmsnorm {
namespace {

Instance select_instance(Epilogue kind, Operands operands, std::int32_t sms) {
    const auto [d, rows, aligned] = operands;
    if (aligned < 4) return Instance::Generic;
#if NINFER_TARGET_RTX5090
    if (!gated(kind) && d == 20480 && aligned >= 8) {
        // One CTA per row: widen each CTA once a row wave covers the device. The small
        // route uses wider transfers; naturally aligned operands keep their fallback.
        return rows <= sms && aligned >= 16 ? Instance::D20480Block512 : Instance::D20480Block1024;
    }
#endif
    if (kind == Epilogue::Offset && d == 10240) return Instance::D10240;
    if (gated(kind) && d == 128) {
        // Sixteen rows per CTA cover one wave; smaller CTAs overlap the next wave.
        if (rows <= 384 || (rows > 16LL * sms && rows <= 6144)) return Instance::D128Block128;
        if (rows <= 16LL * sms) return Instance::D128Block512;
        return Instance::D128Block256;
    }
    if (!gated(kind) && d == 5120) return Instance::D5120;
    if (kind == Epilogue::Offset && d == 256) return Instance::D256;
    if (kind == Epilogue::Plain && d == 128) return Instance::D128Block128;
    if (d >= 64 && d <= 256 && d % 64 == 0) {
        const auto blocks = (rows + 15) / 16;
        return gated(kind) && blocks > sms ? Instance::Warp : Instance::WarpPrefetch;
    }
    if (kind == Epilogue::Plain && d == 2048) return Instance::D2048;
    if (d >= 512 && d <= 3072 && d % 512 == 0) return Instance::CtaMedium;
    if (d > 3072 && d <= 8192 && d % 1024 == 0)
        return gated(kind) && rows > sms ? Instance::CtaWide : Instance::CtaWidePrefetch;
    return Instance::Generic;
}

} // namespace

void dispatch(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps, Epilogue kind,
              Tensor& out, Operands operands, CudaExecutionView execution) {
    // Each launch respects grid.x capacity; the semantic row extent remains unrestricted.
    constexpr std::int64_t kMaxRows = std::numeric_limits<int>::max();
    for (std::int64_t first = 0; first < operands.rows;) {
        const auto count  = std::min(operands.rows - first, kMaxRows);
        const auto offset = std::size_t(first) * operands.dimension * sizeof(std::uint16_t);
        Tensor x_chunk = x, out_chunk = out, gate_chunk;
        x_chunk.data   = static_cast<std::byte*>(x.data) + offset;
        out_chunk.data = static_cast<std::byte*>(out.data) + offset;
        if (gate) {
            gate_chunk      = *gate;
            gate_chunk.data = static_cast<std::byte*>(gate->data) + offset;
        }
        const Operands chunk{operands.dimension, count, operands.alignment_bytes};
        launch(x_chunk, weight, gate ? &gate_chunk : nullptr, eps, kind, out_chunk, chunk,
               select_instance(kind, chunk, execution.multiprocessor_count()), execution.stream());
        first += count;
    }
}

} // namespace ninfer::ops::detail::rmsnorm
