// qus::kernels::detail - tuned low-bit linear GEMV launcher (tile-centric).
#include "kernels/linear/gemv/linear_lowbit_gemv.h"

#include "kernels/linear/gemv/linear_lowbit_gemv.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

// Sweep overrides for tuning (read once): QUS_KSPLIT in {16,32}, QUS_ROWS in {16,32}.
int env_int(const char* name) {
    const char* e = std::getenv(name);
    return e ? std::atoi(e) : 0;
}
int ksplit_override() {
    static const int v = env_int("QUS_KSPLIT");
    return v;
}
int rows_override() {
    static const int v = env_int("QUS_ROWS");
    return v;
}

const std::uint8_t* payload_ptr(const Weight& w) {
    const void* payload = w.payload != nullptr ? w.payload : w.qdata;
    return static_cast<const std::uint8_t*>(payload);
}

// K-split factor = warps per CTA. More warps/CTA raises occupancy (warps/SM);
// few-CTA shapes need the most. Only very large N over-fills the grid already.
int select_ksplit(std::int32_t n) {
    if (const int o = ksplit_override(); o != 0) { return o; }
    return (n > 64000) ? 16 : 32;
}

// Rows per CTA. 32 (a full warp of lanes==rows) measured best across all shapes:
// fewer rows wastes decode lanes and raised compute SOL without lifting the
// (occupancy-capped) DRAM throughput. Kept tunable via QUS_ROWS for experiments.
int select_rows(std::int32_t /*n*/) {
    if (const int o = rows_override(); o != 0) { return o; }
    return 32;
}

template <class Codec, int ROWS, int KSPLIT>
void launch_cfg(const __nv_bfloat16* x, const std::uint8_t* payload, __nv_bfloat16* out,
                std::int32_t n, std::int32_t k, std::int32_t padded_k, cudaStream_t stream) {
    const std::int64_t grid64 = (static_cast<std::int64_t>(n) + ROWS - 1) / ROWS;
    if (grid64 > std::numeric_limits<int>::max()) {
        throw std::overflow_error("linear: tuned low-bit GEMV launch grid exceeds CUDA limit");
    }
    const int grid         = static_cast<int>(std::max<std::int64_t>(1, grid64));
    const int block        = 32 * KSPLIT;
    const std::size_t smem = tile_lowbit_gemv_smem_bytes<Codec, ROWS, KSPLIT>();
    auto kernel            = linear_tile_lowbit_gemv_kernel<Codec, ROWS, KSPLIT>;
    if (smem > 48u * 1024u) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem)));
    }
    kernel<<<grid, block, smem, stream>>>(x, payload, out, n, k, padded_k);
}

// Q4 direct kernel: 2 warps per row, 2 rows per 128-thread CTA.
void launch_q4(const Tensor& x, const std::uint8_t* payload, Tensor& out, std::int32_t n,
               std::int32_t k, std::int32_t padded_k, cudaStream_t stream) {
    constexpr int kBlock       = 128;
    constexpr int kRowsPerBlock = (kBlock / 32) / 2;  // 2 warps/row -> 2 rows/CTA
    const std::int64_t grid64   = (static_cast<std::int64_t>(n) + kRowsPerBlock - 1) / kRowsPerBlock;
    if (grid64 > std::numeric_limits<int>::max()) {
        throw std::overflow_error("linear: tuned Q4 GEMV launch grid exceeds CUDA limit");
    }
    const int grid = static_cast<int>(std::max<std::int64_t>(1, grid64));
    linear_tuned_q4_gemv_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), payload, static_cast<__nv_bfloat16*>(out.data), n,
        k, padded_k);
}

template <class Codec>
void launch_tuned(const Tensor& x, const std::uint8_t* payload, Tensor& out, std::int32_t n,
                  std::int32_t k, std::int32_t padded_k, cudaStream_t stream) {
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    auto* op           = static_cast<__nv_bfloat16*>(out.data);
    const int rows     = select_rows(n);
    const int ksplit   = select_ksplit(n);
    if (rows <= 16) {
        if (ksplit >= 32) { launch_cfg<Codec, 16, 32>(xp, payload, op, n, k, padded_k, stream); }
        else              { launch_cfg<Codec, 16, 16>(xp, payload, op, n, k, padded_k, stream); }
    } else {
        if (ksplit >= 32) { launch_cfg<Codec, 32, 32>(xp, payload, op, n, k, padded_k, stream); }
        else              { launch_cfg<Codec, 32, 16>(xp, payload, op, n, k, padded_k, stream); }
    }
}

} // namespace

void linear_tuned_lowbit_gemv_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     LinearFormat fmt, cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = w.padded_shape[1];
    const std::uint8_t* payload = payload_ptr(w);

    switch (fmt) {
    case LinearFormat::Q4G64_N64K64:
        launch_q4(x, payload, out, n, k, padded_k, stream);
        break;
    case LinearFormat::Q5G64_N64K64:
        launch_tuned<Q5Codec>(x, payload, out, n, k, padded_k, stream);
        break;
    case LinearFormat::Q6G64_N64K64:
        launch_tuned<Q6Codec>(x, payload, out, n, k, padded_k, stream);
        break;
    default:
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
