// qus::kernels - embed_gather launcher: variant grid/block/stream setup.
#include "kernels/launcher/embed_gather.h"

#include "kernels/kernel/embed_gather.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace qus::kernels::detail {
namespace {

constexpr int kBlock = 128;

int grid_for(std::int64_t n) {
    return static_cast<int>(std::max<std::int64_t>(1, (n + kBlock - 1) / kBlock));
}

} // namespace

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_dense_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data),
        static_cast<const __nv_bfloat16*>(table.data),
        static_cast<__nv_bfloat16*>(out.data), d, T);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    const void* payload = table.payload != nullptr ? table.payload : table.qdata;
    embed_gather_q6_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(payload),
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
