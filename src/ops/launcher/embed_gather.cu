// ninfer::ops - embedding launcher: variant grid/block/stream setup.
#include "ops/launcher/embed_gather.h"

#include "ops/common/math.h"
#include "ops/kernel/embed_gather.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock          = 128;
constexpr int kQ6GroupedBlock = kEmbedGatherQ6Group * kEmbedGatherQ6GroupsPerBlock;
constexpr int kW8GroupedBlock = 32;
constexpr int kW8RowBlock     = 256;

int grid_for(std::int64_t n) {
    return static_cast<int>(
        std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
}

int grid_for_q6_grouped(std::int32_t d, std::int32_t T) {
    const std::int32_t kg           = d / kEmbedGatherQ6Group;
    const std::int32_t group_blocks = div_up(kg, kEmbedGatherQ6GroupsPerBlock);
    return static_cast<int>(std::max<std::int64_t>(1, static_cast<std::int64_t>(T) *
                                                          static_cast<std::int64_t>(group_blocks)));
}

} // namespace

const char* w8_embed_route_name(W8EmbedRoute route) {
    switch (route) {
    case W8EmbedRoute::Auto:
        return "auto";
    case W8EmbedRoute::Grouped:
        return "grouped-b32";
    case W8EmbedRoute::Row:
        return "row-b256";
    }
    return "unknown";
}

void embed_gather_w8_2048_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 W8EmbedRoute route, cudaStream_t stream) {
    const std::int32_t T = ids.ne[0];
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (route == W8EmbedRoute::Auto) { route = T <= 6 ? W8EmbedRoute::Grouped : W8EmbedRoute::Row; }
    if (route == W8EmbedRoute::Grouped) {
        const int grid = T * kEmbedGatherW8Groups;
        embed_gather_w8_grouped_2048_kernel<<<grid, kW8GroupedBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
    } else {
        embed_gather_w8_row_2048_kernel<<<T, kW8RowBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_dense_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const __nv_bfloat16*>(table.data),
        static_cast<__nv_bfloat16*>(out.data), d, T);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* high     = static_cast<const std::uint8_t*>(table.qhigh);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (d == table.padded_shape[1] && d % kEmbedGatherQ6Group == 0) {
        embed_gather_q6_grouped_kernel<<<grid_for_q6_grouped(d, T), kQ6GroupedBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, high, scales,
            static_cast<__nv_bfloat16*>(out.data), d, T);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    embed_gather_q6_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), codes, high, scales,
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_w8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (d == kEmbedGatherW8D && table.padded_shape[1] == kEmbedGatherW8D) {
        embed_gather_w8_2048_launch(ids, table, out, W8EmbedRoute::Auto, stream);
        return;
    }

    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_w8_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), codes, scales,
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
