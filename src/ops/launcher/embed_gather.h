#pragma once

// ninfer::ops::detail - private launch prototypes for embedding variants.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class W8EmbedRoute {
    Auto,
    Grouped,
    Row,
};

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream);
void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_w8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream);
void embed_gather_w8_2048_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 W8EmbedRoute route, cudaStream_t stream);
const char* w8_embed_route_name(W8EmbedRoute route);

} // namespace ninfer::ops::detail
