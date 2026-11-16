#pragma once
#include "ninfer/ops/dynamic_grouped_conv.h"

namespace ninfer::ops::detail::dynamic_grouped_conv::finish {
void launch(const Tensor& x, const Weight& projection, const Tensor& base, const Tensor& delta,
            Tensor& partial, cudaStream_t stream);
}
