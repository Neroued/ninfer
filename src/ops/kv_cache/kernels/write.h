#pragma once

#include <cuda_runtime_api.h>

#include "core/paged_kv_cache.h"

namespace ninfer::ops::detail::kv {

enum class PagedWriteInstance {
    Copy96,
    Copy128,
    WarpRows256,
    Page8x256,
    BatchRows256,
};

// D256 packed-record writes from validated BF16 operands. The owning Op selects the instance;
// these bindings perform no tuning selection. Copy instances require batch size one;
// BatchRows256 accepts BF16 and quantized storage with device-selected valid prefixes.
void launch_paged_write(PagedWriteInstance instance, const Tensor& k, const Tensor& v,
                        const Tensor& positions, PagedKVLayerView cache, cudaStream_t stream);
void launch_paged_write(PagedWriteInstance instance, const Tensor& k, const Tensor& v,
                        const Tensor& positions, const Tensor& valid_columns,
                        const Tensor& table_rows, PagedKVBatchLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops::detail::kv
