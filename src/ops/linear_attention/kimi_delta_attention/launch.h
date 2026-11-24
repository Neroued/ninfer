#pragma once

#include "core/device.h"
#include "ops/linear_attention/kimi_delta_attention/operands.h"

namespace ninfer::ops::detail::kimi_delta_attention {
struct Chunk;
void launch_recurrent(const Arguments& args, cudaStream_t stream);
void launch_batch_update(const Arguments& args, const std::int64_t* sources,
                         const std::int64_t* destinations, int batch, cudaStream_t stream);
void launch_record(const Arguments&, const std::int64_t* sources, const std::int32_t* valid,
                   RecurrenceRecords, int batch, cudaStream_t stream);
void launch_prepare(const Arguments& args, Chunk* workspace, cudaStream_t stream);
void launch_chunk_recurrence(const Arguments& args, const Chunk* workspace, int value_tile,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail::kimi_delta_attention
