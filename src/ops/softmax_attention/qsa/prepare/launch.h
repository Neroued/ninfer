#pragma once

#include "ninfer/ops/qsa.h"

namespace ninfer::ops::detail::qsa {

void launch_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const QsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                    CudaExecutionView execution);

void launch_prepare_kv(const QsaKvProjectionView& projected,
                       const QsaKvPrepareParameters& parameters, const QsaBatchView& batch,
                       const Tensor& rope_positions, const QsaTailStateView& before,
                       QsaCacheView cache, QsaRawRecordWriteView records,
                       CudaExecutionView execution);

} // namespace ninfer::ops::detail::qsa
