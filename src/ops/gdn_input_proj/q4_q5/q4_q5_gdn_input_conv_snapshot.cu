#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden      = 5120;
constexpr int kQueryRows   = 2048;
constexpr int kKeyRows     = 2048;
constexpr int kValueRows   = 6144;
constexpr int kZRows       = 6144;
constexpr int kValueZRows  = kValueRows + kZRows;
constexpr int kQkRows      = kQueryRows + kKeyRows;
constexpr int kChannels    = kQkRows + kValueRows;
constexpr int kValueOffset = kQkRows;

using Q4ScheduleC4 = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4ScheduleC8 = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

enum class PdlOrder {
    Q4ThenQ5,
    Q5ThenQ4,
};

template <class Publish>
GdnConvEpilogue<Publish> make_epilogue(const Tensor& conv_weight, const Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       Tensor& query, Tensor& key, Tensor& value,
                                       int global_row_offset, Publish publish) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        kChannels,
        kQueryRows,
        kKeyRows,
        kValueRows,
        global_row_offset,
        static_cast<std::int32_t>(query.ne[1]),
        0,
        publish,
    };
}

template <class Publish>
struct Q4GdnDecodeEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        const float projected[1]{value};
        conv.store(row, projected);
    }
};

template <int Tokens, class Publish>
struct Q4GdnSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <bool, int, int TileCols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t row,
               std::int32_t, std::int32_t active_cols, const float (&values)[TileCols]) const {
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) { projected[token] = values[token]; }
        if (active_cols == Tokens) { conv.store(row, projected); }
    }
};

template <class Publish>
struct Q5GdnDecodeEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        if (row < kValueRows) {
            const float projected[1]{value};
            conv.store(row, projected);
        } else {
            z[row - kValueRows] = __float2bfloat16_rn(value);
        }
    }
};

template <int Tokens, class Publish>
struct Q5GdnSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <bool, int, int ProducedTokens>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                               std::int32_t, std::int32_t row,
                                               const float (&values)[ProducedTokens]) const {
        static_assert(ProducedTokens == Tokens);
        if (row < kValueRows) {
            conv.store(row, values);
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                z[static_cast<std::int64_t>(token) * kZRows + row - kValueRows] =
                    __float2bfloat16_rn(values[token]);
            }
        }
    }
};

template <class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q4_t1(const Tensor& x, const Weight& qk_weight,
                  const GdnConvEpilogue<Publish>& qk_epilogue, Tensor& query, cudaStream_t stream) {
    constexpr int q4_threads = Q4GemvR1Q8DirectSchedule::kThreads;
    constexpr int q4_blocks  = kQkRows / Q4GemvR1Q8DirectSchedule::kRowsPerCta;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q4_blocks), dim3(q4_threads), 0, stream},
            q4_rowsplit_gemv_kernel<Q4GemvR1Q8DirectSchedule, false, 0,
                                    Q4GdnDecodeEpilogue<Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, kQkRows, kHidden,
            Q4GdnDecodeEpilogue<Publish>{qk_epilogue}));
    } else {
        q4_rowsplit_gemv_kernel<Q4GemvR1Q8DirectSchedule, false, 0, Q4GdnDecodeEpilogue<Publish>,
                                TriggerPdl, JoinPdl><<<q4_blocks, q4_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, kQkRows, kHidden,
            Q4GdnDecodeEpilogue<Publish>{qk_epilogue});
    }
}

template <class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_t1(const Tensor& x, const Weight& value_z_weight,
                  const GdnConvEpilogue<Publish>& value_epilogue, Tensor& value, Tensor& z,
                  cudaStream_t stream) {
    constexpr int q5_rows_per_block = 16;
    constexpr int q5_threads        = q5_rows_per_block * 32;
    constexpr int q5_blocks         = kValueZRows / q5_rows_per_block;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q5_blocks), dim3(q5_threads), 0, stream},
            q5_rowsplit_gemv_kernel<kValueZRows, kHidden, q5_rows_per_block, 2, true, true,
                                    kValueRows, Q5GdnDecodeEpilogue<Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Q5GdnDecodeEpilogue<Publish>{value_epilogue, static_cast<__nv_bfloat16*>(z.data)}));
    } else {
        q5_rowsplit_gemv_kernel<kValueZRows, kHidden, q5_rows_per_block, 2, true, true, kValueRows,
                                Q5GdnDecodeEpilogue<Publish>, TriggerPdl, JoinPdl>
            <<<q5_blocks, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                Q5GdnDecodeEpilogue<Publish>{value_epilogue, static_cast<__nv_bfloat16*>(z.data)});
    }
}

template <int Tokens, class Q4Schedule, class Publish, bool TriggerPdl, bool JoinPdl,
          bool Dependent>
void launch_q4_ksplit(const Tensor& x, const Weight& qk_weight,
                      const GdnConvEpilogue<Publish>& qk_epilogue, Tensor& query,
                      cudaStream_t stream) {
    const dim3 q4_grid(kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
            q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0,
                                         Q4GdnSmallTEpilogue<Tokens, Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, kQueryRows, 0, kQkRows, kHidden,
            Tokens, kHidden, Q4GdnSmallTEpilogue<Tokens, Publish>{qk_epilogue}));
    } else {
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0,
                                     Q4GdnSmallTEpilogue<Tokens, Publish>, TriggerPdl, JoinPdl>
            <<<q4_grid, Q4Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data), nullptr, kQueryRows, 0, kQkRows, kHidden,
                Tokens, kHidden, Q4GdnSmallTEpilogue<Tokens, Publish>{qk_epilogue});
    }
}

template <int Tokens, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_small_t(const Tensor& x, const Weight& value_z_weight,
                       const GdnConvEpilogue<Publish>& value_epilogue, Tensor& value, Tensor& z,
                       cudaStream_t stream) {
    constexpr int q5_threads = 4 * 32;
    const dim3 q5_grid(kValueZRows, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q5_grid, dim3(q5_threads), 0, stream},
            q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, 5, kHidden, true,
                                                kValueRows, Q5GdnSmallTEpilogue<Tokens, Publish>,
                                                TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            kValueZRows, kValueRows, kHidden, Tokens, kHidden, 5,
            Q5GdnSmallTEpilogue<Tokens, Publish>{
                value_epilogue,
                static_cast<__nv_bfloat16*>(z.data),
            }));
    } else {
        q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, 5, kHidden, true,
                                            kValueRows, Q5GdnSmallTEpilogue<Tokens, Publish>,
                                            TriggerPdl, JoinPdl>
            <<<q5_grid, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                kValueZRows, kValueRows, kHidden, Tokens, kHidden, 5,
                Q5GdnSmallTEpilogue<Tokens, Publish>{
                    value_epilogue,
                    static_cast<__nv_bfloat16*>(z.data),
                });
    }
}

template <PdlOrder Order, class Publish>
void launch_t1(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
               const GdnConvEpilogue<Publish>& qk_epilogue,
               const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    // The Q4 and Q5 sides read the same activation but write disjoint output/state rows. The
    // dependent side therefore computes before waiting, then joins the producer at kernel exit.
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_t1<Publish, true, false, false>(x, value_z_weight, value_epilogue, value, z,
                                                  stream);
        launch_q4_t1<Publish, false, true, true>(x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_t1<Publish, true, false, false>(x, qk_weight, qk_epilogue, query, stream);
        launch_q5_t1<Publish, false, true, true>(x, value_z_weight, value_epilogue, value, z,
                                                 stream);
    }
}

template <int Tokens, class Q4Schedule, PdlOrder Order, class Publish>
void launch_small_t_schedule(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const GdnConvEpilogue<Publish>& qk_epilogue,
                             const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_small_t<Tokens, Publish, true, false, false>(x, value_z_weight, value_epilogue,
                                                               value, z, stream);
        launch_q4_ksplit<Tokens, Q4Schedule, Publish, false, true, true>(x, qk_weight, qk_epilogue,
                                                                         query, stream);
    } else {
        launch_q4_ksplit<Tokens, Q4Schedule, Publish, true, false, false>(x, qk_weight, qk_epilogue,
                                                                          query, stream);
        launch_q5_small_t<Tokens, Publish, false, true, true>(x, value_z_weight, value_epilogue,
                                                              value, z, stream);
    }
}

template <int Tokens, PdlOrder Order, class Publish>
void launch_small_t(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    const GdnConvEpilogue<Publish>& qk_epilogue,
                    const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
                    Tensor& z, cudaStream_t stream) {
    if constexpr (Tokens <= 4) {
        launch_small_t_schedule<Tokens, Q4ScheduleC4, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    } else {
        launch_small_t_schedule<Tokens, Q4ScheduleC8, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    }
}

template <PdlOrder Order, class Publish>
void launch_conv(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
                 const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                 Publish publish, cudaStream_t stream) {
    const GdnConvEpilogue<Publish> qk_epilogue = make_epilogue(
        conv_weight, conv_states, valid_columns, initial_slot, query, key, value, 0, publish);
    const GdnConvEpilogue<Publish> value_epilogue =
        make_epilogue(conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
                      kValueOffset, publish);

    switch (x.ne[1]) {
    case 1:
        launch_t1<Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query,
                                  value, z, stream);
        break;
    case 2:
        launch_small_t<2, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                          query, value, z, stream);
        break;
    case 3:
        launch_small_t<3, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                          query, value, z, stream);
        break;
    case 5:
        launch_small_t<5, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                          query, value, z, stream);
        break;
    case 6:
        launch_small_t<6, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                          query, value, z, stream);
        break;
    default:
        throw std::invalid_argument("Q4/Q5 projection-epilogue GDN conv requires T=1..3 or 5..6");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& valid_columns,
                                          const Tensor& initial_slot,
                                          const Tensor& snapshot_base_slot, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream) {
    if (x.ne[1] == 2) {
        launch_conv<PdlOrder::Q4ThenQ5>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
            query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   kChannels},
            stream);
    } else {
        launch_conv<PdlOrder::Q5ThenQ4>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
            query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   kChannels},
            stream);
    }
}

void q4_q5_gdn_input_conv_record_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, const Tensor& conv_weight,
                                        const Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_slot, Tensor& conv_record,
                                        Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                        cudaStream_t stream) {
    const RecordColumnPublish publish{static_cast<__nv_bfloat16*>(conv_record.data), kChannels,
                                      x.ne[1]};
    if (x.ne[1] == 2) {
        launch_conv<PdlOrder::Q4ThenQ5>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                        valid_columns, initial_slot, query, key, value, z, publish,
                                        stream);
    } else {
        launch_conv<PdlOrder::Q5ThenQ4>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                        valid_columns, initial_slot, query, key, value, z, publish,
                                        stream);
    }
}

} // namespace ninfer::ops::detail
