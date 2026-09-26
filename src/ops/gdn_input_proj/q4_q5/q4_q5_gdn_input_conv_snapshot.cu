#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q4/q4_instances.cuh"
#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/q4/q4_simt_launch.cuh"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/linear/q5/q5_gemv_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden      = 5120;
constexpr int kQueryRows   = 2048;
constexpr int kKeyRows     = 2048;
constexpr int kValueRows   = 6144;
constexpr int kQkRows      = kQueryRows + kKeyRows;
constexpr int kChannels    = kQkRows + kValueRows;
constexpr int kValueOffset = kQkRows;

using Q4ScheduleT4 = Q4A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1, true>;
using Q4ScheduleT8 = Q4A16SimtSchedule<8, 8, 1, 16, 2, Cache::ca, 1, true>;

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

    template <class Output, int Tokens>
    __device__ __forceinline__ void apply_row(const Output&, int row, int,
                                              const float (&values)[Tokens], int) const {
        static_assert(Tokens == 1);
        conv.store(row, values);
    }
};

template <int Tokens, class Publish>
struct Q4GdnSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <class Output, int TileTokens>
    __device__ __forceinline__ void apply_row(const Output&, int row, int token_begin,
                                              const float (&values)[TileTokens], int active) const {
        static_assert(Tokens <= TileTokens);
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) projected[token] = values[token];
        if (token_begin == 0 && active == Tokens) conv.store(row, projected);
    }
};

// The row consumer publishes value rows through convolution and maps the
// remaining projection rows to this fixed-layout Z tensor.
struct Q5GdnZOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[std::int64_t(token) * kValueRows + row] = __float2bfloat16_rn(value);
    }
};

template <int Tokens, class Publish>
struct Q5GdnProjectionEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <class Output, int ProducedTokens>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int token_begin,
                                              const float (&values)[ProducedTokens],
                                              int active_tokens) const {
        static_assert(ProducedTokens == Tokens);
        if (row < kValueRows) {
            if (token_begin == 0 && active_tokens == Tokens) conv.store(row, values);
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
                if (token < active_tokens)
                    output.store(row - kValueRows, token_begin + token, values[token]);
        }
    }
};

template <class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q4_t1(const Tensor& x, const Weight& qk_weight,
                  const GdnConvEpilogue<Publish>& qk_epilogue, Tensor& query, cudaStream_t stream) {
    launch_q4_a16_gemv<q4_instances::GemvR1W8K5120, TriggerPdl, JoinPdl, Dependent>(
        q4_linear_operands(x, qk_weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(query.data), query.ne[0]},
        Q4GdnDecodeEpilogue<Publish>{qk_epilogue}, stream);
}

template <class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_t1(const Tensor& x, const Weight& value_z_weight,
                  const GdnConvEpilogue<Publish>& value_epilogue, Tensor& z, cudaStream_t stream) {
    launch_q5_a16_gemv<q5_instances::GemvR16W1G16S2XK5120, TriggerPdl, JoinPdl, Dependent>(
        q5_linear_operands(x, value_z_weight), Q5GdnZOutput{static_cast<__nv_bfloat16*>(z.data)},
        Q5GdnProjectionEpilogue<1, Publish>{value_epilogue}, stream);
}

template <int Tokens, class Q4Schedule, class Publish, bool TriggerPdl, bool JoinPdl,
          bool Dependent>
void launch_q4_simt(const Tensor& x, const Weight& qk_weight,
                    const GdnConvEpilogue<Publish>& qk_epilogue, Tensor& query,
                    cudaStream_t stream) {
    launch_q4_a16_simt<Q4Schedule, TriggerPdl, JoinPdl, Dependent>(
        q4_linear_operands(x, qk_weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(query.data), query.ne[0]},
        Q4GdnSmallTEpilogue<Tokens, Publish>{qk_epilogue}, stream);
}

template <int Tokens, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_small_t(const Tensor& x, const Weight& value_z_weight,
                       const GdnConvEpilogue<Publish>& value_epilogue, Tensor& z,
                       cudaStream_t stream) {
    constexpr int kRows      = Tokens == 5 ? 2 : 1;
    constexpr int kWarps     = Tokens >= 5 ? 2 : 4;
    constexpr int kMinBlocks = Tokens == 5 ? 8 : Tokens == 6 ? 16 : 10;
    using Schedule =
        Q5A16DirectSimtSchedule<kRows, Tokens, kWarps, 16 / kWarps, kMinBlocks, kHidden, true>;
    launch_q5_a16_direct_simt<Schedule, TriggerPdl, JoinPdl, Dependent>(
        q5_linear_operands(x, value_z_weight), Q5GdnZOutput{static_cast<__nv_bfloat16*>(z.data)},
        Q5GdnProjectionEpilogue<Tokens, Publish>{value_epilogue}, stream);
}

template <PdlOrder Order, class Publish>
void launch_t1(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
               const GdnConvEpilogue<Publish>& qk_epilogue,
               const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    // The Q4 and Q5 sides read the same activation but write disjoint output/state rows. The
    // dependent side therefore computes before waiting, then joins the producer at kernel exit.
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_t1<Publish, true, false, false>(x, value_z_weight, value_epilogue, z, stream);
        launch_q4_t1<Publish, false, true, true>(x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_t1<Publish, true, false, false>(x, qk_weight, qk_epilogue, query, stream);
        launch_q5_t1<Publish, false, true, true>(x, value_z_weight, value_epilogue, z, stream);
    }
}

template <int Tokens, class Q4Schedule, PdlOrder Order, class Publish>
void launch_small_t_schedule(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const GdnConvEpilogue<Publish>& qk_epilogue,
                             const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_small_t<Tokens, Publish, true, false, false>(x, value_z_weight, value_epilogue, z,
                                                               stream);
        launch_q4_simt<Tokens, Q4Schedule, Publish, false, true, true>(x, qk_weight, qk_epilogue,
                                                                       query, stream);
    } else {
        launch_q4_simt<Tokens, Q4Schedule, Publish, true, false, false>(x, qk_weight, qk_epilogue,
                                                                        query, stream);
        launch_q5_small_t<Tokens, Publish, false, true, true>(x, value_z_weight, value_epilogue, z,
                                                              stream);
    }
}

template <int Tokens, PdlOrder Order, class Publish>
void launch_small_t(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    const GdnConvEpilogue<Publish>& qk_epilogue,
                    const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
                    Tensor& z, cudaStream_t stream) {
    if constexpr (Tokens <= 4) {
        launch_small_t_schedule<Tokens, Q4ScheduleT4, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    } else {
        launch_small_t_schedule<Tokens, Q4ScheduleT8, Order, Publish>(
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
