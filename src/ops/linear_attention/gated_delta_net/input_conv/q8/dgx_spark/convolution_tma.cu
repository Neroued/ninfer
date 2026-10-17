#include "ops/linear/kernels/q8/q8_a16_tma.cuh"
#include "ops/linear_attention/gated_delta_net/input_conv/kernels/output.cuh"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/dgx_spark/launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using Geometry = GdnConvGeometry<6144>;

// The complete row consumer owns every observable store.
struct ProjectionSink {};

template <class Publish>
struct BatchEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <class Cfg>
    static constexpr int kSharedBytes =
        Cfg::kBlockTokens * (Cfg::kBlockRows + 4) * static_cast<int>(sizeof(float));

    template <class Cfg, bool Full, class Output>
    __device__ __forceinline__ void
    finish_tile(const Output&, unsigned char* scratch,
                const float (&acc)[Cfg::kMmaRows][Cfg::kMmaTokens][4], int m0, int n0, int rows,
                int tokens) const {
        static_assert(Cfg::kBlockRows == 64 && Cfg::kWarpRows == 64 && Cfg::kWarpTokens == 16 &&
                      Geometry::channels % Cfg::kBlockRows == 0);
        const int tid  = static_cast<int>(threadIdx.x);
        const int warp = tid >> 5, lane = tid & 31;
        const int wm = warp / Cfg::kWarpGridTokens, wn = warp % Cfg::kWarpGridTokens;
        const int gid = lane >> 2, lid = lane & 3;

        // Every row tile belongs wholly to QKV or Z. Keep Z's coalesced MMA stores
        // and write every physical column, independently of valid prefixes.
        if (m0 >= Geometry::channels) {
            const auto store = [&](int row, int token, float v) {
                if (Full || (row < rows && token < tokens))
                    z[std::int64_t(token) * Geometry::z_rows + row - Geometry::channels] =
                        __float2bfloat16_rn(v);
            };
#pragma unroll
            for (int mi = 0; mi < Cfg::kMmaRows; ++mi) {
                const int row = m0 + wm * Cfg::kWarpRows + mi * 16 + gid;
#pragma unroll
                for (int ni = 0; ni < Cfg::kMmaTokens; ++ni) {
                    const int col = n0 + wn * Cfg::kWarpTokens + ni * 8 + 2 * lid;
                    store(row, col, acc[mi][ni][0]);
                    store(row, col + 1, acc[mi][ni][1]);
                    store(row + 8, col, acc[mi][ni][2]);
                    store(row + 8, col + 1, acc[mi][ni][3]);
                }
            }
            return;
        }

        // Drain the final MMA reads before reusing pipeline storage. A column
        // stride of BM+4 gives contiguous row loads and avoids the MMA store's
        // repeated banks. FP32 projection values never cross a BF16 boundary.
        __syncthreads();
        constexpr int Stride = Cfg::kBlockRows + 4;
        float* projected     = reinterpret_cast<float*>(scratch);
#pragma unroll
        for (int mi = 0; mi < Cfg::kMmaRows; ++mi) {
            const int row = wm * Cfg::kWarpRows + mi * 16 + gid;
#pragma unroll
            for (int ni = 0; ni < Cfg::kMmaTokens; ++ni) {
                const int col                           = wn * Cfg::kWarpTokens + ni * 8 + 2 * lid;
                projected[col * Stride + row]           = acc[mi][ni][0];
                projected[(col + 1) * Stride + row]     = acc[mi][ni][1];
                projected[col * Stride + row + 8]       = acc[mi][ni][2];
                projected[(col + 1) * Stride + row + 8] = acc[mi][ni][3];
            }
        }
        __syncthreads();
        const int live = min(Cfg::kBlockTokens, tokens - n0);
        // Every batch window has independent initial state. Spread those
        // consumers over all producer warps instead of serializing B windows
        // on just the first two warps of the CTA.
        for (int item = tid; item < (live / conv.width) * Cfg::kBlockRows; item += Cfg::kThreads) {
            const int local_row = item % Cfg::kBlockRows;
            const int row       = m0 + local_row;
            const int first     = (item / Cfg::kBlockRows) * conv.width;
            if (row >= rows) continue;
            const auto destination =
                CausalConvSplitOutput3<__nv_bfloat16, 2048, 2048, 6144>{conv.query, conv.key,
                                                                        conv.value}
                    .column(row);
            const float w0 = __bfloat162float(conv.conv_weight[row]);
            const float w1 = __bfloat162float(conv.conv_weight[Geometry::channels + row]);
            const float w2 = __bfloat162float(conv.conv_weight[2 * Geometry::channels + row]);
            const float w3 = __bfloat162float(conv.conv_weight[3 * Geometry::channels + row]);
            // Admission requires one complete physical block in this CTA's token
            // tile. A row has a single owner, including its initial history reads,
            // so an own-source/destination overlap remains valid.
            const int batch            = (n0 + first) / conv.width;
            const std::int64_t initial = std::int64_t(conv.initial_slots[batch]) * conv.slot_stride;
            float s0                   = __bfloat162float(conv.state_read[initial + row]);
            float s1  = __bfloat162float(conv.state_read[initial + Geometry::channels + row]);
            float s2  = __bfloat162float(conv.state_read[initial + 2LL * Geometry::channels + row]);
            int valid = conv.valid_columns == nullptr ? conv.width : conv.valid_columns[batch];
            valid     = max(0, min(valid, conv.width));
            for (int token = 0; token < conv.width; ++token) {
                const int column = n0 + first + token;
                if (token >= valid) {
                    *destination.at(column) = __float2bfloat16_rn(0.0F);
                    continue;
                }
                const float p           = projected[(first + token) * Stride + local_row];
                *destination.at(column) = causal_conv4_silu(w0, w1, w2, w3, s0, s1, s2, p);
                conv.publish.publish(token, batch, row, s1, s2, p);
                s0 = s1;
                s1 = s2;
                s2 = p;
            }
        }
    }
};

using Mid96    = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using Large128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;

template <class Publish>
void launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight, const Tensor& state,
            const Tensor& valid, const Tensor& initial, Tensor& query, Tensor& key, Tensor& value,
            Tensor& z, Publish publish, cudaStream_t stream) {
    const int columns = x.ne[1] * x.ne[2];
    if (weight.n != 16384 || weight.k != 2560 || x.ne[2] < 2 || x.ne[2] > 8 || x.ne[1] < 1 ||
        x.ne[1] > 16 || columns <= 64 || columns > 128)
        throw std::invalid_argument("Q8 fused batch TMA: unsupported physical block");
    const auto flat     = x.view({2560, columns});
    const auto operands = q8_linear_operands(flat, weight);
    auto adapter = make_gdn_conv_output<Geometry, 1>(conv_weight, state, valid, initial, query, key,
                                                     value, z, publish);
    adapter.conv.width = x.ne[1];
    const BatchEpilogue<Publish> epilogue{adapter.conv, adapter.z};
    if (columns <= 96)
        return launch_q8_a16_tma_mma<Mid96>(operands, ProjectionSink{}, epilogue, stream);
    launch_q8_a16_tma_mma<Large128>(operands, ProjectionSink{}, epilogue, stream);
}
} // namespace

void launch_q8_gdn_tma_n16384_k2560_snapshot(const Tensor& x, const Weight& weight,
                                             const Tensor& conv_weight, Tensor& state,
                                             const Tensor& valid, const Tensor& initial,
                                             const Tensor& bases, Tensor& query, Tensor& key,
                                             Tensor& value, Tensor& z, cudaStream_t stream) {
    launch(x, weight, conv_weight, state, valid, initial, query, key, value, z,
           SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(state.data), state.nb[2] / 2,
                                  static_cast<const std::int32_t*>(bases.data), Geometry::channels},
           stream);
}

void launch_q8_gdn_tma_n16384_k2560_record(const Tensor& x, const Weight& weight,
                                           const Tensor& conv_weight, const Tensor& state,
                                           const Tensor& valid, const Tensor& initial,
                                           Tensor& record, Tensor& query, Tensor& key,
                                           Tensor& value, Tensor& z, cudaStream_t stream) {
    launch(x, weight, conv_weight, state, valid, initial, query, key, value, z,
           RecordColumnPublish{static_cast<__nv_bfloat16*>(record.data), Geometry::channels,
                               static_cast<std::int32_t>(x.ne[1])},
           stream);
}
} // namespace ninfer::ops::detail
