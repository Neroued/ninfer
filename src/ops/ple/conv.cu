#include "ninfer/ops/ple.h"
#include "ops/ple/conv_kernel.cuh"
#include "core/platform.h"

namespace ninfer::ops::detail::ple {
namespace {
constexpr int kChannelTiles = kChannels / 256;

inline ConvArguments conv_arguments(const Tensor& residual, const Tensor& gated,
                                    const Tensor& kernel, const PleBatchView& batch,
                                    const ConstPleConvStateView& before,
                                    const PleConvRecordWindowView& current, Tensor& updated,
                                    const Tensor* norm, float epsilon, Tensor* normalized) {
    return {static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<const __nv_bfloat16*>(gated.data),
            static_cast<const __nv_bfloat16*>(kernel.data),
            common::operand::state_field_base<const __nv_bfloat16>(before.history),
            static_cast<const __nv_bfloat16*>(current.values.data),
            norm ? static_cast<const __nv_bfloat16*>(norm->data) : nullptr,
            static_cast<__nv_bfloat16*>(updated.data),
            normalized ? static_cast<__nv_bfloat16*>(normalized->data) : nullptr,
            static_cast<const std::int32_t*>(batch.valid_columns.data),
            static_cast<const std::int32_t*>(current.offsets.data),
            static_cast<const std::int64_t*>(before.history.records.byte_offsets.data),
            kChannels,
            current.values.nb[1] / 2,
            current.values.nb[2] / 2,
            residual.ne[2],
            residual.ne[3],
            epsilon};
}

inline unsigned conv_grid(std::int64_t count) {
    return static_cast<unsigned>(count < 65535 ? count : 65535);
}

template <bool Ragged>
void launch_tiled_conv(const ConvArguments& a, cudaStream_t stream) {
    const auto tiles = (std::int64_t(a.tokens) + 7) / 8;
    conv_add_tiled_kernel<128, 2, 8, Ragged>
        <<<conv_grid(tiles * a.batch * kChannelTiles), 128, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

template <int Block, int Width, bool Ragged>
void launch_conv_add(const ConvArguments& a, cudaStream_t stream) {
    const auto columns = std::int64_t(a.tokens) * a.batch;
    conv_add_kernel<Block, Width, Ragged>
        <<<conv_grid(columns * kChannelTiles), Block, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

template <int Block, int Width, bool WithConv, bool Ragged>
void launch_conv_norm(const ConvArguments& a, cudaStream_t stream) {
    if constexpr (!WithConv) launch_tiled_conv<Ragged>(a, stream);
    const auto groups = std::int64_t(a.tokens) * a.batch * kStreams;
    conv_norm_kernel<Block, Width, WithConv, Ragged><<<conv_grid(groups), Block, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

using ConvLaunch = void (*)(const ConvArguments&, cudaStream_t);

template <bool Ragged>
ConvLaunch select_conv(const ConvArguments& a, bool paired, bool wide_history,
                       int multiprocessors) {
    const auto columns = std::int64_t(a.tokens) * a.batch;
    if (a.normalized) {
        if (!paired) return launch_conv_norm<256, 1, true, Ragged>;
        if (columns * kStreams <= 2LL * multiprocessors)
            return launch_conv_norm<512, 2, true, Ragged>;
#if NINFER_TARGET_SPARK
        if (columns >= 1024) {
            bool aligned = wide_history;
            for (const auto* p : {a.kernel, a.history, a.records, a.norm})
                aligned &= reinterpret_cast<std::uintptr_t>(p) % 8 == 0;
            aligned &= a.history_column % 4 == 0 && a.record_column % 4 == 0;
            if constexpr (!Ragged) aligned &= a.record_batch % 4 == 0;
            if (aligned) return launch_conv_norm<512, 4, true, Ragged>;
        }
#else
        // Reuse across tokens pays for a separate norm launch in this extent range.
        if (columns >= 1024 && columns <= 2304) return launch_conv_norm<256, 2, false, Ragged>;
#endif
        return launch_conv_norm<256, 2, true, Ragged>;
    }
    if (paired && a.tokens >= 128) return launch_tiled_conv<Ragged>;
    return paired ? launch_conv_add<128, 2, Ragged> : launch_conv_add<256, 1, Ragged>;
}

void execute_conv(const Tensor& residual, const Tensor& gated, const Tensor& kernel,
                  const PleBatchView& batch, const ConstPleConvStateView& before,
                  const PleConvRecordWindowView& current, Tensor& updated,
                  const Tensor* norm_weight, float epsilon, Tensor* normalized,
                  CudaExecutionView execution, const RaggedSequenceView* ragged = nullptr) {
    validate_conv(residual, gated, kernel, batch, before, current, updated, norm_weight, epsilon,
                  normalized, execution, ragged);
    auto args         = conv_arguments(residual, gated, kernel, batch, before, current, updated,
                                       norm_weight, epsilon, normalized);
    const bool paired = pair_aligned(kernel) && field_aligned(before.history, 4) &&
                        pair_aligned(current.values) &&
                        (!norm_weight || pair_aligned(*norm_weight));
    if (ragged && ragged->num_seqs() > 1) {
        args.cu_seqlens   = static_cast<const std::int32_t*>(ragged->cu_seqlens.data);
        args.token_to_seq = static_cast<const std::int32_t*>(ragged->token_to_seq.data);
    }
    const bool wide   = field_aligned(before.history, 8);
    const int sms     = execution.multiprocessor_count();
    const auto launch = args.token_to_seq ? select_conv<true>(args, paired, wide, sms)
                                          : select_conv<false>(args, paired, wide, sms);
    launch(args, execution.stream());
}
} // namespace
} // namespace ninfer::ops::detail::ple

namespace ninfer::ops {

void ple_conv_add(const Tensor& residual, const Tensor& gated_value, const Tensor& kernel,
                  const PleBatchView& batch, const ConstPleConvStateView& before,
                  const PleConvRecordWindowView& current, Tensor& updated_residual,
                  CudaExecutionView execution) {
    detail::ple::execute_conv(residual, gated_value, kernel, batch, before, current,
                              updated_residual, nullptr, 0, nullptr, execution);
}

void ple_conv_add_norm(const Tensor& residual, const Tensor& gated_value, const Tensor& kernel,
                       const PleBatchView& batch, const ConstPleConvStateView& before,
                       const PleConvRecordWindowView& current, const Tensor& next_hc_norm_weight,
                       float next_hc_epsilon, Tensor& updated_residual, Tensor& normalized,
                       CudaExecutionView execution) {
    detail::ple::execute_conv(residual, gated_value, kernel, batch, before, current,
                              updated_residual, &next_hc_norm_weight, next_hc_epsilon, &normalized,
                              execution);
}

namespace {
void ragged_conv(const Tensor& residual, const Tensor& gated, const Tensor& kernel,
                 const RaggedSequenceView& ragged, const ConstPleConvStateView& before,
                 const Tensor& current, Tensor& updated, const Tensor* norm, float epsilon,
                 Tensor* normalized, CudaExecutionView execution) {
    common::operand::ragged_sequence(ragged, "ple_conv_add", execution.device());
    const auto offset = ragged.num_seqs() == 1 ? ragged.cu_seqlens.slice(0, 0, 1) : Tensor{};
    detail::ple::execute_conv(residual, gated, kernel, {ragged.seq_lengths}, before,
                              {current, offset}, updated, norm, epsilon, normalized, execution,
                              &ragged);
}
} // namespace

void ple_conv_add(const Tensor& residual, const Tensor& gated_value, const Tensor& kernel,
                  const RaggedSequenceView& ragged, const ConstPleConvStateView& before,
                  const Tensor& current_values, Tensor& updated_residual,
                  CudaExecutionView execution) {
    ragged_conv(residual, gated_value, kernel, ragged, before, current_values, updated_residual,
                nullptr, 0, nullptr, execution);
}

void ple_conv_add_norm(const Tensor& residual, const Tensor& gated_value, const Tensor& kernel,
                       const RaggedSequenceView& ragged, const ConstPleConvStateView& before,
                       const Tensor& current_values, const Tensor& next_hc_norm_weight,
                       float next_hc_epsilon, Tensor& updated_residual, Tensor& normalized,
                       CudaExecutionView execution) {
    ragged_conv(residual, gated_value, kernel, ragged, before, current_values, updated_residual,
                &next_hc_norm_weight, next_hc_epsilon, &normalized, execution);
}

} // namespace ninfer::ops
