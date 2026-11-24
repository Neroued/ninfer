#include "ops/common/host/launch.h"
#include "ops/ple/operands.h"

#include <cstdint>

namespace ninfer::ops::detail::ple {
namespace {

struct ExtractArguments {
    const std::uint16_t *before, *records;
    std::uint16_t* after;
    const std::int64_t *source, *destination;
    const std::int32_t *accepted, *live;
    std::int64_t before_column, record_column, record_batch, after_column;
    const std::int32_t* record_starts = nullptr;
};

template <class Word, bool Ragged>
__global__ void extract_kernel(const __grid_constant__ ExtractArguments a) {
    constexpr int width = sizeof(Word) / 2;
    const int c         = (blockIdx.x * blockDim.x + threadIdx.x) * width;
    if (c >= kChannels) return;
    const int b = blockIdx.y;
    if (a.live && !a.live[b]) return;
    const int accepted = a.accepted[b];
    const auto* before = a.before + a.source[b] / 2 + c;
    auto* after        = a.after + a.destination[b] / 2 + c;
    if (accepted == 0 && before == after) return;
    const auto record_start = Ragged ? std::int64_t(a.record_starts[b]) * a.record_column
                                     : std::int64_t(b) * a.record_batch;
    const auto* records     = a.records + record_start + c;
    // Load the complete result before writing: this row may update its source in place.
    Word values[kHistory];
#pragma unroll
    for (int j = 0; j < kHistory; ++j) {
        const int k = accepted - kHistory + j;
        const auto* p =
            k < 0 ? before + (k + kHistory) * a.before_column : records + k * a.record_column;
        values[j] = *reinterpret_cast<const Word*>(p);
    }
#pragma unroll
    for (int j = 0; j < kHistory; ++j)
        *reinterpret_cast<Word*>(after + j * a.after_column) = values[j];
}

template <class Word, bool Ragged>
void launch_extract(const ExtractArguments& args, int batch, cudaStream_t stream) {
    constexpr int threads = sizeof(Word) == 4 ? 128 : 256;
    common::for_each_token_slice(batch, 1, [&](int offset, int count) {
        auto slice = args;
        slice.source += offset;
        slice.destination += offset;
        slice.accepted += offset;
        if (slice.live) slice.live += offset;
        if constexpr (Ragged)
            slice.record_starts += offset;
        else
            slice.records += std::int64_t(offset) * args.record_batch;
        extract_kernel<Word, Ragged><<<dim3(kChannels / 256, count), threads, 0, stream>>>(slice);
    });
    CUDA_CHECK(cudaGetLastError());
}

using ExtractLaunch = void (*)(const ExtractArguments&, int, cudaStream_t);

template <bool Ragged>
ExtractLaunch select_extract(const ConstStateFieldView& before, const Tensor& records,
                             const StateFieldView& after) {
    return field_aligned(before, 4) && field_aligned(after, 4) && pair_aligned(records)
               ? launch_extract<std::uint32_t, Ragged>
               : launch_extract<std::uint16_t, Ragged>;
}

void execute_extract(const ConstPleConvStateView& before, const Tensor& records,
                     const Tensor& accepted, const PleConvStateView& after,
                     const std::int32_t* live, const std::int32_t* record_starts,
                     CudaExecutionView execution) {
    const ExtractArguments a{
        common::operand::state_field_base<const std::uint16_t>(before.history),
        static_cast<const std::uint16_t*>(records.data),
        common::operand::state_field_base<std::uint16_t>(after.history),
        static_cast<const std::int64_t*>(before.history.records.byte_offsets.data),
        static_cast<const std::int64_t*>(after.history.records.byte_offsets.data),
        static_cast<const std::int32_t*>(accepted.data),
        live,
        kChannels,
        records.nb[1] / 2,
        records.nb[2] / 2,
        kChannels,
        record_starts};
    const auto launch = record_starts
                            ? select_extract<true>(before.history, records, after.history)
                            : select_extract<false>(before.history, records, after.history);
    launch(a, accepted.ne[0], execution.stream());
}
} // namespace
} // namespace ninfer::ops::detail::ple

namespace ninfer::ops {

void ple_state_extract(const ConstPleConvStateView& before, const PleConvRecordReadView& records,
                       const Tensor& accepted, PleConvStateView after, const Tensor& live_rows,
                       CudaExecutionView execution) {
    using namespace detail::ple;
    validate_extract(before, records, accepted, after, live_rows, execution);
    execute_extract(before, records.values, accepted, after,
                    static_cast<const std::int32_t*>(live_rows.data), nullptr, execution);
}

void ple_state_extract(const ConstPleConvStateView& before, const Tensor& records,
                       const RaggedSequenceView& ragged, PleConvStateView after,
                       CudaExecutionView execution) {
    using namespace detail::ple;
    validate_extract(before, records, ragged, after, execution);
    // The only segment starts at column zero, so it can use the dense kernel.
    const auto* starts =
        ragged.num_seqs() > 1 ? static_cast<const std::int32_t*>(ragged.cu_seqlens.data) : nullptr;
    execute_extract(before, records, ragged.seq_lengths, after, nullptr, starts, execution);
}

} // namespace ninfer::ops
