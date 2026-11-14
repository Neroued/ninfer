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
};

template <class Word>
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
    const auto* records = a.records + b * a.record_batch + c;
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

template <class Word>
void launch_extract(const ExtractArguments& args, int batch, cudaStream_t stream) {
    constexpr int threads = sizeof(Word) == 4 ? 128 : 256;
    common::for_each_token_slice(batch, 1, [&](int offset, int count) {
        auto slice = args;
        slice.source += offset;
        slice.destination += offset;
        slice.accepted += offset;
        if (slice.live) slice.live += offset;
        slice.records += std::int64_t(offset) * args.record_batch;
        extract_kernel<Word><<<dim3(kChannels / 256, count), threads, 0, stream>>>(slice);
    });
    CUDA_CHECK(cudaGetLastError());
}

using ExtractLaunch = void (*)(const ExtractArguments&, int, cudaStream_t);

ExtractLaunch select_extract(const ConstStateFieldView& before, const Tensor& records,
                             const StateFieldView& after) {
    return field_aligned(before, 4) && field_aligned(after, 4) && pair_aligned(records)
               ? launch_extract<std::uint32_t>
               : launch_extract<std::uint16_t>;
}
} // namespace
} // namespace ninfer::ops::detail::ple

namespace ninfer::ops {

void ple_state_extract(const ConstPleConvStateView& before, const PleConvRecordReadView& records,
                       const Tensor& accepted, PleConvStateView after, const Tensor& live_rows,
                       CudaExecutionView execution) {
    using namespace detail::ple;
    validate_extract(before, records, accepted, after, live_rows, execution);
    const ExtractArguments a{
        common::operand::state_field_base<const std::uint16_t>(before.history),
        static_cast<const std::uint16_t*>(records.values.data),
        common::operand::state_field_base<std::uint16_t>(after.history),
        static_cast<const std::int64_t*>(before.history.records.byte_offsets.data),
        static_cast<const std::int64_t*>(after.history.records.byte_offsets.data),
        static_cast<const std::int32_t*>(accepted.data),
        static_cast<const std::int32_t*>(live_rows.data),
        kChannels,
        records.values.nb[1] / 2,
        records.values.nb[2] / 2,
        kChannels};
    select_extract(before.history, records.values, after.history)(a, accepted.ne[0],
                                                                  execution.stream());
}

} // namespace ninfer::ops
