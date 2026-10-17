#include "ops/ple/operands.h"

#include <cstdint>

namespace ninfer::ops::detail::ple {
namespace {

struct ExtractArguments {
    const std::uint16_t *before, *records;
    std::uint16_t* after;
    const std::int32_t *source, *destination, *accepted;
    std::int64_t before_column, before_slot, record_column, record_batch, after_column, after_slot;
};

template <class Word>
__global__ void extract_kernel(const __grid_constant__ ExtractArguments a) {
    constexpr int width = sizeof(Word) / 2;
    const int c         = (blockIdx.x * blockDim.x + threadIdx.x) * width;
    if (c >= kChannels) return;
    const int b = blockIdx.y, accepted = a.accepted[b];
    const auto* before = a.before + a.source[b] * a.before_slot + c;
    auto* after        = a.after + a.destination[b] * a.after_slot + c;
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
    extract_kernel<Word><<<dim3(kChannels / 256, batch), threads, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

using ExtractLaunch = void (*)(const ExtractArguments&, int, cudaStream_t);

ExtractLaunch select_extract(const Tensor& before, const Tensor& records, const Tensor& after) {
    return pair_aligned(before) && pair_aligned(after) && pair_aligned(records)
               ? launch_extract<std::uint32_t>
               : launch_extract<std::uint16_t>;
}
} // namespace
} // namespace ninfer::ops::detail::ple

namespace ninfer::ops {

void ple_state_extract(const PleConvStateView& before, const PleConvRecordReadView& records,
                       const Tensor& accepted, PleConvStateView after,
                       CudaExecutionView execution) {
    using namespace detail::ple;
    validate_extract(before, records, accepted, after, execution);
    const ExtractArguments a{static_cast<const std::uint16_t*>(before.history.data),
                             static_cast<const std::uint16_t*>(records.values.data),
                             static_cast<std::uint16_t*>(after.history.data),
                             static_cast<const std::int32_t*>(before.slots.data),
                             static_cast<const std::int32_t*>(after.slots.data),
                             static_cast<const std::int32_t*>(accepted.data),
                             before.history.nb[1] / 2,
                             before.history.nb[2] / 2,
                             records.values.nb[1] / 2,
                             records.values.nb[2] / 2,
                             after.history.nb[1] / 2,
                             after.history.nb[2] / 2};
    select_extract(before.history, records.values, after.history)(a, accepted.ne[0],
                                                                  execution.stream());
}

} // namespace ninfer::ops
