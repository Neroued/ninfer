#include "ninfer/ops/collective.h"

#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void validate(const Tensor& value, const distributed::Group& group, CudaExecutionView execution) {
    if (!value.data || value.numel() <= 0 || !value.is_contiguous())
        throw std::invalid_argument("collective requires a nonempty contiguous tensor");
    if (value.device != group.device() || value.device != execution.device())
        throw std::invalid_argument("collective tensor, group and execution devices differ");
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.data);
    const auto y = reinterpret_cast<std::uintptr_t>(b.data);
    return x >= y ? x - y < b.bytes() : y - x < a.bytes();
}

void pair(const Tensor& input, const Tensor& output, const distributed::Group& group,
          CudaExecutionView execution, std::size_t input_factor, std::size_t output_factor,
          std::size_t input_offset, std::size_t output_offset) {
    validate(input, group, execution);
    validate(output, group, execution);
    if (input.dtype != output.dtype || std::uint64_t(input.numel()) > UINT64_MAX / input_factor ||
        std::uint64_t(output.numel()) > UINT64_MAX / output_factor ||
        std::uint64_t(input.numel()) * input_factor !=
            std::uint64_t(output.numel()) * output_factor)
        throw std::invalid_argument("collective input/output extents or types differ");
    if (overlaps(input, output) &&
        reinterpret_cast<std::uintptr_t>(input.data) + input_offset !=
            reinterpret_cast<std::uintptr_t>(output.data) + output_offset)
        throw std::invalid_argument("unsupported collective input/output overlap");
}

} // namespace

void all_reduce(const Tensor& input, Tensor& output, distributed::Reduction reduction,
                const distributed::Group& group, CudaExecutionView execution) {
    pair(input, output, group, execution, 1, 1, 0, 0);
    group.enqueue(distributed::Collective::AllReduce, input.data, output.data, input.numel(),
                  input.dtype, reduction, 0, execution);
}

void broadcast(Tensor& value, std::uint32_t root, const distributed::Group& group,
               CudaExecutionView execution) {
    validate(value, group, execution);
    group.enqueue(distributed::Collective::Broadcast, value.data, value.data, value.numel(),
                  value.dtype, distributed::Reduction::Sum, root, execution);
}

void all_gather(const Tensor& input, Tensor& output, const distributed::Group& group,
                CudaExecutionView execution) {
    pair(input, output, group, execution, group.size(), 1, 0, group.rank() * input.bytes());
    group.enqueue(distributed::Collective::AllGather, input.data, output.data, input.numel(),
                  input.dtype, distributed::Reduction::Sum, 0, execution);
}

void reduce_scatter(const Tensor& input, Tensor& output, distributed::Reduction reduction,
                    const distributed::Group& group, CudaExecutionView execution) {
    pair(input, output, group, execution, 1, group.size(), group.rank() * output.bytes(), 0);
    group.enqueue(distributed::Collective::ReduceScatter, input.data, output.data, output.numel(),
                  output.dtype, reduction, 0, execution);
}

} // namespace ninfer::ops
