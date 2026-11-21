#pragma once

#include "ops/common/host/operands.h"
#include "ninfer/ops/sampling.h"

namespace ninfer::ops::sampling_detail {

inline void sparse_candidate_plane(const Tensor& plane, DType dtype, int steps, int rows,
                                   const char* op, const char* name, Device device) {
    common::operand::strided_tensor(plane, dtype, {16, steps, rows}, 4, op, name, device);
    if (plane.nb[0] != 4)
        throw std::invalid_argument(std::string(op) + ": candidates must be contiguous in " + name);
}

inline void check_outputs(std::initializer_list<Tensor*> outputs,
                          std::initializer_list<const Tensor*> inputs,
                          const SamplingConfig* configs, int batch, const char* op,
                          MemorySpan scratch = {}) {
    using namespace common::operand;
    const auto check_write = [&](Range write) {
        for (const Tensor* input : inputs)
            if (input->data) {
                const auto read =
                    input->is_contiguous() ? range(*input, op) : strided_range(*input, op);
                disjoint(write, read, op);
            }
        disjoint(write, range(configs, sizeof(SamplingConfig) * batch, op), op);
    };
    if (scratch.data) check_write(range(scratch.data, scratch.bytes, op));
    for (auto current = outputs.begin(); current != outputs.end(); ++current) {
        const auto write = range(**current, op);
        check_write(write);
        if (scratch.data) disjoint(write, range(scratch.data, scratch.bytes, op), op);
        for (auto earlier = outputs.begin(); earlier != current; ++earlier)
            disjoint(write, range(**earlier, op), op);
    }
}

} // namespace ninfer::ops::sampling_detail
