#include "ninfer/ops/scalar.h"
#include "core/device.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
__global__ void set_i32_scalar_kernel(std::int32_t* destination, std::int32_t value) {
    destination[0] = value;
}

__global__ void add_i32_scalars_kernel(const std::int32_t* lhs, const std::int32_t* rhs,
                                       std::int32_t* destination) {
    destination[0] = lhs[0] + rhs[0];
}

__global__ void increment_i32_scalar_kernel(std::int32_t* scalar) { ++scalar[0]; }

__global__ void increment_i64_scalar_kernel(std::int64_t* scalar) { ++scalar[0]; }

void require_scalar(const Tensor& tensor, DType dtype, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != 1 || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(name) + " must be a non-null contiguous scalar");
    }
}

} // namespace

void set_i32_scalar(Tensor& destination, std::int32_t value, CudaExecutionView execution) {
    if (destination.device != execution.device()) {
        throw std::invalid_argument("set_i32_scalar: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_scalar(destination, DType::I32, "set_i32_scalar destination");
    set_i32_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(destination.data), value);
    CUDA_CHECK(cudaGetLastError());
}

void assign_i32_scalar(const Tensor& source, Tensor& destination, CudaExecutionView execution) {
    if (source.device != execution.device() || destination.device != execution.device()) {
        throw std::invalid_argument(
            "assign_i32_scalar: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_scalar(source, DType::I32, "assign_i32_scalar source");
    require_scalar(destination, DType::I32, "assign_i32_scalar destination");
    if (source.data == destination.data) {
        throw std::invalid_argument("assign_i32_scalar: source and destination must not alias");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void add_i32_scalars(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                     CudaExecutionView execution) {
    if (lhs.device != execution.device() || rhs.device != execution.device() ||
        destination.device != execution.device()) {
        throw std::invalid_argument(
            "add_i32_scalars: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_scalar(lhs, DType::I32, "add_i32_scalars lhs");
    require_scalar(rhs, DType::I32, "add_i32_scalars rhs");
    require_scalar(destination, DType::I32, "add_i32_scalars destination");
    if (destination.data == lhs.data || destination.data == rhs.data) {
        throw std::invalid_argument("add_i32_scalars: inputs must not alias destination");
    }
    add_i32_scalars_kernel<<<1, 1, 0, stream>>>(static_cast<const std::int32_t*>(lhs.data),
                                                static_cast<const std::int32_t*>(rhs.data),
                                                static_cast<std::int32_t*>(destination.data));
    CUDA_CHECK(cudaGetLastError());
}

void increment_i32_scalar(Tensor& scalar, CudaExecutionView execution) {
    if (scalar.device != execution.device()) {
        throw std::invalid_argument(
            "increment_i32_scalar: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_scalar(scalar, DType::I32, "increment_i32_scalar scalar");
    increment_i32_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(scalar.data));
    CUDA_CHECK(cudaGetLastError());
}

void increment_i64_scalar(Tensor& scalar, CudaExecutionView execution) {
    if (scalar.device != execution.device()) {
        throw std::invalid_argument(
            "increment_i64_scalar: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_scalar(scalar, DType::I64, "increment_i64_scalar scalar");
    increment_i64_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int64_t*>(scalar.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
