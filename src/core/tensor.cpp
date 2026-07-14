#include "ninfer/core/tensor.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {

std::int64_t checked_mul_i64(std::int64_t a, std::int64_t b) {
    if (a < 0 || b < 0) { throw std::invalid_argument("tensor dimensions must be positive"); }
    if (b != 0 && a > std::numeric_limits<std::int64_t>::max() / b) {
        throw std::overflow_error("tensor size overflows int64");
    }
    return a * b;
}

std::array<std::int32_t, 4> normalize_shape(std::initializer_list<std::int32_t> shape) {
    if (shape.size() == 0 || shape.size() > 4) {
        throw std::invalid_argument("tensor rank must be in [1,4]");
    }

    std::array<std::int32_t, 4> out{1, 1, 1, 1};
    std::size_t i = 0;
    for (std::int32_t dim : shape) {
        if (dim <= 0) { throw std::invalid_argument("tensor dimensions must be positive"); }
        out[i++] = dim;
    }
    return out;
}

std::int64_t shape_numel(const std::int32_t (&shape)[4]) {
    std::int64_t total = 1;
    for (std::int32_t dim : shape) {
        if (dim <= 0) { throw std::invalid_argument("tensor dimensions must be positive"); }
        total = checked_mul_i64(total, dim);
    }
    return total;
}

std::int64_t shape_numel(const std::array<std::int32_t, 4>& shape) {
    std::int64_t total = 1;
    for (std::int32_t dim : shape) {
        if (dim <= 0) { throw std::invalid_argument("tensor dimensions must be positive"); }
        total = checked_mul_i64(total, dim);
    }
    return total;
}

void set_contiguous_strides(Tensor& tensor) {
    std::int64_t stride = static_cast<std::int64_t>(dtype_size(tensor.dtype));
    tensor.nb[0]        = stride;
    for (int i = 1; i < 4; ++i) {
        stride       = checked_mul_i64(stride, tensor.ne[i - 1]);
        tensor.nb[i] = stride;
    }
}

} // namespace

Tensor::Tensor(void* data_in, DType dtype_in, std::initializer_list<std::int32_t> shape)
    : data(data_in), dtype(dtype_in) {
    const auto normalized = normalize_shape(shape);
    for (int i = 0; i < 4; ++i) { ne[i] = normalized[i]; }
    set_contiguous_strides(*this);
}

std::int64_t Tensor::numel() const { return shape_numel(ne); }

std::size_t Tensor::bytes() const {
    const std::int64_t elements = numel();
    const std::size_t elem_size = dtype_size(dtype);
    if (elements < 0 ||
        static_cast<unsigned long long>(elements) >
            static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("tensor element count overflows size_t");
    }
    const auto unsigned_elements = static_cast<std::size_t>(elements);
    if (elem_size != 0 && unsigned_elements > std::numeric_limits<std::size_t>::max() / elem_size) {
        throw std::overflow_error("tensor byte size overflows size_t");
    }
    return unsigned_elements * elem_size;
}

bool Tensor::is_contiguous() const {
    std::int64_t expected = static_cast<std::int64_t>(dtype_size(dtype));
    for (int i = 0; i < 4; ++i) {
        if (ne[i] <= 0) { return false; }
        if (ne[i] != 1 && nb[i] != expected) { return false; }
        if (i < 3 && ne[i] != 1) {
            try {
                expected = checked_mul_i64(expected, ne[i]);
            } catch (const std::overflow_error&) { return false; }
        }
    }
    return true;
}

Tensor Tensor::view(std::initializer_list<std::int32_t> shape) const {
    if (!is_contiguous()) { throw std::invalid_argument("view requires a contiguous tensor"); }

    const auto normalized = normalize_shape(shape);
    if (shape_numel(normalized) != numel()) {
        throw std::invalid_argument("view element count mismatch");
    }

    return Tensor(data, dtype, shape);
}

Tensor Tensor::reshape(std::initializer_list<std::int32_t> shape) const { return view(shape); }

Tensor Tensor::slice(int dim, std::int32_t start, std::int32_t len) const {
    if (dim < 0 || dim >= 4) { throw std::invalid_argument("slice dim out of range"); }
    if (start < 0 || len <= 0 || start > ne[dim] || len > ne[dim] - start) {
        throw std::invalid_argument("slice range out of bounds");
    }

    Tensor out                = *this;
    const std::int64_t offset = checked_mul_i64(static_cast<std::int64_t>(start), out.nb[dim]);
    auto* bytes_ptr           = static_cast<unsigned char*>(out.data);
    if (bytes_ptr != nullptr) { bytes_ptr += offset; }
    out.data    = bytes_ptr;
    out.ne[dim] = len;
    return out;
}

Tensor Tensor::permute(std::initializer_list<int> order) const {
    if (order.size() != 4) { throw std::invalid_argument("permute requires four dimensions"); }

    bool seen[4] = {false, false, false, false};
    int dims[4]  = {0, 1, 2, 3};
    int i        = 0;
    for (int dim : order) {
        if (dim < 0 || dim >= 4 || seen[dim]) {
            throw std::invalid_argument("invalid permutation");
        }
        seen[dim] = true;
        dims[i++] = dim;
    }

    Tensor out = *this;
    for (int j = 0; j < 4; ++j) {
        out.ne[j] = ne[dims[j]];
        out.nb[j] = nb[dims[j]];
    }
    return out;
}

} // namespace ninfer
