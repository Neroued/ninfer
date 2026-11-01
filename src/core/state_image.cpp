#include "core/state_image.h"

#include "core/transfer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {

constexpr std::size_t kAlignment = 256;

std::size_t product(std::size_t a, std::size_t b) {
    if (b != 0 && a > static_cast<std::size_t>(INT64_MAX) / b) {
        throw std::overflow_error("StateImage extent exceeds int64");
    }
    return a * b;
}

} // namespace

TransferWork StateImageLayout::transfer_work() const noexcept {
    return {.payload_bytes = image_bytes, .copy_operations = 1};
}

StateField StateImageBuilder::add(DType dtype, std::initializer_list<std::int32_t> shape,
                                  std::uint32_t layers, std::string_view label) {
    if (shape.size() == 0 || shape.size() > 4 || layers == 0 || fields_.size() >= UINT32_MAX ||
        std::any_of(shape.begin(), shape.end(), [](auto extent) { return extent <= 0; })) {
        throw std::invalid_argument(
            "StateImage field requires 1..4 inner axes and positive layers");
    }
    LayoutBuilder field;
    TensorRegion tensor  = field.add_tensor(dtype, shape, kAlignment, label);
    const auto stride    = field.finish(kAlignment, label);
    const auto region    = image_.add(product(stride, layers), kAlignment, label);
    tensor.region.offset = region.offset;
    const StateField id{static_cast<std::uint32_t>(fields_.size())};
    fields_.push_back({tensor, layers, stride});
    return id;
}

LayoutRegion StateImageBuilder::group(StateField first, StateField last) const {
    if (first.index > last.index || last.index >= fields_.size()) {
        throw std::out_of_range("StateImage field group is out of range");
    }
    const auto begin = fields_[first.index].tensor.region.offset;
    const auto& end  = fields_[last.index];
    return {begin, end.tensor.region.offset + product(end.layer_stride, end.layers) - begin,
            kAlignment};
}

StateImageLayout StateImageBuilder::finish() const {
    if (fields_.empty()) { throw std::invalid_argument("StateImage requires fields"); }
    return {image_.finish(kAlignment, "StateImage"), kAlignment, fields_};
}

StateImageStorage::StateImageStorage(MemorySpan backing, const StateImageLayout& layout)
    : layout_(layout), storage_(backing) {
    if (layout.image_bytes == 0 || layout.fields.empty()) {
        throw std::invalid_argument("StateImage geometry is empty");
    }
    const auto record = LayoutRegion{0, layout.image_bytes, layout.alignment}.bind(backing);
    std::size_t end   = 0;
    for (const auto& field : layout.fields) {
        if (field.layers == 0 || field.tensor.region.offset < end ||
            field.layer_stride < field.tensor.region.bytes) {
            throw std::invalid_argument("StateImage fields overlap or have invalid geometry");
        }
        const auto extent = product(field.layer_stride, field.layers);
        if (field.tensor.region.offset > layout.image_bytes ||
            extent > layout.image_bytes - field.tensor.region.offset) {
            throw std::invalid_argument("StateImage field exceeds its image");
        }
        const auto tensor = field.bind(record);
        if (field.layer_stride % field.tensor.region.alignment != 0 ||
            field.layer_stride % tensor.nb[0] != 0) {
            throw std::invalid_argument("StateImage field layer stride is misaligned");
        }
        end = field.tensor.region.offset + extent;
    }
}

StateFieldView StateImageStorage::field(StateField id, const Tensor& byte_offsets) const {
    return {{storage_, byte_offsets}, layout_.fields.at(id.index)};
}

StateFieldView StateImageStorage::field(StateField id, const Tensor& byte_offsets,
                                        std::uint32_t layer) const {
    return field(id, byte_offsets).layer(layer);
}

Tensor StateImageStorage::tensor(StateField id, std::int64_t byte_offset,
                                 std::uint32_t layer) const {
    return layout_.fields.at(id.index).layer(layer).bind(image(byte_offset));
}

std::int64_t StateImageStorage::layer_stride(StateField field) const {
    return static_cast<std::int64_t>(layout_.fields.at(field.index).layer_stride);
}

MemorySpan StateImageStorage::image(std::int64_t byte_offset) const {
    if (byte_offset < 0) { throw std::out_of_range("StateImage record offset is negative"); }
    return LayoutRegion{static_cast<std::size_t>(byte_offset), image_bytes(), layout_.alignment}
        .bind(storage_);
}

void StateImageStorage::clear(std::int64_t byte_offset, CudaExecutionView execution) const {
    const auto memory = image(byte_offset);
    if (memory.device != execution.device()) {
        throw std::invalid_argument("StateImage clearing requires its CUDA execution device");
    }
    CudaDeviceScope scope(execution.device());
    CUDA_CHECK(cudaMemsetAsync(memory.data, 0, memory.bytes, execution.stream()));
}

void StateImageStorage::copy(std::int64_t source, std::int64_t destination,
                             CudaExecutionView execution) const {
    copy_bytes_async(image(destination), image(source), execution);
}

void StateImageStorage::copy_group(LayoutRegion group, std::int64_t source,
                                   std::int64_t destination, CudaExecutionView execution) const {
    copy_bytes_async(group.bind(image(destination)), group.bind(image(source)), execution);
}

void StateImageStorage::copy_to_host(std::int64_t source, MemorySpan destination,
                                     CudaExecutionView execution) const {
    if (destination.device != Device::cpu()) {
        throw std::invalid_argument("StateImage Host destination must be CPU memory");
    }
    copy_bytes_async(destination, image(source), execution);
}

void StateImageStorage::copy_from_host(ConstMemorySpan source, std::int64_t destination,
                                       CudaExecutionView execution) const {
    if (source.device != Device::cpu()) {
        throw std::invalid_argument("StateImage Host source must be CPU memory");
    }
    copy_bytes_async(image(destination), source, execution);
}

} // namespace ninfer
