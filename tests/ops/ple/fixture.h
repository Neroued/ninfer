#pragma once

#include "core/buffer.h"
#include "core/record_view.h"
#include "ops/op_tester.h"

#include <array>
#include <memory>

namespace ninfer::test::ple {

inline constexpr int H = 2560, C = 4, D = H * C;
inline constexpr float epsilon = 1e-6F;

// Test-owned logical indexing; gaps and guards are retained in host snapshots.
struct Words {
    Tensor tensor;
    std::size_t leading;
    std::vector<std::uint16_t> host;
    std::unique_ptr<GuardedDeviceBuffer> device;

    Words(std::initializer_list<std::int32_t> shape, int leading_words = 0,
          std::array<std::int64_t, 4> word_strides = {})
        : tensor(Tensor::metadata(DType::BF16, shape)), leading(leading_words) {
        if (word_strides[0])
            for (int d = 0; d < 4; ++d) tensor.nb[d] = 2 * word_strides[d];
        std::size_t span = 1;
        for (int d = 0; d < 4; ++d) span += (tensor.ne[d] - 1) * (tensor.nb[d] / 2);
        host.resize(leading + span, 0x7fa1);
        device        = std::make_unique<GuardedDeviceBuffer>(host.size() * 2);
        tensor.data   = static_cast<std::uint16_t*>(device->data()) + leading;
        tensor.device = device->device();
    }

    std::size_t index(int x, int y = 0, int z = 0, int w = 0) const {
        return leading + x * (tensor.nb[0] / 2) + y * (tensor.nb[1] / 2) + z * (tensor.nb[2] / 2) +
               w * (tensor.nb[3] / 2);
    }

    void fill(std::uint32_t seed, float scale = 1.f, bool raw_words = false) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> values(-scale, scale);
        for (int w = 0; w < tensor.ne[3]; ++w)
            for (int z = 0; z < tensor.ne[2]; ++z)
                for (int y = 0; y < tensor.ne[1]; ++y)
                    for (int x = 0; x < tensor.ne[0]; ++x)
                        host[index(x, y, z, w)] =
                            raw_words ? std::uint16_t(rng()) : f32_to_bf16(values(rng));
    }

    void upload() const {
        cuda_check(cudaMemcpy(device->data(), host.data(), host.size() * 2, cudaMemcpyHostToDevice),
                   "PLE fixture upload");
    }

    std::vector<std::uint16_t> download() const {
        return from_device<std::uint16_t>(device->data(), host.size());
    }

    int preserved(const char* label) const {
        return verify_exact(label, download(), host) + device->verify_guards(label);
    }
};

inline std::array<std::int64_t, 4> column_strides(int columns, int outer, int layout) {
    if (layout == 0) return {1, D, std::int64_t(D) * columns, std::int64_t(D) * columns * outer};
    if (layout == 1) return {1, D + 3, std::int64_t(D + 3) * columns + 5, 1};
    // Slot/batch interleaving is also a legal positive-stride view.
    return {1, std::int64_t(D + 8) * outer + 16, D + 8, 1};
}

inline std::array<std::int64_t, 4> history_strides(int records, int layout) {
    const auto stride = std::int64_t(D) * 9 + (layout ? 23 : 0);
    return {1, D, stride, stride * records};
}

inline StateFieldView history_field(const Tensor& values, MemorySpan storage, Tensor offsets) {
    const auto bytes = std::size_t(D) * 9 * 2;
    const auto offset =
        static_cast<std::byte*>(values.data) - static_cast<std::byte*>(storage.data);
    const std::size_t alignment =
        (reinterpret_cast<std::uintptr_t>(values.data) % 4 == 0 && values.nb[2] % 4 == 0) ? 4 : 2;
    return {{storage, offsets},
            {{{std::size_t(offset), bytes, alignment}, DType::BF16, {D, 9, 1, 1}}, 1, bytes}};
}

inline std::vector<std::int64_t> history_offsets(const Tensor& values,
                                                 const std::vector<int>& indices) {
    std::vector<std::int64_t> result;
    for (int index : indices) result.push_back(index < 0 ? -1 : index * values.nb[2]);
    return result;
}

inline void upload_offsets(DeviceBuffer& destination, const Tensor& values,
                           const std::vector<int>& indices) {
    const auto offsets = history_offsets(values, indices);
    destination.copy_from_host(offsets.data(), offsets.size() * sizeof(std::int64_t));
}

inline double value(const Words& words, int x, int y = 0, int z = 0, int w = 0) {
    return bf16_to_f32(words.host[words.index(x, y, z, w)]);
}

inline void upload_i32(DeviceBuffer& dst, const std::vector<int>& src) {
    dst.copy_from_host(src.data(), src.size() * sizeof(int));
}

template <class F>
int rejects(const char* label, F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": did not reject invalid operands\n";
    return 1;
}

} // namespace ninfer::test::ple
