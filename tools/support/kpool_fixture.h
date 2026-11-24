#pragma once

#include "core/buffer.h"
#include "core/tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <vector>

namespace ninfer::tool::kpool {
inline float decode(std::uint16_t x) { return std::bit_cast<float>(std::uint32_t(x) << 16); }

inline std::uint16_t encode(float x) {
    auto u = std::bit_cast<std::uint32_t>(x);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}

inline std::uint32_t hash(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

inline std::uint16_t sample(std::uint32_t x, float scale) {
    return encode((int(hash(x) % 1025) - 512) * scale);
}

template <class T>
struct Buffer {
    Tensor view;
    std::vector<T> host;
    DeviceBuffer storage;

    Buffer(Device device, DType type, std::initializer_list<int> shape, bool padded = false)
        : view(Tensor::metadata(type, shape)) {
        if (padded)
            for (int d = 1; d < 4; ++d)
                view.nb[d] = view.nb[d - 1] * view.ne[d - 1] + 32 * sizeof(T);
        std::size_t count = 1;
        for (int d = 0; d < 4; ++d) count += (view.ne[d] - 1) * view.nb[d] / sizeof(T);
        host.resize(count, T(0xa5a5));
        storage = DeviceBuffer(device, count * sizeof(T) + 512);
        storage.fill(0xa5);
        view.data   = static_cast<std::byte*>(storage.data()) + 256;
        view.device = device;
    }

    std::size_t index(int x, int y = 0, int z = 0, int w = 0) const {
        return (x * view.nb[0] + y * view.nb[1] + z * view.nb[2] + w * view.nb[3]) / sizeof(T);
    }

    T& at(int x, int y = 0, int z = 0, int w = 0) { return host[index(x, y, z, w)]; }

    T at(int x, int y = 0, int z = 0, int w = 0) const { return host[index(x, y, z, w)]; }

    void upload() { storage.copy_from_host(host.data(), host.size() * sizeof(T), 256); }

    std::vector<T> download() const {
        std::vector<T> result(host.size());
        storage.copy_to_host(result.data(), result.size() * sizeof(T), 256);
        return result;
    }

    bool guards() const {
        std::array<std::byte, 256> prefix, suffix;
        storage.copy_to_host(prefix.data(), 256);
        storage.copy_to_host(suffix.data(), 256, storage.size() - 256);
        const auto ok = [](auto x) { return x == std::byte{0xa5}; };
        return std::all_of(prefix.begin(), prefix.end(), ok) &&
               std::all_of(suffix.begin(), suffix.end(), ok);
    }
};

} // namespace ninfer::tool::kpool
