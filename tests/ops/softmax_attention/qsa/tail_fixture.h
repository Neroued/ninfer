#pragma once

#include "ops/op_tester.h"
#include "ninfer/ops/qsa.h"

#include <array>
#include <memory>
#include <type_traits>

namespace ninfer::test::qsa {

inline constexpr int kFeatures = 128;

template <class Word>
struct Plane {
    Tensor view;
    std::size_t leading;
    std::vector<Word> host;
    std::unique_ptr<GuardedDeviceBuffer> device;

    Plane(int features, int columns, int outer, int layout, bool state = false,
          bool coordinates = false)
        : view(Tensor::metadata(std::is_same_v<Word, std::uint16_t> ? DType::BF16 : DType::I32,
                                {features, columns, outer})),
          leading(layout == 1 ? 1 : 0) {
        if (state) {
            const int axis = coordinates ? 1 : 2;
            if (layout == 3) {
                leading = std::size_t(features) * (coordinates ? 1 : columns) * 5;
                view.nb[axis] *= 12;
            } else if (layout) {
                view.nb[axis] += (layout == 1 ? 5 : 17) * sizeof(Word);
            }
        } else if (layout == 1) {
            view.nb[1] = (features + 3) * sizeof(Word);
            view.nb[2] = ((features + 3) * columns + 5) * sizeof(Word);
        } else if (layout == 2) {
            view.nb[1] = (std::int64_t(features + 8) * outer + 16) * sizeof(Word);
            view.nb[2] = (features + 8) * sizeof(Word);
        } else if (layout == 3) {
            // A layer view in a twelve-layer slab; preserve gaps and the preceding layers.
            const int axis = outer == 1 ? 1 : 2;
            view.nb[axis] *= 12;
            leading = std::size_t(features) * (axis == 1 ? 1 : columns) * 5;
        }
        std::size_t span = features + (columns - 1) * (view.nb[1] / sizeof(Word)) +
                           (outer - 1) * (view.nb[2] / sizeof(Word));
        host.resize(leading + span);
        device      = std::make_unique<GuardedDeviceBuffer>(host.size() * sizeof(Word));
        view.data   = static_cast<Word*>(device->data()) + leading;
        view.device = device->device();
    }

    std::size_t index(int feature, int column, int outer = 0) const {
        return leading + feature + std::size_t(column) * (view.nb[1] / sizeof(Word)) +
               std::size_t(outer) * (view.nb[2] / sizeof(Word));
    }

    void fill(std::uint32_t seed) {
        for (std::size_t i = 0; i < host.size(); ++i) {
            std::uint32_t x = std::uint32_t(i) + seed;
            x ^= x >> 16;
            x *= 0x7feb352dU;
            x ^= x >> 15;
            host[i] = Word(x & 0x7fffffffU);
        }
        if constexpr (std::is_same_v<Word, std::uint16_t>) {
            constexpr std::uint16_t special[]{0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0x7fa1, 0xffff, 1};
            for (int b = 0; b < view.ne[2]; ++b)
                for (int j = 0; j < view.ne[1]; ++j)
                    for (int c = 0; c < 8; ++c) host[index(c, j, b)] = special[(c + j + b) % 8];
        }
    }

    void upload() const { device->copy_from_host(host.data(), host.size() * sizeof(Word)); }

    std::vector<Word> download() const { return from_device<Word>(device->data(), host.size()); }

    int check(const std::string& label, const std::vector<Word>& expected) const {
        return verify_exact(label.c_str(), download(), expected) + device->verify_guards(label);
    }
};

inline StateFieldView state_field(const Tensor& values, MemorySpan storage, const Tensor& offsets,
                                  bool coordinates = false) {
    const int columns       = coordinates ? 1 : 3;
    const std::size_t bytes = values.ne[0] * columns * dtype_size(values.dtype);
    const auto offset =
        static_cast<std::byte*>(values.data) - static_cast<std::byte*>(storage.data);
    const int axis = coordinates ? 1 : 2;
    const auto alignment =
        (reinterpret_cast<std::uintptr_t>(values.data) % 4 == 0 && values.nb[axis] % 4 == 0) ? 4U
                                                                                             : 2U;
    return {{storage, offsets},
            {{{std::size_t(offset), bytes, alignment}, values.dtype, {values.ne[0], columns, 1, 1}},
             1,
             bytes}};
}

inline std::vector<std::int64_t>
state_offsets(const Tensor& values, const std::vector<int>& indices, bool coordinates = false) {
    std::vector<std::int64_t> result;
    for (int index : indices) result.push_back(index * values.nb[coordinates ? 1 : 2]);
    return result;
}

struct Tail {
    std::array<std::array<std::uint16_t, kFeatures>, 3> raw{};
    std::array<std::int32_t, 3> rope{};
};

// Independent chronological oracle: append one raw token at a time and discard complete
// four-token blocks. No direct final-column selection or production helper is used.
inline Tail reference(const Plane<std::uint16_t>& before, const Plane<std::int32_t>& first,
                      const Plane<std::uint16_t>& records, const Plane<std::int32_t>& positions,
                      int source, int row, int base, int accepted) {
    std::vector<std::array<std::uint16_t, kFeatures>> pending(base % 4);
    for (int j = 0; j < base % 4; ++j)
        for (int c = 0; c < kFeatures; ++c) pending[j][c] = before.host[before.index(c, j, source)];
    Tail result;
    if (!pending.empty())
        for (int axis = 0; axis < 3; ++axis)
            result.rope[axis] = first.host[first.index(axis, source)];
    for (int i = 0; i < accepted; ++i) {
        if (pending.empty())
            for (int axis = 0; axis < 3; ++axis)
                result.rope[axis] = positions.host[positions.index(axis, i, row)];
        std::array<std::uint16_t, kFeatures> token;
        for (int c = 0; c < kFeatures; ++c) token[c] = records.host[records.index(c, i, row)];
        pending.push_back(token);
        if (pending.size() == 4) {
            pending.clear();
            result.rope.fill(0);
        }
    }
    for (std::size_t j = 0; j < pending.size(); ++j) result.raw[j] = pending[j];
    return result;
}

} // namespace ninfer::test::qsa
