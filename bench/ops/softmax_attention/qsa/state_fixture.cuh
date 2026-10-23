#pragma once

#include "core/buffer.h"
#include "core/record_view.h"
#include "options.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::bench::qsa {

inline std::vector<int> broadcast(std::vector<int> values, int batch) {
    if (values.size() == 1) values.resize(batch, values[0]);
    if (values.size() != std::size_t(batch)) throw std::invalid_argument("metadata must match B");
    return values;
}

inline std::string csv_list(const std::vector<int>& values) {
    std::string result;
    for (int value : values) {
        if (!result.empty()) result += ':';
        result += std::to_string(value);
    }
    return result;
}

template <class Word>
struct StatePlane {
    DeviceBuffer storage;
    Tensor view;
    std::vector<Word> host;
    std::size_t leading = 0;

    StatePlane(int features, int count, int outer, DType dtype, const std::string& layout,
               bool state_coordinates = false, bool state = false)
        : view(Tensor::metadata(dtype, {features, count, outer})) {
        if (state) {
            const int axis = state_coordinates ? 1 : 2;
            if (layout == "layered")
                view.nb[axis] *= 12;
            else if (layout == "unaligned" && dtype == DType::BF16) {
                leading = 1;
                view.nb[axis] += sizeof(Word);
            }
        } else if (layout == "layered") {
            const int axis = state_coordinates ? 1 : 2;
            view.nb[axis] *= 12;
        } else if (layout == "unaligned" && dtype == DType::BF16) {
            leading    = 1;
            view.nb[1] = (features + 1) * sizeof(Word);
            view.nb[2] = ((features + 1) * count + 1) * sizeof(Word);
        }
        const auto words = leading + features + (count - 1) * (view.nb[1] / sizeof(Word)) +
                           (outer - 1) * (view.nb[2] / sizeof(Word));
        host.resize(words);
        for (std::size_t i = 0; i < words; ++i) host[i] = Word((i * 37 + 101) & 0x7fff);
        storage     = DeviceBuffer(ninfer::bench::device_context().device(), words * sizeof(Word));
        view.data   = static_cast<Word*>(storage.data()) + leading;
        view.device = storage.device();
    }

    StateFieldView field(const Tensor& offsets, bool coordinates = false) {
        const int columns       = coordinates ? 1 : 3;
        const std::size_t bytes = view.ne[0] * columns * sizeof(Word);
        const int axis          = coordinates ? 1 : 2;
        const std::size_t alignment =
            (reinterpret_cast<std::uintptr_t>(view.data) % 4 == 0 && view.nb[axis] % 4 == 0) ? 4
                                                                                             : 2;
        return {
            {storage.span(), offsets},
            {{{leading * sizeof(Word), bytes, alignment}, view.dtype, {view.ne[0], columns, 1, 1}},
             1,
             bytes}};
    }

    std::vector<std::int64_t> offsets(const std::vector<int>& indices,
                                      bool coordinates = false) const {
        std::vector<std::int64_t> result;
        for (int index : indices) result.push_back(index * view.nb[coordinates ? 1 : 2]);
        return result;
    }

    Word& at(int feature, int column, int outer = 0) {
        return host[leading + feature + std::size_t(column) * (view.nb[1] / sizeof(Word)) +
                    std::size_t(outer) * (view.nb[2] / sizeof(Word))];
    }

    void upload() { storage.copy_from_host(host.data(), host.size() * sizeof(Word)); }
};

} // namespace ninfer::bench::qsa
