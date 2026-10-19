#pragma once

#include "models/common/weights.h"
#include "artifact/binder.h"
#include "artifact/encoded_slice.h"

namespace ninfer::models {

struct IndexRange {
    std::uint32_t begin = 0, size = 0;
    friend bool operator==(IndexRange, IndexRange) = default;
};

[[nodiscard]] IndexRange partition_range(std::uint32_t total, std::uint32_t parts,
                                         std::uint32_t rank);

// Matrix coordinates use the parameter's logical Binding, independent of its source parents.
struct ParameterSlice {
    WeightId parameter;
    std::uint64_t row = 0, column = 0, rows = 0, columns = 0;
    std::uint64_t destination_row = 0, destination_column = 0;
};

struct ParameterAssembly {
    std::vector<std::uint64_t> shape;
    std::vector<ParameterSlice> slices;

    struct Binding {
        WeightId parameter;
        std::vector<std::uint64_t> shape;
        std::uint64_t begin = 0, end = 0;
    };

    std::vector<Binding> bindings;
};

void append_parameter_slice(std::vector<artifact::MatrixSlice>&, const artifact::Reader&,
                            const artifact::ParameterReference&, const ParameterSlice&);

} // namespace ninfer::models
