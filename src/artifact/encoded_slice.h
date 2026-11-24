#pragma once

#include "artifact/schema.h"
#include "core/weight_view.h"

#include <optional>

namespace ninfer::artifact {

class Reader;

struct StorageId {
    std::size_t index                            = 0;
    friend bool operator==(StorageId, StorageId) = default;
};

// A rectangular selection in flattened matrix rows. A vector is one row.
struct MatrixSlice {
    ObjectHandle object;
    std::uint64_t row = 0, column = 0, rows = 0, columns = 0;
    std::uint64_t destination_row = 0, destination_column = 0;
};

// Repeated encoded byte ranges, without decoding or changing scale/divisor domains.
struct EncodedCopy {
    ObjectHandle object;
    std::uint64_t source = 0, destination = 0, bytes = 0;
    std::uint64_t count = 1, source_stride = 0, destination_stride = 0;
    friend bool operator==(const EncodedCopy&, const EncodedCopy&) = default;
};

struct EncodedTensor {
    WeightGeometry geometry;
    std::vector<EncodedCopy> copies;
    std::optional<ObjectHandle> whole_object;
};

[[nodiscard]] EncodedTensor whole_tensor(const Reader&, ObjectHandle);
// Slices preserve represented values and cover the destination exactly. The compiler repacks
// scale tiles, refines block-FP8 scales when needed, and preserves matrix divisor domains.
[[nodiscard]] EncodedTensor slice_tensor(const Reader&, std::span<const std::uint64_t> shape,
                                         std::span<const MatrixSlice>);

} // namespace ninfer::artifact
