#pragma once

#include "core/state_image.h"
#include "../../tools/support/state_record_fixture.h"
#include "ops/op_tester.h"

namespace ninfer::test {

inline StateFieldView dense_record_field(const Tensor& dense, int record_axis,
                                         const Tensor& byte_offsets) {
    TensorRegion region;
    region.dtype = dense.dtype;
    for (int axis = 0; axis < record_axis; ++axis) region.shape[axis] = dense.ne[axis];
    region.region = {0, static_cast<std::size_t>(dense.nb[record_axis]), 16};
    return {{MemorySpan{dense.data, dense.bytes(), dense.device}, byte_offsets},
            StateFieldLayout{region, 1, 0}};
}

// Test-owned reporting around the common physical record fixture.
class RecordStateFixture {
public:
    RecordStateFixture(const Tensor& dense, int record_axis)
        : records_(std::array<Tensor, 1>{dense}, record_axis) {}

    [[nodiscard]] std::int64_t byte_offset(int record) const {
        return records_.byte_offset(record);
    }

    [[nodiscard]] Tensor tensor(int record) const { return records_.field_tensor(0, record); }

    [[nodiscard]] StateFieldView view(const Tensor& offsets = {}) const {
        return records_.field(0, offsets);
    }

    [[nodiscard]] std::vector<std::int64_t>
    record_offsets(std::span<const std::int32_t> records) const {
        return records_.record_offsets(records);
    }

    void upload() const { records_.upload(); }

    void copy_back() const {
        cuda_synchronize();
        records_.copy_back();
    }

    int verify_guards(std::string_view label) const {
        cuda_synchronize();
        if (records_.guards_intact()) return 0;
        std::cerr << label << ": neighbouring state field or record gap overwritten\n";
        return 1;
    }

private:
    tool::StateRecordFixture records_;
};

} // namespace ninfer::test
