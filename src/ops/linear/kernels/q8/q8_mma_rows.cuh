#pragma once

namespace ninfer::ops::detail {
template <class Schedule>
struct Q8MmaIdentityRows {
    static constexpr int kOutputRowsPerCta = Schedule::kBlockRows;
    static constexpr int kTmaRowsPerPanel  = Schedule::kBlockRows;

    __device__ __forceinline__ int weight_row(int row0, int local_row, int) const {
        return row0 + local_row;
    }
};

} // namespace ninfer::ops::detail
