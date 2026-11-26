#pragma once

namespace ninfer::ops::detail {

enum class Group3Schedule {
    Standard,
    NarrowKeys,
    ParallelKeys,
    ParallelKeysStagedQuery,
};

} // namespace ninfer::ops::detail
