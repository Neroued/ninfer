#pragma once

#include <cstdint>

namespace ninfer::runtime {

using ProductCookie = std::uint64_t;

// Allocates a process-local identity for one loaded product. The value is used
// only to reject a PreparedPrompt submitted to a different Engine instance.
ProductCookie allocate_product_cookie();

} // namespace ninfer::runtime
