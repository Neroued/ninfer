#include "ops/moe/kernels/cpu/groupwise_int/avx512.h"
#include "ops/moe/kernels/cpu/groupwise_int/compute.h"

namespace ninfer::ops::detail::moe_cpu::groupwise_int {
const Kernel avx512 = make_kernel<Avx512>();

} // namespace ninfer::ops::detail::moe_cpu::groupwise_int
