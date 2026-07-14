#include "runtime/engine/product_cookie.h"

#include <atomic>
#include <limits>
#include <stdexcept>

namespace ninfer::runtime {

ProductCookie allocate_product_cookie() {
    static std::atomic<ProductCookie> next{1};
    ProductCookie current = next.load(std::memory_order_relaxed);
    for (;;) {
        if (current == std::numeric_limits<ProductCookie>::max()) {
            throw std::overflow_error("NInfer product cookie space is exhausted");
        }
        if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {
            return current;
        }
    }
}

} // namespace ninfer::runtime
