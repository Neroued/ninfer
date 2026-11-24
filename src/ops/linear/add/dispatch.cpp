#include "ops/linear/add/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
LinearAddLaunch select_q5_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (tokens > 0) {
        if (rows == 5120 && k == 6144) return select_q5_linear_add_n5120_k6144(tokens);
        if (rows == 5120 && k == 17408) return select_q5_linear_add_n5120_k17408(tokens);
    }
    throw std::invalid_argument("q5 linear_add: unsupported shape or token extent");
}

LinearAddLaunch select_q8_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (tokens > 0) {
        if (rows == 2048 && k == 4096) return select_q8_linear_add_n2048_k4096(tokens);
        if (rows == 2048 && k == 6144) return select_q8_linear_add_n2048_k6144(tokens);
        if (rows == 5120 && k == 6144) return select_q8_linear_add_n5120_k6144(tokens);
        if (rows == 5120 && k == 17408) return select_q8_linear_add_n5120_k17408(tokens);
    }
    throw std::invalid_argument("q8 linear_add: unsupported shape or token extent");
}

} // namespace ninfer::ops::detail
