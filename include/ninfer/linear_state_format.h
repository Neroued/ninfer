#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace ninfer {

enum class LinearStateDType : std::uint8_t {
    Float32  = 0,
    BFloat16 = 1,
};

[[nodiscard]] constexpr const char* linear_state_dtype_name(LinearStateDType dtype) {
    switch (dtype) {
    case LinearStateDType::Float32:
        return "fp32";
    case LinearStateDType::BFloat16:
        return "bf16";
    }
    throw std::invalid_argument("unknown linear-state dtype");
}

[[nodiscard]] constexpr LinearStateDType linear_state_dtype_from_option(std::string_view option) {
    if (option == "fp32") return LinearStateDType::Float32;
    if (option == "bf16") return LinearStateDType::BFloat16;
    throw std::invalid_argument("unknown linear-state dtype option");
}

} // namespace ninfer
