#pragma once

namespace ninfer::ops::detail::rmsnorm {
enum class Epilogue { Plain, Offset, SiluGated, SigmoidGated };

constexpr bool gated(Epilogue kind) {
    return kind == Epilogue::SiluGated || kind == Epilogue::SigmoidGated;
}

template <Epilogue Kind>
inline constexpr bool is_gated = gated(Kind);
} // namespace ninfer::ops::detail::rmsnorm
