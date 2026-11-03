#pragma once

#include "artifact/encoded_slice.h"

#include <functional>

namespace ninfer::artifact {

inline constexpr std::size_t kCsfRepackBufferBytes  = 1024 * 1024;
inline constexpr std::size_t kCsfRepackScratchBytes = 4 * kCsfRepackBufferBytes + 192 * 1024;

[[nodiscard]] EncodedTensor select_csf_tensor(const Reader&, std::span<const std::uint64_t> shape,
                                              std::span<const MatrixSlice>);
[[nodiscard]] std::uint64_t csf_repack_bytes(const WeightGeometry&);
// Produces only compressed scale planes. Codes, row bases and matrix scalars use EncodedCopy.
void repack_csf_scales(const Reader&, const EncodedTensor&,
                       const std::function<void(std::uint64_t, std::span<const std::byte>)>& write);

} // namespace ninfer::artifact
