#pragma once

// Teacher-forcing dump (tfdump/1 layout) for qwen4_exp. Off by default; enabled by:
//
//   NINFER_TFDUMP=<dir>    root output directory (unset = off)
//
// On-disk layout under <dir>:
//   hidden_L00.bin    compact records {i32 first_position, i32 columns, i32 rows, bf16[rows][columns]}
//   positions.npy     int32 [T]   token positions (0..T-1)
//   tokens.npy        int64 [T]   token ids
//   meta.json
//
// This is a null-checked, additive consumer: when NINFER_TFDUMP is unset the
// program holds a null pointer and the hooks add zero overhead.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

class TfDump final {
public:
    // Null when NINFER_TFDUMP is unset.
    [[nodiscard]] static std::unique_ptr<TfDump> from_environment();

    explicit TfDump(std::filesystem::path directory);

    // D2H-copy the tapped layer residual (contiguous BF16 [rows, columns]) and write it to disk.
    // `layer` is the tap index (0 for the final-layer residual). `first_position` is the
    // sequence position of column 0.
    void capture_layer(std::int32_t layer, const Tensor& value, std::int32_t first_position,
                       cudaStream_t stream);

    // Write token ids (int64 [T]) to tokens.npy.
    void write_tokens(std::span<const std::int32_t> tokens);

    // Write meta.json.
    void write_meta(const std::string& producer, const std::string& artifact) const;

    [[nodiscard]] bool enabled() const noexcept { return true; }

private:
    std::filesystem::path directory_;
    std::vector<std::uint16_t> scratch_;
};

} // namespace ninfer::models::qwen4_exp::execution
