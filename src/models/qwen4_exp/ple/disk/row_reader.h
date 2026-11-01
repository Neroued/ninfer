#pragma once

#include "models/qwen4_exp/ple/disk/table.h"

#include <exception>

namespace ninfer::models::qwen4_exp::ple::disk {

struct ReadStats {
    std::uint64_t rows         = 0; // Output row demands, including duplicates.
    std::uint64_t cache_hits   = 0; // Unique rows fetched from the persistent raw cache.
    std::uint64_t shared_rows  = 0; // Demands reusing a retained row of a live window.
    std::uint64_t shared_pages = 0; // Fragments joining an already queued/in-flight page.
    std::uint64_t reads        = 0;
    std::uint64_t read_bytes   = 0; // Actual returned bytes, including alignment overhead.
};

// Single-threaded row preparation mechanism. Each owner borrows a contiguous BF16 output;
// destination row i occupies [160*i,160*(i+1)). Each destination is requested at most once.
// Cancel suppresses output writes; release additionally requires all its reads to retire.
// Only wake() is cross-thread. Capacity includes retained rows of all unreleased owners.
class RowReader {
public:
    RowReader(const Table&, std::uint32_t owners, std::uint32_t rows_per_owner,
              std::uint32_t io_depth, std::size_t cache_bytes);
    ~RowReader();
    RowReader(const RowReader&)            = delete;
    RowReader& operator=(const RowReader&) = delete;

    void begin(std::uint32_t owner, std::span<std::uint16_t> output, bool required);
    void request(std::uint32_t owner, std::uint32_t destination, std::uint32_t row);
    void cancel(std::uint32_t owner, std::uint32_t begin, std::uint32_t end);
    void prioritize(std::uint32_t owner);
    void release(std::uint32_t owner);
    void pump();
    [[nodiscard]] bool runnable() const noexcept;
    void wait();
    void wake();

    [[nodiscard]] std::size_t pending(std::uint32_t owner) const;
    [[nodiscard]] bool drained(std::uint32_t owner) const;
    [[nodiscard]] std::exception_ptr error(std::uint32_t owner) const;
    [[nodiscard]] ReadStats stats() const noexcept;
    [[nodiscard]] std::size_t memory_bytes() const noexcept;
    [[nodiscard]] static std::size_t memory_bytes(TableGeometry, std::uint32_t owners,
                                                  std::uint32_t rows_per_owner,
                                                  std::uint32_t io_depth, std::size_t cache_bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp::ple::disk
