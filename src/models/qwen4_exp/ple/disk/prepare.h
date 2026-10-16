#pragma once

#include "models/qwen4_exp/ple/input.h"
#include "models/qwen4_exp/ple/disk/row_reader.h"
#include "core/cpu/system.h"

#include <optional>

namespace ninfer::models::qwen4_exp::ple::disk {

struct PrepareOptions {
    std::uint32_t max_positions = 0; // Maximum T*B, not max(T)*max(B).
    std::uint32_t io_depth      = 128;
    std::size_t cache_bytes     = 64ULL * 1024 * 1024;
    CpuAffinity affinity;
};

enum class PrepareStatus { Preparing, HostReady, Cancelled, Failed };

struct WindowSnapshot {
    InputWindow input;
    PrepareStatus status;
    bool drained;
    std::exception_ptr error;
};

// Program-owned disk preparation service with one I/O worker. Borrows the immutable table
// and disjoint CPU outputs through destruction; each output holds max_positions*2560 BF16
// words and supplies one window slot. Control calls come from one Program thread.
// IDs/identities are copied before returning. The worker hashes, reads, decodes and publishes
// output; it never calls CUDA. The caller owns transfer/consumer ordering for borrowed outputs.
class PrepareService {
public:
    PrepareService(const Table&, std::int32_t eos, PrepareOptions,
                   std::span<const MemorySpan> outputs);
    ~PrepareService();
    PrepareService(const PrepareService&)            = delete;
    PrepareService& operator=(const PrepareService&) = delete;

    // B is rows.size(), in [1,8]. No free slot returns nullopt. Optional prefetch leaves
    // room for a required window; prioritize promotes it when it becomes the current input.
    [[nodiscard]] std::optional<Ticket>
    prepare(std::uint32_t tokens, std::span<const InputRow> rows, bool required = true);
    void prioritize(Ticket);
    void append(Ticket, std::span<const Append>);
    // Final lengths do not exceed known prefixes. Sealing truncates candidates for this
    // input window; it does not commit model state. Inactive E columns become exact +0.
    void seal(Ticket, std::span<const std::uint32_t> valid_columns);
    void cancel(Ticket);
    [[nodiscard]] WindowSnapshot poll(Ticket) const;
    [[nodiscard]] HostEmbedding embedding(Ticket) const;
    // Requires a terminal, Host-drained window and no external users of its output.
    // Invalidates the handle immediately; worker reclamation makes its slot available again.
    // Preparing must first be cancelled, including when all currently known rows are ready.
    void release(Ticket);

    [[nodiscard]] ReadStats stats() const;
    // Owned Host array capacities, excluding borrowed outputs and allocator/system overhead.
    [[nodiscard]] std::size_t memory_bytes() const noexcept;
    [[nodiscard]] static std::size_t memory_bytes(TableGeometry, const PrepareOptions&,
                                                  std::uint32_t slots);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp::ple::disk
