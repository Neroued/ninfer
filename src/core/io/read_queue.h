#pragma once

#include "core/io/file.h"
#include "core/memory.h"

#include <memory>

namespace ninfer::io {

struct DirectRead {
    std::uint32_t file_index;
    std::uint32_t buffer_index;
    std::uint64_t file_offset;
    std::uint32_t bytes;
    std::uint64_t tag;
};

struct ReadCompletion {
    std::uint64_t tag;
    std::uint32_t buffer_index;
    std::uint32_t bytes;
    int error; // Positive errno, or zero. A successful read may be short, including zero bytes.
};

enum class ReadAdmission { Accepted, Full, Closed };
enum class ReadWake { Completion, Signalled };

// Single-owner-thread queue; only wake() may be called concurrently. Files and nonoverlapping
// CPU buffers are borrowed through destruction. Construction registers them once with io_uring.
// No worker or CUDA resource is created. Each Accepted request reserves its buffer until poll
// returns its one completion. The caller finishes consuming those bytes before reusing it.
// Accepted prepares an SQE; submit issues it. Invalid requests throw; Full/Closed are flow control.
class DirectReadQueue {
public:
    DirectReadQueue(std::span<const FileView> files, std::span<const MemorySpan> buffers,
                    std::uint32_t capacity);
    ~DirectReadQueue();
    DirectReadQueue(const DirectReadQueue&)            = delete;
    DirectReadQueue& operator=(const DirectReadQueue&) = delete;

    [[nodiscard]] ReadAdmission try_enqueue(DirectRead request);
    void submit();
    [[nodiscard]] std::size_t poll(std::span<ReadCompletion> output);
    // Waits for a completion or wake(), including when there are no pending reads.
    // Signalled may coincide with completions: inspect both commands and poll after waking.
    [[nodiscard]] ReadWake wait();
    void wake();

    // Close submits accepted work and rejects new admission. Poll delivers its completions;
    // drain instead discards them and waits for all buffer writers. Destruction closes/drains.
    // If teardown cannot establish that writers have ended, it terminates rather than release
    // borrowed storage still in use. Ordinary I/O failures are completion.error, not this case.
    void close();
    void drain();
    [[nodiscard]] std::size_t outstanding() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::io
