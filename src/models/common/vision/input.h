#pragma once

#include "core/memory.h"
#include "distributed/wire.h"
#include "models/common/vision/window.h"

namespace ninfer::models::vision {

// One receive slot borrowing rank-owned pinned staging. Release follows the execution ticket
// that uploaded these bytes, independently of the output window's later consumers.
class InputReceiver {
public:
    void bind(MemorySpan);
    void begin(std::uint64_t id, std::span<const std::size_t> parts);
    void append(std::uint64_t id, std::size_t part, std::size_t offset, std::span<const std::byte>);
    [[nodiscard]] std::vector<ConstMemorySpan> finish(std::uint64_t id) const;
    void release(std::uint64_t id);
private:
    struct Part {
        std::size_t offset, bytes, received;
    };

    MemorySpan storage_;
    std::vector<Part> parts_;
    std::uint64_t id_ = 0;
};

void encode_window(distributed::wire::Encoder&, const WindowUse&);
[[nodiscard]] WindowUse decode_window(distributed::wire::Decoder&);

} // namespace ninfer::models::vision
