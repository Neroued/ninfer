#pragma once

#include "distributed/topology.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer::distributed::wire {

using Bytes                               = std::vector<std::byte>;
inline constexpr std::size_t kHeaderBytes = 20;

enum class Kind : std::uint32_t {
    Join = 1,
    Topology,
    GroupId,
    Connected,
    Prepare,
    Prepared,
    Enter,
    Allocated,
    Ready,
    Shutdown,
    Stopped,
    Failed,
    Progress,
    MemoryQuery,
    MemoryReport,
    InputBegin,
    InputChunk,
    InputEnd,
    InputReady,
    InputRelease,
    Submit = 32,
    Supply,
    Commit,
    Discard,
    Event,
};

struct Message {
    Kind kind;
    std::uint64_t epoch = 0;
    Bytes payload;
};

class Encoder {
public:
    void u8(std::uint8_t);
    void u16(std::uint16_t);
    void u32(std::uint32_t);
    void u64(std::uint64_t);
    void i32(std::int32_t);
    void i64(std::int64_t);
    void f32(float);
    void f64(double);
    void boolean(bool);
    void string(std::string_view);
    void bytes(std::span<const std::byte>);
    void raw(std::span<const std::byte>);
    [[nodiscard]] Bytes take() &&;
private:
    Bytes bytes_;
};

class Decoder {
public:
    explicit Decoder(std::span<const std::byte> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t u8();
    [[nodiscard]] std::uint16_t u16();
    [[nodiscard]] std::uint32_t u32();
    [[nodiscard]] std::uint64_t u64();
    [[nodiscard]] std::int32_t i32();
    [[nodiscard]] std::int64_t i64();
    [[nodiscard]] float f32();
    [[nodiscard]] double f64();
    [[nodiscard]] bool boolean();
    [[nodiscard]] std::string string();
    [[nodiscard]] std::span<const std::byte> bytes();
    [[nodiscard]] std::span<const std::byte> raw(std::size_t);
    // Every encoded element consumes at least minimum_bytes. Check before allocating a vector.
    [[nodiscard]] std::size_t count(std::size_t minimum_bytes = 1);

    [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size(); }

    void finish() const;
private:
    std::span<const std::byte> bytes_;
};

struct Header {
    Kind kind;
    std::uint64_t epoch, bytes;
};

[[nodiscard]] Bytes encode_header(const Message&);
[[nodiscard]] Header decode_header(std::span<const std::byte>);
void encode(Encoder&, const ExecutionTopology&);
[[nodiscard]] ExecutionTopology decode_topology(Decoder&);

} // namespace ninfer::distributed::wire
