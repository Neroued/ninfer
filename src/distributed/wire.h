#pragma once

#include "distributed/topology.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
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
    Measurement,
    MeasurementComplete,
    Observation,
    InputBegin,
    InputChunk,
    InputEnd,
    InputReady,
    InputRelease,
    StartupProgress,
    Submit = 32,
    Supply,
    Commit,
    Discard,
    Event,
};

// Moving a received payload retains its receive-buffer reservation, including when startup
// keeps the bytes after consuming the message envelope. Payload size is fixed once published.
class Payload {
public:
    Payload() = default;

    Payload(Bytes bytes, std::shared_ptr<void> reservation = {})
        : reservation_(std::move(reservation)), bytes_(std::move(bytes)) {}

    Payload(Payload&&) noexcept = default;

    Payload& operator=(Payload&& other) noexcept {
        Payload replacement(std::move(other));
        bytes_.swap(replacement.bytes_);
        reservation_.swap(replacement.reservation_);
        return *this;
    }

    Payload(const Payload&)            = delete;
    Payload& operator=(const Payload&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return bytes_.size(); }

    [[nodiscard]] bool empty() const noexcept { return bytes_.empty(); }

    void consume_prefix(std::size_t bytes) {
        if (bytes > bytes_.size()) throw std::out_of_range("payload prefix exceeds its size");
        bytes_.erase(bytes_.begin(), bytes_.begin() + bytes);
    }

    [[nodiscard]] auto data() noexcept { return bytes_.data(); }

    [[nodiscard]] auto data() const noexcept { return bytes_.data(); }

    [[nodiscard]] auto begin() noexcept { return bytes_.begin(); }

    [[nodiscard]] auto begin() const noexcept { return bytes_.begin(); }

    [[nodiscard]] auto end() noexcept { return bytes_.end(); }

    [[nodiscard]] auto end() const noexcept { return bytes_.end(); }

    [[nodiscard]] auto& back() const { return bytes_.back(); }

    friend bool operator==(const Payload& a, const Bytes& b) { return a.bytes_ == b; }

    friend bool operator==(const Payload& a, const Payload& b) { return a.bytes_ == b.bytes_; }
private:
    std::shared_ptr<void> reservation_;
    Bytes bytes_;
};

struct Message {
    Kind kind;
    std::uint64_t epoch = 0;
    Payload payload;
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
void encode(Encoder&, const RankIdentity&);
[[nodiscard]] RankIdentity decode_rank_identity(Decoder&);
void encode(Encoder&, const ExecutionTopology&);
[[nodiscard]] ExecutionTopology decode_topology(Decoder&);

} // namespace ninfer::distributed::wire
