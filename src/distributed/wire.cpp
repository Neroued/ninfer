#include "distributed/wire.h"

#include <bit>
#include <stdexcept>
#include <utility>

namespace ninfer::distributed::wire {

void Encoder::u8(std::uint8_t v) { bytes_.push_back(static_cast<std::byte>(v)); }

void Encoder::u16(std::uint16_t v) {
    for (unsigned i = 0; i < 2; ++i) u8(v >> (8 * i));
}

void Encoder::u32(std::uint32_t v) {
    for (unsigned i = 0; i < 4; ++i) u8(v >> (8 * i));
}

void Encoder::u64(std::uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) u8(v >> (8 * i));
}

void Encoder::i32(std::int32_t v) { u32(std::bit_cast<std::uint32_t>(v)); }

void Encoder::i64(std::int64_t v) { u64(std::bit_cast<std::uint64_t>(v)); }

void Encoder::f32(float v) { u32(std::bit_cast<std::uint32_t>(v)); }

void Encoder::f64(double v) { u64(std::bit_cast<std::uint64_t>(v)); }

void Encoder::boolean(bool v) { u8(v ? 1 : 0); }

void Encoder::raw(std::span<const std::byte> v) { bytes_.insert(bytes_.end(), v.begin(), v.end()); }

void Encoder::bytes(std::span<const std::byte> v) {
    u64(v.size());
    raw(v);
}

void Encoder::string(std::string_view v) { bytes(std::as_bytes(std::span(v.data(), v.size()))); }

Bytes Encoder::take() && { return std::move(bytes_); }

std::span<const std::byte> Decoder::raw(std::size_t n) {
    if (n > bytes_.size()) throw std::invalid_argument("truncated distributed message");
    const auto out = bytes_.first(n);
    bytes_         = bytes_.subspan(n);
    return out;
}

std::uint8_t Decoder::u8() { return std::to_integer<std::uint8_t>(raw(1)[0]); }

std::uint16_t Decoder::u16() {
    std::uint16_t v = 0;
    for (unsigned i = 0; i < 2; ++i) v |= std::uint16_t(u8()) << (8 * i);
    return v;
}

std::uint32_t Decoder::u32() {
    std::uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i) v |= std::uint32_t(u8()) << (8 * i);
    return v;
}

std::uint64_t Decoder::u64() {
    std::uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) v |= std::uint64_t(u8()) << (8 * i);
    return v;
}

std::int32_t Decoder::i32() { return std::bit_cast<std::int32_t>(u32()); }

std::int64_t Decoder::i64() { return std::bit_cast<std::int64_t>(u64()); }

float Decoder::f32() { return std::bit_cast<float>(u32()); }

double Decoder::f64() { return std::bit_cast<double>(u64()); }

bool Decoder::boolean() {
    const auto v = u8();
    if (v > 1) throw std::invalid_argument("invalid wire boolean");
    return v != 0;
}

std::size_t Decoder::count(std::size_t minimum) {
    const auto n = u64();
    if (!minimum || n > remaining() / minimum)
        throw std::invalid_argument("distributed array exceeds its payload");
    return static_cast<std::size_t>(n);
}

std::span<const std::byte> Decoder::bytes() {
    const auto n = count();
    return raw(n);
}

std::string Decoder::string() {
    const auto v = bytes();
    return {reinterpret_cast<const char*>(v.data()), v.size()};
}

void Decoder::finish() const {
    if (!bytes_.empty()) throw std::invalid_argument("trailing distributed message data");
}

Bytes encode_header(const Message& message) {
    Encoder out;
    out.u32(static_cast<std::uint32_t>(message.kind));
    out.u64(message.epoch);
    out.u64(message.payload.size());
    return std::move(out).take();
}

Header decode_header(std::span<const std::byte> data) {
    Decoder in(data);
    Header out{static_cast<Kind>(in.u32()), in.u64(), in.u64()};
    in.finish();
    return out;
}

void encode(Encoder& out, const ExecutionTopology& t) {
    out.u64(t.plan);
    out.u32(t.output_rank);
    out.u64(t.ranks.size());
    for (const auto& r : t.ranks) {
        out.u32(r.rank);
        out.u32(r.node);
        out.u32(r.local_rank);
        out.i32(r.device);
        for (const auto c : {r.tp, r.ep, r.dp, r.pp}) {
            out.u32(c.rank);
            out.u32(c.size);
        }
    }
    out.u64(t.groups.size());
    for (const auto& g : t.groups) {
        out.u32(g.id);
        out.u64(g.members.size());
        for (auto r : g.members) out.u32(r);
    }
    out.u64(t.roles.size());
    for (const auto& role : t.roles) {
        out.string(role.role);
        out.u32(role.group);
    }
}

ExecutionTopology decode_topology(Decoder& in) {
    ExecutionTopology out;
    out.plan        = in.u64();
    out.output_rank = in.u32();
    out.ranks.resize(in.count(48));
    for (auto& r : out.ranks) {
        r.rank       = in.u32();
        r.node       = in.u32();
        r.local_rank = in.u32();
        r.device     = in.i32();
        for (auto* c : {&r.tp, &r.ep, &r.dp, &r.pp}) {
            c->rank = in.u32();
            c->size = in.u32();
        }
    }
    out.groups.resize(in.count(12));
    for (auto& g : out.groups) {
        g.id = in.u32();
        g.members.resize(in.count(4));
        for (auto& r : g.members) r = in.u32();
    }
    out.roles.resize(in.count(12));
    for (auto& r : out.roles) {
        r.role  = in.string();
        r.group = in.u32();
    }
    return out;
}

} // namespace ninfer::distributed::wire
