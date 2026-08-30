#include "digest.h"

#include <array>
#include <bit>
#include <cstddef>
#include <iomanip>
#include <sstream>

namespace ninfer::perplexity {
namespace {

constexpr std::array<std::uint32_t, 64> kRound{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

void transform(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
        words[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24U) |
                   (static_cast<std::uint32_t>(block[4 * i + 1]) << 16U) |
                   (static_cast<std::uint32_t>(block[4 * i + 2]) << 8U) |
                   static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
        const std::uint32_t s0 =
            std::rotr(words[i - 15], 7) ^ std::rotr(words[i - 15], 18) ^ (words[i - 15] >> 3U);
        const std::uint32_t s1 =
            std::rotr(words[i - 2], 17) ^ std::rotr(words[i - 2], 19) ^ (words[i - 2] >> 10U);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::uint32_t s1       = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const std::uint32_t choice   = (e & f) ^ (~e & g);
        const std::uint32_t t1       = h + s1 + choice + kRound[i] + words[i];
        const std::uint32_t s0       = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2       = s0 + majority;
        h                            = g;
        g                            = f;
        f                            = e;
        e                            = d + t1;
        d                            = c;
        c                            = b;
        b                            = a;
        a                            = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

} // namespace

Sha256Digest sha256(std::span<const std::uint8_t> input) {
    std::array<std::uint32_t, 8> state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                       0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::size_t offset = 0;
    while (input.size() - offset >= 64) {
        transform(state, input.data() + offset);
        offset += 64;
    }
    std::array<std::uint8_t, 128> tail{};
    const std::size_t remaining = input.size() - offset;
    for (std::size_t i = 0; i < remaining; ++i) { tail[i] = input[offset + i]; }
    tail[remaining]          = 0x80U;
    const std::size_t blocks = remaining < 56 ? 1 : 2;
    const std::uint64_t bits = static_cast<std::uint64_t>(input.size()) * 8U;
    for (std::size_t i = 0; i < 8; ++i) {
        tail[blocks * 64 - 1 - i] = static_cast<std::uint8_t>(bits >> (8U * i));
    }
    for (std::size_t block = 0; block < blocks; ++block) {
        transform(state, tail.data() + block * 64);
    }
    Sha256Digest out{};
    for (std::size_t i = 0; i < state.size(); ++i) {
        out[4 * i]     = static_cast<std::uint8_t>(state[i] >> 24U);
        out[4 * i + 1] = static_cast<std::uint8_t>(state[i] >> 16U);
        out[4 * i + 2] = static_cast<std::uint8_t>(state[i] >> 8U);
        out[4 * i + 3] = static_cast<std::uint8_t>(state[i]);
    }
    return out;
}

Sha256Digest sha256(std::string_view input) {
    return sha256(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(input.data()),
                                                input.size()));
}

std::string sha256_hex(const Sha256Digest& digest) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const std::uint8_t byte : digest) { out << std::setw(2) << static_cast<int>(byte); }
    return out.str();
}

} // namespace ninfer::perplexity
