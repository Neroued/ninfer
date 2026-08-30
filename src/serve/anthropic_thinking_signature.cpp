#include "serve/anthropic_thinking_signature.h"

#include <sys/random.h>

#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace ninfer::serve {
namespace {

using Digest = std::array<std::uint8_t, 32>;

constexpr std::string_view kSignaturePrefix = "sig_ninfer_v1_";
constexpr std::string_view kDomain          = "ninfer.anthropic.thinking.v1";
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

std::uint32_t load_be32(const std::uint8_t* bytes) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
}

void store_be32(std::uint32_t value, std::uint8_t* bytes) {
    bytes[0] = static_cast<std::uint8_t>(value >> 24U);
    bytes[1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[3] = static_cast<std::uint8_t>(value);
}

void process_block(std::array<std::uint32_t, 8>& state, const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
        words[index] = load_be32(block + 4U * index);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
        const std::uint32_t s0 = std::rotr(words[index - 15], 7) ^
                                 std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3U);
        const std::uint32_t s1 = std::rotr(words[index - 2], 17) ^ std::rotr(words[index - 2], 19) ^
                                 (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
        const std::uint32_t sum1     = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const std::uint32_t choose   = (e & f) ^ (~e & g);
        const std::uint32_t t1       = h + sum1 + choose + kRound[index] + words[index];
        const std::uint32_t sum0     = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2       = sum0 + majority;
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

Digest sha256(std::span<const std::uint8_t> input) {
    if (input.size() > std::numeric_limits<std::uint64_t>::max() / 8ULL) {
        throw std::invalid_argument("Thinking signature payload is too large");
    }
    std::array<std::uint32_t, 8> state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                       0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::size_t offset = 0;
    while (input.size() - offset >= 64) {
        process_block(state, input.data() + offset);
        offset += 64;
    }

    std::array<std::uint8_t, 128> tail{};
    const std::size_t remaining = input.size() - offset;
    for (std::size_t index = 0; index < remaining; ++index) { tail[index] = input[offset + index]; }
    tail[remaining]             = 0x80U;
    const std::size_t tail_size = remaining < 56 ? 64 : 128;
    const std::uint64_t bits    = static_cast<std::uint64_t>(input.size()) * 8ULL;
    for (std::size_t index = 0; index < 8; ++index) {
        tail[tail_size - 1U - index] = static_cast<std::uint8_t>(bits >> (8U * index));
    }
    process_block(state, tail.data());
    if (tail_size == 128) { process_block(state, tail.data() + 64); }

    Digest result{};
    for (std::size_t index = 0; index < state.size(); ++index) {
        store_be32(state[index], result.data() + 4U * index);
    }
    return result;
}

void append_be64(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        output.push_back(static_cast<std::uint8_t>(value >> static_cast<unsigned int>(shift)));
    }
}

Digest thinking_mac(const AnthropicThinkingSigner::Key& key, std::string_view thinking,
                    std::size_t block_index) {
    constexpr std::size_t kBlockBytes = 64;
    std::vector<std::uint8_t> inner;
    inner.reserve(kBlockBytes + kDomain.size() + 1U + 16U + thinking.size());
    for (std::size_t index = 0; index < kBlockBytes; ++index) {
        const std::uint8_t value = index < key.size() ? key[index] : 0;
        inner.push_back(static_cast<std::uint8_t>(value ^ 0x36U));
    }
    inner.insert(inner.end(), kDomain.begin(), kDomain.end());
    inner.push_back(0);
    append_be64(inner, static_cast<std::uint64_t>(block_index));
    append_be64(inner, static_cast<std::uint64_t>(thinking.size()));
    inner.insert(inner.end(), thinking.begin(), thinking.end());
    const Digest inner_digest = sha256(inner);

    std::array<std::uint8_t, kBlockBytes + 32U> outer{};
    for (std::size_t index = 0; index < kBlockBytes; ++index) {
        const std::uint8_t value = index < key.size() ? key[index] : 0;
        outer[index]             = static_cast<std::uint8_t>(value ^ 0x5cU);
    }
    for (std::size_t index = 0; index < inner_digest.size(); ++index) {
        outer[kBlockBytes + index] = inner_digest[index];
    }
    return sha256(outer);
}

std::string hex_digest(const Digest& digest) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string result(digest.size() * 2U, '\0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        result[2U * index]     = kHex[digest[index] >> 4U];
        result[2U * index + 1] = kHex[digest[index] & 0x0fU];
    }
    return result;
}

int hex_value(char value) {
    if (value >= '0' && value <= '9') { return value - '0'; }
    if (value >= 'a' && value <= 'f') { return value - 'a' + 10; }
    return -1;
}

bool decode_digest(std::string_view encoded, Digest& digest) {
    if (encoded.size() != digest.size() * 2U) { return false; }
    for (std::size_t index = 0; index < digest.size(); ++index) {
        const int high = hex_value(encoded[2U * index]);
        const int low  = hex_value(encoded[2U * index + 1]);
        if (high < 0 || low < 0) { return false; }
        digest[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return true;
}

AnthropicThinkingSigner::Key random_key() {
    AnthropicThinkingSigner::Key key{};
    std::size_t offset = 0;
    while (offset < key.size()) {
        const ssize_t count = ::getrandom(key.data() + offset, key.size() - offset, 0);
        if (count < 0) {
            if (errno == EINTR) { continue; }
            throw std::system_error(errno, std::generic_category(),
                                    "failed to initialize Anthropic Thinking signer");
        }
        if (count == 0) {
            throw std::runtime_error("operating-system random source returned no key bytes");
        }
        offset += static_cast<std::size_t>(count);
    }
    return key;
}

} // namespace

AnthropicThinkingSigner::AnthropicThinkingSigner() : key_(random_key()) {}

AnthropicThinkingSigner::AnthropicThinkingSigner(Key key) : key_(key) {}

std::string AnthropicThinkingSigner::sign(std::string_view thinking,
                                          std::size_t block_index) const {
    return std::string(kSignaturePrefix) + hex_digest(thinking_mac(key_, thinking, block_index));
}

bool AnthropicThinkingSigner::verify(std::string_view thinking, std::size_t block_index,
                                     std::string_view signature) const {
    if (!signature.starts_with(kSignaturePrefix)) { return false; }
    Digest supplied{};
    if (!decode_digest(signature.substr(kSignaturePrefix.size()), supplied)) { return false; }
    const Digest expected   = thinking_mac(key_, thinking, block_index);
    std::uint8_t difference = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        difference |= static_cast<std::uint8_t>(expected[index] ^ supplied[index]);
    }
    return difference == 0;
}

} // namespace ninfer::serve
